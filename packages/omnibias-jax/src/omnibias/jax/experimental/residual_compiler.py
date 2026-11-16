# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Plan the residual Gram, Gauss-Newton step or NTK diagonal of a PDE loss.

``compile_residual(model, blocks, quantity=...)`` is :func:`~omnibias.jax.experimental.compile_operator` for training:
the blocks are ``(x, Residual)`` pairs whose residuals declare which derivative quantities they read, so the planner
knows which engines can serve them (the first-layer projected streams, the full-gradient streams, or explicit
autodiff rows) and ranks the feasible ones by predicted cost.  The residual functions and the points are the same
objects the engines take; what changes is that the choice is made from the descriptor, recorded in
:meth:`CompiledResidual.explain` and refused with a reason where an engine cannot serve the demand.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any, cast

import numpy as np
from omnibias.core.demand import Demand
from omnibias.core.planner import (
    FIND_TOP_K,
    CostModel,
    Descriptor,
    OperatorRequest,
    Plan,
    PlanCache,
    Refusal,
    select,
)
from omnibias.core.precision import PrecisionPolicy, request_order, resolve_precision
from omnibias.jax.experimental.chain import describe_layers
from omnibias.jax.experimental.coldcost import (
    MemoryBudgetExceeded,
    PolicyChoice,
    StagedProgram,
    compilation_cache_dir,
    is_traced,
    resolve_compile_policy,
)
from omnibias.jax.experimental.compensated import exact_gram
from omnibias.jax.experimental.compiler import (
    _DEFAULT_CACHE,
    ChainMatch,
    _block,
    _default_budget,
    _device_description,
    _is_layer_model,
    _metric_of,
    _Setup,
    _setup_callable,
    _setup_layers,
    _versions,
    match_dense_chain,
)
from omnibias.jax.experimental.conditioning import (
    ConditioningPolicy,
    condition_report,
    damped_solve,
    floored_damping,
    resolve_conditioning,
)
from omnibias.jax.experimental.demand import Quantities, Residual
from omnibias.jax.experimental.projected_gram import (
    projected_residual_gauss_newton,
    projected_residual_gram,
    projected_residual_ntk_diagonal,
)
from omnibias.jax.experimental.residual_gram import (
    deep_residual_gauss_newton,
    deep_residual_gram,
    deep_residual_ntk_diagonal,
)

import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree

Array = jax.Array

__all__ = ["CompiledResidual", "compile_residual"]

_QUANTITIES = ("gram", "gauss_newton", "ntk_diagonal")


@dataclass(frozen=True)
class _Block:
    x: Array
    residual: Residual
    order: int | None


def _normalise_blocks(blocks: Sequence[Any]) -> list[_Block]:
    out: list[_Block] = []
    for i, block in enumerate(blocks):
        if len(block) not in (2, 3):
            raise ValueError(f"block {i} must be (x, residual) or (x, residual, order)")
        x, residual = jnp.asarray(block[0]), block[1]
        if not isinstance(residual, Residual):
            raise TypeError(
                f"block {i}: the residual must be a Residual declaring the quantities it reads; wrap an old "
                "callback with legacy_full_gradient or legacy_projected"
            )
        if x.ndim != 2:
            raise ValueError(f"block {i}: x must have shape (B, D)")
        out.append(_Block(x, residual, None if len(block) == 2 else int(block[2])))
    if not out:
        raise ValueError("at least one block is required")
    return out


def _rows_per_point(block: _Block, channels: int, dim: int, q: int) -> int:
    """Residual outputs per point, found by tracing the residual on abstract quantities."""
    B = int(block.x.shape[0])
    dt = block.x.dtype

    def spec(*shape: int) -> Array:
        return cast(Array, jax.ShapeDtypeStruct(shape, dt))  # type: ignore[no-untyped-call,unused-ignore]

    want = block.residual.demand.quantities
    fields: dict[str, Any] = {"u": spec(B, channels), "x": spec(B, dim)}
    if "grad" in want:
        fields["grad"] = spec(B, dim, channels)
    for name in ("grad_sq", "lap", "bilap"):
        if name in want:
            fields[name] = spec(B, channels)
    if "directional" in want and q:
        fields["directional"] = spec(B, q, channels)
    try:
        out = jax.eval_shape(lambda qs: block.residual(qs), Quantities(**fields))
    except Exception:  # noqa: BLE001 - a residual that cannot be traced abstractly is counted as one row per point
        return 1
    return int(out.shape[1]) if len(out.shape) == 2 else 1


@dataclass(frozen=True)
class _RCtx:
    quantity: str
    blocks: tuple[_Block, ...]
    fn: Callable[[Any, Array], Array]
    layers: Callable[[Any], list[tuple[Array, Array | None, Any]]] | None
    to_params: (
        Callable[[Any, Any], Any] | None
    )  # (layer-convention step, params) -> step in the params structure
    coordinates: np.ndarray | None
    diffusion: Any
    directions: Any
    allow_ae: bool
    strategy: str
    row_chunk: int
    dim: int
    policy: ConditioningPolicy
    gram_precision: str = "standard"
    compute_dtype: str | None = None
    precision: PrecisionPolicy = PrecisionPolicy()

    def xs_of(self, xs: Sequence[Array] | None) -> list[Array]:
        if xs is None:
            return [b.x for b in self.blocks]
        if len(xs) != len(self.blocks):
            raise ValueError(f"expected {len(self.blocks)} point blocks, got {len(xs)}")
        return [jnp.asarray(x) for x in xs]

    def engine_blocks(self, xs: Sequence[Array]) -> list[tuple[Any, ...]]:
        out: list[tuple[Any, ...]] = []
        for b, x in zip(self.blocks, xs, strict=True):
            out.append((x, b.residual) if b.order is None else (x, b.residual, b.order))
        return out


Runner = Callable[[Any, Sequence[Array], Any], tuple[Any, ...]]
_CONDITION = jax.jit(condition_report, static_argnums=(2,))


def _structured(ctx: _RCtx, projected: bool) -> Runner:
    def run(params: Any, xs: Sequence[Array], damping: Any) -> tuple[Any, ...]:
        assert ctx.layers is not None
        layers = ctx.layers(params)
        blocks = ctx.engine_blocks(xs)
        kw: dict[str, Any] = {
            "coordinates": ctx.coordinates,
            "diffusion": ctx.diffusion,
            "strategy": ctx.strategy,
            "allow_almost_everywhere": ctx.allow_ae,
        }
        if projected:
            kw["directions"] = (
                None if ctx.directions is None else jnp.asarray(ctx.directions, xs[0].dtype)
            )
        gram_engine = cast(
            "Callable[..., Any]", projected_residual_gram if projected else deep_residual_gram
        )
        ntk_engine = cast(
            "Callable[..., Any]",
            projected_residual_ntk_diagonal if projected else deep_residual_ntk_diagonal,
        )
        engine = cast(
            "Callable[..., Any]",
            projected_residual_gauss_newton if projected else deep_residual_gauss_newton,
        )
        if ctx.quantity == "gram":
            return tuple(gram_engine(blocks, layers, **kw))
        if ctx.quantity == "ntk_diagonal":
            return tuple(ntk_engine(blocks, layers, **kw))
        if ctx.policy.mode == "floor":
            kw["conditioning"] = ctx.policy
        if not projected and jnp.ndim(damping) == 1:
            # the full-gradient engine takes a scalar: map over the values (r and K do not depend on damping, so
            # they stay unbatched and are formed once)
            r, K, step = jax.vmap(
                lambda d: engine(blocks, layers, d, **kw), out_axes=(None, None, 0)
            )(jnp.asarray(damping))
        else:
            r, K, step = engine(blocks, layers, damping, **kw)
        if ctx.to_params is not None:
            step = ctx.to_params(step, params)
        return r, K, step

    return run


def _autodiff_quantities(ctx: _RCtx, params: Any, block: _Block, x: Array) -> Quantities:
    """The derivative quantities of ``block`` by autodiff (the generic, nested-graph route)."""
    want = block.residual.demand.quantities
    D = ctx.dim
    idx = np.arange(D) if ctx.coordinates is None else np.asarray(ctx.coordinates)
    m = len(idx)
    if ctx.diffusion is None:
        S = np.eye(m)
    else:
        d = np.asarray(ctx.diffusion)
        S = np.diag(d) if d.ndim == 1 else d
    V = np.zeros((D, S.shape[1]))
    V[idx] = S
    A = jnp.asarray(V @ V.T, x.dtype)

    def point(z: Array) -> Array:
        return jnp.reshape(ctx.fn(params, z[None, :]), (-1,))

    u = jax.vmap(point)(x)
    fields: dict[str, Any] = {"u": u, "x": x}
    if want & {"grad", "grad_sq", "directional"}:
        J = jax.vmap(jax.jacrev(point))(x)  # (B, C, D)
        if "grad" in want:
            fields["grad"] = jnp.swapaxes(J, 1, 2)
        if "grad_sq" in want:
            fields["grad_sq"] = jnp.sum(
                jnp.einsum("bcd,dr->bcr", J, jnp.asarray(V, x.dtype)) ** 2, axis=-1
            )
        if "directional" in want and ctx.directions is not None:
            fields["directional"] = jnp.einsum(
                "bcd,dq->bqc", J, jnp.asarray(ctx.directions, x.dtype)
            )

    def laplace(g: Callable[[Array], Array]) -> Callable[[Array], Array]:
        def h(z: Array) -> Array:
            return jnp.einsum("...ij,ij->...", jax.hessian(g)(z), A)

        return h

    if "lap" in want:
        fields["lap"] = jax.vmap(laplace(point))(x)
    if "bilap" in want:
        fields["bilap"] = jax.vmap(laplace(laplace(point)))(x)
    return Quantities(**fields)


def _autodiff_rows(ctx: _RCtx) -> Runner:
    def run(params: Any, xs: Sequence[Array], damping: Any) -> tuple[Any, ...]:
        def rvec(p: Any) -> Array:
            parts = []
            for b, x in zip(ctx.blocks, xs, strict=True):
                parts.append(jnp.reshape(b.residual(_autodiff_quantities(ctx, p, b, x)), (-1,)))
            return jnp.concatenate(parts)

        r, pullback = jax.vjp(rvec, params)
        n = r.shape[0]
        flat0, unravel = ravel_pytree(params)
        chunk = max(1, min(ctx.row_chunk, n))
        pad = (-n) % chunk
        eye_idx = jnp.arange(n + pad).reshape(-1, chunk)

        def rows(idx: Array) -> Array:
            e = jax.nn.one_hot(idx, n, dtype=r.dtype)  # rows beyond n are zero cotangents
            (g,) = jax.vmap(pullback)(e)
            return jax.vmap(lambda t: ravel_pytree(t)[0])(g)

        J = jax.lax.map(rows, eye_idx).reshape(-1, flat0.shape[0])[:n]
        if ctx.quantity == "ntk_diagonal":
            return r, jnp.sum(J * J, axis=1)
        K = exact_gram(J) if ctx.gram_precision == "compensated" else J @ J.T
        if ctx.quantity == "gram":
            return r, K
        d = floored_damping(K, damping, ctx.policy)
        if d.ndim == 0:
            step = unravel(J.T @ damped_solve(K, r, d)[0])
        else:
            evals, vecs = jnp.linalg.eigh(K)
            proj = vecs.T @ r
            coeffs = vecs @ (proj[None, :] / (evals[None, :] + d[:, None])).T  # (n, len(d))
            step = jax.vmap(lambda c: unravel(J.T @ c), in_axes=1)(coeffs)
        return r, K, step

    return run


_RUNNERS: dict[str, Callable[[_RCtx], Runner]] = {
    "projected_streams": lambda c: _structured(c, True),
    "full_streams": lambda c: _structured(c, False),
    "autodiff_rows": _autodiff_rows,
}


class CompiledResidual:
    """A planned residual program.  ``op(params, xs=None, damping=None)``.

    Returns ``(r, K)`` for ``quantity="gram"``, ``(r, diag)`` for ``"ntk_diagonal"`` and ``(r, K, step)`` for
    ``"gauss_newton"``, where ``step`` has the structure of ``params`` (``theta - step`` is the damped update).  ``xs``
    optionally replaces the points of the blocks (same shapes) at call time; residuals that close over per-point data
    keep that data, so it must still match the rows.
    """

    def __init__(
        self,
        ctx: _RCtx,
        *,
        plan: Plan,
        matched: bool,
        fallback_reason: str | None,
        chain: Mapping[str, Any] | None,
        request: OperatorRequest,
        prepare: Callable[[Any], Any],
        jit: bool,
        policy: PolicyChoice | None = None,
        alternatives: Sequence[str] = (),
    ) -> None:
        self._ctx = ctx
        self.precision_notes: list[str] = []
        self.plan = plan
        self._jit_enabled = jit
        self._alternatives = list(alternatives)
        self.matched = matched
        self.fallback_reason = fallback_reason
        self._chain = None if chain is None else dict(chain)
        self.request = request
        self._prepare = prepare
        self.policy = (
            policy
            if policy is not None
            else resolve_compile_policy(
                "default",
                platform="cpu",
                predicted_seconds=None,
                cold_seconds=None,
                expected_uses=1,
            )
        )
        self._rebuild()
        self.first_call_seconds: float | None = None
        self._last_condition: Any = None
        self.executor_failures: list[Refusal] = []
        self.descriptor: Descriptor | None = (
            None  # the planner's problem description (None for an unmatched model)
        )

    def _rebuild(self) -> None:
        self._run = _RUNNERS[self.plan.algorithm](self._ctx)
        self._staged: StagedProgram | None = (
            StagedProgram(
                self._call_impl,
                options=self.policy.options,
                memory_budget=self.plan.memory_budget,
                retry=self.policy.requested == "auto",
            )
            if self._jit_enabled
            else None
        )
        self._apply: Callable[..., Any] = self._call_impl
        if self._staged is not None:
            self._apply = self._staged

    def _advance(self, err: MemoryBudgetExceeded) -> bool:
        """Drop the current program because its compiled size exceeds the budget; ``False`` when none is left."""
        self.executor_failures.append(
            Refusal(self.plan.algorithm, f"refused before running: {err}")
        )
        if not self._alternatives:
            return False
        name = self._alternatives.pop(0)
        self.plan = replace(self.plan, algorithm=name, mode=self.plan.mode + "+memory_retry")
        self._rebuild()
        return True

    def _guarded(self, call: Callable[[], Any]) -> Any:
        """Run ``call``; a program refused for its compiled memory is replaced by the next feasible candidate."""
        while True:
            try:
                return call()
            except MemoryBudgetExceeded as err:
                if not self._advance(err):
                    err.add_note(
                        "every candidate was refused: "
                        + "; ".join(f"{f.algorithm}: {f.reason}" for f in self.executor_failures)
                    )
                    raise

    @property
    def algorithm(self) -> str:
        return self.plan.algorithm

    @property
    def last_conditioning(self) -> dict[str, Any] | None:
        """Conditioning diagnostics of the most recent Gauss-Newton call (fetches device values), or ``None``."""
        return None if self._last_condition is None else self._last_condition.to_config()

    def _call_impl(self, params: Any, xs: tuple[Array, ...], damping: Any) -> tuple[Any, ...]:
        name = self._ctx.precision.jax_name
        if name is None:
            return self._call_wide(params, xs, damping)
        with jax.default_matmul_precision(name):
            return self._call_wide(params, xs, damping)

    def _call_wide(self, params: Any, xs: tuple[Array, ...], damping: Any) -> tuple[Any, ...]:
        if self._ctx.compute_dtype is None:
            return self._run(params, xs, damping)
        wide = jnp.dtype(self._ctx.compute_dtype)
        lift = lambda a: a.astype(wide) if jnp.issubdtype(jnp.result_type(a), jnp.floating) else a  # noqa: E731
        ref = jax.tree_util.tree_leaves(xs)[0].dtype
        out = self._run(
            jax.tree_util.tree_map(lift, params),
            jax.tree_util.tree_map(lift, xs),
            None if damping is None else jax.tree_util.tree_map(lift, damping),
        )
        return jax.tree_util.tree_map(lambda a: a.astype(ref) if a.dtype == wide else a, out)

    def warmup(
        self, params: Any, xs: Sequence[Array] | None = None, damping: Any = None
    ) -> dict[str, Any]:
        """Build the executable for these arguments now, without evaluating; returns its stage times.

        Changing batch sizes recompile; pad blocks with :func:`~omnibias.jax.experimental.pad_block` and
        :func:`~omnibias.jax.experimental.masked_residual` so that they share an executable.
        """
        if self._staged is None:
            raise ValueError("this program was built with jit=False; there is nothing to compile")
        if self._ctx.quantity == "gauss_newton" and damping is None:
            raise ValueError("quantity='gauss_newton' needs `damping`")
        pts = tuple(self._ctx.xs_of(xs))
        prepared = self._prepare(params)

        def compile_now() -> dict[str, Any]:
            staged = self._staged
            assert staged is not None
            return staged.compile(prepared, pts, damping).to_config()

        result: dict[str, Any] = self._guarded(compile_now)
        return result

    def __call__(
        self, params: Any, xs: Sequence[Array] | None = None, damping: Any = None
    ) -> tuple[Any, ...]:
        if self._ctx.quantity == "gauss_newton" and damping is None:
            raise ValueError("quantity='gauss_newton' needs `damping`")
        pts = tuple(self._ctx.xs_of(xs))
        first = self.first_call_seconds is None
        t0 = time.perf_counter()
        prepared = self._prepare(params)
        out: tuple[Any, ...] = self._guarded(lambda: self._apply(prepared, pts, damping))
        if first and not is_traced(out):
            _block(out)
            self.first_call_seconds = time.perf_counter() - t0
        if (
            self._ctx.quantity == "gauss_newton"
            and self._ctx.policy.mode != "off"
            and not is_traced(out)
        ):
            self._last_condition = _CONDITION(out[1], damping, self._ctx.policy)
        return out

    def _conditioning_entry(self) -> dict[str, Any]:
        pol = self._ctx.policy
        base: dict[str, Any] = {"mode": pol.mode, "step_tolerance": pol.step_tolerance}
        if self._ctx.quantity != "gauss_newton" or pol.mode == "off":
            return {**base, "status": "not applicable"}
        last = self.last_conditioning
        return {**base, "status": "no call yet"} if last is None else last

    def explain(self) -> dict[str, Any]:
        out = self.plan.explain()
        out.update(
            {
                "backend": "jax",
                "matched_patterns": ["dense_chain"] if self.matched else [],
                "fallback_reason": self.fallback_reason,
                "chain": self._chain,
                "request": self.request.to_config(),
                "first_call_seconds": self.first_call_seconds,
                "executor_failures": [
                    {"algorithm": r.algorithm, "reason": r.reason} for r in self.executor_failures
                ],
                "conditioning": self._conditioning_entry(),
                "gram_precision": self._ctx.gram_precision,
                "compute_dtype": self._ctx.compute_dtype,
                "precision": {**self._ctx.precision.to_config(), "notes": self.precision_notes},
                "cold_cost": {
                    "predicted_seconds": self.plan.compile_seconds,
                    "compile_policy": self.policy.to_config(),
                    "persistent_cache_dir": compilation_cache_dir(),
                    "executables": 0 if self._staged is None else self._staged.signatures(),
                    "stages": [] if self._staged is None else self._staged.report(),
                },
            }
        )
        return out


def _step_to_params(match: ChainMatch, template: Any) -> Callable[[Any, Any], Any]:
    """Map a layer-convention step ``[(dW, db)]`` onto the structure of the model's parameters."""
    leaves0, treedef = jax.tree_util.tree_flatten(template)

    def convert(step: Any, params: Any) -> Any:
        leaves, _ = jax.tree_util.tree_flatten(params)
        out = [jnp.zeros_like(leaf) for leaf in leaves]
        for lm, (dW, db) in zip(match.layers, step, strict=True):
            leaf_shape = jnp.shape(leaves[lm.weight])
            if lm.weight_form == "vector":
                dW = dW[..., 0, :]
            elif lm.weight_form == "in_out":
                dW = jnp.swapaxes(dW, -1, -2)
            out[lm.weight] = jnp.reshape(dW, dW.shape[: dW.ndim - len(leaf_shape)] + leaf_shape)
            if lm.bias is not None and db is not None:
                bias_shape = jnp.shape(leaves[lm.bias])
                out[lm.bias] = jnp.reshape(db, db.shape[: db.ndim - 1] + bias_shape)
        return jax.tree_util.tree_unflatten(treedef, out)

    del leaves0
    return convert


def compile_residual(
    model: Any,
    blocks: Sequence[Any],
    *,
    example_params: Any = None,
    quantity: str = "gauss_newton",
    coordinates: Sequence[int] | np.ndarray | None = None,
    diffusion: Any = None,
    directions: Any = None,
    strategy: str = "auto",
    mode: str = "heuristic",
    memory_budget: float | None = None,
    expected_uses: int = 1,
    force: str | None = None,
    allow_almost_everywhere: bool = False,
    row_chunk: int = 64,
    device: Any | None = None,
    cache: PlanCache | str | None = None,
    jit: bool = True,
    cost_model: CostModel | None = None,
    compile_policy: str = "default",
    conditioning: str | ConditioningPolicy | None = "report",
    gram_precision: str = "standard",
    compute_dtype: str | None = None,
    precision: str | PrecisionPolicy | None = None,
    allow_low_precision: bool = False,
    top_k: int | None = FIND_TOP_K,
) -> CompiledResidual:
    r"""Plan the residual Gram ``K = J J^T``, its damped Gauss-Newton step or its diagonal.

    Parameters
    ----------
    model : callable ``model(params, x)`` (with ``example_params``), a layer list or an object with ``layer_specs()``.
    blocks : sequence of ``(x, Residual)`` or ``(x, Residual, order)``; ``x`` of shape ``(B, D)``.  Each residual
        declares the quantities it reads, so the planner knows which engines can serve it.
    quantity : ``"gauss_newton"`` (step ``J^T (K + damping I)^{-1} r``), ``"gram"`` or ``"ntk_diagonal"``.
    coordinates, diffusion, directions : as for :func:`compile_operator`.
    strategy : the engines' per-layer choice between stream identities and explicit rows (``"auto"``).
    mode, memory_budget, expected_uses, force, cache, cost_model, compile_policy : as for :func:`compile_operator`.
    top_k : how many candidates ``mode="find"`` times: ``2`` (default) or ``None`` for every feasible program, as for
        :func:`compile_operator`.  A program that fails when timed is listed in ``explain()["executor_failures"]`` and
        ranked last; it no longer raises out of ``compile_residual``.
    row_chunk : rows of the explicit Jacobian formed at once by the autodiff-rows program.
    conditioning : ``"report"`` (default) estimates the condition of ``K + damping I`` after each Gauss-Newton call and
        stores it in ``explain()["conditioning"]`` without changing any result; ``"floor"`` also raises the damping to
        the value at which the step is reproducible to the tolerance of a :class:`~omnibias.jax.experimental.conditioning.ConditioningPolicy`;
        ``"off"`` does nothing.
    gram_precision : ``"compensated"`` assembles ``K`` from explicit Jacobian rows by an error-free product
        (:func:`~omnibias.jax.experimental.compensated.exact_matmul`), so the step is limited by the rows and not by the
        assembly; it uses the autodiff-rows program (float64 only, ``O(n P)`` memory). ``"standard"`` is the default.
    compute_dtype : ``"float64"`` evaluates a float32 problem in float64 and returns results in the input dtype
        (requires ``jax_enable_x64``); ``None`` computes in the dtype of the inputs.
    precision, allow_low_precision : the matrix-product precision (``"highest"`` by default, ``"tf32"`` or ``"default"``)
        and whether half-precision inputs are admitted for residuals of derivative order two and above; see
        :func:`~omnibias.jax.experimental.compile_operator`.

    Raises
    ------
    ValueError
        If no program is feasible; the message lists every refusal reason.
    """
    if quantity not in _QUANTITIES:
        raise ValueError(f"quantity must be one of {_QUANTITIES}")
    norm = _normalise_blocks(blocks)
    first_x = norm[0].x
    if _is_layer_model(model):
        setup: _Setup = _setup_layers(model, first_x)
        params0 = setup.params
    else:
        if example_params is None:
            raise ValueError("a callable model needs example_params")
        setup = _setup_callable(model, (example_params, first_x), 0)
        params0 = example_params
    dim = int(first_x.shape[1])
    coords = None if coordinates is None else np.asarray(coordinates, dtype=int)
    dirs = None if directions is None else np.asarray(directions)
    union = Demand.of()
    for b in norm:
        union = union.union(b.residual.demand)
    if dirs is not None and "directional" not in union:
        union = union.union(Demand.of("directional"))
    channels = int(jnp.reshape(setup.fn(params0, first_x[:1]), (1, -1)).shape[1])
    q = 0 if dirs is None else int(dirs.shape[-1])
    rows = sum(int(b.x.shape[0]) * _rows_per_point(b, channels, dim, q) for b in norm)
    request = OperatorRequest(
        quantity,
        selected=None if coords is None else int(len(coords)),
        metric=_metric_of(diffusion),
        directions=q,
        allow_almost_everywhere=allow_almost_everywhere,
    )
    to_params: Callable[[Any, Any], Any] | None = None
    if setup.layers is not None and not _is_layer_model(model):
        match, _ = match_dense_chain(model, (example_params, first_x), params_argnum=0)
        assert match is not None
        to_params = _step_to_params(match, example_params)
    if gram_precision not in ("standard", "compensated"):
        raise ValueError("gram_precision must be 'standard' or 'compensated'")
    if gram_precision == "compensated":
        if force not in (None, "autodiff_rows"):
            raise ValueError(
                "gram_precision='compensated' uses the autodiff_rows program; it cannot be combined with force="
                + repr(force)
            )
        if first_x.dtype != jnp.float64 and compute_dtype != "float64":
            raise ValueError(
                "gram_precision='compensated' needs float64 (set compute_dtype='float64' or enable x64)"
            )
        force = "autodiff_rows"
    if compute_dtype not in (None, "float64"):
        raise ValueError("compute_dtype must be None or 'float64'")
    if compute_dtype == "float64" and not jax.config.read("jax_enable_x64"):
        raise ValueError("compute_dtype='float64' needs jax_enable_x64")
    policy = resolve_precision(precision, allow_low_precision)
    precision_notes = policy.check(
        jnp.float64 if compute_dtype == "float64" else first_x.dtype,
        request_order(quantity, demand_order=union.order),
        what=f"residual demanding {sorted(union.quantities)}",
    )
    ctx = _RCtx(
        quantity=quantity,
        blocks=tuple(norm),
        fn=setup.fn,
        layers=setup.layers,
        to_params=to_params,
        coordinates=coords,
        diffusion=diffusion,
        directions=dirs,
        allow_ae=allow_almost_everywhere,
        strategy=strategy,
        row_chunk=row_chunk,
        dim=dim,
        policy=resolve_conditioning(conditioning),
        gram_precision=gram_precision,
        compute_dtype=compute_dtype,
        precision=policy,
    )
    dev_class, dev = _device_description(device)
    budget = memory_budget if memory_budget is not None else _default_budget(dev)
    plan_cache = (
        cache
        if isinstance(cache, PlanCache)
        else PlanCache(cache)
        if isinstance(cache, str)
        else _DEFAULT_CACHE
    )
    versions = _versions()
    xs0 = tuple(b.x for b in norm)
    damping0 = jnp.asarray(1e-3, first_x.dtype) if quantity == "gauss_newton" else None

    find_failures: list[Refusal] = []

    def timer(name: str) -> float:
        """Median seconds of one call; a program that cannot compile or run is a failed measurement, not a failed plan."""
        try:
            compiled = jax.jit(_RUNNERS[name](ctx))
            _block(compiled(params0, xs0, damping0))
            times = []
            for _ in range(3):
                t0 = time.perf_counter()
                _block(compiled(params0, xs0, damping0))
                times.append(time.perf_counter() - t0)
            return float(np.median(times))
        except Exception as err:  # noqa: BLE001 - an exception here used to reach the caller of compile_residual
            find_failures.append(Refusal(name, f"find: {type(err).__name__}: {str(err)[:200]}"))
            return float("inf")  # may be transient (host memory): the plan is not cached

    chain_config: Mapping[str, Any] | None = None
    descriptor: Descriptor | None = None
    if setup.layers is not None:
        chain = describe_layers(setup.layers(params0))
        chain_config = chain.to_config()
        descriptor = Descriptor(
            request, chain, sum(int(b.x.shape[0]) for b in norm), demand=union, rows=rows
        )
        plan = select(
            descriptor,
            backend="jax",
            device=dev_class,
            cost_model=cost_model,
            memory_budget=budget,
            mode=mode,
            timer=timer,
            top_k=top_k,
            expected_uses=expected_uses,
            force=force,
            cache=plan_cache,
            versions=versions,
        )
        order = [plan.algorithm] + [
            c["name"] for c in plan.candidates if c["name"] != plan.algorithm
        ]
    else:
        if force not in (None, "autodiff_rows"):
            raise ValueError(
                f"cannot force {force!r} for a model that is not a dense chain; only autodiff_rows applies"
            )
        plan = Plan(
            descriptor="unmatched",
            operator=quantity,
            algorithm="autodiff_rows",
            mode="forced" if force else "fallback",
            device=dev_class,
            candidates=(
                {
                    "name": "autodiff_rows",
                    "seconds": None,
                    "peak_bytes": None,
                    "compile_seconds": None,
                    "generic": True,
                },
            ),
            refusals=(),
            memory_budget=budget,
            expected_uses=max(1, int(expected_uses)),
            versions={
                **versions,
                "cost_model": "none: unmatched model, generic executors are not priced",
            },
        )
        order = ["autodiff_rows"]
    if force is not None:
        order = [
            plan.algorithm
        ]  # a forced program that cannot run is an error, not a silent substitution
    failures: list[Refusal] = list(find_failures)
    chosen = None
    for name in order:
        try:
            run = _RUNNERS[name](ctx)
            jax.eval_shape(run, params0, xs0, damping0)
        except Exception as err:  # noqa: BLE001 - an engine that cannot trace this residual is a refusal
            failures.append(
                Refusal(
                    name,
                    f"program could not trace this problem: {type(err).__name__}: {str(err)[:200]}",
                )
            )
            continue
        chosen = name
        break
    if chosen is None:
        raise ValueError(
            "no program could run this problem: "
            + "; ".join(f"{f.algorithm}: {f.reason}" for f in failures)
        )
    if chosen != plan.algorithm:
        plan = replace(plan, algorithm=chosen, mode=plan.mode + "+retry")
    op = CompiledResidual(
        ctx,
        plan=plan,
        matched=setup.matched,
        fallback_reason=setup.fallback_reason,
        chain=chain_config,
        request=request,
        prepare=setup.prepare,
        jit=jit,
        alternatives=order[order.index(chosen) + 1 :],
        policy=resolve_compile_policy(
            compile_policy,
            platform=str(getattr(dev, "platform", "cpu")),
            predicted_seconds=plan.predicted_seconds,
            cold_seconds=plan.compile_seconds,
            expected_uses=max(1, int(expected_uses)),
        ),
    )
    op.executor_failures = failures
    op.descriptor = descriptor
    op.precision_notes = precision_notes
    return op
