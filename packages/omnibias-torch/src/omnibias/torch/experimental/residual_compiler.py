# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Plan the residual Gram, Gauss-Newton step or NTK diagonal of a PDE loss (PyTorch).

``compile_residual(model, blocks, quantity=...)`` is the training counterpart of
:func:`~omnibias.torch.experimental.planned.compile_planned_operator`: blocks are ``(x, Residual)`` pairs whose residuals
declare the quantities they read; the planner ranks the engines that can serve them. PyTorch implements ``full_streams``
(the stream identity of :mod:`omnibias.torch.experimental.residual_gram`) and ``autodiff_rows`` (explicit Jacobian rows
by ``torch.func.jacrev``, the generic route). The first-layer projected streams of the JAX backend are refused with that
reason. The planner, the refusal reasons and the keys of ``explain()`` are those of the JAX front end.

The model owns its parameters. The program returns values (not differentiable in the parameters); ``step`` is a list
aligned with :attr:`CompiledResidual.parameters`, and :meth:`CompiledResidual.apply_step` performs ``theta -= scale * step``.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any, cast

import numpy as np
from omnibias.core.chain import ChainDescriptor
from omnibias.core.demand import Demand, Quantities, Residual
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
from omnibias.torch.experimental._chain import describe_layers
from omnibias.torch.experimental.conditioning import (
    ConditioningPolicy,
    condition_report,
    damped_solve,
    resolve_conditioning,
)
from omnibias.torch.experimental.planned import (
    _DEFAULT_CACHE,
    _default_budget,
    _device_description,
    _is_layer_model,
    _metric_of,
    _numpy,
    _sync,
    _versions,
    chain_from_module,
    layers_of,
    matmul_precision,
)
from omnibias.torch.experimental.residual_gram import (
    deep_residual_gauss_newton,
    deep_residual_gn_diagonal,
    deep_residual_gram,
    deep_residual_ntk_diagonal,
    deep_residual_values,
    dual_step,
    gram_with_streams,
)

import torch
from torch import Tensor, nn

__all__ = ["CompiledResidual", "compile_residual"]

_QUANTITIES = ("gram", "gauss_newton", "ntk_diagonal")


@dataclass(frozen=True)
class _Block:
    x: Tensor
    residual: Residual
    order: int | None


def _normalise_blocks(blocks: Sequence[Any]) -> list[_Block]:
    out: list[_Block] = []
    for i, block in enumerate(blocks):
        if len(block) not in (2, 3):
            raise ValueError(f"block {i} must be (x, residual) or (x, residual, order)")
        x, residual = torch.as_tensor(block[0]), block[1]
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
    """Residual outputs per point, found by calling the residual on zero quantities of the block's shapes."""
    B = int(block.x.shape[0])
    x = block.x
    want = block.residual.demand.quantities

    def zeros(*shape: int) -> Tensor:
        return torch.zeros(shape, dtype=x.dtype, device=x.device)

    fields: dict[str, Any] = {"u": zeros(B, channels), "x": x}
    if "grad" in want:
        fields["grad"] = zeros(B, dim, channels)
    for name in ("grad_sq", "lap", "bilap"):
        if name in want:
            fields[name] = zeros(B, channels)
    if "directional" in want and q:
        fields["directional"] = zeros(B, q, channels)
    try:
        with torch.no_grad():
            out = torch.as_tensor(block.residual(Quantities(**fields)))
    except Exception:  # noqa: BLE001 - a residual that cannot be probed is counted as one row per point
        return 1
    return int(out.shape[1]) if out.ndim == 2 else 1


@dataclass(frozen=True)
class _RCtx:
    quantity: str
    blocks: tuple[_Block, ...]
    fn: Callable[[list[Tensor], Tensor], Tensor]  # (params, batched x) -> (B, C) or (B,)
    layers: Callable[[list[Tensor]], list[tuple[Tensor, Tensor | None, Any]]] | None
    coordinates: np.ndarray | None
    diffusion: Any
    directions: Any
    allow_ae: bool
    strategy: str
    row_chunk: int
    dim: int
    policy: ConditioningPolicy
    precision: PrecisionPolicy = PrecisionPolicy()

    def xs_of(self, xs: Sequence[Tensor] | None) -> list[Tensor]:
        if xs is None:
            return [b.x for b in self.blocks]
        if len(xs) != len(self.blocks):
            raise ValueError(f"expected {len(self.blocks)} point blocks, got {len(xs)}")
        return [torch.as_tensor(x) for x in xs]

    def engine_blocks(self, xs: Sequence[Tensor]) -> list[tuple[Any, ...]]:
        return [
            (x, b.residual) if b.order is None else (x, b.residual, b.order)
            for b, x in zip(self.blocks, xs, strict=True)
        ]


Runner = Callable[[list[Tensor], Sequence[Tensor], Any], tuple[Any, ...]]


def _flat_step(step: Sequence[tuple[Tensor, Tensor | None]]) -> list[Tensor]:
    out: list[Tensor] = []
    for dW, db in step:
        out.append(dW)
        if db is not None:
            out.append(db)
    return out


def _structured(ctx: _RCtx) -> Runner:
    def run(params: list[Tensor], xs: Sequence[Tensor], damping: Any) -> tuple[Any, ...]:
        assert ctx.layers is not None
        layers = ctx.layers(params)
        blocks = ctx.engine_blocks(xs)
        kw: dict[str, Any] = {
            "coordinates": ctx.coordinates,
            "diffusion": ctx.diffusion,
            "strategy": ctx.strategy,
            "allow_almost_everywhere": ctx.allow_ae,
        }
        if ctx.quantity == "gram":
            return tuple(deep_residual_gram(blocks, layers, **kw))
        if ctx.quantity == "ntk_diagonal":
            return tuple(deep_residual_ntk_diagonal(blocks, layers, **kw))
        conditioning = ctx.policy if ctx.policy.mode == "floor" else None
        values = torch.as_tensor(damping)
        if values.ndim == 0:
            r, K, step = deep_residual_gauss_newton(
                blocks, layers, values, conditioning=conditioning, **kw
            )
            return r, K, _flat_step(step)
        r, K, norm, data = gram_with_streams(
            blocks, layers, ctx.coordinates, ctx.diffusion, ctx.strategy, ctx.allow_ae
        )
        steps = []
        for d in values:
            alpha, _ = damped_solve(K, r, d, conditioning)
            steps.append(_flat_step(dual_step(alpha, norm, data, layers)))
        return r, K, [torch.stack([s[k] for s in steps]) for k in range(len(steps[0]))]

    return run


def _autodiff_quantities(
    ctx: _RCtx, params: Sequence[Tensor], block: _Block, x: Tensor
) -> Quantities:
    """The derivative quantities of ``block`` by autodiff (the generic, nested-graph route)."""
    want = block.residual.demand.quantities
    D = ctx.dim
    idx = np.arange(D) if ctx.coordinates is None else np.asarray(ctx.coordinates)
    m = len(idx)
    if ctx.diffusion is None:
        S = np.eye(m)
    else:
        d = _numpy(ctx.diffusion)
        S = np.diag(d) if d.ndim == 1 else d
    V = np.zeros((D, S.shape[1]))
    V[idx] = S
    Vt = torch.as_tensor(V, dtype=x.dtype, device=x.device)
    A = Vt @ Vt.T
    plist = list(params)

    def point(z: Tensor) -> Tensor:
        return ctx.fn(plist, z[None, :]).reshape(-1)

    u = torch.func.vmap(point)(x)
    fields: dict[str, Any] = {"u": u, "x": x}
    if want & {"grad", "grad_sq", "directional"}:
        J = torch.func.vmap(torch.func.jacrev(point))(x)  # (B, C, D)
        if "grad" in want:
            fields["grad"] = J.transpose(1, 2)
        if "grad_sq" in want:
            fields["grad_sq"] = torch.sum(torch.einsum("bcd,dr->bcr", J, Vt) ** 2, dim=-1)
        if "directional" in want and ctx.directions is not None:
            dirs = torch.as_tensor(ctx.directions, dtype=x.dtype, device=x.device)
            fields["directional"] = torch.einsum("bcd,dq->bqc", J, dirs)

    def laplace(g: Callable[[Tensor], Tensor]) -> Callable[[Tensor], Tensor]:
        def h(z: Tensor) -> Tensor:
            return torch.einsum("...ij,ij->...", torch.func.jacrev(torch.func.jacrev(g))(z), A)

        return h

    if "lap" in want:
        fields["lap"] = torch.func.vmap(laplace(point))(x)
    if "bilap" in want:
        fields["bilap"] = torch.func.vmap(laplace(laplace(point)))(x)
    return Quantities(**fields)


def _autodiff_rvec(ctx: _RCtx, xs: Sequence[Tensor]) -> Callable[..., Tensor]:
    def rvec(*ps: Tensor) -> Tensor:
        parts = [
            torch.as_tensor(b.residual(_autodiff_quantities(ctx, ps, b, x))).reshape(-1)
            for b, x in zip(ctx.blocks, xs, strict=True)
        ]
        return torch.cat(parts)

    return rvec


def _autodiff_jacobian(
    ctx: _RCtx, params: Sequence[Tensor], xs: Sequence[Tensor]
) -> tuple[Tensor, Tensor]:
    rvec = _autodiff_rvec(ctx, xs)
    detached = tuple(p.detach() for p in params)
    r = rvec(*detached).detach()
    n = r.shape[0]
    chunk = max(1, min(ctx.row_chunk, n))
    Js = torch.func.jacrev(rvec, argnums=tuple(range(len(detached))), chunk_size=chunk)(*detached)
    return r, torch.cat([j.reshape(n, -1) for j in Js], dim=1)


def _autodiff_rows(ctx: _RCtx) -> Runner:
    def run(params: list[Tensor], xs: Sequence[Tensor], damping: Any) -> tuple[Any, ...]:
        r, J = _autodiff_jacobian(ctx, params, xs)
        if ctx.quantity == "ntk_diagonal":
            return r, torch.sum(J * J, dim=1)
        K = J @ J.T
        if ctx.quantity == "gram":
            return r, K
        conditioning = ctx.policy if ctx.policy.mode == "floor" else None

        def one(d: Tensor) -> list[Tensor]:
            alpha, _ = damped_solve(K, r, d, conditioning)
            flat = J.T @ alpha
            out, start = [], 0
            for p in params:
                out.append(flat[start : start + p.numel()].reshape(p.shape))
                start += p.numel()
            return out

        values = torch.as_tensor(damping, dtype=K.dtype, device=K.device)
        if values.ndim == 0:
            return r, K, one(values)
        steps = [one(d) for d in values]
        return r, K, [torch.stack([s[k] for s in steps]) for k in range(len(steps[0]))]

    return run


_RUNNERS: dict[str, Callable[[_RCtx], Runner]] = {
    "full_streams": _structured,
    "autodiff_rows": _autodiff_rows,
}


def _structured_kw(ctx: _RCtx) -> dict[str, Any]:
    return {
        "coordinates": ctx.coordinates,
        "diffusion": ctx.diffusion,
        "allow_almost_everywhere": ctx.allow_ae,
    }


def _values(ctx: _RCtx, algorithm: str, params: list[Tensor], xs: Sequence[Tensor]) -> Tensor:
    """The residual vector alone, by the plan's own program (no Jacobian, no Gram)."""
    if algorithm == "full_streams":
        assert ctx.layers is not None
        return deep_residual_values(
            ctx.engine_blocks(xs), ctx.layers(params), **_structured_kw(ctx)
        )
    return _autodiff_rvec(ctx, xs)(*[p.detach() for p in params]).detach()


def _parameter_diagonal(
    ctx: _RCtx, algorithm: str, params: list[Tensor], xs: Sequence[Tensor]
) -> tuple[Tensor, list[Tensor]]:
    """``(r, [sum_n (d r_n / d theta)^2 per parameter tensor])`` by the plan's own program."""
    if algorithm == "full_streams":
        assert ctx.layers is not None
        r, acc = deep_residual_gn_diagonal(
            ctx.engine_blocks(xs), ctx.layers(params), **_structured_kw(ctx)
        )
        return r, _flat_step(acc)
    r, J = _autodiff_jacobian(ctx, params, xs)
    flat = torch.sum(J * J, dim=0)
    out, start = [], 0
    for p in params:
        out.append(flat[start : start + p.numel()].reshape(p.shape))
        start += p.numel()
    return r, out


class CompiledResidual:
    """A planned residual program.  ``op(xs=None, damping=None)``.

    Returns ``(r, K)`` for ``quantity="gram"``, ``(r, diag)`` for ``"ntk_diagonal"`` and ``(r, K, step)`` for
    ``"gauss_newton"``, where ``step`` is a list aligned with :attr:`parameters` (``theta - step`` is the damped update; a
    1-D ``damping`` stacks the steps along a new leading axis). ``xs`` optionally replaces the points of the blocks.
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
        parameters: Callable[[], list[Tensor]],
        alternatives: Sequence[str] = (),
    ) -> None:
        self._ctx = ctx
        self.precision_notes: list[str] = []
        self.plan = plan
        self._alternatives = list(alternatives)
        self.matched = matched
        self.fallback_reason = fallback_reason
        self._chain = None if chain is None else dict(chain)
        self.request = request
        self._parameters = parameters
        self._run = _RUNNERS[plan.algorithm](ctx)
        self.first_call_seconds: float | None = None
        self._last_condition: dict[str, Any] | None = None
        self._pending_condition: tuple[Tensor, Any, Any] | None = None
        self.executor_failures: list[Refusal] = []
        self.descriptor: Descriptor | None = None

    @property
    def algorithm(self) -> str:
        return self.plan.algorithm

    @property
    def parameters(self) -> list[Tensor]:
        """The parameter tensors the step is aligned with (the current ones of the model)."""
        return self._parameters()

    @property
    def last_conditioning(self) -> dict[str, Any] | None:
        """Conditioning diagnostics of the most recent Gauss-Newton call, or ``None``.

        The report is formed when it is read, from the Gram matrix of that call: it takes a power iteration and about ten
        device-to-host reads, which a step that never looks at it should not pay.
        """
        if self._pending_condition is not None:
            K, damping, policy = self._pending_condition
            self._pending_condition = None
            with torch.no_grad():
                self._last_condition = condition_report(K, damping, policy)
        return self._last_condition

    def apply_step(self, step: Sequence[Tensor], scale: float = 1.0) -> None:
        """``theta -= scale * step`` on the model's parameters (in place, outside autograd)."""
        with torch.no_grad():
            for p, s in zip(self.parameters, step, strict=True):
                p.sub_(scale * s)

    def _advance(self, err: BaseException) -> bool:
        self.executor_failures.append(
            Refusal(self.plan.algorithm, f"refused at run time: {err}"[:300])
        )
        if not self._alternatives:
            return False
        name = self._alternatives.pop(0)
        self.plan = replace(self.plan, algorithm=name, mode=self.plan.mode + "+memory_retry")
        self._run = _RUNNERS[name](self._ctx)
        return True

    def __call__(self, xs: Sequence[Tensor] | None = None, damping: Any = None) -> tuple[Any, ...]:
        if self._ctx.quantity == "gauss_newton" and damping is None:
            raise ValueError("quantity='gauss_newton' needs `damping`")
        pts = tuple(self._ctx.xs_of(xs))
        params = self.parameters
        first = self.first_call_seconds is None
        t0 = time.perf_counter()
        while True:
            try:
                with matmul_precision(self._ctx.precision):
                    out = self._run(params, pts, damping)
                break
            except torch.OutOfMemoryError as err:
                if pts[0].is_cuda:
                    torch.cuda.empty_cache()
                if not self._advance(err):
                    err.add_note(
                        "every candidate was refused: "
                        + "; ".join(f"{f.algorithm}: {f.reason}" for f in self.executor_failures)
                    )
                    raise
        if first:
            _sync(pts[0].device)
            self.first_call_seconds = time.perf_counter() - t0
        if self._ctx.quantity == "gauss_newton" and self._ctx.policy.mode != "off":
            self._last_condition = None
            self._pending_condition = (
                out[1].detach(),
                damping.detach().clone() if isinstance(damping, Tensor) else damping,
                self._ctx.policy,
            )
        return out

    def _guarded(self, work: Callable[[], Any]) -> Any:
        while True:
            try:
                return work()
            except torch.OutOfMemoryError as err:
                if self._ctx.blocks[0].x.is_cuda:
                    torch.cuda.empty_cache()
                if not self._advance(err):
                    raise

    def residual(self, xs: Sequence[Tensor] | None = None) -> Tensor:
        """The residual vector ``r`` at the current parameters, without a Jacobian or Gram (for trial steps)."""
        pts = self._ctx.xs_of(xs)
        params = self.parameters
        return cast(
            Tensor, self._guarded(lambda: _values(self._ctx, self.plan.algorithm, params, pts))
        )

    def parameter_diagonal(self, xs: Sequence[Tensor] | None = None) -> tuple[Tensor, list[Tensor]]:
        """``(r, [sum_n (d r_n / d theta_i)^2])``: the exact per-parameter Gauss-Newton diagonal (a sum over rows).

        The list is aligned with :attr:`parameters`; divide by ``r.numel()`` for the row mean.
        """
        pts = self._ctx.xs_of(xs)
        params = self.parameters
        return cast(
            tuple[Tensor, list[Tensor]],
            self._guarded(lambda: _parameter_diagonal(self._ctx, self.plan.algorithm, params, pts)),
        )

    def _conditioning_entry(self) -> dict[str, Any]:
        pol = self._ctx.policy
        base: dict[str, Any] = {"mode": pol.mode, "step_tolerance": pol.step_tolerance}
        if self._ctx.quantity != "gauss_newton" or pol.mode == "off":
            return {**base, "status": "not applicable"}
        return (
            {**base, "status": "no call yet"}
            if self.last_conditioning is None
            else self.last_conditioning
        )

    def explain(self) -> dict[str, Any]:
        out = self.plan.explain()
        out.update(
            {
                "backend": "torch",
                "matched_patterns": ["dense_chain"] if self.matched else [],
                "fallback_reason": self.fallback_reason,
                "chain": self._chain,
                "request": self.request.to_config(),
                "first_call_seconds": self.first_call_seconds,
                "executor_failures": [
                    {"algorithm": r.algorithm, "reason": r.reason} for r in self.executor_failures
                ],
                "conditioning": self._conditioning_entry(),
                "gram_precision": "standard",
                "compute_dtype": None,
                "precision": {**self._ctx.precision.to_config(), "notes": self.precision_notes},
                "cold_cost": {
                    "predicted_seconds": self.plan.compile_seconds,
                    "compile_policy": {"policy": "eager"},
                    "persistent_cache_dir": None,
                    "executables": 0,
                    "stages": [],
                },
            }
        )
        return out


def _leaves(layers: Sequence[tuple[Tensor, Tensor | None, Any]]) -> list[Tensor]:
    out: list[Tensor] = []
    for W, b, _ in layers:
        out.append(torch.as_tensor(W))
        if b is not None:
            out.append(torch.as_tensor(b))
    return out


def _rebuild(
    template: Sequence[tuple[Tensor, Tensor | None, Any]], params: Sequence[Tensor]
) -> list[tuple[Tensor, Tensor | None, Any]]:
    out, k = [], 0
    for _, b, spec in template:
        W = params[k]
        k += 1
        bias = None
        if b is not None:
            bias = params[k]
            k += 1
        out.append((W, bias, spec))
    return out


def _chain_value(layers: Sequence[tuple[Tensor, Tensor | None, Any]], x: Tensor) -> Tensor:
    from omnibias.torch.activations.registry import get_activation

    a = x
    for W, b, spec in layers:
        a = a @ W.T
        if b is not None:
            a = a + b
        if spec is not None:
            a = get_activation(spec).forward(a)
    return a


def compile_residual(
    model: Any,
    blocks: Sequence[Any],
    *,
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
    cost_model: CostModel | None = None,
    conditioning: str | ConditioningPolicy | None = "report",
    precision: str | PrecisionPolicy | None = None,
    allow_low_precision: bool = False,
    top_k: int | None = FIND_TOP_K,
) -> CompiledResidual:
    r"""Plan the residual Gram ``K = J J^T``, its damped Gauss-Newton step or its diagonal.

    ``model`` is a layer list, an object with ``_layer_specs()``, an ``nn.Sequential`` of ``nn.Linear`` and recognised
    activations, or any ``nn.Module`` ``model(x)`` (planned with ``autodiff_rows`` only). ``blocks`` are
    ``(x, Residual)`` or ``(x, Residual, order)``; the other parameters are those of the JAX ``compile_residual``, except
    that the model owns its parameters and ``gram_precision``/``compute_dtype`` are not provided. ``precision`` is the
    float32 matrix-product mode (``"highest"`` by default, ``"tf32"`` opt-in, ``"default"``); half-precision points are
    refused for residuals of derivative order two and above unless ``allow_low_precision`` is set. ``top_k`` is how many
    candidates ``mode="find"`` times: ``2`` (default) or ``None`` for every feasible program. A program that fails when
    timed is listed in ``explain()["executor_failures"]`` and ranked last.

    Raises
    ------
    ValueError
        If no program is feasible; the message lists every refusal reason.
    """
    if quantity not in _QUANTITIES:
        raise ValueError(f"quantity must be one of {_QUANTITIES}")
    norm = _normalise_blocks(blocks)
    first_x = norm[0].x
    layer_accessor: Callable[[], list[tuple[Tensor, Tensor | None, Any]]] | None = None
    fallback_reason: str | None = None
    if _is_layer_model(model):

        def layer_accessor() -> list[tuple[Tensor, Tensor | None, Any]]:  # noqa: F811
            return layers_of(model)

    elif isinstance(model, nn.Module):
        layer_accessor = chain_from_module(model)
        if layer_accessor is None:
            fallback_reason = "the module is not a Sequential of Linear and recognised activations"
        else:
            with torch.no_grad():
                out = model(first_x)
                chain_out = _chain_value(layer_accessor(), first_x)
            tol = float(torch.finfo(first_x.dtype).eps) ** 0.5
            if out.reshape(chain_out.shape).shape != chain_out.shape or not torch.allclose(
                chain_out, out.reshape(chain_out.shape), rtol=tol, atol=tol
            ):
                layer_accessor = None
                fallback_reason = (
                    "the dense chain read from the module does not reproduce its forward pass"
                )
    else:
        raise TypeError(
            "model must be a layer list, an object with _layer_specs(), or an nn.Module"
        )
    template = layer_accessor() if layer_accessor is not None else None
    if template is not None:

        def parameters() -> list[Tensor]:
            assert layer_accessor is not None
            return _leaves(layer_accessor())

        def fn(params: list[Tensor], x: Tensor) -> Tensor:
            assert template is not None
            return _chain_value(_rebuild(template, params), x)

        def layers_from(params: list[Tensor]) -> list[tuple[Tensor, Tensor | None, Any]]:
            assert template is not None
            return _rebuild(template, params)

        layers_fn: Callable[[list[Tensor]], list[tuple[Tensor, Tensor | None, Any]]] | None = (
            layers_from
        )
    else:
        module = model
        names = [n for n, _ in module.named_parameters()]

        def parameters() -> list[Tensor]:
            return [p for _, p in module.named_parameters()]

        def fn(params: list[Tensor], x: Tensor) -> Tensor:
            return torch.func.functional_call(module, dict(zip(names, params, strict=True)), (x,))  # type: ignore[no-any-return]

        layers_fn = None
    dim = int(first_x.shape[1])
    coords = None if coordinates is None else np.asarray(coordinates, dtype=int)
    dirs = None if directions is None else _numpy(directions)
    union = Demand.of()
    for b in norm:
        union = union.union(b.residual.demand)
    if dirs is not None and "directional" not in union:
        union = union.union(Demand.of("directional"))
    params0 = parameters()
    with torch.no_grad():
        channels = int(fn(params0, first_x[:1]).reshape(1, -1).shape[1])
    q = 0 if dirs is None else int(dirs.shape[-1])
    rows = sum(int(b.x.shape[0]) * _rows_per_point(b, channels, dim, q) for b in norm)
    request = OperatorRequest(
        quantity,
        selected=None if coords is None else int(len(coords)),
        metric=_metric_of(diffusion),
        directions=q,
        allow_almost_everywhere=allow_almost_everywhere,
    )
    precision_policy = resolve_precision(precision, allow_low_precision)
    precision_notes = precision_policy.check(
        first_x.dtype,
        request_order(quantity, demand_order=union.order),
        what=f"residual demanding {sorted(union.quantities)}",
    )
    ctx = _RCtx(
        quantity=quantity,
        blocks=tuple(norm),
        fn=fn,
        layers=layers_fn,
        coordinates=coords,
        diffusion=diffusion,
        directions=dirs,
        allow_ae=allow_almost_everywhere,
        strategy=strategy,
        row_chunk=row_chunk,
        dim=dim,
        policy=resolve_conditioning(conditioning),
        precision=precision_policy,
    )
    dev_class, dev = _device_description(first_x, device)
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
    damping0 = (
        torch.tensor(1e-3, dtype=first_x.dtype, device=first_x.device)
        if quantity == "gauss_newton"
        else None
    )

    find_failures: list[Refusal] = []

    def timer(name: str) -> float:
        """Median seconds of one call; a program that cannot run is a failed measurement, not a failed plan."""
        try:
            run = _RUNNERS[name](ctx)
            with matmul_precision(precision_policy):
                run(params0, xs0, damping0)
                times = []
                for _ in range(3):
                    _sync(dev)
                    t0 = time.perf_counter()
                    run(params0, xs0, damping0)
                    _sync(dev)
                    times.append(time.perf_counter() - t0)
            return float(np.median(times))
        except Exception as err:  # noqa: BLE001 - an exception here used to reach the caller of compile_residual
            find_failures.append(Refusal(name, f"find: {type(err).__name__}: {str(err)[:200]}"))
            return float("inf")  # may be transient (device memory): the plan is not cached

    chain_config: Mapping[str, Any] | None = None
    descriptor: Descriptor | None = None
    if template is not None:
        chain: ChainDescriptor = describe_layers(template)
        chain_config = chain.to_config()
        descriptor = Descriptor(
            request, chain, sum(int(b.x.shape[0]) for b in norm), demand=union, rows=rows
        )
        plan = select(
            descriptor,
            backend="torch",
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
            with matmul_precision(precision_policy):
                _RUNNERS[name](ctx)(params0, xs0, damping0)
        except Exception as err:  # noqa: BLE001 - an engine that cannot run this residual is a refusal
            failures.append(
                Refusal(
                    name,
                    f"program could not run this problem: {type(err).__name__}: {str(err)[:200]}",
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
        matched=template is not None,
        fallback_reason=fallback_reason,
        chain=chain_config,
        request=request,
        parameters=parameters,
        alternatives=order[order.index(chosen) + 1 :],
    )
    op.precision_notes = precision_notes
    op.executor_failures = failures
    op.descriptor = descriptor
    return op
