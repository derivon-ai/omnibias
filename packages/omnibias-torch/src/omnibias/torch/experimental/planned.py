# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Planned derivative operators for PyTorch: descriptor, planner, executors, ``explain()``.

The PyTorch counterpart of :func:`omnibias.jax.experimental.compile_operator`. A model is read as a dense chain when
it is a layer list ``[(W, b, spec), ...]``, an object that provides ``_layer_specs()`` (the architectures of
:mod:`omnibias.torch.architectures`), or an ``nn.Sequential`` of ``nn.Linear`` and recognised activation modules whose
forward pass the chain reproduces on the example input. Any other callable ``model(x)`` is planned with the generic
executors only (nested autodiff, forward-mode jets, dense Jacobians).

The planner, the algorithm names, the refusal reasons and the keys of ``explain()`` are those of the JAX front end;
only the executors differ. Executors are eager PyTorch programs: parameter gradients flow through autograd, and the
cold cost is the first call (no compilation unless ``compile=True``).

Registered for PyTorch: ``projected`` (second order; the Laplace powers 2 and 3 are refused with that reason),
``forward_laplacian``, ``forward_polylaplacian`` (exact support enumeration), ``taylor_jet`` (one directional
second derivative per input direction, the generic second-order engine), ``nested_autodiff`` (reverse-mode ``torch.func.jacrev`` nested twice per
power), ``hutchinson`` (inexact), divergence ``projected``/``jacobian_trace``, log-determinant ``projected``/
``dense_jacobian`` and solve ``woodbury``/``dense_jacobian``.
"""

from __future__ import annotations

import contextlib
import functools
import importlib.metadata
import operator as _operator
import os
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import Any, NamedTuple, TypeVar, cast

import numpy as np
from omnibias.core.chain import ChainDescriptor
from omnibias.core.engines import engine_algorithm
from omnibias.core.planner import (
    FIND_TOP_K,
    Algorithm,
    CostModel,
    Descriptor,
    OperatorRequest,
    Plan,
    PlanCache,
    Refusal,
    Work,
    default_plan_cache_path,
    device_class,
    estimator_probes,
    host_memory_limit,
    measurement_context,
    register_algorithm,
    resolve_planning,
    select,
    unregister_algorithm,
)
from omnibias.core.precision import PrecisionPolicy, request_order, resolve_precision
from omnibias.torch.activations.registry import get_activation
from omnibias.torch.experimental._chain import describe_layers
from omnibias.torch.experimental.projected import (
    capture_safe_branches,
    projected_deep_divergence,
    projected_deep_laplacian,
    projected_deep_logdet,
)
from omnibias.torch.experimental.projected_polylaplacian import (
    projected_deep_bilaplacian,
    projected_deep_trilaplacian,
)
from omnibias.torch.experimental.streams import bilaplacian_by_jets, directional_jet_streams
from omnibias.torch.experimental.woodbury import residual_solve
from omnibias.torch.laplacian import deep_field_polylaplacian, deep_field_value_grad_laplacian

import torch
import torch.nn.functional as F
from torch import Tensor, nn

__all__ = [
    "CompiledOperator",
    "OperatorOutput",
    "analyze_input_weights",
    "chain_from_module",
    "compile_planned_operator",
    "layers_of",
    "register_engine",
    "unregister_engine",
]

_T = TypeVar("_T")

#: ``nn`` activation modules whose forward pass is reproduced by a registry activation of that name.
_MODULE_ACTIVATIONS: dict[type, str] = {
    nn.Tanh: "tanh",
    nn.SiLU: "silu",
    nn.Sigmoid: "sigmoid",
    nn.Softplus: "softplus",
    nn.GELU: "gelu",
    nn.ReLU: "relu",
}

#: Generic executors by operator, in the order the fallback tries them when the model is not a recognised chain.
_GENERIC_ORDER: dict[str, tuple[str, ...]] = {
    "laplacian": ("taylor_jet", "nested_autodiff"),
    "divergence": ("jacobian_trace",),
    "logdet": ("dense_jacobian",),
    "solve": ("dense_jacobian",),
}


# ------------------------------------------------------------------------------------------------- model adapters


def chain_from_module(
    module: nn.Module,
) -> Callable[[], list[tuple[Tensor, Tensor | None, Any]]] | None:
    """A live layer accessor for an ``nn.Sequential`` of ``nn.Linear`` and recognised activations, else ``None``.

    The accessor returns the current parameters on every call, so optimizer updates are seen.
    """
    if not isinstance(module, nn.Sequential) or len(module) == 0:
        return None
    steps: list[tuple[nn.Linear, str | None]] = []
    for child in module:
        if isinstance(child, nn.Linear):
            steps.append((child, None))
        elif type(child) in _MODULE_ACTIVATIONS and steps and steps[-1][1] is None:
            name = _MODULE_ACTIVATIONS[type(child)]
            if isinstance(child, nn.Softplus) and (child.beta != 1 or child.threshold != 20):
                return None
            if isinstance(child, nn.GELU) and child.approximate != "none":
                return None
            steps[-1] = (steps[-1][0], name)
        else:
            return None

    def layers() -> list[tuple[Tensor, Tensor | None, Any]]:
        return [(lin.weight, lin.bias, name) for lin, name in steps]

    return layers


def analyze_input_weights(
    module: nn.Module,
) -> tuple[Callable[[], Tensor] | None, str | None]:
    """Whether ``module`` reads its input only through ``nn.Linear`` / ``F.linear`` / ``x @ W`` with its own parameters.

    Returns ``(accessor, None)`` where ``accessor()`` stacks the live first-layer matrices ``(rows, D)``, or
    ``(None, reason)``.  The module is traced with :mod:`torch.fx`; any other read of the input values (an elementwise
    use, a slice, a concatenation, a raw-coordinate skip) refuses with the operation that reads it.  Reading the
    shape of the input is allowed.
    """
    try:
        gm = torch.fx.symbolic_trace(module)
    except Exception as err:  # noqa: BLE001 - untraceable control flow is a refusal
        return (
            None,
            f"the module cannot be traced by torch.fx: {type(err).__name__}: {str(err)[:120]}",
        )
    placeholders = [n for n in gm.graph.nodes if n.op == "placeholder"]
    if len(placeholders) != 1:
        return None, "the module must take exactly one input"
    x = placeholders[0]
    modules = dict(gm.named_modules())
    sources: list[Callable[[], Tensor]] = []

    def param(node: Any) -> Callable[[], Tensor] | None:
        if getattr(node, "op", None) == "get_attr":
            target = node.target
            return lambda: cast(Tensor, gm.get_parameter(target))
        return None

    for n in gm.graph.nodes:
        if n.op in ("placeholder", "output") or x not in n.all_input_nodes:
            continue
        if (
            n.op == "call_function"
            and n.target is getattr
            and n.args[1] in ("shape", "ndim", "dtype", "device")
        ):
            continue
        if n.op == "call_method" and n.target in ("size", "dim"):
            continue
        if n.op == "call_module" and isinstance(modules[n.target], nn.Linear) and n.args[0] is x:
            lin = modules[n.target]
            sources.append(lambda lin=lin: lin.weight)
            continue
        if n.op == "call_function" and n.target is F.linear and n.args[0] is x and len(n.args) >= 2:
            getter = param(n.args[1])
            if getter is not None:
                sources.append(getter)
                continue
        if (
            n.op == "call_function"
            and n.target in (_operator.matmul, torch.matmul)
            and n.args[0] is x
        ):
            getter = param(n.args[1])
            if getter is not None:
                sources.append(lambda getter=getter: getter().T)
                continue
        return (
            None,
            f"the input is also read by {n.op} {getattr(n.target, '__name__', n.target)!s}; it must enter only through products with the module's parameters",
        )
    if not sources:
        return None, "the input is not read by any weight product"
    return (lambda: torch.cat([src() for src in sources], dim=0)), None


def layers_of(model: Any) -> list[tuple[Tensor, Tensor | None, Any]]:
    """The layer list of ``model`` (a sequence of ``(W, b, spec)``, an object with ``_layer_specs()``/``layer_specs()``)."""
    if isinstance(model, Sequence) and not isinstance(model, str):
        layers = list(model)
    else:
        provider = getattr(model, "layer_specs", None) or getattr(model, "_layer_specs", None)
        if provider is None:
            raise TypeError(
                "expected a sequence of (W, b, spec) or an object with layer_specs(); "
                f"got {type(model).__name__}"
            )
        layers = list(provider())
    if not layers:
        raise ValueError("layers must be non-empty")
    previous: int | None = None
    for i, layer in enumerate(layers):
        if len(layer) != 3:
            raise ValueError(f"layer {i} must be a (W, b, spec) triple")
        W = torch.as_tensor(layer[0])
        if W.ndim != 2 or (previous is not None and W.shape[1] != previous):
            raise ValueError(
                f"layer {i} weight must have shape (out, {previous if previous is not None else 'in'})"
            )
        previous = W.shape[0]
    return layers


def _chain_value(layers: Sequence[tuple[Tensor, Tensor | None, Any]], x: Tensor) -> Tensor:
    a = x
    for W, b, spec in layers:
        a = a @ torch.as_tensor(W).T
        if b is not None:
            a = a + torch.as_tensor(b)
        if spec is not None:
            a = get_activation(spec).forward(a)
    return a


# ------------------------------------------------------------------------------------------------------ context


class OperatorOutput(NamedTuple):
    """Result of :meth:`CompiledOperator.evaluate`: the operator value and the by-products the program formed."""

    value: Tensor
    aux: dict[str, Tensor]


@dataclass(frozen=True)
class _Ctx:
    fn: Callable[[Tensor], Tensor]  # batched model: (B, D) -> (B, C) or (B,)
    layers: Callable[[], list[tuple[Tensor, Tensor | None, Any]]] | None
    squeeze: bool
    batched: bool
    dim: int
    power: int
    coordinates: np.ndarray | None
    diffusion: Any
    directions: Any
    allow_ae: bool
    transpose: bool
    n_probes: int | None
    input_rows: Callable[[], Tensor] | None = None
    differentiate: str = "none"
    accuracy: float | None = None
    matmul: str = "highest"  # the PrecisionPolicy mode of the call: "highest", "tf32" or "default" (the process setting)
    _memo: dict[Any, Any] = field(default_factory=dict, repr=False, compare=False)

    def constant(self, name: str, source: Any, x: Tensor) -> Tensor | None:
        """``source`` as a tensor of ``x``'s dtype and device, converted once while ``source`` is unchanged.

        Every call used to convert and upload these constants again (a host to device copy that also waits for the stream).
        An entry stays valid while ``source`` is the same object with the same contents: a tensor by its version counter,
        an array or list by its bytes. A source that requires grad is never held, because converting it can build a graph
        node that belongs to a single backward pass.
        """
        if source is None:
            return None
        if isinstance(source, Tensor) and source.requires_grad:
            return torch.as_tensor(source, dtype=x.dtype, device=x.device)
        key = (name, x.dtype, x.device)
        stamp = _fingerprint(source)
        held = self._memo.get(key)
        if held is not None and held[0] == stamp and (held[1] is source or not isinstance(source, Tensor)):
            return cast(Tensor, held[2])
        value = torch.as_tensor(source, dtype=x.dtype, device=x.device)
        self._memo[key] = (stamp, source, value)
        return value

    def like_diffusion(self, x: Tensor) -> Tensor | None:
        return self.constant("diffusion", self.diffusion, x)

    def like_directions(self, x: Tensor) -> Tensor | None:
        return self.constant("directions", self.directions, x)

    def device_columns(self, x: Tensor) -> Tensor:
        """:meth:`columns` as a tensor like ``x``, rebuilt only when the metric, the coordinates or the dimension change."""
        key = ("columns", x.dtype, x.device)
        stamp = (self.dim, _fingerprint(self.diffusion), _fingerprint(self.coordinates))
        held = self._memo.get(key)
        if held is not None and held[0] == stamp and held[1] == (self.diffusion, self.coordinates):
            return cast(Tensor, held[2])
        value = torch.as_tensor(self.columns(), dtype=x.dtype, device=x.device)
        self._memo[key] = (stamp, (self.diffusion, self.coordinates), value)
        return value

    def point(self, xi: Tensor) -> Tensor:
        return self.fn(xi[None, :])[0]

    def reduced_columns(self, like: Tensor) -> Tensor:
        """Directions ``V'`` ``(D, k)``, ``k = min(rows, q)``, with ``W V' V'^T W^T = W V V^T W^T`` (held constant)."""
        assert self.input_rows is not None
        with torch.no_grad():
            W = self.input_rows().detach().to(like.dtype)
            if self.diffusion is None:
                idx = (
                    np.arange(self.dim)
                    if self.coordinates is None
                    else np.asarray(self.coordinates)
                )
                Q = torch.linalg.qr(W[:, torch.as_tensor(idx, device=W.device)].T, mode="reduced")[
                    0
                ]
                if self.coordinates is None:
                    return Q
                out = torch.zeros(self.dim, Q.shape[1], dtype=Q.dtype, device=Q.device)
                out[torch.as_tensor(idx, device=Q.device)] = Q
                return out
            V = torch.as_tensor(self.columns(), dtype=like.dtype, device=like.device)
            return V @ torch.linalg.qr((W @ V).T, mode="reduced")[0]

    def columns(self) -> np.ndarray:
        """``V`` with ``V V^T`` the metric embedded in the ``D`` inputs: the operator is ``sum_c v_c^T H v_c``."""
        idx = np.arange(self.dim) if self.coordinates is None else np.asarray(self.coordinates)
        m = len(idx)
        if self.diffusion is None:
            S = np.eye(m)
        else:
            d = _numpy(self.diffusion)
            S = np.diag(d) if d.ndim == 1 else d
        V = np.zeros((self.dim, S.shape[1]))
        V[idx] = S
        return V

    def pairing(self, outputs: int) -> np.ndarray:
        if self.coordinates is not None:
            return np.asarray(self.coordinates)
        if outputs != self.dim:
            raise ValueError(
                "a vector field needs as many outputs as inputs (or `coordinates` to pair them)"
            )
        return np.arange(self.dim)


def _fingerprint(a: Any) -> Any:
    """What must be unchanged for a converted copy of ``a`` to stay valid: a tensor's identity and version, an array's bytes."""
    if a is None:
        return None
    if isinstance(a, Tensor):
        return ("tensor", id(a), a._version)
    arr = np.asarray(a)
    return ("array", arr.shape, arr.dtype.str, arr.tobytes())


def _numpy(a: Any) -> np.ndarray:
    return a.detach().cpu().numpy() if isinstance(a, Tensor) else np.asarray(a)


def _squeeze(ctx: _Ctx, tree: tuple[Tensor, dict[str, Tensor]]) -> tuple[Tensor, dict[str, Tensor]]:
    if not ctx.squeeze:
        return tree
    value, aux = tree
    return value[..., 0], {k: v[..., 0] for k, v in aux.items()}


def _points(ctx: _Ctx, g: Callable[[Tensor], Any], x: Tensor) -> Any:
    return torch.func.vmap(g)(x) if ctx.batched else g(x)


def _layers(ctx: _Ctx) -> list[tuple[Tensor, Tensor | None, Any]]:
    if ctx.layers is None:
        raise ValueError("this program needs a recognised dense chain")
    return ctx.layers()


Runner = Callable[[Tensor, tuple[Any, ...], Any], tuple[Tensor, dict[str, Tensor]]]


# ----------------------------------------------------------------------------------------------------- executors


def _lap_projected(ctx: _Ctx) -> Runner:
    def run(x: Tensor, extra: tuple[Any, ...], key: Any) -> tuple[Tensor, dict[str, Tensor]]:
        layers = _layers(ctx)
        dirs = ctx.like_directions(x)
        if ctx.power == 1:
            u, gs, lap, dv = projected_deep_laplacian(
                x,
                layers,
                diffusion=ctx.like_diffusion(x),
                coordinates=ctx.coordinates,
                directions=dirs,
                allow_almost_everywhere=ctx.allow_ae,
            )
            value, aux = lap, {"value": u, "grad_sq": gs}
        elif ctx.power == 2:
            u, gs, lap, bil, dv = projected_deep_bilaplacian(
                x,
                layers,
                diffusion=ctx.like_diffusion(x),
                coordinates=ctx.coordinates,
                directions=dirs,
                allow_almost_everywhere=ctx.allow_ae,
            )
            value, aux = bil, {"value": u, "grad_sq": gs, "laplacian": lap}
        else:
            u, gs, lap, bil, tri, dv = projected_deep_trilaplacian(
                x,
                layers,
                coordinates=ctx.coordinates,
                directions=dirs,
                allow_almost_everywhere=ctx.allow_ae,
            )
            value, aux = tri, {"value": u, "grad_sq": gs, "laplacian": lap, "bilaplacian": bil}
        if dv is not None:
            aux["directional"] = dv
        return _squeeze(ctx, (value, aux))

    return run


def _lap_fused(ctx: _Ctx) -> Runner:
    """One Triton kernel per forward and per backward pass (CUDA, first-order parameter derivatives)."""

    def run(x: Tensor, extra: tuple[Any, ...], key: Any) -> tuple[Tensor, dict[str, Tensor]]:
        from omnibias.torch.experimental.fused_streams import fused_second_order

        r = fused_second_order(
            x,
            _layers(ctx),
            diffusion=ctx.like_diffusion(x),
            coordinates=ctx.coordinates,
            allow_almost_everywhere=ctx.allow_ae,
            precision=ctx.matmul,  # the policy of this operator: TF32 only when it was asked for
        )
        return _squeeze(ctx, (r.laplacian, {"value": r.value, "grad_sq": r.grad_sq}))

    return run


def _lap_taylor_closed(ctx: _Ctx) -> Runner:
    """Closed-form Taylor streams: ``Delta u = sum_c D^2_{v_c} u`` and ``Delta^2 u`` by polarized fourth-order jets."""

    def run(x: Tensor, extra: tuple[Any, ...], key: Any) -> tuple[Tensor, dict[str, Tensor]]:
        layers = _layers(ctx)
        if ctx.power == 1:
            V = ctx.device_columns(x)
            jet = directional_jet_streams(x, layers, V, 2, allow_almost_everywhere=ctx.allow_ae)
            return _squeeze(ctx, (jet.derivatives[..., 1, :].sum(dim=-2), {"value": jet.value}))
        u, bil = bilaplacian_by_jets(
            x,
            layers,
            coordinates=None if ctx.coordinates is None else [int(c) for c in ctx.coordinates],
            allow_almost_everywhere=ctx.allow_ae,
        )
        return _squeeze(ctx, (bil, {"value": u}))

    return run


def _lap_forward(ctx: _Ctx) -> Runner:
    def run(x: Tensor, extra: tuple[Any, ...], key: Any) -> tuple[Tensor, dict[str, Tensor]]:
        u, g, lap = deep_field_value_grad_laplacian(
            x, _layers(ctx), allow_almost_everywhere=ctx.allow_ae
        )
        return _squeeze(ctx, (lap, {"value": u, "grad": g}))

    return run


def _lap_polylaplacian(ctx: _Ctx) -> Runner:
    def run(x: Tensor, extra: tuple[Any, ...], key: Any) -> tuple[Tensor, dict[str, Tensor]]:
        out = deep_field_polylaplacian(x, _layers(ctx), ctx.power, mode="support")
        return _squeeze(ctx, (out, {}))

    return run


def _second_derivative_along(ctx: _Ctx, xi: Tensor, v: Tensor) -> tuple[Tensor, Tensor]:
    """``(f(xi), v^T H(xi) v)``: the directional first derivative (one tangent pass) differentiated in reverse mode.

    Nested forward mode would need a second forward rule for every activation backward (``silu_backward`` has none);
    reverse-over-forward differentiates every activation twice with the rules autograd has.
    """

    def f(z: Tensor) -> Tensor:
        return ctx.point(z)

    def df(z: Tensor) -> Tensor:
        return cast(Tensor, torch.func.jvp(f, (z,), (v,))[1])

    primal = f(xi)
    d2 = torch.func.jacrev(df)(xi) @ v
    return primal, d2


def _lap_jet(ctx: _Ctx, *, reduced: bool = False) -> Runner:
    def run(x: Tensor, extra: tuple[Any, ...], key: Any) -> tuple[Tensor, dict[str, Tensor]]:
        V = (
            ctx.reduced_columns(x)
            if reduced
            else ctx.device_columns(x)
        )

        def g(xi: Tensor) -> tuple[Tensor, Tensor]:
            u, d2 = torch.func.vmap(lambda v: _second_derivative_along(ctx, xi, v))(V.T)
            return u[0], d2.sum(dim=0)

        u, lap = _points(ctx, g, x)
        return lap, {"value": u}

    return run


def _lap_subspace_nested(ctx: _Ctx) -> Runner:
    """``Delta^p`` as ``p`` nested sums of reverse-over-forward second derivatives along the reduced directions."""

    def run(x: Tensor, extra: tuple[Any, ...], key: Any) -> tuple[Tensor, dict[str, Tensor]]:
        V = ctx.reduced_columns(x).T

        def second(g: Callable[[Tensor], Tensor]) -> Callable[[Tensor], Tensor]:
            def h(z: Tensor) -> Tensor:
                def along(v: Tensor) -> Tensor:
                    def df(w: Tensor) -> Tensor:
                        return cast(Tensor, torch.func.jvp(g, (w,), (v,))[1])

                    return cast(Tensor, torch.func.jacrev(df)(z) @ v)

                return torch.func.vmap(along)(V).sum(dim=0)

            return h

        g: Callable[[Tensor], Tensor] = ctx.point
        for _ in range(ctx.power):
            g = second(g)
        lap = _points(ctx, g, x)
        u = _points(ctx, ctx.point, x)
        return lap, {"value": u}

    return run


def _estimator(ctx: _Ctx, kind: str) -> Runner:
    """Stochastic Laplacian with probes independent across points (see the JAX program of the same name).

    ``hutchinson``: Rademacher combinations of the metric columns; ``stde``: Gaussian ones; ``sdgd``: ``s`` of the ``r``
    columns without replacement, rescaled by ``r / s``. The probe count follows ``accuracy`` unless ``n_probes`` was given.
    A training objective needs ``key`` (a generator or an integer seed): a silent fixed seed would repeat the probes.
    """

    def run(x: Tensor, extra: tuple[Any, ...], key: Any) -> tuple[Tensor, dict[str, Tensor]]:
        if key is None and ctx.differentiate != "none":
            raise ValueError(
                f"the {kind} estimator draws random probes: pass key=, and a new one every training step"
            )
        gen = (
            key
            if isinstance(key, torch.Generator)
            else torch.Generator(device=x.device).manual_seed(0 if key is None else int(key))
        )
        V = ctx.device_columns(x)
        r = V.shape[1]
        n = ctx.n_probes if ctx.n_probes is not None else estimator_probes(kind, ctx.accuracy, r)
        xb = x if ctx.batched else x[None, :]
        B = xb.shape[0]
        if kind == "hutchinson":
            z = torch.randint(0, 2, (B, n, r), generator=gen, device=x.device).to(x.dtype) * 2 - 1
            probes, scale = z @ V.T, 1.0 / n
        elif kind == "stde":
            z = torch.randn((B, n, r), generator=gen, device=x.device, dtype=x.dtype)
            probes, scale = z @ V.T, 1.0 / n
        else:
            s = min(n, r)
            pick = torch.rand((B, r), generator=gen, device=x.device).argsort(dim=1)[:, :s]
            probes, scale = V.T[pick], float(r) / s

        def g(xi: Tensor, pr: Tensor) -> tuple[Tensor, Tensor]:
            u, d2 = torch.func.vmap(lambda v: _second_derivative_along(ctx, xi, v))(pr)
            return u[0], scale * d2.sum(dim=0)

        u, lap = torch.func.vmap(g)(xb, probes)
        if not ctx.batched:
            u, lap = u[0], lap[0]
        return lap, {"value": u}

    return run


def _lap_hutchinson(ctx: _Ctx) -> Runner:
    return _estimator(ctx, "hutchinson")


def _lap_nested(ctx: _Ctx) -> Runner:
    def run(x: Tensor, extra: tuple[Any, ...], key: Any) -> tuple[Tensor, dict[str, Tensor]]:
        V = ctx.columns()
        A = torch.as_tensor(V @ V.T, dtype=x.dtype, device=x.device)

        def once(g: Callable[[Tensor], Tensor]) -> Callable[[Tensor], Tensor]:
            def h(z: Tensor) -> Tensor:
                return torch.einsum("...ij,ij->...", torch.func.jacrev(torch.func.jacrev(g))(z), A)

            return h

        g: Callable[[Tensor], Tensor] = ctx.point
        for _ in range(ctx.power):
            g = once(g)
        lap = _points(ctx, g, x)
        u = _points(ctx, ctx.point, x)
        return lap, {"value": u}

    return run


def _div_projected(ctx: _Ctx) -> Runner:
    def run(x: Tensor, extra: tuple[Any, ...], key: Any) -> tuple[Tensor, dict[str, Tensor]]:
        u, div = projected_deep_divergence(x, _layers(ctx), coordinates=ctx.coordinates)
        return _squeeze(ctx, (div, {"value": u}))

    return run


def _jacobian(ctx: _Ctx, xi: Tensor) -> Tensor:
    return cast(Tensor, torch.func.jacrev(ctx.point)(xi))


def _div_jacobian(ctx: _Ctx) -> Runner:
    def run(x: Tensor, extra: tuple[Any, ...], key: Any) -> tuple[Tensor, dict[str, Tensor]]:
        def g(xi: Tensor) -> tuple[Tensor, Tensor]:
            J = _jacobian(ctx, xi)
            idx = torch.as_tensor(ctx.pairing(J.shape[0]), device=xi.device)
            return ctx.point(xi), J[torch.arange(J.shape[0], device=xi.device), idx].sum()

        u, div = _points(ctx, g, x)
        return div, {"value": u}

    return run


def _logdet_projected(ctx: _Ctx) -> Runner:
    def run(x: Tensor, extra: tuple[Any, ...], key: Any) -> tuple[Tensor, dict[str, Tensor]]:
        u, sign, logabs = projected_deep_logdet(x, _layers(ctx), coordinates=ctx.coordinates)
        return logabs, {"value": u, "sign": sign}

    return run


def _logdet_dense(ctx: _Ctx) -> Runner:
    def run(x: Tensor, extra: tuple[Any, ...], key: Any) -> tuple[Tensor, dict[str, Tensor]]:
        def g(xi: Tensor) -> tuple[Tensor, Tensor, Tensor]:
            J = _jacobian(ctx, xi)
            idx = torch.as_tensor(ctx.pairing(J.shape[0]), device=xi.device)
            M = torch.eye(J.shape[0], dtype=J.dtype, device=J.device) + J[:, idx]
            sign, logabs = torch.linalg.slogdet(M)
            return ctx.point(xi), sign, logabs

        u, sign, logabs = _points(ctx, g, x)
        return logabs, {"value": u, "sign": sign}

    return run


def _solve_woodbury(ctx: _Ctx) -> Runner:
    def run(x: Tensor, extra: tuple[Any, ...], key: Any) -> tuple[Tensor, dict[str, Tensor]]:
        if not extra:
            raise ValueError("operator='solve' needs the right-hand side: op(x, rhs)")
        u, y = residual_solve(x, _layers(ctx), torch.as_tensor(extra[0]), transpose=ctx.transpose)
        return y, {"value": u}

    return run


def _solve_dense(ctx: _Ctx) -> Runner:
    def run(x: Tensor, extra: tuple[Any, ...], key: Any) -> tuple[Tensor, dict[str, Tensor]]:
        if not extra:
            raise ValueError("operator='solve' needs the right-hand side: op(x, rhs)")
        rhs = torch.as_tensor(extra[0])

        def g(xi: Tensor, ri: Tensor) -> tuple[Tensor, Tensor]:
            J = _jacobian(ctx, xi)
            M = torch.eye(J.shape[0], dtype=J.dtype, device=J.device) + (
                J.T if ctx.transpose else J
            )
            return ctx.point(xi), torch.linalg.solve(M, ri)

        u, y = torch.func.vmap(g)(x, rhs) if ctx.batched else g(x, rhs)
        return y, {"value": u}

    return run


_EXECUTORS: dict[tuple[str, str], Callable[[_Ctx], Runner]] = {
    ("laplacian", "projected"): _lap_projected,
    ("laplacian", "forward_laplacian"): _lap_forward,
    ("laplacian", "forward_polylaplacian"): _lap_polylaplacian,
    ("laplacian", "taylor_jet"): _lap_jet,
    ("laplacian", "subspace_jet"): lambda ctx: _lap_jet(ctx, reduced=True),
    ("laplacian", "subspace_nested"): _lap_subspace_nested,
    ("laplacian", "nested_autodiff"): _lap_nested,
    ("laplacian", "hutchinson"): _lap_hutchinson,
    ("laplacian", "stde"): lambda ctx: _estimator(ctx, "stde"),
    ("laplacian", "sdgd"): lambda ctx: _estimator(ctx, "sdgd"),
    ("laplacian", "fused_streams"): _lap_fused,
    ("laplacian", "taylor_closed"): _lap_taylor_closed,
    ("divergence", "projected"): _div_projected,
    ("divergence", "jacobian_trace"): _div_jacobian,
    ("logdet", "projected"): _logdet_projected,
    ("logdet", "dense_jacobian"): _logdet_dense,
    ("solve", "woodbury"): _solve_woodbury,
    ("solve", "dense_jacobian"): _solve_dense,
}


# ----------------------------------------------------------------------------------------- external engines


def _engine_runner(operator: str, engine: Callable[..., Any]) -> Callable[[_Ctx], Runner]:
    def factory(ctx: _Ctx) -> Runner:
        def run(x: Tensor, extra: tuple[Any, ...], key: Any) -> tuple[Tensor, dict[str, Tensor]]:
            def f(z: Tensor) -> Tensor:
                out = ctx.point(z)
                return out.reshape(()) if operator == "laplacian" else out

            xb = x if ctx.batched else x[None, :]
            value = torch.as_tensor(
                engine(f, xb, power=ctx.power) if operator == "laplacian" else engine(f, xb)
            )
            u = torch.func.vmap(ctx.point)(xb)  # the model's own output shape
            if operator == "laplacian":
                value = value.reshape(u.shape)
            if not ctx.batched:
                value, u = value[0], u[0]
            return value, {"value": u}

        return run

    return factory


def register_engine(
    name: str,
    operator: str,
    engine: Callable[..., Any],
    *,
    work: Callable[[Descriptor], Work] | str,
    requires: Callable[[Descriptor], str | None] | None = None,
    max_power: int = 1,
    exact: bool = True,
    generic: bool = False,
    replace: bool = False,
) -> Algorithm:
    """Let the planner choose a third-party engine for ``operator`` (``"laplacian"`` or ``"divergence"``).

    ``engine(f, x, power=p)`` (``engine(f, x)`` for the divergence) receives the pointwise model ``f`` (one input point, a
    scalar or, for the divergence, a vector) and a batch ``x`` of shape ``(B, D)`` and returns the operator at every point,
    shape ``(B,)``; autograd must flow through it if gradients are wanted. ``work`` prices it: a function
    ``descriptor -> Work`` or the name of a registered algorithm to be priced like. The engine is refused, with a reason,
    for requests it cannot serve (see :func:`omnibias.core.engines.engine_algorithm`), competes with the built-in programs
    on the cost model, can be timed by ``mode="find"`` and chosen by ``force=name``. ``generic=True`` also makes it
    available for models that are not a recognised dense chain (after the built-in generic executors, or by ``force``).
    """
    algorithm = engine_algorithm(
        name,
        operator,
        work=work,
        backend="torch",
        requires=requires,
        max_power=max_power,
        exact=exact,
        generic=generic,
    )
    if (operator, name) in _EXECUTORS and not replace:
        raise ValueError(f"an executor named {name!r} already serves {operator!r}")
    register_algorithm(algorithm, replace=replace)
    _EXECUTORS[(operator, name)] = _engine_runner(operator, engine)
    if generic and name not in _GENERIC_ORDER[operator]:
        _GENERIC_ORDER[operator] = (*_GENERIC_ORDER[operator], name)
    return algorithm


def unregister_engine(name: str, operator: str) -> None:
    """Remove an engine added by :func:`register_engine`; the built-in programs cannot be removed."""
    if name in _BUILTIN_EXECUTORS.get(operator, ()):
        raise ValueError(f"{name!r} is a built-in program of {operator!r}, not a registered engine")
    unregister_algorithm(name, frozenset({operator}))
    _EXECUTORS.pop((operator, name), None)
    _GENERIC_ORDER[operator] = tuple(a for a in _GENERIC_ORDER[operator] if a != name)


_BUILTIN_EXECUTORS: dict[str, tuple[str, ...]] = {}
for _op, _name in _EXECUTORS:
    _BUILTIN_EXECUTORS[_op] = (*_BUILTIN_EXECUTORS.get(_op, ()), _name)


def _widen(run: Runner, compute_dtype: str | None) -> Runner:
    """``run`` evaluated in ``compute_dtype`` (inputs lifted, results returned in the input dtype)."""
    if compute_dtype is None:
        return run
    wide = getattr(torch, compute_dtype)

    def lifted(x: Tensor, extra: tuple[Any, ...], key: Any) -> tuple[Tensor, dict[str, Tensor]]:
        ref = x.dtype
        cast_extra = tuple(
            e.to(wide) if isinstance(e, Tensor) and e.is_floating_point() else e for e in extra
        )
        value, aux = run(x.to(wide), cast_extra, key)
        return value.to(ref), {k: v.to(ref) for k, v in aux.items()}

    return lifted


# ----------------------------------------------------------------------------------------------------- front end


@functools.cache
def _versions_once() -> tuple[tuple[str, str], ...]:
    """Installed versions, read once per process: the package metadata is read from disk on every ``version`` call."""
    out = [("torch", torch.__version__)]
    for pkg in ("omnibias-core", "omnibias-torch"):
        try:
            out.append((pkg, importlib.metadata.version(pkg)))
        except importlib.metadata.PackageNotFoundError:
            out.append((pkg, "unknown"))
    return tuple(out)


def _versions() -> dict[str, str]:
    return dict(_versions_once())


_DEFAULT_CACHE = PlanCache()
_PERSISTENT: dict[str, PlanCache] = {}


def _persistent_cache() -> PlanCache:
    """The on-disk plan cache of ``mode="auto"`` (``$OMNIBIAS_PLAN_CACHE=off`` keeps it in process)."""
    path = default_plan_cache_path()
    if path is None:
        return _DEFAULT_CACHE
    if str(path) not in _PERSISTENT:
        _PERSISTENT[str(path)] = PlanCache(path)
    return _PERSISTENT[str(path)]


def _device_description(x: Tensor, device: Any | None) -> tuple[str, torch.device]:
    dev = torch.device(device) if device is not None else x.device
    if dev.type == "cuda":
        return device_class(torch.cuda.get_device_name(dev)), dev
    if os.environ.get("TRITON_INTERPRET", "") == "1":
        # tests and debugging: accelerator kernel programs stay candidates and run in the Triton interpreter
        return "cpu-interpret", dev
    return "cpu", dev


def _default_budget(dev: torch.device) -> float | None:
    if dev.type != "cuda":
        return None
    try:
        return 0.75 * float(torch.cuda.get_device_properties(dev).total_memory)
    except Exception:  # noqa: BLE001 - not every build reports memory
        return None


def _metric_of(diffusion: Any) -> str:
    if diffusion is None:
        return "euclidean"
    return "diagonal" if np.ndim(_numpy(diffusion)) == 1 else "factor"


def _sync(dev: torch.device) -> None:
    if dev.type == "cuda":
        torch.cuda.synchronize(dev)


def _out_of_memory(err: BaseException) -> bool:
    return isinstance(err, torch.OutOfMemoryError)


def _matmul_mode() -> tuple[str, str]:
    """The process's float32 matrix-product setting and which API holds it (``"new"`` from PyTorch 2.9)."""
    matmul = torch.backends.cuda.matmul
    if hasattr(matmul, "fp32_precision"):
        return "new", str(matmul.fp32_precision)
    return "old", torch.get_float32_matmul_precision()


def _set_matmul_mode(api: str, value: str) -> None:
    if api == "new":
        torch.backends.cuda.matmul.fp32_precision = value
    else:
        torch.set_float32_matmul_precision(value)


@contextlib.contextmanager
def matmul_precision(policy: PrecisionPolicy) -> Any:
    """Apply a policy's float32 matrix-product mode for the duration of a call and restore the process setting.

    ``"highest"`` disables TF32 and ``"tf32"`` enables it for matrix products; ``"default"`` leaves the process
    setting alone.  Tensor-core precision is a process-wide flag in PyTorch, so concurrent threads that need
    different modes must serialise.  PyTorch 2.9 and later hold it in ``torch.backends.cuda.matmul.fp32_precision``;
    earlier versions in ``torch.set_float32_matmul_precision``.
    """
    if policy.matmul == "default":
        yield
        return
    api, previous = _matmul_mode()
    names = (
        {"highest": "ieee", "tf32": "tf32"}
        if api == "new"
        else {"highest": "highest", "tf32": "high"}
    )
    _set_matmul_mode(api, names[policy.matmul])
    try:
        yield
    finally:
        _set_matmul_mode(api, previous)


@dataclass
class _GraphEntry:
    """One captured evaluation: the graph, its static input and outputs, and the parameter addresses it was captured with."""

    graph: Any
    static_x: Tensor
    value: Tensor
    aux: dict[str, Tensor]
    ptrs: tuple[int, ...]

    def release(self) -> None:
        self.graph.reset()


def _abandon_capture(graph: Any, device: torch.device) -> None:
    """Undo a capture that failed part way.

    The default CUDA generator stays flagged as capturing and every later draw from it raises; giving it a fresh state
    clears the flag. See :mod:`omnibias.torch.experimental.training_step`, which does the same for its step.
    """
    for undo in (torch.cuda.synchronize, graph.reset, lambda: _renew_generator(device), torch.cuda.synchronize, torch.cuda.empty_cache):
        try:
            undo()
        except Exception:  # noqa: BLE001, S110 - recovery is best effort; the capture error is the one to report
            pass


def _renew_generator(device: torch.device) -> None:
    try:
        torch.rand(1, device=device)
        return
    except RuntimeError:
        pass
    index = device.index if device.index is not None else torch.cuda.current_device()
    torch.cuda.default_generators[index].graphsafe_set_state(torch.Generator(device=device).graphsafe_get_state())


class CompiledOperator:
    """A planned derivative operator.  Call it like the model: ``op(x)`` (``op(x, rhs)`` for ``"solve"``).

    ``__call__`` returns the operator value; :meth:`evaluate` also returns the by-products the chosen program formed.
    :meth:`explain` reports what was matched, what was chosen and why, what the alternatives were predicted to cost,
    and what was refused, with the keys of the JAX front end.
    """

    def __init__(
        self,
        ctx: _Ctx,
        *,
        plan: Plan,
        matched: bool,
        fallback_reason: str | None,
        chain: Mapping[str, Any] | None,
        request: OperatorRequest,
        compute_dtype: str | None,
        compile: bool,
        alternatives: Sequence[str] = (),
        precision: PrecisionPolicy | None = None,
        precision_notes: Sequence[str] = (),
        graph: bool | str = False,
    ) -> None:
        if graph not in (False, True, "auto"):
            raise ValueError("graph must be False, True or 'auto'")
        self._graph_mode = graph
        self._graph_entries: dict[tuple[Any, ...], _GraphEntry] = {}
        self._graph_why: str | None = None
        self._ctx = ctx
        self.precision = PrecisionPolicy() if precision is None else precision
        self.precision_notes = list(precision_notes)
        self.plan = plan
        self.matched = matched
        self.fallback_reason = fallback_reason
        self._chain = None if chain is None else dict(chain)
        self.request = request
        self.compute_dtype = compute_dtype
        self._compile = compile
        self._alternatives = list(alternatives)
        self.first_call_seconds: float | None = None
        self.executor_failures: list[Refusal] = []
        self.descriptor: Descriptor | None = None
        self._rebuild()

    def _rebuild(self) -> None:
        run = _EXECUTORS[(self.request.kind, self.plan.algorithm)](self._ctx)
        self._run = _widen(run, self.compute_dtype)
        self._apply: Callable[..., Any] = self._call_impl
        if self._compile:
            self._apply = torch.compile(self._call_impl, dynamic=False)

    @property
    def algorithm(self) -> str:
        return self.plan.algorithm

    def _call_impl(
        self, x: Tensor, extra: tuple[Any, ...], key: Any
    ) -> tuple[Tensor, dict[str, Tensor]]:
        with matmul_precision(self.precision):
            return self._run(x, extra, key)

    def _advance(self, err: BaseException) -> bool:
        self.executor_failures.append(
            Refusal(self.plan.algorithm, f"refused at run time: {err}"[:300])
        )
        if not self._alternatives:
            return False
        name = self._alternatives.pop(0)
        self.plan = replace(self.plan, algorithm=name, mode=self.plan.mode + "+memory_retry")
        self._rebuild()
        return True

    def _replay(self, x: Tensor, extra: tuple[Any, ...], key: Any) -> OperatorOutput | None:
        """The value from a captured CUDA graph, or ``None`` when this call runs eagerly (the reason is in ``explain()``).

        A graph records kernels, not autograd, and holds the addresses of its inputs: it serves a call that needs no
        gradient, with no extra argument or random key, on a recognised chain. The parameters are read in place (an
        optimizer step updates them); a replaced parameter tensor triggers a new capture.
        """
        if not x.is_cuda:
            self._graph_why = "the input is not on a CUDA device"
        elif extra or key is not None:
            self._graph_why = "the call has extra arguments or a random key"
        elif self._ctx.layers is None:
            self._graph_why = "the model is not a recognised dense chain (its parameters are not known)"
        else:
            tensors = [t for W, b, _ in self._ctx.layers() for t in (W, b) if isinstance(t, Tensor)]
            if torch.is_grad_enabled() and (x.requires_grad or any(t.requires_grad for t in tensors)):
                self._graph_why = "the call records gradients (run it under torch.no_grad() for graph replay)"
            else:
                sig = (tuple(x.shape), x.dtype, x.device, self.plan.algorithm)
                ptrs = tuple(t.data_ptr() for t in tensors)
                entry = self._graph_entries.get(sig)
                if entry is not None and entry.ptrs != ptrs:
                    entry.release()
                    entry = None
                if entry is None:
                    entry = self._capture(x, ptrs)
                    if entry is None:
                        return None
                    self._graph_entries[sig] = entry
                entry.static_x.copy_(x)
                entry.graph.replay()
                self._graph_why = None
                return OperatorOutput(entry.value.clone(), {k: v.clone() for k, v in entry.aux.items()})
        if self._graph_mode is True:
            raise RuntimeError(f"graph=True but this call cannot be replayed: {self._graph_why}")
        return None

    def _capture(self, x: Tensor, ptrs: tuple[int, ...]) -> _GraphEntry | None:
        static_x = x.detach().clone()
        graph = torch.cuda.CUDAGraph()
        try:
            with torch.no_grad(), capture_safe_branches():
                side = torch.cuda.Stream()  # type: ignore[no-untyped-call]
                side.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(side):
                    for _ in range(3):  # warm-up: library handles and workspaces are created outside the capture
                        self._call_impl(static_x, (), None)
                torch.cuda.current_stream().wait_stream(side)
                torch.cuda.synchronize()
                with torch.cuda.graph(graph):
                    value, aux = self._call_impl(static_x, (), None)
            torch.cuda.synchronize()
        except BaseException as err:
            _abandon_capture(graph, x.device)
            self._graph_why = f"capture failed: {type(err).__name__}: {str(err)[:200]}"
            if self._graph_mode is True or not isinstance(err, Exception):
                raise
            return None
        return _GraphEntry(graph, static_x, value, aux, ptrs)

    def evaluate(self, x: Tensor, *extra: Any, key: Any = None) -> OperatorOutput:
        x = torch.as_tensor(x)
        first = self.first_call_seconds is None
        t0 = time.perf_counter()
        if self._graph_mode:
            replayed = self._replay(x, tuple(extra), key)
            if replayed is not None:
                if first:
                    _sync(x.device)
                    self.first_call_seconds = time.perf_counter() - t0
                return replayed
        while True:
            try:
                value, aux = self._apply(x, tuple(extra), key)
                break
            except torch.OutOfMemoryError as err:
                if x.is_cuda:
                    torch.cuda.empty_cache()
                if not self._advance(err):
                    err.add_note(
                        "every candidate was refused: "
                        + "; ".join(f"{f.algorithm}: {f.reason}" for f in self.executor_failures)
                    )
                    raise
        if first:
            _sync(x.device)
            self.first_call_seconds = time.perf_counter() - t0
        return OperatorOutput(value, aux)

    def __call__(self, x: Tensor, *extra: Any, key: Any = None) -> Tensor:
        return self.evaluate(x, *extra, key=key).value

    def explain(self) -> dict[str, Any]:
        out = self.plan.explain()
        out.update(
            {
                "backend": "torch",
                "matched_patterns": ["dense_chain"] if self.matched else [],
                "fallback_reason": self.fallback_reason,
                "input_products": (
                    None
                    if self._ctx.input_rows is None
                    else {"rows": int(self._ctx.input_rows().shape[0])}
                ),
                "chain": self._chain,
                "request": self.request.to_config(),
                "first_call_seconds": self.first_call_seconds,
                "compute_dtype": self.compute_dtype,
                "precision": {**self.precision.to_config(), "notes": self.precision_notes},
                "executor_failures": [
                    {"algorithm": r.algorithm, "reason": r.reason} for r in self.executor_failures
                ],
                "cold_cost": {
                    "predicted_seconds": self.plan.compile_seconds,
                    "bucket": None,
                    "compile_policy": {"policy": "torch.compile" if self._compile else "eager"},
                    "persistent_cache_dir": None,
                    "executables": 0,
                    "stages": [],
                },
            }
        )
        if self._graph_mode:
            out["graph"] = {
                "mode": self._graph_mode,
                "captured": len(self._graph_entries),
                "eager_reason": self._graph_why,
            }
        return out


@dataclass(frozen=True)
class _Setup:
    fn: Callable[[Tensor], Tensor]
    layers: Callable[[], list[tuple[Tensor, Tensor | None, Any]]] | None
    x: Tensor
    squeeze: bool
    fallback_reason: str | None
    input_rows: Callable[[], Tensor] | None = None
    input_reason: str | None = None

    @property
    def matched(self) -> bool:
        return self.layers is not None


def _is_layer_model(model: Any) -> bool:
    return (
        hasattr(model, "layer_specs")
        or hasattr(model, "_layer_specs")
        or (isinstance(model, Sequence) and not isinstance(model, str | nn.Module))
    )


def _setup(model: Any, example: Tensor) -> _Setup:
    if _is_layer_model(model):
        layers0 = layers_of(model)

        def accessor() -> list[tuple[Tensor, Tensor | None, Any]]:
            return layers_of(model)

        def fn(x: Tensor) -> Tensor:
            return _chain_value(accessor(), x)

        del layers0
        return _Setup(fn, accessor, example, False, None)
    if not callable(model):
        raise TypeError("model must be a layer list, an object with _layer_specs(), or a callable")
    reason: str | None = (
        "the model is not a layer list, a Sequential of Linear and activations, or a chain provider"
    )
    accessor_m = chain_from_module(model) if isinstance(model, nn.Module) else None
    squeeze = False
    probe = example if example.ndim == 2 else example[None, :]
    with torch.no_grad():
        out = model(probe)
    squeeze = out.ndim == 1
    if accessor_m is not None:
        with torch.no_grad():
            chain_out = _chain_value(accessor_m(), probe)
        tol = float(torch.finfo(probe.dtype).eps) ** 0.5
        if chain_out.shape == out.reshape(chain_out.shape).shape and torch.allclose(
            chain_out, out.reshape(chain_out.shape), rtol=tol, atol=tol
        ):
            return _Setup(model, accessor_m, example, squeeze, None)
        reason = "the dense chain read from the module does not reproduce its forward pass on the example"
    rows, rows_reason = (
        analyze_input_weights(model) if isinstance(model, nn.Module) else (None, "not an nn.Module")
    )
    return _Setup(model, None, example, squeeze, reason, rows, rows_reason)


def compile_planned_operator(
    model: Any,
    example_args: Any,
    *,
    operator: str = "laplacian",
    power: int = 1,
    coordinates: Sequence[int] | np.ndarray | None = None,
    diffusion: Any = None,
    directions: Any = None,
    transpose: bool = False,
    mode: str = "heuristic",
    memory_budget: float | None = None,
    expected_uses: int | str = 1,
    force: str | None = None,
    exact: bool = True,
    allow_almost_everywhere: bool = False,
    n_probes: int | None = None,
    device: Any | None = None,
    cache: PlanCache | str | None = None,
    cost_model: CostModel | None = None,
    compute_dtype: str | None = None,
    precision: str | PrecisionPolicy | None = None,
    allow_low_precision: bool = False,
    compile: bool = False,
    differentiate: str = "none",
    accuracy: float | None = None,
    uses_hint: int | None = None,
    graph: bool | str = False,
    top_k: int | None = FIND_TOP_K,
) -> CompiledOperator:
    r"""Plan ``operator`` for ``model`` on ``example_args`` and return a callable.

    Parameters mirror :func:`omnibias.jax.experimental.compile_operator`; the model is called as ``model(x)`` (batched
    ``(B, D)``) or read as a dense chain. ``example_args`` is the example input ``x`` or ``(x,)``.

    ``operator`` is ``"laplacian"`` (``power`` > 1 for ``Delta^p``), ``"divergence"``, ``"logdet"`` or ``"solve"``
    (``(I + J) y = rhs`` with ``rhs`` the second call argument). ``mode="find"`` times the leading candidates on the
    example (``top_k=2`` of them by default, or every feasible candidate with ``top_k=None``; a non-default ``top_k`` is
    part of the cache key); ``mode="auto"`` finds once ``expected_uses`` reaches :data:`omnibias.core.planner.AUTO_FIND_MIN_USES` (and
    caches the plan on disk), ``expected_uses="auto"`` takes ``uses_hint`` or 1000 for ``differentiate="params"``;
    ``differentiate`` (``"none"``, ``"params"``, ``"params2"``) names the objective that is ranked and timed (a forward
    pass or value-and-gradient into the model parameters); ``accuracy`` sizes the stochastic estimators; ``memory_budget`` (bytes) refuses candidates predicted to exceed it (75 % of device memory by default on an
    accelerator); a candidate that runs out of memory at run time is replaced by the next feasible one.
    ``compute_dtype="float64"`` evaluates a chain in double precision and returns the input dtype.
    ``compile=True`` wraps the program in ``torch.compile`` (opt-in; the cold cost then includes compilation).
    ``graph="auto"`` (or ``True``, which raises instead of falling back) serves a no-gradient CUDA call on a recognised
    chain from a captured CUDA graph, one replay for the whole program; any other call runs eagerly and ``explain()["graph"]``
    gives the reason. The outputs are copies, so they stay valid after the next call. Default: off.
    ``precision`` is the float32 matrix-product mode: ``"highest"`` (default; TF32 is off), ``"tf32"`` (opt-in) or
    ``"default"`` (leave the process setting); half-precision inputs are refused for operators of derivative order two
    and above unless ``allow_low_precision`` is set.

    Raises
    ------
    ValueError
        If no program is feasible for the request; the message lists every refusal reason.
    """
    args = tuple(example_args) if isinstance(example_args, tuple | list) else (example_args,)
    example = torch.as_tensor(args[0])
    if example.ndim not in (1, 2):
        raise ValueError("x must have shape (D,) or (B, D)")
    setup = _setup(model, example)
    batched = example.ndim == 2
    coords = None if coordinates is None else np.asarray(coordinates, dtype=int)
    dirs = None if directions is None else _numpy(directions)
    if compute_dtype not in (None, "float64"):
        raise ValueError("compute_dtype must be None or 'float64'")
    if compute_dtype is not None and setup.layers is None:
        raise ValueError("compute_dtype is available for dense chains only")
    request = OperatorRequest(
        operator,
        power=power,
        selected=None if coords is None else int(len(coords)),
        metric=_metric_of(diffusion),
        directions=0 if dirs is None else int(dirs.shape[-1]),
        exact=exact,
        allow_almost_everywhere=allow_almost_everywhere,
        differentiate=differentiate,
        accuracy=accuracy,
    )
    if request.kind not in _GENERIC_ORDER:
        raise ValueError(
            f"operator {operator!r} is not an operator program; residual Gram, Gauss-Newton and NTK programs "
            "are planned by compile_residual"
        )
    policy = resolve_precision(precision, allow_low_precision)
    precision_notes = policy.check(
        torch.float64 if compute_dtype == "float64" else example.dtype,
        request_order(request.kind, request.power),
        what=f"operator {operator!r}",
    )
    layer_accessor = setup.layers
    if compute_dtype is not None and layer_accessor is not None:
        wide = getattr(torch, compute_dtype)
        base_accessor = layer_accessor

        def layer_accessor() -> list[tuple[Tensor, Tensor | None, Any]]:  # noqa: F811
            return [
                (W.to(wide), None if b is None else b.to(wide), s) for W, b, s in base_accessor()
            ]

    ctx = _Ctx(
        fn=setup.fn,
        layers=layer_accessor,
        squeeze=setup.squeeze,
        batched=batched,
        dim=int(example.shape[-1]),
        power=request.power,
        coordinates=coords,
        diffusion=diffusion,
        directions=dirs,
        allow_ae=allow_almost_everywhere,
        transpose=transpose,
        n_probes=n_probes,
        input_rows=setup.input_rows,
        differentiate=differentiate,
        accuracy=accuracy,
        matmul=policy.matmul,
    )
    dev_class, dev = _device_description(example, device)
    budget = memory_budget if memory_budget is not None else _default_budget(dev)
    resolved_mode, uses, uses_source = resolve_planning(
        mode, expected_uses, differentiate=differentiate, uses_hint=uses_hint
    )
    plan_cache = (
        cache
        if isinstance(cache, PlanCache)
        else PlanCache(cache)
        if isinstance(cache, str)
        else _persistent_cache()
        if resolved_mode == "find"
        else _DEFAULT_CACHE
    )
    versions = _versions()
    rhs_probe: tuple[Any, ...] = (torch.zeros_like(example),) if request.kind == "solve" else ()
    x0 = example.to(dev)
    trainable: list[Tensor] = []
    if differentiate != "none":
        if setup.layers is not None:
            trainable = [
                t for W, b, _ in setup.layers() for t in (W, b) if t is not None and t.requires_grad
            ]
        elif isinstance(model, nn.Module):
            trainable = [p for p in model.parameters() if p.requires_grad]
        if not trainable:
            raise ValueError(
                "differentiate='params' needs a model with parameters that require grad"
            )

    def objective(run: Runner) -> None:
        if differentiate == "none":
            with torch.no_grad():
                run(x0, rhs_probe, 0)
            return
        value, aux = run(x0, rhs_probe, 0)
        total = (value * value).sum() + sum(
            (a.sum() for a in aux.values() if a.is_floating_point()), value.new_zeros(())
        )
        torch.autograd.grad(total, trainable, allow_unused=True)

    find_failures: list[Refusal] = []

    def timer(name: str, deadline: float | None = None) -> float:
        """A candidate that cannot be built or run (out of memory included) is a failed measurement, not a failed plan.

        What is timed is what the operator runs: the executor widened to the compute dtype, under the matmul precision,
        through ``torch.compile`` when the operator is compiled.  The CUDA allocator cache is emptied after every candidate
        so that the blocks one program held are not charged to the next.
        """
        try:
            run = _widen(_EXECUTORS[(request.kind, name)](ctx), compute_dtype)

            def under_precision(x: Tensor, extra: tuple[Any, ...], key: Any) -> tuple[Tensor, dict[str, Tensor]]:
                with matmul_precision(policy):
                    return run(x, extra, key)

            call: Runner = torch.compile(under_precision, dynamic=False) if compile else under_precision
            objective(call)
            _sync(dev)
            times = []
            for i in range(5):
                _sync(dev)
                t0 = time.perf_counter()
                objective(call)
                _sync(dev)
                times.append(time.perf_counter() - t0)
                if i == 0 and deadline is not None and times[0] > deadline:
                    return times[0]  # slower by the cut factor than the best so far: it cannot be the minimum
            return float(np.median(times))
        except Exception as err:  # noqa: BLE001
            find_failures.append(Refusal(name, f"find: {type(err).__name__}: {str(err)[:200]}"))
            return float("inf")
        finally:
            if dev.type == "cuda":
                torch.cuda.empty_cache()

    chain_config: Mapping[str, Any] | None = None
    descriptor: Descriptor | None = None
    if setup.layers is not None:
        chain: ChainDescriptor = describe_layers(setup.layers())
        chain_config = chain.to_config()
        descriptor = Descriptor(request, chain, int(example.shape[0]) if batched else 1)
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
            uses_hint=uses_hint,
            force=force,
            cache=plan_cache,
            versions=versions,
            key_extra=measurement_context(policy.matmul, compute_dtype, "torch.compile" if compile else None),
            find_memory_limit=None if budget is not None else host_memory_limit(),
        )
        order = [plan.algorithm] + [
            c["name"] for c in plan.candidates if c["name"] != plan.algorithm
        ]
    else:
        plan, order = _fallback_plan(
            request,
            exact,
            force,
            dev_class,
            budget,
            uses,
            versions,
            ctx=ctx,
            reason=setup.input_reason,
            uses_source=uses_source,
        )
    if force is not None:
        # a forced program that cannot run is an error, never a silent substitution of another program
        order = [plan.algorithm]
    failures: list[Refusal] = list(find_failures)
    chosen = None
    for name in order:
        try:
            run = _widen(_EXECUTORS[(request.kind, name)](ctx), compute_dtype)
            with torch.no_grad():
                run(x0, rhs_probe, 0)
        except Exception as err:  # noqa: BLE001 - an executor that cannot run this model is a refusal, not a crash
            failures.append(
                Refusal(
                    name,
                    f"executor could not run this model: {type(err).__name__}: {str(err)[:200]}",
                )
            )
            continue
        chosen = name
        break
    if chosen is None:
        raise ValueError(
            "no executor could run this model: "
            + "; ".join(f"{f.algorithm}: {f.reason}" for f in failures)
        )
    if chosen != plan.algorithm:
        plan = replace(plan, algorithm=chosen, mode=plan.mode + "+retry")
    op = CompiledOperator(
        ctx,
        plan=plan,
        matched=setup.matched,
        fallback_reason=setup.fallback_reason,
        chain=chain_config,
        request=request,
        compute_dtype=compute_dtype,
        compile=compile,
        graph=graph,
        alternatives=order[order.index(chosen) + 1 :],
        precision=policy,
        precision_notes=precision_notes,
    )
    op.executor_failures = failures
    op.descriptor = descriptor
    return op


def _fallback_plan(
    request: OperatorRequest,
    exact: bool,
    force: str | None,
    device: str,
    budget: float | None,
    expected_uses: int,
    versions: Mapping[str, str],
    *,
    ctx: _Ctx | None = None,
    reason: str | None = None,
    uses_source: str = "explicit",
) -> tuple[Plan, list[str]]:
    """Plan for a model that is not a recognised chain: only the generic executors, in a fixed order.

    A module that reads its input only through its own weight products is served first by the subspace programs,
    which differentiate along ``min(rows, columns)`` directions instead of every input coordinate.
    """
    generic = [
        a for a in _GENERIC_ORDER[request.kind] if not (a == "taylor_jet" and request.power > 1)
    ]
    refusals: list[Refusal] = []
    notes: dict[str, str] = {}
    if request.kind == "laplacian":
        if ctx is not None and ctx.input_rows is not None:
            with torch.no_grad():
                rows = int(ctx.input_rows().shape[0])
            q = int(ctx.columns().shape[1])
            k = min(rows, q)
            if k < q:
                names = (
                    ["subspace_jet", "subspace_nested"]
                    if request.power == 1
                    else ["subspace_nested"]
                )
                generic = [*names, *generic]
                for name in names:
                    notes[name] = (
                        f"the input enters only through weight products of {rows} rows: {k} directions replace {q}"
                    )
            else:
                refusals.append(
                    Refusal(
                        "subspace",
                        f"no reduction: the input products have {rows} rows for {q} direction columns",
                    )
                )
        elif reason is not None and reason != "not an nn.Module":
            refusals.append(Refusal("subspace", reason))
    if not exact and request.kind == "laplacian" and request.power == 1:
        generic = ["hutchinson", "stde", "sdgd", *generic]
    if force is not None:
        if force not in generic:
            raise ValueError(
                f"cannot force {force!r} for a model that is not a dense chain; use one of {generic}"
            )
        generic = [force]
    plan = Plan(
        descriptor="unmatched",
        operator=request.kind,
        algorithm=generic[0],
        mode="forced" if force else "fallback",
        device=device,
        candidates=tuple(
            {
                "name": a,
                "seconds": None,
                "peak_bytes": None,
                "compile_seconds": None,
                "generic": True,
                **({"note": notes[a]} if a in notes else {}),
            }
            for a in generic
        ),
        refusals=tuple(refusals),
        memory_budget=budget,
        expected_uses=max(1, int(expected_uses)),
        versions={
            **versions,
            "cost_model": "none: unmatched model, generic executors are not priced",
        },
        objective="forward" if request.differentiate == "none" else request.differentiate,
        uses_source=uses_source,
    )
    return plan, generic
