# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Torch CNF operators: exact trace-of-Jacobian via omnibias-fields divergence."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

import torch
from omnibias.fields._core.components import ComponentSpec
from omnibias.fields._core.coords import CoordinateSpec
from omnibias.fields._core.sigma_cache import SigmaCache
from omnibias.fields._core.state import FieldState
from omnibias.fields.torch.ops.basic import divergence
from torch import Tensor

if TYPE_CHECKING:  # pragma: no cover
    from types import ModuleType

VelocityFn = Callable[[Tensor | float, Tensor], Tensor]
#: A trace-of-Jacobian estimator ``(velocity_fn, t, x) -> tr(dv/dx)`` of shape ``(...,)``.
#: :func:`exact_trace_jacobian` is one (deterministic); a fixed-noise
#: :func:`hutchinson_trace_jacobian` closure is the stochastic FFJORD baseline.
TraceFn = Callable[["VelocityFn", "Tensor | float", Tensor], Tensor]


class _CallableVelocityField:
    """Spectral-dispatch field wrapping a user velocity callable.

    Each spatial component of ``v(t, x)`` is exposed as a named scalar
    component; first derivatives are obtained via torch autograd so the
    omnibias-fields ``divergence`` op yields ``tr(partial v / partial x)``.
    """

    _omnibias_dispatch = "spectral"

    def __init__(
        self,
        velocity_fn: VelocityFn,
        t: Tensor | float,
        dim: int,
        ops_module: ModuleType,
    ) -> None:
        self._velocity_fn = velocity_fn
        self._t = t
        self.coordinate_spec = CoordinateSpec(
            tuple(f"x{i}" for i in range(dim)),
            time_axis=None,
        )
        self.components = ComponentSpec(
            tuple(f"v{i}" for i in range(dim)),
            groups={"velocity": tuple(f"v{i}" for i in range(dim))},
        )
        self._ops = ops_module

    def evaluate(self, coords: Tensor) -> FieldState:
        return FieldState(
            coords=coords,
            field=self,
            components=self.components,
            coordinate_spec=self.coordinate_spec,
            ops=self._ops,
            sigma_cache=SigmaCache(z=coords),
        )

    __call__ = evaluate

    def _comp_index(self, name: str) -> int:
        return self.components.index(name)

    def value_component(self, state: FieldState, name: str) -> Tensor:
        v = self._velocity_fn(self._t, state.coords)
        return v[..., self._comp_index(name)]

    def derivative(
        self,
        state: FieldState,
        name: str,
        *,
        axis: int,
        order: int,
    ) -> Tensor:
        if order == 0:
            return self.value_component(state, name)
        if order != 1:
            raise NotImplementedError(f"order {order} not supported for callable velocity")
        comp = self._comp_index(name)
        x = state.coords
        if not x.requires_grad:
            x = x.detach().requires_grad_(True)
            state = FieldState(
                coords=x,
                field=self,
                components=self.components,
                coordinate_spec=self.coordinate_spec,
                ops=self._ops,
                sigma_cache=SigmaCache(z=x),
            )
        v_comp = self._velocity_fn(self._t, x)[..., comp]
        grad = torch.autograd.grad(
            v_comp.sum(),
            x,
            create_graph=torch.is_grad_enabled(),
            retain_graph=True,
        )[0]
        return grad[..., axis]

    def mixed_partial(
        self,
        state: FieldState,
        name: str,
        axes: tuple[int, ...],
        orders: tuple[int, ...],
    ) -> Tensor:
        raise NotImplementedError(
            "mixed_partial is not supported for callable CNF velocity fields; "
            "only first-order single-axis derivatives (via autograd) are available. "
            f"Got name={name!r}, axes={axes!r}, orders={orders!r}."
        )


def _flatten_leading(x: Tensor) -> tuple[Tensor, tuple[int, ...]]:
    if x.ndim < 2:
        raise ValueError(f"x must have shape (..., d) with d >= 1, got {tuple(x.shape)}")
    batch_shape = x.shape[:-1]
    if not batch_shape:
        return x.unsqueeze(0), ()
    return x.reshape(-1, x.shape[-1]), batch_shape


def _restore_leading(y: Tensor, batch_shape: tuple[int, ...]) -> Tensor:
    if not batch_shape:
        return y.squeeze(0)
    return y.reshape(*batch_shape)


def _velocity_state(
    velocity_fn: VelocityFn,
    t: Tensor | float,
    x: Tensor,
) -> FieldState:
    from omnibias.fields.torch import _ops_dispatch

    x_flat, batch_shape = _flatten_leading(x)
    if not x_flat.requires_grad:
        x_flat = x_flat.detach().requires_grad_(True)
    dim = x_flat.shape[-1]
    field = _CallableVelocityField(velocity_fn, t, dim, _ops_dispatch)
    state = field(x_flat)
    state.extra["batch_shape"] = batch_shape
    return state


def exact_trace_jacobian(
    velocity_fn: VelocityFn,
    t: Tensor | float,
    x: Tensor,
) -> Tensor:
    r"""Return ``tr(partial v / partial x)`` of shape ``(...,)``.

    Computed exactly via the omnibias-fields ``divergence`` operator applied
    to the velocity vector field ``v(t, x)``.
    """
    state = _velocity_state(velocity_fn, t, x)
    names = state.components.names
    div = divergence(state, names)
    return _restore_leading(div, state.extra["batch_shape"])


def hutchinson_trace_jacobian(
    velocity_fn: VelocityFn,
    t: Tensor | float,
    x: Tensor,
    noise: Tensor,
) -> Tensor:
    r"""Single-probe Hutchinson estimate of ``tr(partial v / partial x)``.

    Returns ``noise^T (partial v / partial x) noise`` of shape ``(...,)`` -- the
    standard FFJORD stochastic trace, **one** vector-Jacobian product (one backward
    pass) *regardless of dimension*, versus the exact ``O(d)``-pass
    :func:`exact_trace_jacobian`. It is an **unbiased** estimator of the trace when the
    entries of ``noise`` are i.i.d. mean-zero, unit-variance (Rademacher ``+-1`` or
    standard normal): ``E[noise noise^T] = I`` so ``E[noise^T J noise] = tr(J)``. The
    price of the single pass is variance ``Var = sum_{i != j} J_ij^2`` (Rademacher),
    which this study measures against the zero-variance exact op. Differentiable in the
    velocity's parameters (``create_graph`` follows the ambient grad mode), so it drops
    into the same augmented integrator for maximum-likelihood training.
    """
    x_flat, batch_shape = _flatten_leading(x)
    noise_flat, _ = _flatten_leading(noise)
    x_req = x_flat if x_flat.requires_grad else x_flat.detach().requires_grad_(True)
    v = velocity_fn(t, x_req)
    (vjp,) = torch.autograd.grad(
        v,
        x_req,
        grad_outputs=noise_flat,
        create_graph=torch.is_grad_enabled(),
        retain_graph=True,
    )
    est = (vjp * noise_flat).sum(dim=-1)
    return _restore_leading(est, batch_shape)


def cnf_dynamics(
    velocity_fn: VelocityFn,
    t: Tensor | float,
    state: tuple[Tensor, Tensor],
    *,
    trace_fn: TraceFn | None = None,
) -> tuple[Tensor, Tensor]:
    r"""Augmented CNF ODE right-hand side ``(dx/dt, d log_p/dt)``.

    ``trace_fn`` selects the divergence estimator: ``None`` uses the exact
    :func:`exact_trace_jacobian`; pass a fixed-noise :func:`hutchinson_trace_jacobian`
    closure for the stochastic baseline.
    """
    x, _log_p = state
    v = velocity_fn(t, x)
    div = exact_trace_jacobian(velocity_fn, t, x) if trace_fn is None else trace_fn(velocity_fn, t, x)
    return v, -div


def _add_augmented(
    state: tuple[Tensor, Tensor],
    delta: tuple[Tensor, Tensor],
    scale: float,
) -> tuple[Tensor, Tensor]:
    x, lp = state
    dx, dlp = delta
    return x + scale * dx, lp + scale * dlp


def _cnf_step(
    velocity_fn: VelocityFn,
    t: float,
    state: tuple[Tensor, Tensor],
    dt: float,
    *,
    method: str,
    trace_fn: TraceFn | None = None,
) -> tuple[Tensor, Tensor]:
    if method == "euler":
        return _add_augmented(state, cnf_dynamics(velocity_fn, t, state, trace_fn=trace_fn), dt)
    if method == "rk4":
        k1 = cnf_dynamics(velocity_fn, t, state, trace_fn=trace_fn)
        k2 = cnf_dynamics(
            velocity_fn, t + 0.5 * dt, _add_augmented(state, k1, 0.5 * dt), trace_fn=trace_fn
        )
        k3 = cnf_dynamics(
            velocity_fn, t + 0.5 * dt, _add_augmented(state, k2, 0.5 * dt), trace_fn=trace_fn
        )
        k4 = cnf_dynamics(velocity_fn, t + dt, _add_augmented(state, k3, dt), trace_fn=trace_fn)
        inc = (
            (k1[0] + 2.0 * k2[0] + 2.0 * k3[0] + k4[0]) / 6.0,
            (k1[1] + 2.0 * k2[1] + 2.0 * k3[1] + k4[1]) / 6.0,
        )
        return _add_augmented(state, inc, dt)
    raise ValueError(f"unsupported integrator method {method!r}; use 'rk4' or 'euler'")


def integrate_cnf(
    velocity_fn: VelocityFn,
    x0: Tensor,
    t0: float,
    t1: float,
    *,
    steps: int = 20,
    log_p0: Tensor | float | None = None,
    method: str = "rk4",
    trace_fn: TraceFn | None = None,
) -> tuple[Tensor, Tensor] | tuple[Tensor, Tensor, Tensor]:
    r"""Fixed-step integration of the augmented CNF system.

    Returns ``(x1, delta_log_p)`` where ``delta_log_p = log p(x1) - log p(x0)``.
    When ``log_p0`` is supplied, also returns the terminal ``log_p1``. ``trace_fn``
    selects the divergence estimator (default exact; pass a fixed-noise Hutchinson
    closure for the stochastic baseline -- fix the probe once per solve so the
    integrated ``delta_log_p`` stays an unbiased log-det estimate).
    """
    if steps < 1:
        raise ValueError(f"steps must be >= 1, got {steps}")
    batch_shape = x0.shape[:-1]
    if log_p0 is None:
        lp = torch.zeros(batch_shape, dtype=x0.dtype, device=x0.device)
        lp0: Tensor | None = None
    else:
        lp0 = log_p0 if isinstance(log_p0, Tensor) else torch.full(
            batch_shape, float(log_p0), dtype=x0.dtype, device=x0.device,
        )
        lp = lp0.clone()
    t = float(t0)
    dt = (float(t1) - float(t0)) / steps
    state: tuple[Tensor, Tensor] = (x0, lp)
    for _ in range(steps):
        state = _cnf_step(velocity_fn, t, state, dt, method=method, trace_fn=trace_fn)
        t += dt
    x1, log_p1 = state
    if lp0 is None:
        return x1, log_p1
    return x1, log_p1 - lp0, log_p1


def log_prob(
    velocity_fn: VelocityFn,
    x1: Tensor,
    t1: float,
    t0: float,
    base_log_prob: Callable[[Tensor], Tensor],
    *,
    steps: int = 20,
    method: str = "rk4",
    trace_fn: TraceFn | None = None,
) -> Tensor:
    r"""Likelihood via backward integration to the base density.

    Integrates from ``t1`` to ``t0`` and returns ``log p(x1)`` under the
    implied transport map, i.e. ``base_log_prob(x0) - delta_log_p_rev``.
    ``trace_fn`` selects the divergence estimator (default exact).
    """
    result = integrate_cnf(
        velocity_fn, x1, t1, t0, steps=steps, method=method, trace_fn=trace_fn,
    )
    x0 = result[0]
    delta_rev = result[1]
    return base_log_prob(x0) - delta_rev


__all__ = [
    "TraceFn",
    "VelocityFn",
    "cnf_dynamics",
    "exact_trace_jacobian",
    "hutchinson_trace_jacobian",
    "integrate_cnf",
    "log_prob",
]
