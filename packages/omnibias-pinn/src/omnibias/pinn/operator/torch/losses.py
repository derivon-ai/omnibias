# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""DeepONet training losses with closed-form physics residuals (torch).

The physics residual is assembled from the existing field ops, so ``u_t``,
``u_x``, ``u_xx`` are closed-form trunk-jet reads -- no finite difference in
time.
"""

from __future__ import annotations

from typing import cast

import torch
from omnibias.pinn.operator.torch.data import OperatorSlab
from omnibias.pinn.operator.torch.deeponet import DeepONetOperator
from omnibias.pinn.torch import ops as tops
from torch import Tensor


def data_loss(
    operator: DeepONetOperator,
    slab: OperatorSlab,
    *,
    component: str = "u",
) -> Tensor:
    """Mean-square error of ``G(u_0)`` against the reference slab values."""
    field = operator.condition(slab.sensors)
    state = field.on_grid(slab.coords)
    pred = tops.value(state, component).reshape(slab.values.shape[0], -1)
    target = slab.values[..., 0] if slab.values.ndim == 3 else slab.values
    return torch.mean((pred - target) ** 2)


def heat_residual_loss(
    operator: DeepONetOperator,
    sensors: Tensor,
    coords: Tensor,
    *,
    diffusivity: float = 0.1,
    component: str = "u",
) -> Tensor:
    """Closed-form heat residual ``||u_t - D u_xx||^2`` on a shared query grid.

    ``coords`` columns are ``(x, t)`` matching :class:`DeepONetOperator` built
    with ``CoordinateSpec(("x", "t"))``.
    """
    field = operator.condition(sensors)
    state = field.on_grid(coords)
    u_t = tops.derivative(state, component, axis=1, order=1)
    u_xx = tops.derivative(state, component, axis=0, order=2)
    return torch.mean((u_t - float(diffusivity) * u_xx) ** 2)


def burgers_residual_loss(
    operator: DeepONetOperator,
    sensors: Tensor,
    coords: Tensor,
    *,
    viscosity: float = 0.05,
    component: str = "u",
) -> Tensor:
    """Closed-form Burgers residual ``||u_t + u u_x - nu u_xx||^2``."""
    field = operator.condition(sensors)
    state = field.on_grid(coords)
    u = tops.value(state, component)
    u_t = tops.derivative(state, component, axis=1, order=1)
    u_x = tops.derivative(state, component, axis=0, order=1)
    u_xx = tops.derivative(state, component, axis=0, order=2)
    return torch.mean((u_t + u * u_x - float(viscosity) * u_xx) ** 2)


def ks_residual_loss(
    operator: DeepONetOperator,
    sensors: Tensor,
    coords: Tensor,
    *,
    component: str = "u",
) -> Tensor:
    """Closed-form KS residual via the shipped :class:`KuramotoSivashinsky` equation.

    Requires the operator's ``CoordinateSpec`` to declare ``time_axis`` (e.g.
    ``CoordinateSpec(("x", "t"), time_axis="t")``) and ``jet_order >= 4``.
    The residual is assembled by the equation object itself -- no re-derivation.
    """
    from omnibias.pinn.torch.equations.kuramoto_sivashinsky import KuramotoSivashinsky

    field = operator.condition(sensors)
    state = field.on_grid(coords)
    out = KuramotoSivashinsky(component=component)(state)
    return torch.mean(out.residual**2)


def ks_residual_loss_fd(
    operator: DeepONetOperator,
    sensors: Tensor,
    coords: Tensor,
    *,
    h: float,
    component: str = "u",
) -> Tensor:
    """KS residual with ``u_xxxx`` by a 5-point central stencil in ``x``.

    ``u``, ``u_x``, ``u_xx``, and ``u_t`` stay closed-form (order-2 jet). Only
    ``u_xxxx`` is finite-differenced by shifting the query grid in ``x`` and
    re-evaluating the field value -- no periodic wrap, so this is FD's best
    case on a mesh-free DeepONet. Canonical KS coefficients (all 1).
    """
    if float(h) <= 0.0:
        raise ValueError(f"h must be positive, got {h}")
    field = operator.condition(sensors)
    state = field.on_grid(coords)
    u = tops.value(state, component)
    u_t = tops.derivative(state, component, axis=1, order=1)
    u_x = tops.derivative(state, component, axis=0, order=1)
    u_xx = tops.derivative(state, component, axis=0, order=2)

    def _value_at_shift(dx: float) -> Tensor:
        shifted = coords.clone()
        shifted[:, 0] = shifted[:, 0] + float(dx)
        return cast(Tensor, tops.value(field.on_grid(shifted), component))

    hh = float(h)
    um2 = _value_at_shift(-2.0 * hh)
    um1 = _value_at_shift(-hh)
    up1 = _value_at_shift(hh)
    up2 = _value_at_shift(2.0 * hh)
    u_xxxx = (um2 - 4.0 * um1 + 6.0 * u - 4.0 * up1 + up2) / (hh**4)
    resid = u_t + u * u_x + u_xx + u_xxxx
    return torch.mean(resid**2)


def heat_residual_loss_fd(
    operator: DeepONetOperator,
    sensors: Tensor,
    coords: Tensor,
    *,
    diffusivity: float = 0.1,
    dt: float,
    component: str = "u",
) -> Tensor:
    """Physics residual with ``u_t`` by a backward finite difference (the §3a convention).

    ``coords`` must lie on a uniform time grid with spacing ``dt``; the FD is
    taken against the same spatial points one step earlier. Kept as the named
    baseline against which the closed-form residual is measured.
    """
    field = operator.condition(sensors)
    state = field.on_grid(coords)
    u = tops.value(state, component)
    u_xx = tops.derivative(state, component, axis=0, order=2)
    # Backward difference in t: reshape to (F, T, n_x) assuming product grid.
    # Infer T, n_x from unique x / t counts.
    x = coords[:, 0]
    t = coords[:, 1]
    n_x = int(torch.unique(x).numel())
    n_t = int(torch.unique(t).numel())
    F = sensors.shape[0]
    u_b = u.reshape(F, n_t, n_x)
    u_xx_b = u_xx.reshape(F, n_t, n_x)
    u_t_fd = (u_b[:, 1:, :] - u_b[:, :-1, :]) / float(dt)
    resid = u_t_fd - float(diffusivity) * u_xx_b[:, 1:, :]
    return torch.mean(resid**2)


__all__ = [
    "burgers_residual_loss",
    "data_loss",
    "heat_residual_loss",
    "heat_residual_loss_fd",
    "ks_residual_loss",
    "ks_residual_loss_fd",
]
