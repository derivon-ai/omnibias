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


def causal_operator_loss(
    residual: Tensor,
    coords: Tensor,
    *,
    epsilon: float = 1.0,
    n_time_bins: int | None = None,
    time_axis: int = 1,
) -> Tensor:
    """Wang-Perdikaris causal weighting for an operator residual on a slab.

    The whole space-time slab is otherwise fit at once, which is the same
    temporal-causality violation as a standard PINN. This helper bins the
    residual by the time coordinate and applies
    :func:`~omnibias.pinn.torch.losses.causal_residual_loss`.

    Parameters
    ----------
    residual
        Flat residual of shape ``(F*Q,)`` or ``(F*Q, ...)`` aligned with
        ``coords`` of shape ``(Q, D)`` tiled across ``F`` samples (samples
        slow, queries fast -- the :meth:`DeepONetField.on_grid` layout).
    coords
        Shared query grid ``(Q, D)``.
    epsilon
        Causal sharpness.
    n_time_bins
        Number of time bins. Defaults to the number of unique time values
        in ``coords[:, time_axis]`` (so a product grid bins exactly).
    time_axis
        Column of ``coords`` holding time (default ``1`` for ``(x, t)``).
    """
    from omnibias.pinn.torch.losses.causal import causal_residual_loss

    if coords.ndim != 2:
        raise ValueError(f"coords must be 2-D (Q, D); got {tuple(coords.shape)}")
    Q = int(coords.shape[0])
    t = coords[:, int(time_axis)].contiguous()
    # Unique sorted times define the natural bins on a product grid.
    times_sorted, _ = torch.sort(torch.unique(t))
    n_unique = int(times_sorted.numel())
    if n_unique < 1:
        raise ValueError("coords time column is empty")
    bins = int(n_time_bins) if n_time_bins is not None else n_unique
    if bins < 1:
        raise ValueError(f"n_time_bins must be >= 1, got {bins}")

    flat = residual.reshape(-1)
    if flat.numel() % Q != 0:
        raise ValueError(
            f"residual numel {flat.numel()} is not a multiple of Q={Q}"
        )
    F = flat.numel() // Q
    # Assign each query to a time bin by nearest unique time / equal-width.
    t0 = float(times_sorted[0])
    t1 = float(times_sorted[-1])
    if t1 <= t0:
        # Degenerate: single time -> no causality to enforce.
        return torch.mean(flat**2)
    # Bin index in [0, bins) for each of the Q query points.
    edges = torch.linspace(t0, t1, bins + 1, dtype=t.dtype, device=t.device)
    # right=False so the closing edge lands in the last bin.
    idx = torch.bucketize(t, edges[1:-1].contiguous(), right=False)
    idx = torch.clamp(idx, 0, bins - 1)

    # Build (bins, F * per_bin_varies) by scattering; pad short bins with zeros
    # and a mask so the mean is over real entries only.
    resid_FQ = flat.reshape(F, Q)
    # Gather per-bin: stack as (bins, F, max_count) with NaN pad, then nanmean.
    counts = torch.bincount(idx, minlength=bins)
    max_c = int(counts.max().item()) if bins > 0 else 0
    if max_c == 0:
        return torch.mean(flat**2)
    cube = flat.new_zeros((bins, F, max_c))
    # Fill each bin's columns.
    for b in range(bins):
        sel = idx == b
        n_b = int(sel.sum().item())
        if n_b == 0:
            continue
        cube[b, :, :n_b] = resid_FQ[:, sel]
    # causal_residual_loss wants (n_t, ...spatial); use (bins, F*max_c) and
    # zero-fill empty slots (they contribute 0 to the MSE of that bin -- a
    # mild bias when bins are unbalanced, acceptable for the product-grid
    # default where every bin has the same count).
    resid_t = cube.reshape(bins, F * max_c)
    return cast(Tensor, causal_residual_loss(resid_t, epsilon=float(epsilon)))


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
    "causal_operator_loss",
    "data_loss",
    "heat_residual_loss",
    "heat_residual_loss_fd",
    "ks_residual_loss",
    "ks_residual_loss_fd",
]
