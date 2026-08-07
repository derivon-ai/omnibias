# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""DeepONet training losses with closed-form physics residuals (JAX)."""

from __future__ import annotations

from typing import cast

import jax.numpy as jnp
from jax import Array
from omnibias.pinn.jax import ops as jops
from omnibias.pinn.operator.jax.data import OperatorSlab
from omnibias.pinn.operator.jax.deeponet import DeepONetOperator


def data_loss(
    operator: DeepONetOperator,
    slab: OperatorSlab,
    *,
    component: str = "u",
) -> Array:
    field = operator.condition(slab.sensors)
    state = field.on_grid(slab.coords)
    pred = jops.value(state, component).reshape(slab.values.shape[0], -1)
    target = slab.values[..., 0] if slab.values.ndim == 3 else slab.values
    return jnp.mean((pred - target) ** 2)


def heat_residual_loss(
    operator: DeepONetOperator,
    sensors: Array,
    coords: Array,
    *,
    diffusivity: float = 0.1,
    component: str = "u",
) -> Array:
    field = operator.condition(sensors)
    state = field.on_grid(coords)
    u_t = jops.derivative(state, component, axis=1, order=1)
    u_xx = jops.derivative(state, component, axis=0, order=2)
    return jnp.mean((u_t - float(diffusivity) * u_xx) ** 2)


def burgers_residual_loss(
    operator: DeepONetOperator,
    sensors: Array,
    coords: Array,
    *,
    viscosity: float = 0.05,
    component: str = "u",
) -> Array:
    field = operator.condition(sensors)
    state = field.on_grid(coords)
    u = jops.value(state, component)
    u_t = jops.derivative(state, component, axis=1, order=1)
    u_x = jops.derivative(state, component, axis=0, order=1)
    u_xx = jops.derivative(state, component, axis=0, order=2)
    return jnp.mean((u_t + u * u_x - float(viscosity) * u_xx) ** 2)


def ks_residual_loss(
    operator: DeepONetOperator,
    sensors: Array,
    coords: Array,
    *,
    component: str = "u",
) -> Array:
    """Closed-form KS residual via the shipped :class:`KuramotoSivashinsky` equation."""
    from omnibias.pinn.jax.equations.kuramoto_sivashinsky import KuramotoSivashinsky

    field = operator.condition(sensors)
    state = field.on_grid(coords)
    out = KuramotoSivashinsky(component=component)(state)
    return jnp.mean(out.residual**2)


def ks_residual_loss_fd(
    operator: DeepONetOperator,
    sensors: Array,
    coords: Array,
    *,
    h: float,
    component: str = "u",
) -> Array:
    """KS residual with ``u_xxxx`` by a 5-point central stencil in ``x``."""
    if float(h) <= 0.0:
        raise ValueError(f"h must be positive, got {h}")
    field = operator.condition(sensors)
    state = field.on_grid(coords)
    u = jops.value(state, component)
    u_t = jops.derivative(state, component, axis=1, order=1)
    u_x = jops.derivative(state, component, axis=0, order=1)
    u_xx = jops.derivative(state, component, axis=0, order=2)

    def _value_at_shift(dx: float) -> Array:
        shifted = coords.at[:, 0].add(float(dx))
        return cast(Array, jops.value(field.on_grid(shifted), component))

    hh = float(h)
    um2 = _value_at_shift(-2.0 * hh)
    um1 = _value_at_shift(-hh)
    up1 = _value_at_shift(hh)
    up2 = _value_at_shift(2.0 * hh)
    u_xxxx = (um2 - 4.0 * um1 + 6.0 * u - 4.0 * up1 + up2) / (hh**4)
    resid = u_t + u * u_x + u_xx + u_xxxx
    return jnp.mean(resid**2)


def heat_residual_loss_fd(
    operator: DeepONetOperator,
    sensors: Array,
    coords: Array,
    *,
    diffusivity: float = 0.1,
    dt: float,
    component: str = "u",
) -> Array:
    """Physics residual with ``u_t`` by a backward finite difference (§3a convention)."""
    field = operator.condition(sensors)
    state = field.on_grid(coords)
    u = jops.value(state, component)
    u_xx = jops.derivative(state, component, axis=0, order=2)
    x = coords[:, 0]
    t = coords[:, 1]
    n_x = int(jnp.unique(x).size)
    n_t = int(jnp.unique(t).size)
    F = int(sensors.shape[0])
    u_b = u.reshape(F, n_t, n_x)
    u_xx_b = u_xx.reshape(F, n_t, n_x)
    u_t_fd = (u_b[:, 1:, :] - u_b[:, :-1, :]) / float(dt)
    resid = u_t_fd - float(diffusivity) * u_xx_b[:, 1:, :]
    return jnp.mean(resid**2)


__all__ = [
    "burgers_residual_loss",
    "data_loss",
    "heat_residual_loss",
    "heat_residual_loss_fd",
    "ks_residual_loss",
    "ks_residual_loss_fd",
]
