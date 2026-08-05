# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""JAX kinetic-theory ops (twin of :mod:`omnibias.fields.torch.ops.kinetic`).

Vlasov transport, BGK relaxation, the Maxwellian equilibrium and velocity
moments on a phase-space ``(x, v)`` field ``f(t, x, v)``. See the torch twin for
the operator catalogue and the closed-form / numerical honesty split.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import jax.numpy as jnp
from jax import Array
from omnibias.fields._core.quadrature import QuadratureSpec
from omnibias.fields.jax.ops.basic import derivative, stack_components, value

if TYPE_CHECKING:  # pragma: no cover
    from omnibias.fields._core.state import FieldState

#: The full Boltzmann collision integral is a non-local velocity-space integral
#: (quadrature), not a closed-form sigma-tower object. Enforcement tests assert
#: no closed-form Boltzmann-collision op is exported from this module.
BOLTZMANN_COLLISION_IS_NUMERICAL = True


def _require_time(state: FieldState, op: str) -> str:
    ta = state.coordinate_spec.time_axis
    if ta is None:
        raise ValueError(f"{op} requires a time axis on the spec")
    return ta


def _velocity_columns(state: FieldState, velocity_axes: tuple[str, ...]) -> Array:
    """Stack the raw coordinate values along ``velocity_axes``, shape ``(B, d)``."""
    idx = [state.coordinate_spec.axis_index(a) for a in velocity_axes]
    return state.coords[:, jnp.asarray(idx)]


def vlasov_residual(
    state: FieldState,
    name: str,
    *,
    position_axes: tuple[str, ...],
    velocity_axes: tuple[str, ...],
    force: tuple[str, ...] | Array | None = None,
    mass: float = 1.0,
) -> Array:
    r"""Vlasov residual :math:`\partial_t f + v\cdot\nabla_x f + (F/m)\cdot\nabla_v f`.

    Shape ``(B,)``. See the torch twin for the full contract.
    """
    if len(position_axes) != len(velocity_axes):
        raise ValueError(
            f"position_axes ({len(position_axes)}) and velocity_axes "
            f"({len(velocity_axes)}) must have equal length"
        )
    ta = _require_time(state, "vlasov_residual")
    v_cols = _velocity_columns(state, velocity_axes)
    res = derivative(state, name, axis=ta, order=1)
    for i, xa in enumerate(position_axes):
        res = res + v_cols[:, i] * derivative(state, name, axis=xa, order=1)
    if force is not None:
        f = stack_components(state, force) if isinstance(force, tuple) else force
        for i, va in enumerate(velocity_axes):
            res = res + (f[:, i] / mass) * derivative(state, name, axis=va, order=1)
    return res


def bgk_collision(
    state: FieldState,
    name: str,
    *,
    equilibrium: str | Array,
    tau: float,
) -> Array:
    r"""BGK relaxation source :math:`-(f-f_{eq})/\tau`, shape ``(B,)``."""
    if tau <= 0.0:
        raise ValueError(f"relaxation time tau must be > 0, got {tau}")
    f = value(state, name)
    f_eq = value(state, equilibrium) if isinstance(equilibrium, str) else equilibrium
    return -(f - f_eq) / tau


def bgk_vlasov_residual(
    state: FieldState,
    name: str,
    *,
    position_axes: tuple[str, ...],
    velocity_axes: tuple[str, ...],
    equilibrium: str | Array,
    tau: float,
    force: tuple[str, ...] | Array | None = None,
    mass: float = 1.0,
) -> Array:
    r"""Vlasov-BGK residual :math:`\mathcal L f + (f-f_{eq})/\tau`, shape ``(B,)``."""
    lhs = vlasov_residual(
        state, name, position_axes=position_axes, velocity_axes=velocity_axes,
        force=force, mass=mass,
    )
    return lhs - bgk_collision(state, name, equilibrium=equilibrium, tau=tau)


def maxwellian(
    state: FieldState,
    *,
    velocity_axes: tuple[str, ...],
    density: float | Array = 1.0,
    bulk_velocity: tuple[float, ...] | Array | None = None,
    temperature: float | Array = 1.0,
    mass: float = 1.0,
) -> Array:
    r"""Local Maxwell-Boltzmann equilibrium, shape ``(B,)`` (natural units, :math:`k_B=1`)."""
    v = _velocity_columns(state, velocity_axes)
    d = len(velocity_axes)
    if bulk_velocity is None:
        dv = v
    else:
        u = jnp.asarray(bulk_velocity, dtype=v.dtype)
        dv = v - u
    speed_sq = (dv * dv).sum(axis=-1)
    prefactor = density * (mass / (2.0 * math.pi * temperature)) ** (d / 2.0)
    return prefactor * jnp.exp(-mass * speed_sq / (2.0 * temperature))


def _quad_weights(rule: QuadratureSpec, ref: Array) -> Array:
    return jnp.asarray(rule.weights, dtype=ref.dtype)


def _check_nodes(state: FieldState, rule: QuadratureSpec, op: str) -> Array:
    vals = value(state, state.components.names[0])
    if vals.shape[0] != rule.n_nodes:
        raise ValueError(
            f"{op}: state has {vals.shape[0]} points but rule has {rule.n_nodes} "
            "nodes; evaluate the field at quadrature_nodes(rule)"
        )
    return vals


def number_density(state: FieldState, name: str, *, rule: QuadratureSpec) -> Array:
    r"""Zeroth velocity moment :math:`n=\int f\,dv` (scalar array)."""
    _check_nodes(state, rule, "number_density")
    vals = value(state, name)
    w = _quad_weights(rule, vals)
    return jnp.tensordot(w, vals, axes=([0], [0]))


def momentum_density(
    state: FieldState, name: str, *, rule: QuadratureSpec, velocity_axes: tuple[str, ...],
) -> Array:
    r"""First velocity moment :math:`\int v\,f\,dv`, shape ``(d,)``."""
    _check_nodes(state, rule, "momentum_density")
    vals = value(state, name)
    w = _quad_weights(rule, vals)
    v = _velocity_columns(state, velocity_axes)
    cols = [jnp.tensordot(w, v[:, i] * vals, axes=([0], [0])) for i in range(v.shape[-1])]
    return jnp.stack(cols, axis=-1)


def kinetic_energy_density(
    state: FieldState, name: str, *, rule: QuadratureSpec, velocity_axes: tuple[str, ...],
) -> Array:
    r"""Second velocity moment :math:`\int \tfrac12|v|^2 f\,dv` (scalar array)."""
    _check_nodes(state, rule, "kinetic_energy_density")
    vals = value(state, name)
    w = _quad_weights(rule, vals)
    v = _velocity_columns(state, velocity_axes)
    speed_sq = (v * v).sum(axis=-1)
    return jnp.tensordot(w, 0.5 * speed_sq * vals, axes=([0], [0]))


__all__ = [
    "BOLTZMANN_COLLISION_IS_NUMERICAL",
    "bgk_collision",
    "bgk_vlasov_residual",
    "kinetic_energy_density",
    "maxwellian",
    "momentum_density",
    "number_density",
    "vlasov_residual",
]
