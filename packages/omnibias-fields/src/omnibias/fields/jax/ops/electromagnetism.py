# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""JAX electromagnetism ops (mirrors :mod:`omnibias.fields.torch.ops.electromagnetism`).

The 3-D Maxwell system in natural units (``c = epsilon_0 = mu_0 = 1``):
Faraday, Ampere-Maxwell, and the two Gauss laws as PINN residuals, the Poynting
vector ``E x B``, the potential reconstructions ``B = curl A`` /
``E = -grad phi - d_t A``, the Lorenz-gauge residual, and the vector
d'Alembertian (so ``box A = -J`` / ``box phi = -rho`` are one call). Every term
is a closed-form composition of ``curl`` / ``divergence`` / ``gradient`` /
``vector_derivative`` and the scalar :func:`dalembertian`, so the torch twin is
bit-identical by construction.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax.numpy as jnp
from jax import Array
from omnibias.fields.jax.ops.basic import (
    derivative,
    divergence,
    gradient,
    stack_components,
    value,
    vector_derivative,
)
from omnibias.fields.jax.ops.conservation import dalembertian
from omnibias.fields.jax.ops.vector import curl

if TYPE_CHECKING:  # pragma: no cover
    from omnibias.fields._core.state import FieldState


def _spatial(state: FieldState) -> tuple[str, ...]:
    return tuple(state.coordinate_spec.spatial_axes)


def _require_time(state: FieldState, op: str) -> str:
    ta = state.coordinate_spec.time_axis
    if ta is None:
        raise ValueError(f"{op} requires a time axis on the spec")
    return ta


def faraday_residual(
    state: FieldState, *, electric: tuple[str, ...], magnetic: tuple[str, ...],
) -> Array:
    r"""Faraday's law residual :math:`\partial_t B + \nabla\times E`."""
    ta = _require_time(state, "faraday_residual")
    return vector_derivative(state, magnetic, axis=ta, order=1) + curl(state, electric)


def ampere_residual(
    state: FieldState,
    *,
    electric: tuple[str, ...],
    magnetic: tuple[str, ...],
    current: tuple[str, ...] | None = None,
) -> Array:
    r"""Ampere-Maxwell residual :math:`\partial_t E - \nabla\times B + J` (natural units)."""
    ta = _require_time(state, "ampere_residual")
    res = vector_derivative(state, electric, axis=ta, order=1) - curl(state, magnetic)
    if current is not None:
        res = res + stack_components(state, current)
    return res


def gauss_residual(
    state: FieldState, *, electric: tuple[str, ...], charge: str | float | None = None,
) -> Array:
    r"""Gauss's law residual :math:`\nabla\cdot E - \rho`."""
    res = divergence(state, electric)
    if charge is not None:
        rho = value(state, charge) if isinstance(charge, str) else float(charge)
        res = res - rho
    return res


def gauss_magnetic_residual(
    state: FieldState, *, magnetic: tuple[str, ...],
) -> Array:
    r"""No-magnetic-monopole residual :math:`\nabla\cdot B`."""
    return divergence(state, magnetic)


def poynting_vector(
    state: FieldState, *, electric: tuple[str, ...], magnetic: tuple[str, ...],
) -> Array:
    r"""Poynting flux :math:`S = E \times B` (natural units), shape ``(B, 3)``."""
    if len(electric) != 3 or len(magnetic) != 3:
        raise ValueError("poynting_vector requires 3-component E and B fields")
    ex, ey, ez = (value(state, n) for n in electric)
    bx, by, bz = (value(state, n) for n in magnetic)
    sx = ey * bz - ez * by
    sy = ez * bx - ex * bz
    sz = ex * by - ey * bx
    return jnp.stack([sx, sy, sz], axis=-1)


def magnetic_field_from_potential(
    state: FieldState, *, potential: tuple[str, ...],
) -> Array:
    r"""Magnetic field from the vector potential :math:`B = \nabla\times A`."""
    return curl(state, potential)


def electric_field_from_potentials(
    state: FieldState,
    *,
    scalar_potential: str,
    vector_potential: tuple[str, ...] | None = None,
) -> Array:
    r"""Electric field :math:`E = -\nabla\phi - \partial_t A`.

    With ``vector_potential=None`` this is the electrostatic field
    :math:`E = -\nabla\phi`.
    """
    e = -gradient(state, scalar_potential, axes=_spatial(state))
    if vector_potential is not None:
        ta = _require_time(state, "electric_field_from_potentials")
        e = e - vector_derivative(state, vector_potential, axis=ta, order=1)
    return e


def lorenz_gauge_residual(
    state: FieldState, *, scalar_potential: str, vector_potential: tuple[str, ...],
) -> Array:
    r"""Lorenz-gauge condition residual :math:`\partial_t \phi + \nabla\cdot A` (natural units)."""
    ta = _require_time(state, "lorenz_gauge_residual")
    return derivative(state, scalar_potential, axis=ta, order=1) + divergence(
        state, vector_potential
    )


def vector_dalembertian(
    state: FieldState,
    names: tuple[str, ...],
    *,
    c: float = 1.0,
    signature: str = "mostly_plus",
) -> Array:
    r"""Componentwise d'Alembertian :math:`\Box u_i`, shape ``(B, len(names))``.

    The wave equation for a vector potential: ``box A = -J`` in the Lorenz gauge.
    """
    cols = [dalembertian(state, n, c=c, signature=signature) for n in names]
    return jnp.stack(cols, axis=-1)


__all__ = [
    "ampere_residual",
    "electric_field_from_potentials",
    "faraday_residual",
    "gauss_magnetic_residual",
    "gauss_residual",
    "lorenz_gauge_residual",
    "magnetic_field_from_potential",
    "poynting_vector",
    "vector_dalembertian",
]
