# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""JAX magnetohydrodynamics ops (twin of :mod:`omnibias.fields.torch.ops.mhd`).

Single-fluid resistive/ideal MHD in Alfven units (``mu_0 = rho_0 = 1``), built
as closed-form compositions of the existing ``curl`` / ``divergence`` /
``gradient`` / ``advection`` / ``vector_laplacian`` primitives, so this file is
bit-identical to the torch twin. See that module for the operator catalogue and
the ``curl(u x B)`` identity used by :func:`induction_residual`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax.numpy as jnp
from jax import Array
from omnibias.fields.jax.ops.basic import (
    divergence,
    gradient,
    stack_components,
    vector_derivative,
)
from omnibias.fields.jax.ops.high_order import vector_laplacian
from omnibias.fields.jax.ops.nonlinear import advection
from omnibias.fields.jax.ops.vector import curl

if TYPE_CHECKING:  # pragma: no cover
    from omnibias.fields._core.state import FieldState


def _require_time(state: FieldState, op: str) -> str:
    ta = state.coordinate_spec.time_axis
    if ta is None:
        raise ValueError(f"{op} requires a time axis on the spec")
    return ta


def _cross(a: Array, b: Array) -> Array:
    """Row-wise 3-vector cross product ``a x b`` on ``(B, 3)`` tensors."""
    ax, ay, az = a[..., 0], a[..., 1], a[..., 2]
    bx, by, bz = b[..., 0], b[..., 1], b[..., 2]
    return jnp.stack(
        [ay * bz - az * by, az * bx - ax * bz, ax * by - ay * bx], axis=-1,
    )


def current_density(state: FieldState, *, magnetic: tuple[str, ...]) -> Array:
    r"""MHD current density :math:`J = \nabla\times B` (Ampere, natural units).

    For a 3-D field this is shape ``(B, 3)``; the 2-D scalar curl is ``(B, 1)``.
    """
    return curl(state, magnetic)


def lorentz_force(
    state: FieldState,
    *,
    magnetic: tuple[str, ...],
    current: tuple[str, ...] | None = None,
) -> Array:
    r"""Lorentz force density :math:`J\times B`, shape ``(B, 3)``.

    With ``current=None`` the current is closed from Ampere's law
    :math:`J=\nabla\times B`; otherwise ``current`` names an explicit 3-vector.
    """
    if len(magnetic) != 3:
        raise ValueError("lorentz_force requires a 3-component magnetic field")
    b = stack_components(state, magnetic)
    j = stack_components(state, current) if current is not None else curl(state, magnetic)
    return _cross(j, b)


def magnetic_pressure(state: FieldState, *, magnetic: tuple[str, ...]) -> Array:
    r"""Magnetic pressure :math:`p_B = |B|^2/2`, shape ``(B,)`` (natural units)."""
    b = stack_components(state, magnetic)
    return 0.5 * (b * b).sum(axis=-1)


def maxwell_stress_tensor(
    state: FieldState,
    *,
    magnetic: tuple[str, ...],
    electric: tuple[str, ...] | None = None,
) -> Array:
    r"""Maxwell stress tensor, shape ``(B, 3, 3)``.

    :math:`T_{ij}=E_iE_j+B_iB_j-\tfrac12\delta_{ij}(|E|^2+|B|^2)` in natural
    units. With ``electric=None`` only the magnetic part is returned.
    """
    if len(magnetic) != 3:
        raise ValueError("maxwell_stress_tensor requires a 3-component B field")
    b = stack_components(state, magnetic)
    eye = jnp.eye(3, dtype=b.dtype)
    t = b[..., :, None] * b[..., None, :] - 0.5 * (b * b).sum(-1)[..., None, None] * eye
    if electric is not None:
        if len(electric) != 3:
            raise ValueError("maxwell_stress_tensor requires a 3-component E field")
        e = stack_components(state, electric)
        t = t + e[..., :, None] * e[..., None, :] - 0.5 * (e * e).sum(-1)[..., None, None] * eye
    return t


def magnetic_divergence(state: FieldState, *, magnetic: tuple[str, ...]) -> Array:
    r"""Solenoidal-constraint residual :math:`\nabla\cdot B`, shape ``(B,)``."""
    return divergence(state, magnetic)


def _curl_u_cross_b(
    state: FieldState, velocity: tuple[str, ...], magnetic: tuple[str, ...],
) -> Array:
    r""":math:`\nabla\times(u\times B)` via the exact vector identity.

    :math:`\nabla\times(u\times B)=u(\nabla\cdot B)-B(\nabla\cdot u)
    +(B\cdot\nabla)u-(u\cdot\nabla)B`.
    """
    u = stack_components(state, velocity)
    b = stack_components(state, magnetic)
    div_u = divergence(state, velocity)[..., None]
    div_b = divergence(state, magnetic)[..., None]
    b_grad_u = advection(state, velocity=magnetic, target=velocity)
    u_grad_b = advection(state, velocity=velocity, target=magnetic)
    return u * div_b - b * div_u + b_grad_u - u_grad_b


def induction_residual(
    state: FieldState,
    *,
    velocity: tuple[str, ...],
    magnetic: tuple[str, ...],
    resistivity: float = 0.0,
) -> Array:
    r"""Induction-equation residual :math:`\partial_t B-\nabla\times(u\times B)-\eta\nabla^2 B`.

    Shape ``(B, 3)``. ``resistivity = 0`` gives the ideal (frozen-flux) limit.
    """
    if len(magnetic) != 3 or len(velocity) != 3:
        raise ValueError("induction_residual requires 3-component u and B fields")
    ta = _require_time(state, "induction_residual")
    res = vector_derivative(state, magnetic, axis=ta, order=1) - _curl_u_cross_b(
        state, velocity, magnetic,
    )
    if resistivity != 0.0:
        res = res - resistivity * vector_laplacian(state, magnetic)
    return res


def ideal_mhd_momentum_residual(
    state: FieldState,
    *,
    velocity: tuple[str, ...],
    magnetic: tuple[str, ...],
    pressure: str,
    density: float = 1.0,
    viscosity: float = 0.0,
    current: tuple[str, ...] | None = None,
    forcing: tuple[str, ...] | Array | None = None,
) -> Array:
    r"""Incompressible MHD momentum residual, shape ``(B, 3)``.

    :math:`\rho(\partial_t u+(u\cdot\nabla)u)+\nabla p-J\times B-\nu\nabla^2u-f`,
    with the thermal pressure ``p`` and the Lorentz force ``J x B`` (magnetic
    pressure is carried inside ``J x B``). ``B = 0`` recovers Navier-Stokes.
    """
    if len(velocity) != 3 or len(magnetic) != 3:
        raise ValueError("ideal_mhd_momentum_residual requires 3-component fields")
    ta = _require_time(state, "ideal_mhd_momentum_residual")
    u_t = vector_derivative(state, velocity, axis=ta, order=1)
    adv = advection(state, velocity=velocity)
    grad_p = gradient(state, pressure)
    res = density * (u_t + adv) + grad_p - lorentz_force(
        state, magnetic=magnetic, current=current,
    )
    if viscosity != 0.0:
        res = res - viscosity * vector_laplacian(state, velocity)
    if forcing is not None:
        f = stack_components(state, forcing) if isinstance(forcing, tuple) else forcing
        res = res - f
    return res


__all__ = [
    "current_density",
    "ideal_mhd_momentum_residual",
    "induction_residual",
    "lorentz_force",
    "magnetic_divergence",
    "magnetic_pressure",
    "maxwell_stress_tensor",
]
