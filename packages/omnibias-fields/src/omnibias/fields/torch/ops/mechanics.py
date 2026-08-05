# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Torch continuum-mechanics ops (twin of :mod:`omnibias.fields.jax.ops.mechanics`).

Stream-function / velocity / vorticity relations, Newtonian-fluid and linear-elastic
stress tensors, viscous dissipation, and the closed-form Stokes / Navier-Cauchy
momentum residuals (built from the Phase-1 vector-calculus identities).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from omnibias.fields.torch.ops.basic import (
    derivative,
    divergence,
    gradient,
    laplacian,
    stack_components,
)
from omnibias.fields.torch.ops.high_order import vector_laplacian
from omnibias.fields.torch.ops.tensor import tensor_double_dot
from omnibias.fields.torch.ops.vector import (
    gradient_of_divergence,
    strain_rate,
)
from torch import Tensor

if TYPE_CHECKING:  # pragma: no cover
    from omnibias.fields._core.state import FieldState


def _spatial(state: FieldState) -> tuple[str, ...]:
    return tuple(state.coordinate_spec.spatial_axes)


def _body_force(state: FieldState, body_force, d: int):  # type: ignore[no-untyped-def]
    if body_force is None:
        return 0.0
    if isinstance(body_force, tuple):
        return stack_components(state, body_force)
    return torch.as_tensor(body_force)


def _scalar(state: FieldState, name: str) -> Tensor:
    from omnibias.fields.torch.ops.basic import value
    return value(state, name)


def velocity_from_streamfunction(state: FieldState, psi: str) -> Tensor:
    r"""2-D velocity from a scalar stream function: ``u = d psi/dy, v = -d psi/dx``.

    Divergence-free by construction. Returns shape ``(B, 2)``.
    """
    sa = _spatial(state)
    if len(sa) != 2:
        raise ValueError("velocity_from_streamfunction is defined for 2D fields only")
    x, y = sa
    u = derivative(state, psi, axis=y, order=1)
    v = -derivative(state, psi, axis=x, order=1)
    return torch.stack([u, v], dim=-1)


def vorticity_from_streamfunction(state: FieldState, psi: str) -> Tensor:
    r"""2-D scalar vorticity from a stream function: ``omega = -Delta psi``."""
    sa = _spatial(state)
    if len(sa) != 2:
        raise ValueError("vorticity_from_streamfunction is defined for 2D fields only")
    return -laplacian(state, psi, axes=sa)


def newtonian_stress(
    state: FieldState,
    velocity: tuple[str, ...],
    *,
    viscosity: float = 1.0,
    pressure: str | None = None,
) -> Tensor:
    r"""Cauchy stress of an incompressible Newtonian fluid.

    :math:`\sigma_{ij} = -p\,\delta_{ij} + 2\mu\,\varepsilon_{ij}` with
    :math:`\varepsilon` the :func:`strain_rate`. Returns shape ``(B, d, d)``.
    """
    s = strain_rate(state, velocity)
    sigma = 2.0 * viscosity * s
    if pressure is not None:
        d = s.shape[-1]
        p = _scalar(state, pressure)
        eye = torch.eye(d, dtype=s.dtype, device=s.device)
        sigma = sigma - p[..., None, None] * eye
    return sigma


def linear_elastic_stress(
    state: FieldState,
    displacement: tuple[str, ...],
    *,
    lam: float = 1.0,
    mu: float = 1.0,
) -> Tensor:
    r"""Isotropic linear-elastic (Hooke) stress.

    :math:`\sigma_{ij} = \lambda\,(\nabla\cdot u)\,\delta_{ij} + 2\mu\,\varepsilon_{ij}`.
    Returns shape ``(B, d, d)``.
    """
    s = strain_rate(state, displacement)
    d = s.shape[-1]
    tr = divergence(state, displacement)
    eye = torch.eye(d, dtype=s.dtype, device=s.device)
    return 2.0 * mu * s + lam * tr[..., None, None] * eye


def viscous_dissipation(
    state: FieldState, velocity: tuple[str, ...], *, viscosity: float = 1.0,
) -> Tensor:
    r"""Viscous dissipation rate :math:`\Phi = 2\mu\,\varepsilon:\varepsilon \ge 0`."""
    s = strain_rate(state, velocity)
    return 2.0 * viscosity * tensor_double_dot(s, s)


def stress_divergence(state: FieldState, sigma_names) -> Tensor:  # type: ignore[no-untyped-def]
    r"""Divergence of a named stress tensor (alias of :func:`tensor_divergence`)."""
    from omnibias.fields.torch.ops.tensor import tensor_divergence
    return tensor_divergence(state, sigma_names)


def stokes_residual(
    state: FieldState,
    *,
    velocity: tuple[str, ...],
    pressure: str,
    viscosity: float = 1.0,
    body_force=None,  # type: ignore[no-untyped-def]
) -> Tensor:
    r"""Stokes momentum residual :math:`\mu\,\Delta u - \nabla p + f`, shape ``(B, d)``.

    Pair with ``divergence(velocity)`` for the incompressibility constraint.
    """
    sa = _spatial(state)
    lap_u = vector_laplacian(state, velocity)
    grad_p = gradient(state, pressure, axes=sa)
    f = _body_force(state, body_force, len(sa))
    return viscosity * lap_u - grad_p + f


def navier_cauchy_residual(
    state: FieldState,
    *,
    displacement: tuple[str, ...],
    lam: float = 1.0,
    mu: float = 1.0,
    body_force=None,  # type: ignore[no-untyped-def]
) -> Tensor:
    r"""Linear-elastostatics (Navier-Cauchy) residual.

    :math:`(\lambda+\mu)\,\nabla(\nabla\cdot u) + \mu\,\Delta u + f`, the
    closed-form divergence of the linear-elastic stress. Returns shape ``(B, d)``.
    """
    grad_div = gradient_of_divergence(state, displacement)
    lap_u = vector_laplacian(state, displacement)
    f = _body_force(state, body_force, len(displacement))
    return (lam + mu) * grad_div + mu * lap_u + f


__all__ = [
    "linear_elastic_stress",
    "navier_cauchy_residual",
    "newtonian_stress",
    "stokes_residual",
    "stress_divergence",
    "velocity_from_streamfunction",
    "viscous_dissipation",
    "vorticity_from_streamfunction",
]
