# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Torch chemistry / transport ops (twin of :mod:`omnibias.fields.jax.ops.chemistry`).

Constitutive fluxes (Fick, Nernst-Planck, Darcy) and the reaction-diffusion /
Poisson / Nernst-Planck residuals. Every term is a closed-form composition of
the Phase-1 gradient / diffusion primitives, so the jax twin is bit-identical
by construction. Poisson-Nernst-Planck is the coupled system
``nernst_planck_residual + poisson_residual``.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

from omnibias.fields.torch.ops.basic import derivative, gradient, laplacian, value
from omnibias.fields.torch.ops.conservation import (
    diffusive_flux,
    variable_coefficient_diffusion,
)
from torch import Tensor

if TYPE_CHECKING:  # pragma: no cover
    from omnibias.fields._core.state import FieldState


def _spatial(state: FieldState) -> tuple[str, ...]:
    return tuple(state.coordinate_spec.spatial_axes)


def fickian_flux(
    state: FieldState, name: str, *, diffusivity: float | str = 1.0,
) -> Tensor:
    r"""Fick's first law :math:`J = -D\,\nabla c` (alias of :func:`diffusive_flux`)."""
    return diffusive_flux(state, name, diffusivity=diffusivity)


def nernst_planck_flux(
    state: FieldState,
    concentration: str,
    potential: str,
    *,
    diffusivity: float | str = 1.0,
    valence: float = 1.0,
    mobility: float = 1.0,
    faraday: float = 1.0,
) -> Tensor:
    r"""Nernst-Planck molar flux of a charged species, shape ``(B, n_spatial)``.

    :math:`J = -D\,\nabla c - z\,\mu\,F\,c\,\nabla\phi` (diffusion + electromigration).
    """
    sa = _spatial(state)
    j_diff = diffusive_flux(state, concentration, diffusivity=diffusivity)
    grad_phi = gradient(state, potential, axes=sa)
    c = value(state, concentration)
    drift = (valence * mobility * faraday) * c[..., None] * grad_phi
    return j_diff - drift


def darcy_flux(
    state: FieldState,
    pressure: str,
    *,
    permeability: float | str = 1.0,
    viscosity: float = 1.0,
) -> Tensor:
    r"""Darcy seepage velocity :math:`q = -(k/\mu)\,\nabla p`, shape ``(B, n_spatial)``."""
    sa = _spatial(state)
    grad_p = gradient(state, pressure, axes=sa)
    if isinstance(permeability, str):
        k = value(state, permeability)[..., None]
        return -(k / viscosity) * grad_p
    return -(float(permeability) / viscosity) * grad_p


def nernst_planck_residual(
    state: FieldState,
    *,
    concentration: str,
    potential: str,
    diffusivity: float | str = 1.0,
    valence: float = 1.0,
    mobility: float = 1.0,
    faraday: float = 1.0,
    source: str | float | None = None,
) -> Tensor:
    r"""Transient Nernst-Planck species-continuity residual.

    :math:`\partial_t c + \nabla\cdot J - s` with the flux of
    :func:`nernst_planck_flux`. In divergence form this is closed-form:

    .. math::
        \partial_t c - \nabla\cdot(D\nabla c)
        - zF\mu\,\bigl(\nabla c\cdot\nabla\phi + c\,\Delta\phi\bigr) - s,

    so coupling this with :func:`poisson_residual` gives the full
    Poisson-Nernst-Planck system with no autodiff on the closed-form path.
    """
    ta = state.coordinate_spec.time_axis
    if ta is None:
        raise ValueError("nernst_planck_residual requires a time axis on the spec")
    sa = _spatial(state)
    dt_c = derivative(state, concentration, axis=ta, order=1)
    diff = variable_coefficient_diffusion(state, concentration, diffusivity=diffusivity)
    grad_c = gradient(state, concentration, axes=sa)
    grad_phi = gradient(state, potential, axes=sa)
    lap_phi = laplacian(state, potential, axes=sa)
    c = value(state, concentration)
    migration = (valence * mobility * faraday) * (
        (grad_c * grad_phi).sum(dim=-1) + c * lap_phi
    )
    res = dt_c - diff - migration
    if source is not None:
        s = value(state, source) if isinstance(source, str) else float(source)
        res = res - s
    return res


def reaction_diffusion_residual(
    state: FieldState,
    *,
    scalar: str,
    diffusivity: float | str = 1.0,
    reaction: Callable[[Tensor], Tensor] | str | float | None = None,
    source: str | float | None = None,
) -> Tensor:
    r""":math:`\partial_t c - \nabla\cdot(D\nabla c) - R(c) - s`.

    ``reaction`` is a callable applied to ``c`` (e.g. Fisher-KPP ``c(1-c)``), a
    field-component name, or a constant.
    """
    ta = state.coordinate_spec.time_axis
    if ta is None:
        raise ValueError("reaction_diffusion_residual requires a time axis on the spec")
    dt_c = derivative(state, scalar, axis=ta, order=1)
    diff = variable_coefficient_diffusion(state, scalar, diffusivity=diffusivity)
    res = dt_c - diff
    if reaction is not None:
        if callable(reaction):
            r = reaction(value(state, scalar))
        elif isinstance(reaction, str):
            r = value(state, reaction)
        else:
            r = float(reaction)
        res = res - r
    if source is not None:
        s = value(state, source) if isinstance(source, str) else float(source)
        res = res - s
    return res


def poisson_residual(
    state: FieldState,
    potential: str,
    *,
    source: str | float | None = None,
    permittivity: float | str = 1.0,
) -> Tensor:
    r"""Poisson residual :math:`\nabla\cdot(\varepsilon\nabla\phi) + \rho`.

    For constant ``permittivity`` this is :math:`\varepsilon\,\Delta\phi + \rho`;
    set ``source = rho`` (the free charge density / forcing). Steady (no time axis).
    """
    res = variable_coefficient_diffusion(state, potential, diffusivity=permittivity)
    if source is not None:
        s = value(state, source) if isinstance(source, str) else float(source)
        res = res + s
    return res


__all__ = [
    "darcy_flux",
    "fickian_flux",
    "nernst_planck_flux",
    "poisson_residual",
    "reaction_diffusion_residual",
]
