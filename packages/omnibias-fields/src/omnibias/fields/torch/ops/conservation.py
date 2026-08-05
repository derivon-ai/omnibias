# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Torch conservation / flux / wave ops (twin of :mod:`omnibias.fields.jax.ops.conservation`).

Closed-form compositions of the existing partial-derivative reducers; bit-identical
with the jax twin by construction (the math lives in the shared sigma tower, not here).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

from omnibias.fields.torch.ops.basic import (
    derivative,
    divergence,
    gradient,
    laplacian,
    value,
)
from omnibias.fields.torch.ops.nonlinear import advection
from torch import Tensor

if TYPE_CHECKING:  # pragma: no cover
    from omnibias.fields._core.state import FieldState


def _spatial(state: FieldState) -> tuple[str, ...]:
    return tuple(state.coordinate_spec.spatial_axes)


def grad_squared_norm(state: FieldState, name: str) -> Tensor:
    r""":math:`|\nabla u|^2 = \sum_i (\partial_i u)^2` (spatial gradient)."""
    g = gradient(state, name, axes=_spatial(state))
    return (g * g).sum(dim=-1)


def gradient_of_composition(
    state: FieldState, name: str, fprime: Callable[[Tensor], Tensor],
) -> Tensor:
    r""":math:`\nabla f(u) = f'(u)\,\nabla u`. ``fprime`` is applied to the value."""
    u = value(state, name)
    g = gradient(state, name, axes=_spatial(state))
    return fprime(u)[..., None] * g


def laplacian_of_composition(
    state: FieldState,
    name: str,
    fprime: Callable[[Tensor], Tensor],
    fsecond: Callable[[Tensor], Tensor],
) -> Tensor:
    r""":math:`\Delta f(u) = f'(u)\,\Delta u + f''(u)\,|\nabla u|^2`.

    The chain rule for the Laplacian of a pointwise nonlinearity (Cahn-Hilliard /
    Allen-Cahn style). ``fprime`` / ``fsecond`` are callables applied to ``u``.
    """
    u = value(state, name)
    lap = laplacian(state, name, axes=_spatial(state))
    gsq = grad_squared_norm(state, name)
    return fprime(u) * lap + fsecond(u) * gsq


def diffusive_flux(
    state: FieldState, name: str, *, diffusivity: float | str = 1.0,
) -> Tensor:
    r"""Fickian flux :math:`F = -D\,\nabla u`, shape ``(B, n_spatial)``.

    ``diffusivity`` is a constant or the name of a scalar field component.
    """
    g = gradient(state, name, axes=_spatial(state))
    if isinstance(diffusivity, str):
        d = value(state, diffusivity)[..., None]
        return -d * g
    return -float(diffusivity) * g


def flux_divergence(state: FieldState, names: tuple[str, ...]) -> Tensor:
    r"""Divergence of a named flux field :math:`\nabla\cdot F`.

    Conservation-law spelling of :func:`omnibias.fields.torch.ops.basic.divergence`
    (the flux components are model outputs / named fields).
    """
    return divergence(state, names)


def variable_coefficient_diffusion(
    state: FieldState, name: str, *, diffusivity: float | str = 1.0,
) -> Tensor:
    r""":math:`\nabla\cdot(D\,\nabla u)`.

    For a constant ``D`` this is ``D * laplacian(u)``; for ``D`` a scalar field
    component it expands to ``grad(D).grad(u) + D * laplacian(u)``.
    """
    sa = _spatial(state)
    lap = laplacian(state, name, axes=sa)
    if isinstance(diffusivity, str):
        grad_d = gradient(state, diffusivity, axes=sa)
        grad_u = gradient(state, name, axes=sa)
        d_val = value(state, diffusivity)
        return (grad_d * grad_u).sum(dim=-1) + d_val * lap
    return float(diffusivity) * lap


def conservation_residual(
    state: FieldState,
    *,
    density: str,
    flux: tuple[str, ...],
    source: str | float | None = None,
) -> Tensor:
    r"""Local conservation law residual :math:`\partial_t \rho + \nabla\cdot F - s`."""
    ta = state.coordinate_spec.time_axis
    if ta is None:
        raise ValueError("conservation_residual requires a time axis on the spec")
    res = derivative(state, density, axis=ta, order=1) + divergence(state, flux)
    if source is not None:
        s = value(state, source) if isinstance(source, str) else float(source)
        res = res - s
    return res


def advection_diffusion_residual(
    state: FieldState,
    *,
    scalar: str,
    velocity: tuple[str, ...],
    diffusivity: float | str = 1.0,
    source: str | float | None = None,
) -> Tensor:
    r""":math:`\partial_t c + (u\cdot\nabla)c - \nabla\cdot(D\nabla c) - s`."""
    ta = state.coordinate_spec.time_axis
    if ta is None:
        raise ValueError("advection_diffusion_residual requires a time axis on the spec")
    dt_c = derivative(state, scalar, axis=ta, order=1)
    adv = advection(state, velocity=velocity, scalar=scalar)
    diff = variable_coefficient_diffusion(state, scalar, diffusivity=diffusivity)
    res = dt_c + adv - diff
    if source is not None:
        s = value(state, source) if isinstance(source, str) else float(source)
        res = res - s
    return res


def dalembertian(
    state: FieldState, name: str, *, c: float = 1.0, signature: str = "mostly_plus",
) -> Tensor:
    r"""d'Alembert / wave operator :math:`\Box u`.

    ``signature="mostly_plus"`` (the :math:`(-,+,+,+)` metric) gives
    :math:`\Box u = \Delta u - c^{-2}\,\partial_{tt} u`, so the wave equation is
    :math:`\Box u = 0 \iff \partial_{tt} u = c^2 \Delta u`. ``"mostly_minus"``
    flips the overall sign. Requires a time axis.
    """
    ta = state.coordinate_spec.time_axis
    if ta is None:
        raise ValueError("dalembertian requires a time axis on the spec")
    if c == 0.0:
        raise ValueError("wave speed c must be non-zero")
    # axes=None -> spatial Laplacian via the fast closed-form path when available.
    spatial_lap = laplacian(state, name)
    dtt = derivative(state, name, axis=ta, order=2)
    inv_c2 = 1.0 / (c * c)
    if signature == "mostly_plus":
        return spatial_lap - inv_c2 * dtt
    if signature == "mostly_minus":
        return inv_c2 * dtt - spatial_lap
    raise ValueError(
        f"signature must be 'mostly_plus' or 'mostly_minus', got {signature!r}"
    )


def wave_operator(
    state: FieldState, name: str, *, c: float = 1.0, signature: str = "mostly_plus",
) -> Tensor:
    """Alias for :func:`dalembertian`."""
    return dalembertian(state, name, c=c, signature=signature)


__all__ = [
    "advection_diffusion_residual",
    "conservation_residual",
    "dalembertian",
    "diffusive_flux",
    "flux_divergence",
    "grad_squared_norm",
    "gradient_of_composition",
    "laplacian_of_composition",
    "variable_coefficient_diffusion",
    "wave_operator",
]
