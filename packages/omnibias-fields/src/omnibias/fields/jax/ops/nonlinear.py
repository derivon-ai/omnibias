# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""JAX nonlinear ops (mirrors :mod:`omnibias.fields.torch.ops.nonlinear`)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax.numpy as jnp
from jax import Array
from omnibias.fields.jax.ops.basic import (
    derivative,
    divergence,
    gradient,
    laplacian,
    stack_components,
    value,
)
from omnibias.fields.jax.ops.high_order import hessian

if TYPE_CHECKING:  # pragma: no cover
    from omnibias.fields._core.state import FieldState


def advection(
    state: FieldState,
    *,
    velocity: tuple[str, ...],
    target: tuple[str, ...] | None = None,
    scalar: str | None = None,
) -> Array:
    sa = state.coordinate_spec.spatial_axes
    if len(velocity) != len(sa):
        raise ValueError(
            f"advection: velocity has {len(velocity)} components, "
            f"but coordinate spec has {len(sa)} spatial axes ({sa!r})"
        )
    u_vec = stack_components(state, velocity)
    if scalar is not None:
        if target is not None:
            raise ValueError("advection: pass scalar OR target, not both")
        grad_phi = gradient(state, scalar)
        return (u_vec * grad_phi).sum(axis=-1)
    target = target if target is not None else velocity
    cols = []
    for n in target:
        grad_n = gradient(state, n)
        cols.append((u_vec * grad_n).sum(axis=-1))
    return jnp.stack(cols, axis=-1)


def material_derivative(
    state: FieldState,
    *,
    velocity: tuple[str, ...],
    scalar: str | None = None,
) -> Array:
    if state.coordinate_spec.time_axis is None:
        raise ValueError(
            "material_derivative requires a time axis on the coordinate spec"
        )
    time_axis = state.coordinate_spec.time_axis
    assert time_axis is not None  # guarded above
    if scalar is not None:
        dt_target = derivative(state, scalar, axis=time_axis)
        return dt_target + advection(state, velocity=velocity, scalar=scalar)
    dt_target = jnp.stack(
        [derivative(state, n, axis=time_axis) for n in velocity], axis=-1,
    )
    return dt_target + advection(state, velocity=velocity)


def p_laplacian(
    state: FieldState, name: str, *, p: float, eps: float = 1e-8,
) -> Array:
    if p < 1:
        raise ValueError(f"p must be >= 1 for p-Laplacian, got {p}")
    sa = state.coordinate_spec.spatial_axes
    g = gradient(state, name)
    L = laplacian(state, name)
    if abs(p - 2.0) < 1e-15:
        return L
    g2 = (g * g).sum(axis=-1)
    g_norm_sq_reg = g2 + eps * eps
    H_full = hessian(state, name)
    spatial_idx = jnp.array(
        [state.coordinate_spec.axis_index(a) for a in sa]
    )
    H_sp = H_full[..., spatial_idx, :][..., :, spatial_idx]
    gHg = jnp.einsum("bi,bij,bj->b", g, H_sp, g)
    base = g_norm_sq_reg ** ((p - 2.0) / 2.0)
    cross = (p - 2.0) * g_norm_sq_reg ** ((p - 4.0) / 2.0) * gHg
    return base * L + cross


def directional_derivative(
    state: FieldState, name: str, *, direction: Array,
) -> Array:
    g = gradient(state, name)
    direction = jnp.asarray(direction)
    if direction.shape != g.shape:
        raise ValueError(
            f"direction shape {tuple(direction.shape)} != "
            f"gradient shape {tuple(g.shape)}"
        )
    return (g * direction).sum(axis=-1)


def skew_symmetric_advection(
    state: FieldState,
    *,
    velocity: tuple[str, ...],
    scalar: str | None = None,
) -> Array:
    r"""Energy/enstrophy-conserving (skew-symmetric) advection.

    :math:`\tfrac12[(u\cdot\nabla)\phi + \nabla\cdot(u\phi)]
    = (u\cdot\nabla)\phi + \tfrac12(\nabla\cdot u)\phi`. For a divergence-free
    ``u`` this equals the standard advection; the extra ``0.5(div u)phi`` term is
    what makes the discrete operator skew-symmetric (conserves the quadratic
    invariant). With ``scalar=None`` it is the vector self-advection form.
    """
    div_u = divergence(state, velocity)
    if scalar is not None:
        adv = advection(state, velocity=velocity, scalar=scalar)
        return adv + 0.5 * div_u * value(state, scalar)
    adv = advection(state, velocity=velocity)
    u_i = stack_components(state, velocity)
    return adv + 0.5 * div_u[..., None] * u_i


__all__ = [
    "advection",
    "directional_derivative",
    "material_derivative",
    "p_laplacian",
    "skew_symmetric_advection",
]
