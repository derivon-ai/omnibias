# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Probability-current diagnostics for the jax backend."""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax.numpy as jnp
from jax import Array
from omnibias.qpinn._core.complex import psi_value

if TYPE_CHECKING:  # pragma: no cover
    from omnibias.pinn._core.state import FieldState


def probability_current(
    state: FieldState,
    *,
    axes: tuple[int | str, ...] | None = None,
    group: str = "psi",
    hbar: float = 1.0,
    mass: float = 1.0,
) -> Array:
    if mass <= 0:
        raise ValueError(f"mass must be > 0, got {mass}")
    coordinate_spec = state.coordinate_spec
    if axes is None:
        axes = tuple(coordinate_spec.spatial_axes)
    re_name = f"{group}_re"
    im_name = f"{group}_im"
    psi_re, psi_im = psi_value(state, group)
    cols: list[Array] = []
    for a in axes:
        a_idx = coordinate_spec.axis_index(a)
        d_re = state.ops.derivative(state, re_name, axis=a_idx, order=1)
        d_im = state.ops.derivative(state, im_name, axis=a_idx, order=1)
        j_i = (hbar / mass) * (psi_re * d_im - psi_im * d_re)
        cols.append(j_i)
    return jnp.stack(cols, axis=-1)


def current_divergence(
    state: FieldState,
    *,
    group: str = "psi",
    hbar: float = 1.0,
    mass: float = 1.0,
) -> Array:
    if mass <= 0:
        raise ValueError(f"mass must be > 0, got {mass}")
    coordinate_spec = state.coordinate_spec
    spatial = coordinate_spec.spatial_axes
    re_name = f"{group}_re"
    im_name = f"{group}_im"
    div = None
    for a in spatial:
        a_idx = coordinate_spec.axis_index(a)
        psi_re = state.ops.value(state, re_name)
        psi_im = state.ops.value(state, im_name)
        d2_re = state.ops.derivative(state, re_name, axis=a_idx, order=2)
        d2_im = state.ops.derivative(state, im_name, axis=a_idx, order=2)
        contrib = (hbar / mass) * (psi_re * d2_im - psi_im * d2_re)
        div = contrib if div is None else div + contrib
    assert div is not None
    return div


def continuity_residual(
    state: FieldState,
    *,
    group: str = "psi",
    hbar: float = 1.0,
    mass: float = 1.0,
) -> Array:
    time = state.coordinate_spec.time_axis
    if time is None:
        raise ValueError(
            "continuity_residual requires a time axis in the coordinate spec"
        )
    re_name = f"{group}_re"
    im_name = f"{group}_im"
    psi_re, psi_im = psi_value(state, group)
    psi_re_t = state.ops.derivative(state, re_name, axis=time, order=1)
    psi_im_t = state.ops.derivative(state, im_name, axis=time, order=1)
    rho_t = 2.0 * (psi_re * psi_re_t + psi_im * psi_im_t)
    div_j = current_divergence(state, group=group, hbar=hbar, mass=mass)
    return rho_t + div_j


__all__ = [
    "continuity_residual",
    "current_divergence",
    "probability_current",
]
