# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Probability-current diagnostics for the torch backend.

The probability current for a single-particle wavefunction is

.. math::

    j_i(x) = \frac{\hbar}{m}\,\Im(\psi^* \partial_i \psi)
           = \frac{\hbar}{m}\,(\psi_R \partial_i \psi_I
                              - \psi_I \partial_i \psi_R),

with continuity equation :math:`\partial_t \rho + \nabla \cdot j = 0`
where :math:`\rho = |\psi|^2`. Useful both as a physical diagnostic
and as a soft loss term enforcing the continuity equation.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from omnibias.qpinn._core.complex import psi_value
from torch import Tensor

if TYPE_CHECKING:  # pragma: no cover
    from omnibias.pinn._core.state import FieldState


def probability_current(
    state: FieldState,
    *,
    axes: tuple[int | str, ...] | None = None,
    group: str = "psi",
    hbar: float = 1.0,
    mass: float = 1.0,
) -> Tensor:
    r"""Compute the probability current vector :math:`j_i(x)`.

    Parameters
    ----------
    state
        :class:`FieldState` carrying the wavefunction group.
    axes
        Spatial axes along which to compute the current. ``None``
        defaults to all spatial axes (i.e. the time axis is excluded
        if present).
    group
        Wavefunction group name. Default ``"psi"``.
    hbar
        Planck constant. Default 1.0.
    mass
        Particle mass. Default 1.0.

    Returns
    -------
    Tensor
        Shape ``(B, len(axes))``. Each column ``j_i`` is the probability
        current along the corresponding axis.
    """
    if mass <= 0:
        raise ValueError(f"mass must be > 0, got {mass}")
    coordinate_spec = state.coordinate_spec
    if axes is None:
        axes = tuple(coordinate_spec.spatial_axes)
    re_name = f"{group}_re"
    im_name = f"{group}_im"
    psi_re, psi_im = psi_value(state, group)
    cols: list[Tensor] = []
    for a in axes:
        a_idx = coordinate_spec.axis_index(a)
        d_re = state.ops.derivative(state, re_name, axis=a_idx, order=1)
        d_im = state.ops.derivative(state, im_name, axis=a_idx, order=1)
        j_i = (hbar / mass) * (psi_re * d_im - psi_im * d_re)
        cols.append(j_i)
    return torch.stack(cols, dim=-1)


def current_divergence(
    state: FieldState,
    *,
    group: str = "psi",
    hbar: float = 1.0,
    mass: float = 1.0,
) -> Tensor:
    r"""Compute :math:`\nabla\cdot j` on the spatial axes.

    Returns a tensor of shape ``(B,)``. Combined with :math:`\partial_t
    \rho` this is the continuity-equation residual.
    """
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
        # d/da[psi_re * d_a psi_im - psi_im * d_a psi_re]
        #   = d_re * d_im + psi_re * d2_im - d_im * d_re - psi_im * d2_re
        #   = psi_re * d2_im - psi_im * d2_re
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
) -> Tensor:
    r"""Continuity-equation residual :math:`\partial_t \rho + \nabla\cdot j`.

    Returns a tensor of shape ``(B,)``. Should be approximately zero
    for any wavefunction obeying the time-dependent Schrodinger
    equation; useful as a soft physical-consistency loss term.
    """
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
