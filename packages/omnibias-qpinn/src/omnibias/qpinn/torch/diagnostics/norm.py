# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Norm diagnostics for the torch backend.

Diagnostics here read from a :class:`FieldState` and return scalar
quantities that are useful for monitoring training (and for the
norm-conservation soft loss). All routines are pure functions; nothing
in this module mutates the state or the field.
"""

from __future__ import annotations

from omnibias.pinn._core.state import FieldState
from omnibias.qpinn._core.complex import psi_density
from torch import Tensor


def norm_squared(
    state: FieldState,
    *,
    group: str = "psi",
    quadrature_weights: Tensor | None = None,
) -> Tensor:
    r"""Compute :math:`\int |\psi|^2\,dx` on the state's collocation points.

    Parameters
    ----------
    state
        :class:`FieldState` carrying the wavefunction group.
    group
        Wavefunction group name. Default ``"psi"``.
    quadrature_weights
        Quadrature weights of shape ``(B,)``. If ``None``, the
        uniform-mean approximation ``density.mean()`` is returned.

    Returns
    -------
    Tensor
        Zero-dimensional ``Tensor`` with the integrated density.
    """
    density = psi_density(state, group)
    if quadrature_weights is None:
        return density.mean()
    if quadrature_weights.shape != density.shape:
        raise ValueError(
            f"quadrature_weights shape {tuple(quadrature_weights.shape)} "
            f"!= density shape {tuple(density.shape)}"
        )
    return (quadrature_weights * density).sum()


def norm_drift(
    state: FieldState,
    *,
    group: str = "psi",
    quadrature_weights: Tensor | None = None,
    target_norm: float = 1.0,
) -> Tensor:
    r"""Return ``|integral |psi|^2 - target_norm|`` (the unsigned norm drift).

    Parameters
    ----------
    state
        :class:`FieldState`.
    group
        Wavefunction group name.
    quadrature_weights
        Per-collocation weights. If ``None``, uniform.
    target_norm
        Expected :math:`L^2` norm squared. Default 1.0.
    """
    return (norm_squared(state, group=group, quadrature_weights=quadrature_weights)
            - target_norm).abs()


__all__ = ["norm_drift", "norm_squared"]
