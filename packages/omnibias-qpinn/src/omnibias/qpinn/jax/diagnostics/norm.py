# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Norm diagnostics for the jax backend."""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax.numpy as jnp
from jax import Array
from omnibias.qpinn._core.complex import psi_density

if TYPE_CHECKING:  # pragma: no cover
    from omnibias.pinn._core.state import FieldState


def norm_squared(
    state: FieldState,
    *,
    group: str = "psi",
    quadrature_weights: Array | None = None,
) -> Array:
    r"""Compute :math:`\int |\psi|^2\,dx`. JAX twin of
    :func:`omnibias.qpinn.torch.diagnostics.norm_squared`."""
    density = psi_density(state, group)
    if quadrature_weights is None:
        return jnp.mean(density)
    if quadrature_weights.shape != density.shape:
        raise ValueError(
            f"quadrature_weights shape {tuple(quadrature_weights.shape)} "
            f"!= density shape {tuple(density.shape)}"
        )
    return jnp.sum(quadrature_weights * density)


def norm_drift(
    state: FieldState,
    *,
    group: str = "psi",
    quadrature_weights: Array | None = None,
    target_norm: float = 1.0,
) -> Array:
    return jnp.abs(
        norm_squared(state, group=group, quadrature_weights=quadrature_weights)
        - target_norm
    )


__all__ = ["norm_drift", "norm_squared"]
