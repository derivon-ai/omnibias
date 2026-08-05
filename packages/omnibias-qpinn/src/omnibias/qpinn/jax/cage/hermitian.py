# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Hermitian-operator projection helpers (jax twin)."""

from __future__ import annotations

import jax.numpy as jnp
from jax import Array


def hermitian_projection(matrix: Array) -> Array:
    r"""Return the Hermitian part of a complex matrix. JAX twin of
    :func:`omnibias.qpinn.torch.cage.hermitian_projection`."""
    if matrix.ndim < 2:
        raise ValueError(
            f"matrix must have at least 2 dimensions; got shape {tuple(matrix.shape)}"
        )
    if matrix.shape[-1] != matrix.shape[-2]:
        raise ValueError(
            f"last two dims must match; got "
            f"{matrix.shape[-2]} x {matrix.shape[-1]}"
        )
    if jnp.iscomplexobj(matrix):
        return 0.5 * (matrix + jnp.conj(jnp.swapaxes(matrix, -1, -2)))
    return 0.5 * (matrix + jnp.swapaxes(matrix, -1, -2))


def hermiticity_loss(matrix: Array) -> Array:
    r"""Soft loss :math:`||O - O^\dagger||_F^2 / ||O||_F^2`. JAX twin."""
    if matrix.ndim < 2:
        raise ValueError(
            f"matrix must have at least 2 dimensions; got shape {tuple(matrix.shape)}"
        )
    if matrix.shape[-1] != matrix.shape[-2]:
        raise ValueError(
            f"last two dims must match; got "
            f"{matrix.shape[-2]} x {matrix.shape[-1]}"
        )
    if jnp.iscomplexobj(matrix):
        diff = matrix - jnp.conj(jnp.swapaxes(matrix, -1, -2))
    else:
        diff = matrix - jnp.swapaxes(matrix, -1, -2)
    num = jnp.sum(jnp.abs(diff) ** 2)
    den = jnp.sum(jnp.abs(matrix) ** 2)
    eps = jnp.finfo(num.dtype if jnp.issubdtype(num.dtype, jnp.floating) else jnp.float32).tiny
    return num / (den + eps)


__all__ = ["hermitian_projection", "hermiticity_loss"]
