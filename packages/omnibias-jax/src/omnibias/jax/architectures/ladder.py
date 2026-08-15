# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Hermite ladder JAX twin (theory 02-10)."""

from __future__ import annotations

from omnibias.core.ladder import Normalization, hermite_function

import jax.numpy as jnp
from jax import Array


def hermite_basis(
    x: Array,
    n_levels: int,
    *,
    normalization: Normalization,
    scale: float = 1.0,
    centre: float = 0.0,
) -> Array:
    xs = jnp.asarray(x).reshape(-1).tolist()
    rows = [
        [
            hermite_function(
                n, float(xi), normalization=normalization, scale=scale, centre=centre
            )
            for n in range(n_levels)
        ]
        for xi in xs
    ]
    return jnp.asarray(rows, dtype=jnp.asarray(x).dtype).reshape(*jnp.asarray(x).shape, n_levels)


def ladder_apply(coeffs: Array, basis: Array) -> Array:
    return jnp.sum(coeffs * basis, axis=-1)


def apply_operator(coeffs: Array, which: str) -> Array:
    n = coeffs.shape[-1]
    idx = jnp.arange(n, dtype=coeffs.dtype)
    if which == "N":
        return coeffs * idx
    return coeffs * (idx + 0.5)


__all__ = ["apply_operator", "hermite_basis", "ladder_apply"]
