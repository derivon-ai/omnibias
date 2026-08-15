# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Tropical homotopy twins (jax; theory 01-08).

``beta -> inf`` is temperature collapse, not founding ``delta -> 0``.
Do not conflate the two.
"""

from __future__ import annotations

import jax.numpy as jnp
from jax import Array
from omnibias.struct._core.tropical import TropicalLinear
from omnibias.struct.jax._logsumexp import logsumexp_beta, softmax_beta


def scores(poly: TropicalLinear, x: Array) -> Array:
    m = jnp.asarray(poly.exponents, dtype=x.dtype)
    a = jnp.asarray(poly.coeffs, dtype=x.dtype)
    xv = x if x.ndim == 2 else x[None, :]
    return a + xv @ m.T


def relaxed_value(poly: TropicalLinear, x: Array, *, beta: float) -> Array:
    return logsumexp_beta(scores(poly, x), beta, axis=-1)


def relaxed_weights(poly: TropicalLinear, x: Array, *, beta: float) -> Array:
    return softmax_beta(scores(poly, x), beta, axis=-1)


__all__ = ["relaxed_value", "relaxed_weights", "scores"]
