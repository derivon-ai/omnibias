# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""JIT-compatible Gaussian weak measurements of distributional ridges."""

from __future__ import annotations

from math import pi, sqrt

from omnibias.core.polynomials import hermite_coeffs
from omnibias.core.ridge_distribution import RidgeDistribution, gaussian_pairing_orders

import jax.numpy as jnp
from jax import Array
from jax.scipy.special import erfc


def gaussian_weak(
    field: RidgeDistribution,
    mean: Array,
    variance: float | Array,
    bias: float | Array | None = None,
) -> Array:
    r"""Pair with Gaussian probes using projected mean and variance.

    Supply w@mu and w@Sigma@w for the ambient normalized probe N(mu,Sigma).
    The field carries delta(w.x+b), with its pullback normalization, not
    unit surface measure. A scalar bias override stays trainable. Mean and
    variance broadcast; output shape is ``broadcast_shape+(outputs,)``.
    Close over immutable field metadata for ``jit``; mean, variance and bias
    remain runtime inputs compatible with ``vmap`` and parameter gradients.

    Positive finite variance is required; invalid variances return NaNs
    without host-side branching. Analytic real-arithmetic formulas are not
    interval certificates. Extreme tails/high orders/tiny variances can lose
    accuracy or overflow, including cancellation of regular moments.
    This pairs over all of R^d and does not impose boundary conditions.
    """
    regular_order, _ = gaussian_pairing_orders(field)
    if not jnp.issubdtype(mean.dtype, jnp.floating):
        raise TypeError("mean must have a real floating dtype")
    var = jnp.asarray(variance, dtype=mean.dtype)
    offset = jnp.asarray(float(field.bias) if bias is None else bias, dtype=mean.dtype)
    if offset.ndim != 0:
        raise ValueError("bias must be scalar")
    m, var = jnp.broadcast_arrays(mean + offset, var)
    var = jnp.where((var > 0) & jnp.isfinite(var), var, jnp.full_like(var, jnp.nan))
    scale = jnp.sqrt(var)
    standardized = m / scale
    density = jnp.exp(-0.5 * standardized**2) / (sqrt(2 * pi) * scale)
    regular: list[Array] = []
    if regular_order >= 0:
        regular.append(0.5 * erfc(-standardized / sqrt(2)))
    if regular_order >= 1:
        regular.append(m * regular[0] + var * density)
    for degree in range(2, regular_order + 1):
        regular.append((m * regular[-1] + var * regular[-2]) / degree)
    result = jnp.broadcast_to((m * 0 + var * 0)[..., None], (*m.shape, field.outputs))
    for index, row in field.terms:
        if index >= 0:
            value = regular[index]
        else:
            order = -index - 1
            polynomial = jnp.zeros_like(m)
            for coefficient in reversed(hermite_coeffs(order)):
                polynomial = polynomial * standardized + coefficient
            value = (-1)**order * polynomial * density / scale**order
        coefficient_row = jnp.asarray([float(coefficient) for coefficient in row], dtype=mean.dtype)
        result = result + value[..., None] * coefficient_row
    return result


__all__ = ["gaussian_weak"]
