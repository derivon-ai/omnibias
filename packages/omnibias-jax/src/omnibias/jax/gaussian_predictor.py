# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""JIT-compatible one-pass large-age Gaussian inverse predictors."""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax import Array
from jax.typing import ArrayLike

from .gaussian_heat import _real_array


class GaussianHeatPredictorResult(NamedTuple):
    """Numerical predictor and analytical truncation estimates, not certificates.

    Error and validity have the target batch shape; q is a shared scalar.
    The error estimate excludes floating-point evaluation error.
    """

    value: Array
    approximation_error_bound: Array
    contraction_bound: Array
    valid: Array


def gaussian_heat_inverse_predictor(
    target: Array, centers: ArrayLike, biases: ArrayLike, viscosity: ArrayLike = 1.0, time: ArrayLike = 1.0,
    *, order: int = 1,
) -> GaussianHeatPredictorResult:
    r"""Evaluate an order-0 or order-1 predictor without an inverse iteration.

    With r=a-a0, p=softmax(b-b0+r@target/(2*nu)), mu=sum(p*r), and
    T=sum(p*(r-mu)*|r-mu|²), order zero returns ``time*target+a0+mu``;
    order one subtracts ``T/(4*nu*time)``. Ordinary JAX autodiff differentiates
    this explicit predictor, including higher derivatives; these are not
    implicit derivatives of the exact inverse.

    Target ``(...,d)`` is float32/64. Shared centers ``(m,d)``, biases ``(m,)``
    and positive scalar viscosity/time are cast to its dtype. All five inputs
    remain trainable runtime arguments under jit/vmap. ``order`` is static.
    Relative centers and biases remove common gauges before the logits.

    Exact-arithmetic physical Euclidean truncation bounds are D*q/2 for order
    zero and D*q² for order one, where q=D²/(8*nu*time)<1. Their numerical
    evaluations omit rounding error and are NOT outward-rounded certificates.
    Diagnostics are stopped-gradient. Invalid arithmetic or q>=1 gives NaN
    value/error and false validity. Finite q>=1 is retained for diagnosis;
    invalid shared geometry gives NaN q. Only the O(m²*d) shared diameter uses
    a loop; each target's predictor is O(m*d), with no matrix factorization.
    """
    target = jnp.asarray(target)
    if target.dtype not in (jnp.float32, jnp.float64):
        raise TypeError("target must have dtype float32 or float64")
    if target.ndim < 1 or target.shape[-1] < 1:
        raise ValueError("target must have shape (..., positive dimension)")
    if type(order) is not int:
        raise TypeError("order must be the integer 0 or 1")
    if order not in (0, 1):
        raise ValueError("order must be 0 or 1")
    centers, biases = _real_array(centers, target.dtype), _real_array(biases, target.dtype)
    viscosity, time = _real_array(viscosity, target.dtype), _real_array(time, target.dtype)
    if centers.ndim != 2 or centers.shape[0] < 1 or centers.shape[1] != target.shape[-1]:
        raise ValueError("centers must have shape (positive mode count, target dimension)")
    if biases.shape != (centers.shape[0],):
        raise ValueError("biases must have one entry per center")
    if viscosity.ndim or time.ndim:
        raise ValueError("viscosity and time must be scalars")
    shared = (jnp.isfinite(centers).all() & jnp.isfinite(biases).all() & jnp.isfinite(viscosity) & (viscosity > 0)
              & jnp.isfinite(time) & (time > 0))
    centers = jnp.where(jnp.isfinite(centers), centers, 0.0)
    biases = jnp.where(jnp.isfinite(biases), biases, 0.0)
    viscosity = jnp.where(jnp.isfinite(viscosity) & (viscosity > 0), viscosity, 1.0)
    time = jnp.where(jnp.isfinite(time) & (time > 0), time, 1.0)
    relative, offsets = centers - centers[0], biases - biases[0]
    shared = shared & jnp.isfinite(relative).all() & jnp.isfinite(offsets).all()
    relative, offsets = jnp.where(jnp.isfinite(relative), relative, 0.0), jnp.where(jnp.isfinite(offsets), offsets, 0.0)

    def diameter_step(index: int, maximum: Array) -> Array:
        return jnp.maximum(maximum, jnp.sum((relative - relative[index])**2, axis=-1).max())

    squared_diameter = jax.lax.fori_loop(0, centers.shape[0], diameter_step, jnp.zeros((), dtype=target.dtype))
    denominator = 8 * viscosity * time
    shared = shared & jnp.isfinite(squared_diameter) & jnp.isfinite(denominator) & (denominator > 0)
    safe_denominator = jnp.where(jnp.isfinite(denominator) & (denominator > 0), denominator, 1.0)
    q = squared_diameter / safe_denominator
    shared = shared & jnp.isfinite(q)
    valid = shared & (q < 1) & jnp.isfinite(target).all(-1)
    query = jnp.where(jnp.isfinite(target), target, 0.0)
    logits = jnp.einsum("...d,md->...m", query, relative) / (2 * viscosity) + offsets
    valid = valid & jnp.isfinite(logits).all(-1)
    probabilities = jax.nn.softmax(jnp.where(jnp.isfinite(logits), logits, 0.0), axis=-1)
    mean = jnp.einsum("...m,md->...d", probabilities, relative)
    value = time * query + centers[0] + mean
    if order:
        centered = relative - mean[..., None, :]
        scaled_squared = jnp.sum(centered**2, axis=-1) / (safe_denominator / 2)
        correction = jnp.sum((probabilities * scaled_squared)[..., :, None] * centered, axis=-2)
        value = value - correction
        valid = valid & jnp.isfinite(scaled_squared).all(-1) & jnp.isfinite(correction).all(-1)
    diameter = jnp.sqrt(jax.lax.stop_gradient(squared_diameter))
    diagnostic_q = jax.lax.stop_gradient(q)
    estimate = diameter * (diagnostic_q / 2 if order == 0 else diagnostic_q**2)
    valid = valid & jnp.isfinite(value).all(-1) & jnp.isfinite(estimate)
    return GaussianHeatPredictorResult(
        jnp.where(valid[..., None], value, jnp.nan), jnp.where(valid, estimate, jnp.nan),
        jnp.where(shared, diagnostic_q, jnp.nan), jax.lax.stop_gradient(valid),
    )


__all__ = ["GaussianHeatPredictorResult", "gaussian_heat_inverse_predictor"]
