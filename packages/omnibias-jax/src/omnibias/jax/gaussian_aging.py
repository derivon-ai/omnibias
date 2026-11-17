# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""JIT-compatible forward aging series for Gaussian heat inverses."""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax import Array
from jax.typing import ArrayLike

from .gaussian_heat import _real_array


class GaussianHeatAgingResult(NamedTuple):
    """Truncated inverse, ordinary correction coefficients and diagnostics.

    Error and validity have the target batch shape; q is a shared scalar.
    The error estimate excludes floating-point evaluation error.
    Coefficients have shape ``(order+1, ..., dimension)`` with zero order zero.
    """

    value: Array
    correction_coefficients: Array
    approximation_error_bound: Array
    contraction_bound: Array
    valid: Array


def gaussian_heat_inverse_aging(
    target: Array, centers: ArrayLike, biases: ArrayLike, viscosity: ArrayLike = 1.0, time: ArrayLike = 1.0,
    *, order: int = 8,
) -> GaussianHeatAgingResult:
    r"""Evaluate the Gaussian inverse aging series without iteration or solves.

    With p=softmax(b+a@target/(2*nu)), mu=sum(p*a), the ordinary coefficients
    c_n describe ``X(target,time/t)-(time/t)*target-mu = sum(c_n*t**n)``.
    Structural c_0=0; ``value`` adds ``time*target+mu`` to the series at t=1.
    Only this centered correction is analytic at t=0. Each new coefficient
    depends on earlier orders and requires no implicit matrix solve.
    JAX autodiff differentiates this finite formula, including higher
    derivatives; these are not implicit derivatives of the true inverse.

    Target ``(...,d)`` is float32/64. Shared centers ``(m,d)``, biases ``(m,)``
    and positive scalar viscosity/time are cast to its dtype. All five inputs
    remain trainable runtime arguments under jit/vmap. ``order`` is a static
    nonnegative integer.
    Relative centers and biases remove common gauges before the logits.

    Require q=D²/(8*nu*time)<1/3, where D is the center diameter. The analytical
    Euclidean tail bound is ``D/2*(3*q)**(order+1)/(1-3*q)``; orders zero and
    one use the minimum with D*q/2 and D*q², respectively. Identical centers
    give an exact affine result with zero correction. The numerical bounds
    evaluations omit rounding error and are NOT outward-rounded certificates.
    Diagnostics are stopped-gradient. Invalid arithmetic or q>=1/3 gives NaN
    value/coefficients/error and false validity. Finite q is retained for
    diagnosis; invalid shared geometry gives NaN q. N orders use
    O((m+d)*N²+m*d*N) arithmetic and O((m+d)*N+m*d) forward storage per target,
    plus the shared O(m²*d) diameter. Autodiff may retain more storage.
    """
    target = jnp.asarray(target)
    if target.dtype not in (jnp.float32, jnp.float64):
        raise TypeError("target must have dtype float32 or float64")
    if target.ndim < 1 or target.shape[-1] < 1:
        raise ValueError("target must have shape (..., positive dimension)")
    if type(order) is not int:
        raise TypeError("order must be a nonnegative integer")
    if order < 0:
        raise ValueError("order must be nonnegative")
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
    valid = shared & (q < 1 / 3) & jnp.isfinite(target).all(-1)
    query = jnp.where(jnp.isfinite(target), target, 0.0)
    logits = jnp.einsum("...d,md->...m", query, relative) / (2 * viscosity) + offsets
    valid = valid & jnp.isfinite(logits).all(-1)
    probabilities = jax.nn.softmax(jnp.where(jnp.isfinite(logits), logits, 0.0), axis=-1)
    mean = jnp.einsum("...m,md->...d", probabilities, relative)
    centered = relative - mean[..., None, :]
    coefficients = [jnp.zeros_like(query)]
    logarithms = [jnp.zeros_like(probabilities)]
    exponentials = [jnp.ones_like(probabilities)]
    partitions = [jnp.ones_like(valid, dtype=target.dtype)]
    for n in range(1, order + 1):
        logarithm = (-jnp.sum(centered**2, axis=-1) / (safe_denominator / 2) if n == 1
                     else jnp.sum(centered * coefficients[n - 1][..., None, :], axis=-1) / (safe_denominator / 4))
        logarithms.append(logarithm)
        exponential = jnp.zeros_like(probabilities)
        for k in range(1, n + 1):
            exponential = exponential + (k / n) * logarithms[k] * exponentials[n - k]
        exponentials.append(exponential)
        weighted = probabilities * exponential
        partition = jnp.sum(weighted, axis=-1)
        partitions.append(partition)
        coefficient = jnp.sum(weighted[..., :, None] * centered, axis=-2)
        for k in range(1, n + 1):
            coefficient = coefficient - partitions[k][..., None] * coefficients[n - k]
        coefficients.append(coefficient)
        valid = (valid & jnp.isfinite(logarithm).all(-1) & jnp.isfinite(exponential).all(-1)
                 & jnp.isfinite(partition) & jnp.isfinite(coefficient).all(-1))
    stacked = jnp.stack(coefficients)
    value = time * query + centers[0] + mean + jnp.sum(stacked, axis=0)
    diameter = jnp.sqrt(jax.lax.stop_gradient(squared_diameter))
    diagnostic_q = jax.lax.stop_gradient(q)
    safe_q = jnp.where(shared & (diagnostic_q < 1 / 3), diagnostic_q, 0.0)
    estimate = diameter / 2 * (3 * safe_q)**(order + 1) / (1 - 3 * safe_q)
    if order == 0:
        estimate = jnp.minimum(estimate, diameter * safe_q / 2)
    elif order == 1:
        estimate = jnp.minimum(estimate, diameter * safe_q**2)
    valid = valid & jnp.isfinite(value).all(-1) & jnp.isfinite(estimate)
    return GaussianHeatAgingResult(
        jnp.where(valid[..., None], value, jnp.nan),
        jnp.where(valid[None, ..., None], stacked, jnp.nan), jnp.where(valid, estimate, jnp.nan),
        jnp.where(shared, diagnostic_q, jnp.nan), jax.lax.stop_gradient(valid),
    )


__all__ = ["GaussianHeatAgingResult", "gaussian_heat_inverse_aging"]
