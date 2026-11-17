# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Differentiable numerical inverses of contracting Gaussian heat velocities.

This backend uses floating-point contraction diagnostics, not rigorous interval
certificates. Fixed-point iteration proposes an inverse; derivatives use the
implicit equation and do not differentiate through the iteration history.
Higher derivatives differentiate that same implicit rule at the approximate
root. See ``omnibias.core.verified.inverse_gaussian_heat`` for rigorous evidence.
"""

from __future__ import annotations

import math
from functools import partial
from typing import NamedTuple, cast

import jax
import jax.numpy as jnp
from jax import Array
from jax.typing import ArrayLike


class GaussianHeatInverseResult(NamedTuple):
    """Physical inverse and numerical diagnostics; these are not certificates.

    ``value`` has the target shape; ``converged`` and ``residual_bound`` have
    its batch shape. ``contraction_bound`` is shared and scalar. Refused or
    unconverged entries have NaN values; invalid diagnostics have infinite
    residual bounds. The residual bound is a numerical physical L2 estimate.
    """

    value: Array
    converged: Array
    residual_bound: Array
    contraction_bound: Array


def _real_array(value: ArrayLike, dtype: jnp.dtype) -> Array:
    raw = jnp.asarray(value)
    if jnp.issubdtype(raw.dtype, jnp.complexfloating) or raw.dtype == jnp.bool_:
        raise TypeError("Gaussian heat parameters must be real and exclude bool")
    return jnp.asarray(value, dtype=dtype)


def _geometry(centers: Array, biases: Array, viscosity: Array, time: Array) -> tuple[Array, Array, Array, Array]:
    relative = centers - centers[0]
    squared = jnp.sum(relative * relative, axis=-1)
    denominator = 4 * viscosity * time
    shifted = biases - biases[0] - squared / denominator

    def diameter_step(index: int, maximum: Array) -> Array:
        difference = relative - relative[index]
        return jnp.maximum(maximum, jnp.max(jnp.sum(difference * difference, axis=-1)))

    diameter = jax.lax.fori_loop(0, centers.shape[0], diameter_step, jnp.zeros((), dtype=centers.dtype))
    contraction = diameter / (2 * denominator)
    valid = (jnp.all(jnp.isfinite(centers)) & jnp.all(jnp.isfinite(biases))
             & jnp.all(jnp.isfinite(relative)) & jnp.all(jnp.isfinite(squared))
             & jnp.all(jnp.isfinite(shifted)) & jnp.isfinite(viscosity) & (viscosity > 0)
             & jnp.isfinite(time) & (time > 0) & jnp.isfinite(denominator) & (denominator > 0)
             & jnp.isfinite(2 * denominator) & jnp.isfinite(contraction) & (contraction >= 0))
    return relative, shifted, contraction, valid


def _probabilities(z: Array, relative: Array, shifted: Array, viscosity: Array) -> Array:
    logits = jnp.einsum("...d,md->...m", z, relative) / (2 * viscosity) + shifted
    return jax.nn.softmax(logits, axis=-1)


def _fixed_point(z: Array, target: Array, centers: Array, biases: Array, viscosity: Array, time: Array) -> Array:
    relative = centers - centers[0]
    shifted = biases - biases[0] - jnp.sum(relative * relative, axis=-1) / (4 * viscosity * time)
    probabilities = _probabilities(z, relative, shifted, viscosity)
    return target + jnp.einsum("...m,md->...d", probabilities, relative) / time


def _implicit_solve(z: Array, centers: Array, biases: Array, viscosity: Array, time: Array,
                    rhs: Array, adjoint: str) -> Array:
    relative = centers - centers[0]
    shifted = biases - biases[0] - jnp.sum(relative * relative, axis=-1) / (4 * viscosity * time)
    probabilities = _probabilities(z, relative, shifted, viscosity)
    mean = jnp.einsum("...m,md->...d", probabilities, relative)
    deviations = relative - mean[..., None, :]
    count, dimension = centers.shape
    use_low_rank = adjoint == "low_rank" or (adjoint == "auto" and count < dimension)
    if use_low_rank:
        # Underflowed probabilities are exact numerical zeros. Avoid sqrt'(0)
        # leaking NaNs through otherwise zero tangents in higher derivatives.
        positive = probabilities > 0
        roots = jnp.where(positive, jnp.sqrt(jnp.where(positive, probabilities, 1.0)), 0.0)
        rows = roots[..., :, None] * deviations / jnp.sqrt(2 * viscosity * time)
        reduced = jnp.eye(count, dtype=z.dtype) - jnp.einsum("...md,...nd->...mn", rows, rows)
        projected = jnp.einsum("...md,...d->...m", rows, rhs)
        solved = jnp.linalg.solve(reduced, projected[..., None])[..., 0]
        return rhs + jnp.einsum("...md,...m->...d", rows, solved)
    covariance = jnp.einsum("...m,...mi,...mj->...ij", probabilities, deviations, deviations)
    matrix = jnp.eye(dimension, dtype=z.dtype) - covariance / (2 * viscosity * time)
    return cast(Array, jnp.linalg.solve(matrix, rhs[..., None])[..., 0])


@partial(jax.custom_jvp, nondiff_argnums=(6,))
def _implicit_root(z: Array, target: Array, centers: Array, biases: Array, viscosity: Array,
                   time: Array, adjoint: str) -> Array:
    """Attach the implicit derivative to a stopped-gradient numerical root."""
    return z


@_implicit_root.defjvp
def _implicit_root_jvp(
    adjoint: str, primals: tuple[Array, ...], tangents: tuple[Array, ...],
) -> tuple[Array, Array]:
    z, target, centers, biases, viscosity, time = primals
    _, target_dot, centers_dot, biases_dot, viscosity_dot, time_dot = tangents
    root = _implicit_root(z, target, centers, biases, viscosity, time, adjoint)
    _, rhs = jax.jvp(lambda v, a, b, nu, s: _fixed_point(root, v, a, b, nu, s),
                     (target, centers, biases, viscosity, time),
                     (target_dot, centers_dot, biases_dot, viscosity_dot, time_dot))
    derivative = _implicit_solve(root, centers, biases, viscosity, time, rhs, adjoint)
    return root, derivative


def gaussian_heat_inverse(
    target: Array, centers: ArrayLike, biases: ArrayLike, viscosity: ArrayLike = 1.0, time: ArrayLike = 1.0,
    *, max_iterations: int = 128, tolerance: float | None = None, adjoint: str = "auto",
) -> GaussianHeatInverseResult:
    r"""Invert ``U(x)=(x-E[a])/s`` for a shared isotropic Gaussian mixture.

    ``centers`` has shape ``(m,d)``, ``biases`` shape ``(m,)``, and ``target``
    shape ``(...,d)``. All five numerical inputs remain runtime parameters and
    are differentiable. Parameters are cast to the target's float32/64 dtype.
    Scalar positive viscosity and time are shared across target batches.

    The centered iteration solves ``z=target+E[a-a0]/s``, ``x=a0+s*z``.
    It accepts only the sufficient numerical contraction regime
    ``q=max_ij ||a_i-a_j||²/(8*viscosity*time)<1``. After bounded updates it
    checks ``time*||z-T(z)||₂/(1-q) <= tolerance``; default tolerance is
    ``64*finfo(dtype).eps``. Invalid shared data, invalid batch targets, or
    unsuccessful convergence produce false flags and NaN values, not claims of
    nonexistence. Diagnostics are stopped-gradient numerical estimates;
    the implicit derivative contract applies only to converged entries.

    Implicit derivatives solve ``(I-Cov(a)/(2*viscosity*time)) dz = rhs``.
    ``adjoint='low_rank'`` uses an m-by-m Woodbury solve without allocating
    a d-by-d covariance; ``'dense'`` uses d-by-d, and ``'auto'`` uses the smaller.
    Iteration count, tolerance and adjoint selection are static configuration.
    """
    target = jnp.asarray(target)
    if target.dtype not in (jnp.dtype("float32"), jnp.dtype("float64")):
        raise TypeError("target must have dtype float32 or float64")
    if target.ndim < 1 or target.shape[-1] < 1:
        raise ValueError("target must have shape (...,d) with positive d")
    if isinstance(max_iterations, bool) or not isinstance(max_iterations, int):
        raise TypeError("max_iterations must be a nonnegative integer")
    if max_iterations < 0:
        raise ValueError("max_iterations must be nonnegative")
    if adjoint not in ("auto", "dense", "low_rank"):
        raise ValueError("adjoint must be 'auto', 'dense' or 'low_rank'")
    if tolerance is not None:
        if isinstance(tolerance, bool) or not isinstance(tolerance, (int, float)):
            raise TypeError("tolerance must be a positive finite scalar or None")
        if not math.isfinite(tolerance) or tolerance <= 0:
            raise ValueError("tolerance must be positive and finite")
    epsilon = 2.0**(-23 if target.dtype == jnp.dtype("float32") else -52)
    threshold = 64 * epsilon if tolerance is None else tolerance
    centers = _real_array(centers, target.dtype)
    biases = _real_array(biases, target.dtype)
    viscosity = _real_array(viscosity, target.dtype)
    time = _real_array(time, target.dtype)
    if centers.ndim != 2 or centers.shape[0] < 1 or centers.shape[1] != target.shape[-1]:
        raise ValueError("centers must have shape (m,d), with m positive and d matching target")
    if biases.shape != (centers.shape[0],):
        raise ValueError("biases must have shape (m,) matching centers")
    if viscosity.ndim or time.ndim:
        raise ValueError("viscosity and time must be shared scalars")
    relative, shifted, contraction, valid_shared = _geometry(centers, biases, viscosity, time)

    def mapping(z: Array) -> tuple[Array, Array]:
        logits = jnp.einsum("...d,md->...m", z, relative) / (2 * viscosity) + shifted
        probabilities = jax.nn.softmax(logits, axis=-1)
        image = target + jnp.einsum("...m,md->...d", probabilities, relative) / time
        finite = jnp.all(jnp.isfinite(logits), axis=-1) & jnp.all(jnp.isfinite(image), axis=-1)
        return image, finite

    def iterate(_index: int, state: tuple[Array, Array]) -> tuple[Array, Array]:
        image, finite = mapping(state[0])
        return image, state[1] & finite

    proposal, finite_history = jax.lax.fori_loop(
        0, max_iterations, iterate, (target, jnp.ones(target.shape[:-1], dtype=jnp.bool_)),
    )
    image, finite_final = mapping(proposal)
    residual = proposal - image
    magnitude = jnp.max(jnp.abs(residual), axis=-1)
    normalizer = jnp.where(magnitude > 0, magnitude, 1.0)
    residual_norm = magnitude * jnp.sqrt(jnp.sum((residual / normalizer[..., None])**2, axis=-1))
    bound = time * residual_norm / (1 - contraction)
    physical = centers[0] + time * proposal
    valid = (valid_shared & finite_history & finite_final & (contraction < 1)
             & jnp.all(jnp.isfinite(target), axis=-1)
             & jnp.all(jnp.isfinite(proposal), axis=-1) & jnp.all(jnp.isfinite(physical), axis=-1)
             & jnp.isfinite(bound) & (bound >= 0))
    converged = valid & (bound <= threshold)
    accepted = jax.lax.stop_gradient(jnp.where(converged[..., None], proposal, jnp.nan))
    root = _implicit_root(accepted, target, centers, biases, viscosity, time, adjoint)
    value = centers[0] + time * root
    return GaussianHeatInverseResult(
        value, jax.lax.stop_gradient(converged), jax.lax.stop_gradient(jnp.where(valid, bound, jnp.inf)),
        jax.lax.stop_gradient(jnp.where(valid_shared, contraction, jnp.inf)),
    )


__all__ = ["GaussianHeatInverseResult", "gaussian_heat_inverse"]
