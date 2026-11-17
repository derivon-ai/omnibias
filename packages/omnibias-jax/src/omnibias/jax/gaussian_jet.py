# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Forward Taylor jets of Gaussian heat inverses with one reused factorization.

Coefficients are ordinary Taylor coefficients (the raw derivative divided by
its factorial). The constant coefficient is a numerically computed implicit
inverse. Higher coefficients are anchored at the actual velocity of that base
point; the anchor diagnostics expose any discrepancy from the requested base
velocity. This is numerical jet propagation, not an interval certificate.
"""

from __future__ import annotations

from typing import NamedTuple

from omnibias.jax.gaussian_heat import GaussianHeatInverseResult, _real_array, gaussian_heat_inverse

import jax
import jax.numpy as jnp
import jax.scipy.linalg as jsp_linalg
from jax import Array
from jax.typing import ArrayLike


class GaussianHeatInverseJet(NamedTuple):
    """Numerical physical inverse coefficients and their actual base anchor.

    ``coefficients`` has the input's ``(N+1,...,d)`` shape. ``anchor_target``
    and ``base_target_residual=anchor_target-requested_target[0]`` have shape
    ``(...,d)``. The base result's convergence flag concerns only that solve;
    nonfinite higher coefficients or anchor diagnostics also invalidate a jet.
    """

    coefficients: Array
    base_result: GaussianHeatInverseResult
    anchor_target: Array
    base_target_residual: Array


def _factor(
    probabilities: Array, relative: Array, mean: Array, viscosity: Array, time: Array,
    valid: Array, adjoint: str,
) -> tuple[Array, Array | None]:
    count, dimension = relative.shape
    centered = relative - mean[..., None, :]
    low_rank = adjoint == "low_rank" or (adjoint == "auto" and count < dimension)
    if low_rank:
        positive = probabilities > 0
        roots = jnp.where(positive, jnp.sqrt(jnp.where(positive, probabilities, 1.0)), 0.0)
        rows = roots[..., :, None] * centered / jnp.sqrt(2 * viscosity * time)
        rows = jnp.where(valid[..., None, None], rows, 0.0)
        matrix = jnp.eye(count, dtype=relative.dtype) - jnp.einsum("...md,...nd->...mn", rows, rows)
        return jnp.linalg.cholesky(matrix), rows
    covariance = jnp.einsum("...m,...mi,...mj->...ij", probabilities, centered, centered)
    matrix = jnp.eye(dimension, dtype=relative.dtype) - covariance / (2 * viscosity * time)
    matrix = jnp.where(valid[..., None, None], matrix, jnp.eye(dimension, dtype=relative.dtype))
    return jnp.linalg.cholesky(matrix), None


def _solve(factor: Array, rows: Array | None, right: Array) -> Array:
    if rows is None:
        return jsp_linalg.cho_solve((factor, True), right[..., None])[..., 0]
    projected = jnp.einsum("...md,...d->...m", rows, right)
    solved = jsp_linalg.cho_solve((factor, True), projected[..., None])[..., 0]
    return right + jnp.einsum("...md,...m->...d", rows, solved)


def gaussian_heat_inverse_jet(
    target_coefficients: Array, centers: ArrayLike, biases: ArrayLike,
    viscosity: ArrayLike = 1.0, time: ArrayLike = 1.0, *, max_iterations: int = 128,
    tolerance: float | None = None, adjoint: str = "auto",
) -> GaussianHeatInverseJet:
    r"""Propagate a scalar-path inverse jet without nesting autodiff in order.

    Input shape is ``(N+1,...,d)`` and ``v(e)=sum(v_n*e**n)``. The returned
    ``x_n`` are physical-coordinate Taylor coefficients. The computed base
    ``x_0`` comes from :func:`gaussian_heat_inverse`; orders above zero solve
    ``U(x(e))=U(x_0)+sum(v_n*e**n,n>=1)``. Thus their exact mathematical
    anchor is ``U(x_0)``, even when ``x_0`` only approximately inverts ``v_0``.

    One Cholesky factor of ``H=I-Cov(a)/(2*viscosity*time)`` is reused at
    all higher orders. A low-rank Woodbury factor uses an m-by-m matrix;
    ``adjoint='auto'`` chooses the smaller dimension. Order zero needs no
    factorization. Exponential and quotient recurrences cost quadratic work
    in Taylor order and never call nested differentiation to build the jet.

    Every coefficient and mixture parameter is runtime-trainable. The base
    solve retains its implicit derivative; differentiation of the higher
    recurrence is ordinary JAX autodiff. The derivative contract requires
    converged base entries and finite jet arithmetic. Failure produces NaN
    coefficients/anchors; diagnostics are numerical, not rigorous bounds.
    """
    coefficients = jnp.asarray(target_coefficients)
    if coefficients.ndim < 2 or coefficients.shape[0] < 1:
        raise ValueError("target_coefficients must have shape (N+1,...,d) with nonempty order axis")
    base = gaussian_heat_inverse(coefficients[0], centers, biases, viscosity, time,
                                 max_iterations=max_iterations, tolerance=tolerance, adjoint=adjoint)
    centers = _real_array(centers, coefficients.dtype)
    biases = _real_array(biases, coefficients.dtype)
    viscosity = _real_array(viscosity, coefficients.dtype)
    time = _real_array(time, coefficients.dtype)
    relative = centers - centers[0]
    displacement = base.value - centers[0]
    rates = relative / (2 * viscosity * time)
    shifted = biases - biases[0] - jnp.sum(relative * relative, axis=-1) / (4 * viscosity * time)
    logits = jnp.einsum("...d,md->...m", displacement, rates) + shifted
    probabilities = jax.nn.softmax(logits, axis=-1)
    mean = jnp.einsum("...m,md->...d", probabilities, relative)
    anchor = (displacement - mean) / time
    valid = (base.converged & jnp.all(jnp.isfinite(logits), axis=-1)
             & jnp.all(jnp.isfinite(anchor), axis=-1))
    anchor = jnp.where(valid[..., None], anchor, jnp.nan)
    base_residual = anchor - coefficients[0]
    order = coefficients.shape[0] - 1
    if not order:
        return GaussianHeatInverseJet(base.value[None, ...], base, anchor, base_residual)
    factor, rows = _factor(probabilities, relative, mean, viscosity, time, valid, adjoint)
    valid = (valid & jnp.all(jnp.isfinite(factor), axis=(-2, -1))
             & jnp.all(jnp.isfinite(rates)))
    exponentials = [jnp.ones_like(probabilities)]
    log_coefficients = [jnp.zeros_like(probabilities)]
    denominators = [jnp.ones_like(probabilities[..., 0])]
    means = [mean]
    output = [base.value]
    for n in range(1, order + 1):
        exponential_hat = jnp.zeros_like(probabilities)
        for k in range(1, n):
            exponential_hat = exponential_hat + (k/n) * log_coefficients[k] * exponentials[n-k]
        denominator_hat = jnp.sum(probabilities * exponential_hat, axis=-1)
        moment_hat = jnp.einsum("...m,md->...d", probabilities * exponential_hat, relative)
        mean_hat = moment_hat - denominator_hat[..., None] * mean
        for k in range(1, n):
            mean_hat = mean_hat - denominators[k][..., None] * means[n-k]
        physical = _solve(factor, rows, time * coefficients[n] + mean_hat)
        valid = (valid & jnp.all(jnp.isfinite(coefficients[n]), axis=-1)
                 & jnp.all(jnp.isfinite(physical), axis=-1))
        log_coefficient = jnp.einsum("...d,md->...m", physical, rates)
        exponential = exponential_hat + log_coefficient
        denominator = jnp.sum(probabilities * exponential, axis=-1)
        moment = jnp.einsum("...m,md->...d", probabilities * exponential, relative)
        mean_coefficient = moment - denominator[..., None] * mean
        for k in range(1, n):
            mean_coefficient = mean_coefficient - denominators[k][..., None] * means[n-k]
        valid = (valid & jnp.all(jnp.isfinite(exponential), axis=-1) & jnp.isfinite(denominator)
                 & jnp.all(jnp.isfinite(mean_coefficient), axis=-1))
        physical = jnp.where(valid[..., None], physical, jnp.nan)
        log_coefficients.append(log_coefficient)
        exponentials.append(exponential)
        denominators.append(denominator)
        means.append(mean_coefficient)
        output.append(physical)
    return GaussianHeatInverseJet(jnp.stack(output), base, anchor, base_residual)


__all__ = ["GaussianHeatInverseJet", "gaussian_heat_inverse_jet"]
