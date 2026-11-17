# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Differentiable Euclidean geometry of shared affine equality constraints."""

from __future__ import annotations

import math
from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax import Array
from jax.scipy.linalg import solve_triangular


class AffineProjection(NamedTuple):
    """Projection and numerical diagnostics; these are not exact certificates."""

    projected: Array
    correction: Array
    half_squared_distance: Array
    valid: Array
    rcond: Array


class LinearizedEqualityStep(NamedTuple):
    """Minimum-norm correction of linearized equations, not nonlinear distance."""

    value: Array
    correction: Array
    local_half_squared_distance: Array
    valid: Array
    rcond: Array


def _check_query(x: Array) -> None:
    if not hasattr(x, "dtype") or x.dtype not in (jnp.float32, jnp.float64):
        raise TypeError("x must be a float32 or float64 JAX array")
    if x.ndim < 1 or x.shape[-1] < 1:
        raise ValueError("x must have shape (..., positive dimension)")


def _rank_cutoff(rank_rcond: float | None, rows: int, dimension: int, x: Array) -> float:
    if rank_rcond is None:
        epsilon = 2.0 ** (-23 if x.dtype == jnp.float32 else -52)
        return max(rows, dimension) * epsilon
    if type(rank_rcond) not in (int, float) or not math.isfinite(rank_rcond) or not 0 < rank_rcond < 1:
        raise ValueError("rank_rcond must be a finite scalar strictly between zero and one")
    return float(rank_rcond)


def _project(x: Array, matrix: Array, rhs: Array, cutoff: float) -> AffineProjection:
    """Internal projection, also allowing a separate rhs for each query."""
    rows, dimension = matrix.shape
    query_valid = jnp.isfinite(x).all(-1)
    query = jnp.where(jnp.isfinite(x), x, jnp.zeros_like(x))
    rhs_valid = jnp.isfinite(rhs).all(-1)
    offsets = jnp.where(jnp.isfinite(rhs), rhs, jnp.zeros_like(rhs))
    if rows == 0:
        zero = matrix.sum() * 0 + offsets.sum(-1) * 0
        projected = query + zero[..., None]
        correction = query * 0 + zero[..., None]
        valid = query_valid & rhs_valid
        distance = jnp.square(correction).sum(-1) / 2
        return AffineProjection(
            jnp.where(valid[..., None], projected, jnp.nan),
            jnp.where(valid[..., None], correction, jnp.nan),
            jnp.where(valid, distance, jnp.nan), jax.lax.stop_gradient(valid), jnp.ones((), dtype=x.dtype),
        )

    matrix_valid = jnp.isfinite(matrix).all()
    safe_matrix = jnp.where(jnp.isfinite(matrix), matrix, jnp.zeros_like(matrix))
    scales = jnp.abs(safe_matrix).max(-1)
    safe_scales = jnp.where(scales > 0, scales, jnp.ones_like(scales))
    normalized = safe_matrix / safe_scales[:, None]
    normalized_rhs = offsets / safe_scales
    matrix_valid = matrix_valid & (scales > 0).all()
    rhs_valid = rhs_valid & jnp.isfinite(normalized_rhs).all(-1)
    normalized_rhs = jnp.where(jnp.isfinite(normalized_rhs), normalized_rhs, jnp.zeros_like(normalized_rhs))
    singular = jnp.linalg.svd(jax.lax.stop_gradient(normalized), compute_uv=False)
    reciprocal = singular[-1] / jnp.where(singular[0] > 0, singular[0], jnp.ones_like(singular[0]))
    rcond = jax.lax.stop_gradient(jnp.where(matrix_valid, reciprocal, jnp.nan))
    rank_valid = jax.lax.stop_gradient(matrix_valid & jnp.isfinite(reciprocal) & (reciprocal > cutoff))
    identity_rows = jnp.eye(rows, dimension, dtype=x.dtype)
    normalized = jnp.where(rank_valid, normalized, identity_rows)
    normalized_rhs = jnp.where(rank_valid, normalized_rhs, jnp.zeros_like(normalized_rhs))
    normal_basis, triangular = jnp.linalg.qr(normalized.T, mode="reduced")
    eta = solve_triangular(triangular.T, normalized_rhs[..., None], lower=True)[..., 0]
    if rows == dimension:
        projected = eta @ normal_basis.T + query * 0
        correction = projected - query
    else:
        correction = (eta - query @ normal_basis) @ normal_basis.T
        projected = query + correction
    distance = jnp.square(correction).sum(-1) / 2
    valid = (rank_valid & query_valid & rhs_valid & jnp.isfinite(projected).all(-1)
             & jnp.isfinite(correction).all(-1) & jnp.isfinite(distance))
    return AffineProjection(
        jnp.where(valid[..., None], projected, jnp.nan),
        jnp.where(valid[..., None], correction, jnp.nan),
        jnp.where(valid, distance, jnp.nan), jax.lax.stop_gradient(valid), rcond,
    )


def affine_project(x: Array, A: Array, b: Array, *, rank_rcond: float | None = None) -> AffineProjection:
    r"""Project ``x (...,d)`` onto ``A @ y = b`` with shared ``A (m,d)``, ``b (m,)``.

    Correction means ``projected - x``. Reduced QR of row-scaled ``A.T``
    avoids a Gram inverse. Values retain gradients through x, A and b,
    including second derivatives. Empty constraints give the identity;
    square constraints give a point independent of x. Supports jit and vmap.

    Require float32/64, ``0 <= m <= d`` and numerical full row rank. Rows
    are divided by their largest absolute entry; the resulting singular-value
    ratio is returned as stopped-gradient ``rcond``. Its rank cutoff defaults
    to ``max(m,d) * dtype epsilon``. Nonfinite or rank-deficient constraints
    and nonfinite queries produce false ``valid`` and NaN geometric outputs.
    This numerical refusal is not a proof of infeasibility; a successful
    floating-point projection is not an exact membership certificate.
    """
    _check_query(x)
    matrix, rhs = jnp.asarray(A, dtype=x.dtype), jnp.asarray(b, dtype=x.dtype)
    dimension = x.shape[-1]
    if matrix.ndim != 2 or matrix.shape[1] != dimension or matrix.shape[0] > dimension:
        raise ValueError("A must have shape (m,d) with 0 <= m <= d")
    if rhs.shape != (matrix.shape[0],):
        raise ValueError("b must have shape (m,)")
    return _project(x, matrix, rhs, _rank_cutoff(rank_rcond, matrix.shape[0], dimension, x))


def linearized_equality_step(
    x: Array, residual: Array, jacobian: Array, *, rank_rcond: float | None = None,
) -> LinearizedEqualityStep:
    r"""Return ``x + delta`` with minimum-norm ``J delta = -residual``.

    Shared J has shape ``(m,d)``; residual shape is ``x.shape[:-1] + (m,)``.
    This solves only the linearized equations, without promising a nonlinear
    root or nearest point. The local quadratic measure is half ``||delta||²``.
    Gradients include residual and J. Vmap over changing Jacobians when needed.
    Rank and refusal semantics match :func:`affine_project`.
    """
    _check_query(x)
    matrix, values = jnp.asarray(jacobian, dtype=x.dtype), jnp.asarray(residual, dtype=x.dtype)
    dimension = x.shape[-1]
    if matrix.ndim != 2 or matrix.shape[1] != dimension or matrix.shape[0] > dimension:
        raise ValueError("jacobian must have shape (m,d) with 0 <= m <= d")
    if values.shape != x.shape[:-1] + (matrix.shape[0],):
        raise ValueError("residual must have shape x.shape[:-1] + (m,)")
    result = _project(jnp.zeros_like(x), matrix, -values,
                      _rank_cutoff(rank_rcond, matrix.shape[0], dimension, x))
    valid = result.valid & jnp.isfinite(x).all(-1)
    query = jnp.where(jnp.isfinite(x), x, jnp.zeros_like(x))
    value = query + result.correction
    valid = valid & jnp.isfinite(value).all(-1)
    return LinearizedEqualityStep(
        jnp.where(valid[..., None], value, jnp.nan),
        jnp.where(valid[..., None], result.correction, jnp.nan),
        jnp.where(valid, result.half_squared_distance, jnp.nan), jax.lax.stop_gradient(valid), result.rcond,
    )
