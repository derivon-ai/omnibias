# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Differentiable Euclidean geometry of shared affine equality constraints."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import NamedTuple

import torch
from torch import Tensor


class AffineProjection(NamedTuple):
    """Projection and numerical diagnostics; these are not exact certificates."""

    projected: Tensor
    correction: Tensor
    half_squared_distance: Tensor
    valid: Tensor
    rcond: Tensor


class LinearizedEqualityStep(NamedTuple):
    """Minimum-norm correction of linearized equations, not nonlinear distance."""

    value: Tensor
    correction: Tensor
    local_half_squared_distance: Tensor
    valid: Tensor
    rcond: Tensor


def _check_query(x: Tensor) -> None:
    if not isinstance(x, Tensor) or x.dtype not in (torch.float32, torch.float64):
        raise TypeError("x must be a float32 or float64 Tensor")
    if x.ndim < 1 or x.shape[-1] < 1:
        raise ValueError("x must have shape (..., positive dimension)")


def _rank_cutoff(rank_rcond: float | None, rows: int, dimension: int, x: Tensor) -> float:
    if rank_rcond is None:
        return max(rows, dimension) * torch.finfo(x.dtype).eps
    if type(rank_rcond) not in (int, float) or not math.isfinite(rank_rcond) or not 0 < rank_rcond < 1:
        raise ValueError("rank_rcond must be a finite scalar strictly between zero and one")
    return float(rank_rcond)


def _project(x: Tensor, matrix: Tensor, rhs: Tensor, cutoff: float) -> AffineProjection:
    """Internal projection, also allowing a separate rhs for each query."""
    rows, dimension = matrix.shape
    query_valid = torch.isfinite(x).all(-1)
    query = torch.where(torch.isfinite(x), x, torch.zeros_like(x))
    rhs_valid = torch.isfinite(rhs).all(-1)
    offsets = torch.where(torch.isfinite(rhs), rhs, torch.zeros_like(rhs))
    if rows == 0:
        zero = matrix.sum() * 0 + offsets.sum(-1) * 0
        projected = query + zero[..., None]
        correction = query * 0 + zero[..., None]
        valid = query_valid & rhs_valid
        distance = correction.square().sum(-1) / 2
        return AffineProjection(
            torch.where(valid[..., None], projected, float("nan")),
            torch.where(valid[..., None], correction, float("nan")),
            torch.where(valid, distance, float("nan")), valid.detach(), x.new_ones(()),
        )

    matrix_valid = torch.isfinite(matrix).all()
    safe_matrix = torch.where(torch.isfinite(matrix), matrix, torch.zeros_like(matrix))
    scales = safe_matrix.abs().amax(-1)
    safe_scales = torch.where(scales > 0, scales, torch.ones_like(scales))
    normalized = safe_matrix / safe_scales[:, None]
    normalized_rhs = offsets / safe_scales
    matrix_valid = matrix_valid & (scales > 0).all()
    rhs_valid = rhs_valid & torch.isfinite(normalized_rhs).all(-1)
    normalized_rhs = torch.where(torch.isfinite(normalized_rhs), normalized_rhs, torch.zeros_like(normalized_rhs))
    with torch.no_grad():
        singular = torch.linalg.svdvals(normalized)
        reciprocal = singular[-1] / torch.where(singular[0] > 0, singular[0], torch.ones_like(singular[0]))
        rcond = torch.where(matrix_valid, reciprocal, torch.full_like(reciprocal, float("nan")))
        rank_valid = matrix_valid & torch.isfinite(reciprocal) & (reciprocal > cutoff)
    identity_rows = torch.eye(rows, dimension, dtype=x.dtype, device=x.device)
    # Invalid matrices never enter a singular QR/solve, including in backward.
    normalized = torch.where(rank_valid, normalized, identity_rows)
    normalized_rhs = torch.where(rank_valid, normalized_rhs, torch.zeros_like(normalized_rhs))
    normal_basis, triangular = torch.linalg.qr(normalized.mT, mode="reduced")
    eta = torch.linalg.solve_triangular(triangular.mT, normalized_rhs[..., None], upper=False)[..., 0]
    if rows == dimension:
        projected = eta @ normal_basis.mT + query * 0
        correction = projected - query
    else:
        correction = (eta - query @ normal_basis) @ normal_basis.mT
        projected = query + correction
    distance = correction.square().sum(-1) / 2
    valid = (rank_valid & query_valid & rhs_valid & torch.isfinite(projected).all(-1)
             & torch.isfinite(correction).all(-1) & torch.isfinite(distance))
    return AffineProjection(
        torch.where(valid[..., None], projected, float("nan")),
        torch.where(valid[..., None], correction, float("nan")),
        torch.where(valid, distance, float("nan")), valid.detach(), rcond,
    )


def affine_project(
    x: Tensor, A: Tensor | Sequence[Sequence[float]], b: Tensor | Sequence[float], *,
    rank_rcond: float | None = None,
) -> AffineProjection:
    r"""Project ``x (...,d)`` onto ``A @ y = b`` with shared ``A (m,d)``, ``b (m,)``.

    The correction is ``projected - x``. Reduced QR of the row-scaled
    ``A.T`` and triangular solves avoid a Gram inverse. All three geometric
    outputs retain gradients through x, A and b, including double backward.
    Empty constraints give the identity; square constraints give a point
    independent of x. Only float32/64 and ``0 <= m <= d`` are supported.

    Numerical full row rank requires the singular-value ratio of A after
    dividing each row by its largest absolute entry to exceed ``rank_rcond``
    (default ``max(m,d) * dtype epsilon``). ``rcond`` reports that ratio, not
    a condition number of unscaled A. Diagnostics are stopped-gradient.
    Rank failure and nonfinite inputs produce false ``valid`` and NaN
    geometric outputs; this is a visible numerical refusal, not a proof of
    infeasibility. Validity does not certify exact membership in floating point.
    """
    _check_query(x)
    matrix = torch.as_tensor(A, dtype=x.dtype, device=x.device)
    rhs = torch.as_tensor(b, dtype=x.dtype, device=x.device)
    dimension = x.shape[-1]
    if matrix.ndim != 2 or matrix.shape[1] != dimension or matrix.shape[0] > dimension:
        raise ValueError("A must have shape (m,d) with 0 <= m <= d")
    if rhs.shape != (matrix.shape[0],):
        raise ValueError("b must have shape (m,)")
    return _project(x, matrix, rhs, _rank_cutoff(rank_rcond, matrix.shape[0], dimension, x))


def linearized_equality_step(
    x: Tensor, residual: Tensor, jacobian: Tensor, *, rank_rcond: float | None = None,
) -> LinearizedEqualityStep:
    r"""Return ``x + delta`` with minimum-norm ``J delta = -residual``.

    J has shared shape ``(m,d)`` and residual has shape ``(...,m)`` matching
    x's batch. This solves the linearized equations at x. For nonlinear
    residuals it need not reach a root or be the nearest feasible point;
    ``local_half_squared_distance`` is half the squared linearized correction.
    Callers may vmap over changing Jacobians. Refusal and rank semantics match
    :func:`affine_project` and gradients include the supplied residual and J.
    """
    _check_query(x)
    matrix = torch.as_tensor(jacobian, dtype=x.dtype, device=x.device)
    values = torch.as_tensor(residual, dtype=x.dtype, device=x.device)
    dimension = x.shape[-1]
    if matrix.ndim != 2 or matrix.shape[1] != dimension or matrix.shape[0] > dimension:
        raise ValueError("jacobian must have shape (m,d) with 0 <= m <= d")
    if values.shape != x.shape[:-1] + (matrix.shape[0],):
        raise ValueError("residual must have shape x.shape[:-1] + (m,)")
    result = _project(torch.zeros_like(x), matrix, -values,
                      _rank_cutoff(rank_rcond, matrix.shape[0], dimension, x))
    valid = result.valid & torch.isfinite(x).all(-1)
    query = torch.where(torch.isfinite(x), x, torch.zeros_like(x))
    value = query + result.correction
    valid = valid & torch.isfinite(value).all(-1)
    return LinearizedEqualityStep(
        torch.where(valid[..., None], value, float("nan")),
        torch.where(valid[..., None], result.correction, float("nan")),
        torch.where(valid, result.half_squared_distance, float("nan")), valid.detach(), result.rcond,
    )
