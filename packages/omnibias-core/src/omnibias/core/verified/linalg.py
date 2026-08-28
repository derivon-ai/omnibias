# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Verified dense linear algebra: a certified bound on ``||A^{-1}||``.

The radii-polynomial / Newton-Kantorovich machinery needs a *rigorous* upper
bound on the norm of the inverse linearised operator.  The classical tool is the
**Neumann lemma**: if ``B`` approximates ``A^{-1}`` well enough that

.. math::

    \kappa := \| I - B A \| < 1,

then ``A`` is invertible and

.. math::

    \| A^{-1} \| \;\le\; \frac{\|B\|}{1 - \kappa}.

The approximate inverse ``B`` is produced by an ordinary floating-point solver
(e.g. ``numpy.linalg.inv``) by the *caller*; this module only needs ``A`` and
``B`` as nested float lists, so :mod:`omnibias.core` keeps its zero-dependency
contract.  The residual ``I - B A`` and both operator norms are evaluated in
outward-rounded interval arithmetic, so the returned bound is theorem-grade.

Norms are the matrix infinity norm (max absolute row sum) and the matching
vector sup norm.
"""

from __future__ import annotations

from collections.abc import Sequence
from math import fsum, isfinite

from omnibias.core.verified.interval import Interval, IntervalLike, sum_intervals

FloatMatrix = Sequence[Sequence[float]]
IntervalMatrix = list[list[Interval]]
IntervalVector = list[Interval]


def to_interval_matrix(a: Sequence[Sequence[IntervalLike]]) -> IntervalMatrix:
    return [[Interval.from_value(x) for x in row] for row in a]


def identity_matrix(n: int) -> IntervalMatrix:
    return [[Interval.point(1.0 if i == j else 0.0) for j in range(n)] for i in range(n)]


def matmul(a: IntervalMatrix, b: IntervalMatrix) -> IntervalMatrix:
    n, k, m = len(a), len(b), len(b[0]) if b else 0
    if a and len(a[0]) != k:
        raise ValueError("inner dimensions do not match")
    out: IntervalMatrix = []
    for i in range(n):
        row: list[Interval] = []
        for j in range(m):
            acc = Interval.point(0.0)
            for p in range(k):
                acc = acc + a[i][p] * b[p][j]
            row.append(acc)
        out.append(row)
    return out


def mat_sub(a: IntervalMatrix, b: IntervalMatrix) -> IntervalMatrix:
    return [[a[i][j] - b[i][j] for j in range(len(a[i]))] for i in range(len(a))]


def matvec(a: IntervalMatrix, x: Sequence[Interval]) -> list[Interval]:
    out: list[Interval] = []
    for row in a:
        acc = Interval.point(0.0)
        for aij, xj in zip(row, x, strict=True):
            acc = acc + aij * xj
        out.append(acc)
    return out


def interval_triangular_solve(
    matrix: Sequence[Sequence[IntervalLike]],
    rhs: Sequence[IntervalLike],
    *,
    lower: bool,
    unit_diagonal: bool = False,
) -> IntervalVector:
    r"""Solve an interval triangular system by forward or backward substitution.

    The result encloses the solution of every point system represented by
    ``matrix`` and ``rhs``.  Every diagonal interval must exclude zero unless
    ``unit_diagonal=True``.  Only the declared triangular half of ``matrix`` is
    used, so callers can supply a full matrix without copying it.
    """
    a = to_interval_matrix(matrix)
    n = len(a)
    if n == 0 or any(len(row) != n for row in a):
        raise ValueError("triangular solve requires a non-empty square matrix")
    b = [Interval.from_value(value) for value in rhs]
    if len(b) != n:
        raise ValueError("right-hand side length must match matrix dimension")

    solution = [Interval.point(0.0) for _ in range(n)]
    indices = range(n) if lower else range(n - 1, -1, -1)
    for i in indices:
        numerator = b[i]
        known = range(i) if lower else range(i + 1, n)
        for j in known:
            numerator = numerator - a[i][j] * solution[j]
        solution[i] = numerator if unit_diagonal else numerator / a[i][i]
    return solution


def _float_inverse(matrix: FloatMatrix) -> list[list[float]] | None:
    """Gauss--Jordan floating inverse used only as a Krawczyk preconditioner."""
    n = len(matrix)
    if n == 0 or any(len(row) != n for row in matrix):
        return None
    augmented = [
        [float(matrix[i][j]) for j in range(n)]
        + [1.0 if i == j else 0.0 for j in range(n)]
        for i in range(n)
    ]
    for column in range(n):
        pivot = max(range(column, n), key=lambda row: abs(augmented[row][column]))
        if abs(augmented[pivot][column]) < 1e-300:
            return None
        augmented[column], augmented[pivot] = augmented[pivot], augmented[column]
        pivot_value = augmented[column][column]
        augmented[column] = [value / pivot_value for value in augmented[column]]
        for row in range(n):
            if row == column or augmented[row][column] == 0.0:
                continue
            factor = augmented[row][column]
            augmented[row] = [
                value - factor * pivot_entry
                for value, pivot_entry in zip(
                    augmented[row], augmented[column], strict=True
                )
            ]
    return [[augmented[i][n + j] for j in range(n)] for i in range(n)]


def _square_interval_matrix(
    matrix: Sequence[Sequence[IntervalLike]],
) -> IntervalMatrix:
    a = to_interval_matrix(matrix)
    n = len(a)
    if n == 0 or any(len(row) != n for row in a):
        raise ValueError("interval solve requires a non-empty square matrix")
    return a


def _interval_vector_sub(
    left: Sequence[Interval], right: Sequence[Interval]
) -> IntervalVector:
    if len(left) != len(right):
        raise ValueError("interval vector dimensions do not match")
    return [x - y for x, y in zip(left, right, strict=True)]


def _symmetric_ldlt_solve(
    matrix: IntervalMatrix, rhs: IntervalVector
) -> IntervalVector | None:
    """Solve a symmetric system through a certified interval ``LDL^T`` factor."""
    # This import is intentionally lazy: eig_operator imports linalg helpers.
    from omnibias.core.verified.eig_operator import interval_ldlt_factor

    symmetric_matrix = [row[:] for row in matrix]
    for i in range(len(matrix)):
        for j in range(i):
            try:
                entry = matrix[i][j].intersect(matrix[j][i])
            except ValueError as exc:
                raise ValueError(
                    "symmetric=True requires compatible symmetric interval endpoints"
                ) from exc
            symmetric_matrix[i][j] = entry
            symmetric_matrix[j][i] = entry
    factor = interval_ldlt_factor(symmetric_matrix)
    if factor is None:
        return None
    lower, diagonal = factor
    lower_rows = [list(row) for row in lower]
    forward = interval_triangular_solve(
        lower_rows, rhs, lower=True, unit_diagonal=True
    )
    diagonal_solution = [
        forward[i] / diagonal[i] for i in range(len(forward))
    ]
    transpose = [
        [lower_rows[j][i] for j in range(len(lower_rows))]
        for i in range(len(lower_rows))
    ]
    return interval_triangular_solve(
        transpose, diagonal_solution, lower=False, unit_diagonal=True
    )


def interval_solve(
    matrix: Sequence[Sequence[IntervalLike]],
    rhs: Sequence[IntervalLike],
    *,
    approximate_inverse: FloatMatrix | None = None,
    max_iter: int = 8,
    symmetric: bool = False,
) -> IntervalVector:
    r"""Enclose solutions of the interval linear system ``matrix * x = rhs``.

    For a general matrix box, this uses a Krawczyk fixed-point enclosure with a
    floating midpoint inverse as preconditioner.  A contractive comparison
    iteration then sharpens its initial infinity-norm box componentwise.  It
    raises :class:`ValueError` rather than returning a non-rigorous answer if
    the required ``||I - B A||_inf < 1`` test fails.

    Set ``symmetric=True`` to use the interval ``LDL^T`` path instead.  That
    route needs sign-definite pivots and solves the factorized interval system
    by triangular substitution, avoiding a floating preconditioner.
    """
    if max_iter < 0:
        raise ValueError("max_iter must be non-negative")
    a = _square_interval_matrix(matrix)
    n = len(a)
    f = [Interval.from_value(value) for value in rhs]
    if len(f) != n:
        raise ValueError("right-hand side length must match matrix dimension")

    if symmetric:
        solved = _symmetric_ldlt_solve(a, f)
        if solved is None:
            raise ValueError("symmetric interval LDLT factorization is inconclusive")
        return solved

    midpoint = [[entry.mid for entry in row] for row in a]
    if not all(isfinite(value) for row in midpoint for value in row):
        raise ValueError("interval solve requires finite matrix endpoints")
    if approximate_inverse is None:
        preconditioner = _float_inverse(midpoint)
        if preconditioner is None:
            raise ValueError("midpoint matrix is singular")
    else:
        preconditioner = [
            [float(value) for value in row] for row in approximate_inverse
        ]
        if len(preconditioner) != n or any(len(row) != n for row in preconditioner):
            raise ValueError("approximate inverse shape must match matrix")
        if not all(isfinite(value) for row in preconditioner for value in row):
            raise ValueError("approximate inverse entries must be finite")

    f_mid = [entry.mid for entry in f]
    center = [
        fsum(preconditioner[i][j] * f_mid[j] for j in range(n))
        for i in range(n)
    ]
    if not all(isfinite(value) for value in center):
        raise ValueError("interval solve produced a non-finite midpoint center")
    b = to_interval_matrix(preconditioner)
    center_iv = [Interval.point(value) for value in center]
    residual = _interval_vector_sub(f, matvec(a, center_iv))
    correction = matvec(b, residual)
    contraction = mat_sub(identity_matrix(n), matmul(b, a))
    kappa = inf_norm_matrix(contraction)
    if not kappa < 1.0:
        raise ValueError(
            "Krawczyk interval solve is inconclusive: ||I - B A||_inf must be < 1"
        )

    one_minus_kappa = Interval.point(1.0) - Interval.point(kappa)
    global_radius = (
        Interval.point(inf_norm_vector(correction)) * one_minus_kappa.reciprocal()
    ).hi
    radii = [global_radius for _ in range(n)]
    for _ in range(max_iter):
        next_radii: list[float] = []
        for i in range(n):
            bound = correction[i].abs()
            for j in range(n):
                bound = bound + contraction[i][j].abs() * Interval.point(radii[j])
            next_radii.append(min(radii[i], bound.hi))
        radii = next_radii

    enclosure = [
        Interval.point(center[i]) + Interval(-radii[i], radii[i]) for i in range(n)
    ]
    for _ in range(max_iter):
        delta = _interval_vector_sub(enclosure, center_iv)
        image_delta = matvec(contraction, delta)
        image = [
            center_iv[i] + correction[i] + image_delta[i] for i in range(n)
        ]
        try:
            next_enclosure = [
                enclosure[i].intersect(image[i]) for i in range(n)
            ]
        except ValueError:
            break
        if next_enclosure == enclosure:
            break
        enclosure = next_enclosure
    return enclosure


def inf_norm_matrix(a: IntervalMatrix) -> float:
    """Rigorous upper bound on the infinity norm of any matrix in ``a``."""
    best = 0.0
    for row in a:
        row_sum = sum_intervals([x.abs() for x in row])
        best = max(best, row_sum.hi)
    return best


def inf_norm_vector(x: Sequence[Interval]) -> float:
    """Rigorous upper bound on ``max_i |x_i|``."""
    return max((xi.abs().hi for xi in x), default=0.0)


def neumann_inverse_norm_bound(a: FloatMatrix, b: FloatMatrix) -> dict[str, float | bool]:
    """Certify ``||A^{-1}||_inf <= ||B||/(1 - ||I - B A||)`` if ``kappa < 1``.

    Parameters
    ----------
    a
        The (float) matrix whose inverse norm is sought.
    b
        A floating-point approximate inverse of ``a`` (e.g. ``numpy.linalg.inv``).

    Returns
    -------
    dict[str, float | bool]
        ``kappa`` (a rigorous upper bound on ``||I - BA||``), ``norm_b``,
        ``inverse_norm_bound``, and a ``certified`` flag.
    """
    n = len(a)
    a_iv = to_interval_matrix(a)
    b_iv = to_interval_matrix(b)
    residual = mat_sub(identity_matrix(n), matmul(b_iv, a_iv))
    kappa = inf_norm_matrix(residual)
    norm_b = inf_norm_matrix(b_iv)
    certified = kappa < 1.0
    if certified:
        one_minus = Interval(1.0, 1.0) - Interval.point(kappa)  # lower-bounds 1 - kappa
        bound = (Interval.point(norm_b) * one_minus.reciprocal()).hi
    else:
        bound = float("inf")
    return {
        "kappa": kappa,
        "norm_b": norm_b,
        "inverse_norm_bound": bound,
        "certified": certified,
    }


__all__ = [
    "FloatMatrix",
    "IntervalMatrix",
    "IntervalVector",
    "identity_matrix",
    "inf_norm_matrix",
    "inf_norm_vector",
    "interval_solve",
    "interval_triangular_solve",
    "mat_sub",
    "matmul",
    "matvec",
    "neumann_inverse_norm_bound",
    "to_interval_matrix",
]
