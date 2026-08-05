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

from omnibias.core.verified.interval import Interval, IntervalLike, sum_intervals

FloatMatrix = Sequence[Sequence[float]]
IntervalMatrix = list[list[Interval]]


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

    Returns a dict with ``kappa`` (rigorous upper bound on ``||I - BA||``),
    ``norm_b``, ``inverse_norm_bound`` and a ``certified`` flag.
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
    "identity_matrix",
    "inf_norm_matrix",
    "inf_norm_vector",
    "mat_sub",
    "matmul",
    "matvec",
    "neumann_inverse_norm_bound",
    "to_interval_matrix",
]
