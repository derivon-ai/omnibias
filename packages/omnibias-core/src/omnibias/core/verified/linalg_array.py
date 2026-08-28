# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Vectorized outward-rounded symmetric ``LDL^T``: inertia and definiteness.

:mod:`omnibias.core.verified.eig_operator` owns the scalar reference path
(:func:`~omnibias.core.verified.eig_operator.interval_ldlt_inertia` and friends)
built from :class:`~omnibias.core.verified.interval.Interval` objects.  This
module is its :class:`~omnibias.core.verified.interval_array.IntervalArray`
twin: the same left-looking factorization, driven one column at a time with
NumPy endpoint arrays instead of one Python object per matrix entry.

Soundness is inherited rather than re-argued.  Every elementary step reuses the
*same* outward-rounded primitives as the scalar path -- one
:func:`numpy.nextafter` step per computed endpoint, ``min``/``max`` over the four
corner products for a multiplication -- and the pivot loop applies them in the
same order, so each returned interval encloses the corresponding exact factor of
every symmetric point matrix in the input box.  With all pivots sign-definite,
``S = L D L^T`` is a congruence and Sylvester's law of inertia fixes the inertia
of the whole box.  A pivot interval that straddles ``0`` leaves the sign -- hence
the inertia -- uncertified, and the factorization returns ``None`` rather than a
guess.

Only the lower triangle and the diagonal of the input are read, matching the
scalar path: the statement is about symmetric point matrices whose lower
triangle lies in the supplied box.

Every result here is a **fixed finite matrix** statement.  It carries no
continuum, operator-limit, or spectral-asymptotics claim.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import numpy as np
from omnibias.core.verified.eig_operator import Inertia

# The scalar and vectorized paths must round identically, so they share one
# implementation of the outward-rounding primitives.
from omnibias.core.verified.interval_array import (
    FloatArray,
    IntervalArray,
    IntervalArrayLike,
    _mul_bounds,
    _pred,
    _succ,
)


@dataclass(frozen=True)
class IntervalLDLT:
    """A certified interval ``LDL^T`` factorization of a symmetric matrix box.

    ``lower`` is the full ``(n, n)`` unit-lower factor (exact ones on the
    diagonal, exact zeros above it) and ``diagonal`` is the length-``n`` pivot
    vector.  Each entry encloses the corresponding exact factor of every
    symmetric point matrix in the factored box.
    """

    lower: IntervalArray
    diagonal: IntervalArray

    @property
    def size(self) -> int:
        """The matrix dimension ``n``."""
        return self.diagonal.shape[0]


def _square_endpoints(matrix: IntervalArrayLike) -> tuple[FloatArray, FloatArray]:
    """Validate a non-empty square interval matrix and return its endpoints."""
    values = IntervalArray.from_value(matrix)
    if values.ndim != 2:
        raise ValueError("interval LDLT requires a two-dimensional matrix")
    rows, columns = values.shape
    if rows == 0 or rows != columns:
        raise ValueError("interval LDLT requires a non-empty square matrix")
    return values.lo, values.hi


def _entry(values: FloatArray, row: int, column: int) -> FloatArray:
    """A length-one view of one matrix entry, keeping every operand array-typed."""
    return cast(FloatArray, values[row, column : column + 1])


def _item(values: FloatArray, index: int) -> FloatArray:
    """A length-one view of one vector entry."""
    return cast(FloatArray, values[index : index + 1])


def _ldlt_endpoints(
    lo: FloatArray, hi: FloatArray
) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray] | None:
    r"""Endpoint arrays of the interval ``LDL^T`` factors, or ``None``.

    Mirrors ``eig_operator._ldlt_pivots`` column by column: pivot ``j`` is
    ``S[j,j] - sum_{k<j} L[j,k] L[j,k] D[k]`` and the subdiagonal column is
    ``(S[j+1:,j] - sum_{k<j} L[j+1:,k] L[j,k] D[k]) / D[j]``.  The ``k`` sums run
    in the same order as the scalar path, so the accumulated rounding matches.
    """
    n = lo.shape[0]
    lower_lo = np.zeros((n, n), dtype=np.float64)
    lower_hi = np.zeros((n, n), dtype=np.float64)
    pivot_lo = np.zeros(n, dtype=np.float64)
    pivot_hi = np.zeros(n, dtype=np.float64)

    for j in range(n):
        lower_lo[j, j] = 1.0
        lower_hi[j, j] = 1.0

        diagonal_lo = _entry(lo, j, j)
        diagonal_hi = _entry(hi, j, j)
        for k in range(j):
            row_lo = _entry(lower_lo, j, k)
            row_hi = _entry(lower_hi, j, k)
            square_lo, square_hi = _mul_bounds(row_lo, row_hi, row_lo, row_hi)
            term_lo, term_hi = _mul_bounds(
                square_lo, square_hi, _item(pivot_lo, k), _item(pivot_hi, k)
            )
            diagonal_lo, diagonal_hi = (
                _pred(cast(FloatArray, diagonal_lo - term_hi)),
                _succ(cast(FloatArray, diagonal_hi - term_lo)),
            )

        # A NaN endpoint fails both comparisons, so it is reported as
        # uncertified rather than mistaken for a sign-definite pivot.
        if not (float(diagonal_hi[0]) < 0.0 or float(diagonal_lo[0]) > 0.0):
            return None
        pivot_lo[j] = diagonal_lo[0]
        pivot_hi[j] = diagonal_hi[0]

        inverse_lo = _pred(cast(FloatArray, 1.0 / diagonal_hi))
        inverse_hi = _succ(cast(FloatArray, 1.0 / diagonal_lo))

        below = slice(j + 1, n)
        column_lo = cast(FloatArray, np.array(lo[below, j], dtype=np.float64, copy=True))
        column_hi = cast(FloatArray, np.array(hi[below, j], dtype=np.float64, copy=True))
        for k in range(j):
            below_lo = cast(FloatArray, lower_lo[below, k])
            below_hi = cast(FloatArray, lower_hi[below, k])
            pair_lo, pair_hi = _mul_bounds(
                below_lo, below_hi, _entry(lower_lo, j, k), _entry(lower_hi, j, k)
            )
            term_lo, term_hi = _mul_bounds(
                pair_lo, pair_hi, _item(pivot_lo, k), _item(pivot_hi, k)
            )
            column_lo, column_hi = (
                _pred(cast(FloatArray, column_lo - term_hi)),
                _succ(cast(FloatArray, column_hi - term_lo)),
            )
        scaled_lo, scaled_hi = _mul_bounds(column_lo, column_hi, inverse_lo, inverse_hi)
        lower_lo[below, j] = scaled_lo
        lower_hi[below, j] = scaled_hi

    if bool(np.any(np.isnan(lower_lo))) or bool(np.any(np.isnan(lower_hi))):
        return None
    return lower_lo, lower_hi, pivot_lo, pivot_hi


def interval_ldlt_factor_array(matrix: IntervalArrayLike) -> IntervalLDLT | None:
    r"""Certified interval ``LDL^T`` of a symmetric matrix box (or ``None``).

    The vectorized twin of
    :func:`~omnibias.core.verified.eig_operator.interval_ldlt_factor`.  Returns
    ``None`` as soon as a pivot interval straddles ``0``, because the pivot sign
    -- and therefore the inertia of the box -- cannot be certified there.
    """
    factor = _ldlt_endpoints(*_square_endpoints(matrix))
    if factor is None:
        return None
    lower_lo, lower_hi, pivot_lo, pivot_hi = factor
    return IntervalLDLT(
        lower=IntervalArray(lower_lo, lower_hi),
        diagonal=IntervalArray(pivot_lo, pivot_hi),
    )


def interval_ldlt_pivots_array(matrix: IntervalArrayLike) -> IntervalArray | None:
    r"""Certified interval ``LDL^T`` pivots of a symmetric matrix box (or ``None``).

    The vectorized twin of
    :func:`~omnibias.core.verified.eig_operator.interval_ldlt_pivots`: the
    length-``n`` pivot vector ``D_jj`` in order.  The box is certified positive
    definite exactly when the result is not ``None`` and every pivot has
    ``lo > 0``.
    """
    factor = _ldlt_endpoints(*_square_endpoints(matrix))
    return None if factor is None else IntervalArray(factor[2], factor[3])


def interval_ldlt_inertia_array(matrix: IntervalArrayLike) -> Inertia | None:
    r"""Certified inertia of a symmetric matrix box via vectorized ``LDL^T``.

    Returns the same :class:`~omnibias.core.verified.eig_operator.Inertia` the
    scalar :func:`~omnibias.core.verified.eig_operator.interval_ldlt_inertia`
    reports, or ``None`` when a pivot straddles ``0``.  Every symmetric point
    matrix in the box shares the returned signature.
    """
    factor = _ldlt_endpoints(*_square_endpoints(matrix))
    if factor is None:
        return None
    _, _, pivot_lo, pivot_hi = factor
    midpoint = cast(FloatArray, 0.5 * (pivot_lo + pivot_hi))
    midpoint = cast(FloatArray, np.minimum(np.maximum(midpoint, pivot_lo), pivot_hi))
    negative = int(np.count_nonzero(pivot_hi < 0.0))
    return Inertia(
        negative=negative,
        positive=pivot_lo.shape[0] - negative,
        pivots=tuple(float(value) for value in midpoint),
    )


def is_positive_definite_array(matrix: IntervalArrayLike) -> bool:
    """``True`` iff a symmetric matrix box is *certified* positive definite."""
    pivots = interval_ldlt_pivots_array(matrix)
    return pivots is not None and bool(np.all(pivots.lo > 0.0))


__all__ = [
    "IntervalLDLT",
    "interval_ldlt_factor_array",
    "interval_ldlt_inertia_array",
    "interval_ldlt_pivots_array",
    "is_positive_definite_array",
]
