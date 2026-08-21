# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Exact-``Q`` lift: residual identically zero, or ``None``.

A float residual is not a certificate. :func:`residual_identically_zero` and
:func:`integer_null_space` adjudicate over :class:`~fractions.Fraction` only.
This module does not delete the copy in ``omnibias.symbolic.dimensional``.
"""

from __future__ import annotations

from collections.abc import Sequence
from fractions import Fraction
from math import gcd

Number = int | Fraction


def as_fraction(value: Number | float, *, denom_bound: int | None = None) -> Fraction:
    """Exact ``int`` / ``Fraction``, or a bounded-denominator snap of a float."""

    if isinstance(value, Fraction):
        return value
    if isinstance(value, int):
        return Fraction(value)
    snapped = Fraction(value).limit_denominator(denom_bound or 1_000_000)
    if denom_bound is not None:
        return snapped
    return Fraction(value)


def residual_identically_zero(
    design: Sequence[Sequence[Number | float]],
    coeffs: Sequence[Number | float],
    target: Sequence[Number | float],
    *,
    intercept: Number | float = 0,
    denom_bound: int | None = None,
) -> bool:
    """``True`` iff ``intercept + design @ coeffs`` equals ``target`` in ``Q``."""

    if len(design) != len(target):
        raise ValueError("design and target must have the same number of rows")
    snapped = [as_fraction(coef, denom_bound=denom_bound) for coef in coeffs]
    bias = as_fraction(intercept, denom_bound=denom_bound)
    for row, rhs in zip(design, target, strict=True):
        if len(row) != len(snapped):
            raise ValueError("design width must match coefficient count")
        acc = bias
        for coef, entry in zip(snapped, row, strict=True):
            acc += coef * as_fraction(entry, denom_bound=denom_bound)
        if acc != as_fraction(rhs, denom_bound=denom_bound):
            return False
    return True


def rref(matrix: Sequence[Sequence[Number]]) -> tuple[list[list[Fraction]], list[int]]:
    """Reduced row echelon form over the rationals; returns ``(rref, pivot_cols)``."""

    rows = [[as_fraction(value) for value in row] for row in matrix]
    if not rows:
        return rows, []
    n_rows = len(rows)
    n_cols = len(rows[0])
    pivots: list[int] = []
    r = 0
    for c in range(n_cols):
        pivot = None
        for i in range(r, n_rows):
            if rows[i][c] != 0:
                pivot = i
                break
        if pivot is None:
            continue
        rows[r], rows[pivot] = rows[pivot], rows[r]
        head = rows[r][c]
        rows[r] = [val / head for val in rows[r]]
        for i in range(n_rows):
            if i != r and rows[i][c] != 0:
                factor = rows[i][c]
                rows[i] = [a - factor * b for a, b in zip(rows[i], rows[r], strict=True)]
        pivots.append(c)
        r += 1
        if r == n_rows:
            break
    return rows, pivots


def _primitive_integer(vec: Sequence[Fraction]) -> list[int]:
    denom_lcm = 1
    for value in vec:
        denom_lcm = denom_lcm * value.denominator // gcd(denom_lcm, value.denominator)
    ints = [int(value * denom_lcm) for value in vec]
    g = 0
    for entry in ints:
        g = gcd(g, abs(entry))
    if g > 1:
        ints = [entry // g for entry in ints]
    for entry in ints:
        if entry != 0:
            if entry < 0:
                ints = [-x for x in ints]
            break
    return ints


def integer_null_space(matrix: Sequence[Sequence[int]]) -> list[list[int]]:
    """Exact primitive-integer basis of ``{p : matrix @ p = 0}``."""

    if not matrix:
        return []
    reduced, pivots = rref([[Fraction(int(v)) for v in row] for row in matrix])
    n_cols = len(matrix[0])
    pivot_set = set(pivots)
    free_cols = [c for c in range(n_cols) if c not in pivot_set]
    basis: list[list[int]] = []
    for free in free_cols:
        vec = [Fraction(0) for _ in range(n_cols)]
        vec[free] = Fraction(1)
        for row_idx, pivot_col in enumerate(pivots):
            vec[pivot_col] = -reduced[row_idx][free]
        basis.append(_primitive_integer(vec))
    return basis


__all__ = [
    "as_fraction",
    "integer_null_space",
    "residual_identically_zero",
    "rref",
]
