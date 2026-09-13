# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Operand-bound rational rank and generic polynomial Jacobian rank."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from fractions import Fraction
from itertools import combinations
from math import comb

from omnibias.core.proof.lift import rref
from omnibias.core.realization.polynomial import (
    AlgebraBudgetExceeded,
    Rational,
    SparsePolynomial,
    rational,
)

Matrix = tuple[tuple[Fraction, ...], ...]


def matrix(rows: Sequence[Sequence[Rational]]) -> Matrix:
    if not rows or not rows[0] or any(len(row) != len(rows[0]) for row in rows):
        raise ValueError("matrix must be nonempty and rectangular")
    return tuple(tuple(rational(v) for v in row) for row in rows)


def determinant(rows: Sequence[Sequence[Rational]]) -> Fraction:
    if not rows:
        return Fraction(1)
    a = [list(row) for row in matrix(rows)]
    n = len(a)
    if len(a[0]) != n:
        raise ValueError("determinant requires a square matrix")
    out = Fraction(1)
    for j in range(n):
        pivot = next((i for i in range(j, n) if a[i][j]), None)
        if pivot is None:
            return Fraction(0)
        if pivot != j:
            a[j], a[pivot] = a[pivot], a[j]
            out = -out
        d = a[j][j]
        out *= d
        for i in range(j + 1, n):
            q = a[i][j] / d
            for k in range(j + 1, n):
                a[i][k] -= q * a[j][k]
    return out


@dataclass(frozen=True)
class ExactRankWitness:
    source: Matrix
    rank: int
    left: Matrix
    right: Matrix
    minor_rows: tuple[int, ...]
    minor_cols: tuple[int, ...]
    minor_value: Fraction

    def verify(self) -> bool:
        if (
            not self.source
            or not self.source[0]
            or type(self.rank) is not int
            or any(len(row) != len(self.source[0]) for row in self.source)
            or any(type(i) is not int for i in (*self.minor_rows, *self.minor_cols))
        ):
            return False
        try:
            for rows in (self.source, self.left, self.right):
                for row in rows:
                    for value in row:
                        rational(value)
            rational(self.minor_value)
        except TypeError:
            return False
        m, n, r = len(self.source), len(self.source[0]), self.rank
        if r < 0 or r > min(m, n):
            return False
        if len(self.left) != m or any(len(row) != r for row in self.left):
            return False
        if len(self.right) != r or any(len(row) != n for row in self.right):
            return False
        if len(self.minor_rows) != r or len(self.minor_cols) != r:
            return False
        if len(set(self.minor_rows)) != r or len(set(self.minor_cols)) != r:
            return False
        if any(i < 0 or i >= m for i in self.minor_rows) or any(
            j < 0 or j >= n for j in self.minor_cols
        ):
            return False
        if any(
            self.source[i][j]
            != sum((self.left[i][k] * self.right[k][j] for k in range(r)), Fraction(0))
            for i in range(m)
            for j in range(n)
        ):
            return False
        d = determinant([[self.source[i][j] for j in self.minor_cols] for i in self.minor_rows])
        return d == self.minor_value and d != 0


def certify_matrix_rank(rows: Sequence[Sequence[Rational]]) -> ExactRankWitness:
    source = matrix(rows)
    reduced, pivots = rref(source)
    r = len(pivots)
    left = tuple(tuple(row[j] for j in pivots) for row in source)
    right = tuple(tuple(row) for row in reduced[:r])
    if r:
        _, row_pivots = rref(tuple(zip(*left, strict=True)))
        selected = tuple(row_pivots)
    else:
        selected = ()
    minor = determinant([[source[i][j] for j in pivots] for i in selected])
    result = ExactRankWitness(source, r, left, right, selected, tuple(pivots), minor)
    if not result.verify():
        raise ArithmeticError("exact rank witness failed replay")
    return result


def polynomial_determinant(rows: Sequence[Sequence[SparsePolynomial]]) -> SparsePolynomial:
    if not rows or any(len(row) != len(rows) for row in rows):
        raise ValueError("nonempty square polynomial matrix required")
    first = rows[0][0]
    if len(rows) == 1:
        return first
    out = SparsePolynomial.constant(first.nvars, 0, budget=first.budget)
    for j, p in enumerate(rows[0]):
        sub = [[v for k, v in enumerate(row) if k != j] for row in rows[1:]]
        out = out + p * polynomial_determinant(sub) * (-1 if j % 2 else 1)
    return out


@dataclass(frozen=True)
class GenericRankReport:
    lower: int
    upper: int
    sample: tuple[Fraction, ...]
    sample_witness: ExactRankWitness
    vanishing_minors: tuple[SparsePolynomial, ...]
    complete: bool
    detail: str


def certify_generic_rank(
    outputs: Sequence[SparsePolynomial], sample: Sequence[Rational], *, max_minors: int = 1000
) -> GenericRankReport:
    """Generic rank of a polynomial map, never a global constant-rank claim.

    The exact sample minor establishes a lower bound. All next-order minors
    must vanish identically to close the upper bound. Otherwise report bounds.
    """
    if not outputs or not sample or any(p.nvars != len(sample) for p in outputs):
        raise ValueError("nonempty polynomial map and compatible sample required")
    jac = tuple(tuple(p.derivative(j) for j in range(len(sample))) for p in outputs)
    witness = certify_matrix_rank([[p.evaluate(sample) for p in row] for row in jac])
    r = witness.rank
    upper = min(len(outputs), len(sample))
    if r == upper:
        return GenericRankReport(
            r, r, tuple(map(rational, sample)), witness, (), True, "nonzero maximal sample minor"
        )
    size = r + 1
    if comb(len(outputs), size) * comb(len(sample), size) > max_minors:
        return GenericRankReport(
            r,
            upper,
            tuple(map(rational, sample)),
            witness,
            (),
            False,
            "symbolic minor budget exceeded",
        )
    zeros: list[SparsePolynomial] = []
    try:
        for rows in combinations(range(len(outputs)), size):
            for cols in combinations(range(len(sample)), size):
                d = polynomial_determinant([[jac[i][j] for j in cols] for i in rows])
                if d.terms:
                    return GenericRankReport(
                        max(r, size),
                        upper,
                        tuple(map(rational, sample)),
                        witness,
                        tuple(zeros),
                        False,
                        "nonzero symbolic minor gives a stronger generic lower bound; choose another sample",
                    )
                zeros.append(d)
    except AlgebraBudgetExceeded as exc:
        return GenericRankReport(
            r, upper, tuple(map(rational, sample)), witness, tuple(zeros), False, str(exc)
        )
    return GenericRankReport(
        r,
        r,
        tuple(map(rational, sample)),
        witness,
        tuple(zeros),
        True,
        "sample lower minor and all next-order polynomial minors",
    )


__all__ = [
    "ExactRankWitness",
    "GenericRankReport",
    "Matrix",
    "certify_generic_rank",
    "certify_matrix_rank",
    "determinant",
    "matrix",
    "polynomial_determinant",
]
