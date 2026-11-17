# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Bounded exact affine linear elimination shared by PDE coefficient planners."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from fractions import Fraction
from typing import Protocol

Vector = tuple[Fraction, ...]


class RationalBudget(Protocol):
    def _entries(self, count: int) -> None: ...
    def _work(self, count: int) -> None: ...
    def _coefficient(self, value: Fraction) -> Fraction: ...


@dataclass(frozen=True)
class RationalAffineSolution:
    consistent: bool
    particular: Vector | None
    homogeneous_basis: tuple[Vector, ...]
    rank: int


@dataclass(frozen=True)
class RationalLDLT:
    """One exact positive-definite factorization for repeated right-hand sides."""

    lower: tuple[Vector, ...]
    diagonal: Vector

    def solve(self, rhs: Sequence[Fraction], budget: RationalBudget) -> Vector:
        size = len(self.diagonal)
        if len(rhs) != size:
            raise ValueError("exact positive-definite solve has wrong right-hand side size")
        budget._work(8 * size * size + 4 * size)
        checked = budget._coefficient
        forward: list[Fraction] = []
        for i, value in enumerate(rhs):
            total = checked(value)
            for j in range(i):
                total = checked(total - checked(self.lower[i][j] * forward[j]))
            forward.append(total)
        result = [checked(value / pivot) for value, pivot in zip(forward, self.diagonal, strict=True)]
        for i in reversed(range(size)):
            for j in range(i + 1, size):
                result[i] = checked(result[i] - checked(self.lower[j][i] * result[j]))
        return tuple(result)


def factor_positive_definite(matrix: Sequence[Sequence[Fraction]], budget: RationalBudget) -> RationalLDLT:
    """Factor a nonempty symmetric rational SPD matrix once, without radicals."""
    size = len(matrix)
    if not size or any(len(row) != size for row in matrix):
        raise ValueError("positive-definite matrix must be nonempty and square")
    budget._entries(size * size + size)
    budget._work(8 * size**3 + 8 * size**2)
    checked = budget._coefficient
    for i in range(size):
        for j in range(size):
            checked(matrix[i][j])
            if matrix[i][j] != matrix[j][i]:
                raise ValueError("positive-definite matrix must be symmetric")
    lower = [[Fraction(int(i == j)) for j in range(size)] for i in range(size)]
    diagonal: list[Fraction] = []
    for i in range(size):
        for j in range(i):
            total = matrix[i][j]
            for k in range(j):
                product = checked(checked(lower[i][k] * lower[j][k]) * diagonal[k])
                total = checked(total - product)
            lower[i][j] = checked(total / diagonal[j])
        pivot = matrix[i][i]
        for k in range(i):
            pivot = checked(pivot - checked(checked(lower[i][k] * lower[i][k]) * diagonal[k]))
        if pivot <= 0:
            raise ValueError("exact matrix is not positive definite")
        diagonal.append(pivot)
    return RationalLDLT(tuple(tuple(row) for row in lower), tuple(diagonal))


def solve_affine(
    matrix: Sequence[Sequence[Fraction]], rhs: Sequence[Fraction] | None,
    budget: RationalBudget,
) -> RationalAffineSolution:
    """Solve a rectangular rational system, retaining every free variable.

    ``rhs=None`` is homogeneous and avoids allocating an augmented column.
    Callers own exact input-type validation; every elimination intermediate is
    checked here. Rank always refers to the coefficient matrix.
    """
    if not matrix or not matrix[0] or any(len(row) != len(matrix[0]) for row in matrix):
        raise ValueError("exact linear matrix must be nonempty and rectangular")
    rows, columns = len(matrix), len(matrix[0])
    if rhs is not None and len(rhs) != rows:
        raise ValueError("exact linear right-hand side has the wrong size")
    width = columns + int(rhs is not None)
    budget._entries(rows * width)
    budget._work(3 * rows * width * min(rows, columns) + rows * width)
    a = [[budget._coefficient(value) for value in row] for row in matrix]
    if rhs is not None:
        for row, value in zip(a, rhs, strict=True):
            row.append(budget._coefficient(value))
    pivots: list[int] = []
    for column in range(columns):
        pivot = next((i for i in range(len(pivots), rows) if a[i][column]), None)
        if pivot is None:
            continue
        rank = len(pivots)
        a[rank], a[pivot] = a[pivot], a[rank]
        head = a[rank][column]
        a[rank] = [budget._coefficient(value / head) for value in a[rank]]
        for i in range(rows):
            if i == rank or not a[i][column]:
                continue
            factor = a[i][column]
            a[i] = [budget._coefficient(left - budget._coefficient(factor * right))
                    for left, right in zip(a[i], a[rank], strict=True)]
        pivots.append(column)
        if len(pivots) == rows:
            break
    rank = len(pivots)
    consistent = rhs is None or not any(a[i][-1] for i in range(rank, rows))
    budget._entries((columns - rank + int(consistent)) * columns)
    pivot_set = set(pivots)
    basis: list[Vector] = []
    for free in range(columns):
        if free in pivot_set:
            continue
        vector = [Fraction(0)] * columns
        vector[free] = Fraction(1)
        for row_index, pivot_column in enumerate(pivots):
            vector[pivot_column] = -a[row_index][free]
        basis.append(tuple(vector))
    particular: list[Fraction] | None = [Fraction(0)] * columns if consistent else None
    if particular is not None and rhs is not None:
        for row_index, pivot_column in enumerate(pivots):
            particular[pivot_column] = a[row_index][-1]
    return RationalAffineSolution(consistent, tuple(particular) if particular is not None else None,
                                  tuple(basis), rank)
