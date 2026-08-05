# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Exact rational linear algebra: Gaussian elimination, null space, least-norm solve.

Small, dependency-free routines over :class:`~fractions.Fraction`, used by Gosper's
polynomial equation and Zeilberger's telescoping ansatz. Everything is exact -- no floats,
no pivoting heuristics beyond exact non-zero selection.
"""

from __future__ import annotations

from collections.abc import Sequence
from fractions import Fraction

Matrix = list[list[Fraction]]
Vector = list[Fraction]


def _as_fraction(x: Fraction | int) -> Fraction:
    return x if isinstance(x, Fraction) else Fraction(x)


def solve_exact(matrix: Sequence[Sequence[Fraction | int]], rhs: Sequence[Fraction | int]) -> Vector | None:
    r"""A solution ``x`` of ``matrix @ x = rhs`` (any one if under-determined), or ``None``.

    Returns ``None`` iff the system is inconsistent. Free variables are set to zero.
    """
    a: Matrix = [[_as_fraction(v) for v in row] for row in matrix]
    b: Vector = [_as_fraction(v) for v in rhs]
    if not a:
        return []
    rows, cols = len(a), len(a[0])
    aug = [a[i] + [b[i]] for i in range(rows)]
    pivots: list[int] = []
    r = 0
    for c in range(cols):
        piv = next((i for i in range(r, rows) if aug[i][c] != 0), None)
        if piv is None:
            continue
        aug[r], aug[piv] = aug[piv], aug[r]
        inv = Fraction(1) / aug[r][c]
        aug[r] = [v * inv for v in aug[r]]
        for i in range(rows):
            if i != r and aug[i][c] != 0:
                factor = aug[i][c]
                aug[i] = [vi - factor * vr for vi, vr in zip(aug[i], aug[r], strict=True)]
        pivots.append(c)
        r += 1
        if r == rows:
            break
    # consistency: any all-zero LHS row with non-zero RHS is a contradiction.
    for i in range(rows):
        if all(aug[i][c] == 0 for c in range(cols)) and aug[i][cols] != 0:
            return None
    x = [Fraction(0)] * cols
    for i, c in enumerate(pivots):
        x[c] = aug[i][cols]
    return x


def null_space(matrix: Sequence[Sequence[Fraction | int]]) -> list[Vector]:
    r"""A basis of the right null space ``{x : matrix @ x = 0}`` (exact)."""
    a: Matrix = [[_as_fraction(v) for v in row] for row in matrix]
    if not a:
        return []
    rows, cols = len(a), len(a[0])
    aug = [row[:] for row in a]
    pivot_col: list[int] = []
    r = 0
    for c in range(cols):
        piv = next((i for i in range(r, rows) if aug[i][c] != 0), None)
        if piv is None:
            continue
        aug[r], aug[piv] = aug[piv], aug[r]
        inv = Fraction(1) / aug[r][c]
        aug[r] = [v * inv for v in aug[r]]
        for i in range(rows):
            if i != r and aug[i][c] != 0:
                factor = aug[i][c]
                aug[i] = [vi - factor * vr for vi, vr in zip(aug[i], aug[r], strict=True)]
        pivot_col.append(c)
        r += 1
        if r == rows:
            break
    pivot_set = set(pivot_col)
    free = [c for c in range(cols) if c not in pivot_set]
    basis: list[Vector] = []
    for f in free:
        vec = [Fraction(0)] * cols
        vec[f] = Fraction(1)
        for i, pc in enumerate(pivot_col):
            vec[pc] = -aug[i][f]
        basis.append(vec)
    return basis


__all__ = ["null_space", "solve_exact"]
