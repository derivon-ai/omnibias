# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Exact system utilities: GF(2) linear fast-path and assignment verification.

These are the *exact* (pure-Python) helpers the differentiable solver leans on:

* :func:`gf2_solve` -- if every constraint is XOR-linear (ANF degree <= 1) the
  system ``phi_k(x) = 0`` is an affine system over GF(2); Gaussian elimination
  solves it (or proves it inconsistent) in polynomial time, returning a particular
  solution and a null-space basis. This is the honest "search-space collapse" that
  *does* happen -- but only for the linear part.
* :func:`verify_assignment` -- the verification half of propose-and-verify: check a
  candidate bit assignment against the exact Boolean constraints.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from omnibias.boolean._core.anf import algebraic_degree, anf_from_truth_table
from omnibias.boolean._core.truth_table import TruthTable, index_of, num_vars


def verify_assignment(constraints: Sequence[TruthTable], bits: Sequence[int]) -> bool:
    """``True`` iff every constraint ``phi_k`` is ``0`` at the assignment ``bits``."""
    idx = index_of(tuple(bits))
    return all(c[idx] == 0 for c in constraints)


def constraints_are_linear(constraints: Sequence[TruthTable]) -> bool:
    """``True`` iff every constraint is XOR-linear (ANF algebraic degree <= 1)."""
    return all(algebraic_degree(anf_from_truth_table(c)) <= 1 for c in constraints)


def linear_system_rows(
    constraints: Sequence[TruthTable], n: int
) -> list[tuple[tuple[int, ...], int]] | None:
    """Affine GF(2) rows ``(coeffs, rhs)`` for ``XOR_i coeffs_i x_i = rhs``.

    Returns ``None`` if any constraint is non-linear (ANF degree > 1).
    """
    rows: list[tuple[tuple[int, ...], int]] = []
    for c in constraints:
        anf = anf_from_truth_table(c)
        if algebraic_degree(anf) > 1:
            return None
        coeffs = tuple(anf[1 << i] for i in range(n))
        rows.append((coeffs, anf[0]))
    return rows


@dataclass(frozen=True)
class GF2Solution:
    """Solution space of an affine GF(2) system: ``particular + span(basis)``."""

    consistent: bool
    n: int
    particular: tuple[int, ...]
    free_vars: tuple[int, ...]
    basis: tuple[tuple[int, ...], ...]

    def enumerate_solutions(self, limit: int = 1 << 20) -> list[int]:
        """All solution indices (capped at ``limit`` when the null space is large)."""
        if not self.consistent:
            return []
        sols = set()
        k = len(self.free_vars)
        if (1 << k) > limit:
            raise ValueError(
                f"null space has 2**{k} solutions; raise limit to enumerate"
            )
        for combo in range(1 << k):
            bits = list(self.particular)
            for j in range(k):
                if (combo >> j) & 1:
                    for i in range(self.n):
                        bits[i] ^= self.basis[j][i]
            sols.add(index_of(tuple(bits)))
        return sorted(sols)


def gf2_solve(constraints: Sequence[TruthTable]) -> GF2Solution | None:
    """Solve an all-linear Boolean system over GF(2); ``None`` if it is non-linear."""
    if not constraints:
        raise ValueError("need at least one constraint")
    n = num_vars(constraints[0])
    rows = linear_system_rows(constraints, n)
    if rows is None:
        return None
    aug = [list(coeffs) + [rhs] for coeffs, rhs in rows]
    pivot_row_of_col: dict[int, int] = {}
    r = 0
    for col in range(n):
        piv = next((rr for rr in range(r, len(aug)) if aug[rr][col] == 1), None)
        if piv is None:
            continue
        aug[r], aug[piv] = aug[piv], aug[r]
        for rr in range(len(aug)):
            if rr != r and aug[rr][col] == 1:
                aug[rr] = [a ^ b for a, b in zip(aug[rr], aug[r], strict=False)]
        pivot_row_of_col[col] = r
        r += 1
    for row in aug:
        if all(v == 0 for v in row[:n]) and row[n] == 1:
            return GF2Solution(
                consistent=False, n=n, particular=(), free_vars=(), basis=()
            )
    particular = [0] * n
    for col, row in pivot_row_of_col.items():
        particular[col] = aug[row][n]
    free_vars = tuple(col for col in range(n) if col not in pivot_row_of_col)
    basis: list[tuple[int, ...]] = []
    for f in free_vars:
        vec = [0] * n
        vec[f] = 1
        for col, row in pivot_row_of_col.items():
            vec[col] = aug[row][f]
        basis.append(tuple(vec))
    return GF2Solution(
        consistent=True,
        n=n,
        particular=tuple(particular),
        free_vars=free_vars,
        basis=tuple(basis),
    )


__all__ = [
    "GF2Solution",
    "constraints_are_linear",
    "gf2_solve",
    "linear_system_rows",
    "verify_assignment",
]
