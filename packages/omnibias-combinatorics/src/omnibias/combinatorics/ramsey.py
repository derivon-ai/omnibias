# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
r"""Finite Ramsey colouring and saturation-matrix smokes (not Erdős 183).

A triangle-free ``k``-edge-colouring of ``K_n`` is a finite witness that
``R_k(3) > n``. That does **not** prove ``R_k(3) = k^{Θ(k)}``.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from itertools import combinations
from typing import Any

DEFAULT_N_CAP = 8


def _honesty(*, colouring_replay: bool) -> dict[str, bool]:
    return {
        "discovered_by_omnibias": False,
        "erdos_183_claim": False,
        "ramsey_colouring_replay": colouring_replay,
        "ten_proofs_formalization_claim": False,
    }


def is_triangle_free_edge_colouring(
    n: int,
    k: int,
    colours: Mapping[tuple[int, int], int],
    *,
    n_cap: int = DEFAULT_N_CAP,
) -> bool:
    """Whether every triangle of ``K_n`` uses at least two colours."""
    if n < 1 or k < 1:
        raise ValueError("n and k must be positive")
    if n > n_cap:
        raise ValueError(f"n={n} exceeds n_cap={n_cap}")
    def colour_of(u: int, v: int) -> int:
        key = (u, v) if u < v else (v, u)
        if key in colours:
            return colours[key]
        rev = (v, u) if u < v else (u, v)
        if rev in colours:
            return colours[rev]
        raise ValueError(f"missing colour for edge {(u, v)}")

    for i, j in combinations(range(n), 2):
        colour = colour_of(i, j)
        if not 0 <= colour < k:
            raise ValueError(f"colour {colour} is not in 0..{k - 1}")
    for a, b, c in combinations(range(n), 3):
        if colour_of(a, b) == colour_of(b, c) == colour_of(a, c):
            return False
    return True


def pentagon_two_colouring() -> dict[tuple[int, int], int]:
    """Classical ``R(3,3) > 5``: 5-cycle versus its complement."""
    cycle = {(0, 1), (1, 2), (2, 3), (3, 4), (4, 0)}
    colours: dict[tuple[int, int], int] = {}
    for i, j in combinations(range(5), 2):
        edge = (i, j) if (i, j) in cycle or (j, i) in cycle else None
        colours[(i, j)] = 0 if edge is not None else 1
    return colours


def is_saturated(h: int, m: int, s: int, matrix: Sequence[Sequence[int]]) -> bool:
    """Lean ``IsSaturated``: every ``(m+1)``-set of columns is a permutation in some row."""
    if len(matrix) != s:
        raise ValueError("row count must equal s")
    if any(len(row) != h for row in matrix):
        raise ValueError("each row must have H columns")
    symbols = set(range(m + 1))
    for row in matrix:
        if any(entry not in symbols for entry in row):
            raise ValueError("matrix entries must lie in Fin(m+1)")
    if h < m + 1:
        return True
    for cols in combinations(range(h), m + 1):
        covered = False
        for row in matrix:
            if {row[c] for c in cols} == symbols:
                covered = True
                break
        if not covered:
            return False
    return True


def verify_pentagon_colouring() -> dict[str, Any]:
    colours = pentagon_two_colouring()
    ok = is_triangle_free_edge_colouring(5, 2, colours)
    return {
        "kind": "ramsey_triangle_free_colouring",
        "n": 5,
        "k": 2,
        "triangle_free": ok,
        "finite_statement": "R_2(3) > 5",
        "replay_ok": ok,
        "honesty": _honesty(colouring_replay=ok),
    }


def verify_saturated_smoke() -> dict[str, Any]:
    # Vacuous (H < m+1) plus a genuine 3-column permutation row.
    vacuous = is_saturated(2, 2, 1, ((0, 1),))
    genuine = is_saturated(3, 2, 1, ((0, 1, 2),))
    ok = vacuous and genuine
    return {
        "kind": "ramsey_saturated_matrix",
        "vacuous_ok": vacuous,
        "genuine_ok": genuine,
        "replay_ok": ok,
        "honesty": _honesty(colouring_replay=False),
    }


__all__ = [
    "DEFAULT_N_CAP",
    "is_saturated",
    "is_triangle_free_edge_colouring",
    "pentagon_two_colouring",
    "verify_pentagon_colouring",
    "verify_saturated_smoke",
]
