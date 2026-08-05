# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""DP-recurrence discovery probe: recover the exact law a struct DP obeys.

Cross-package probe (omnibias-struct counts -> omnibias-symbolic recovery); skipped when
omnibias-symbolic is not installed so the struct-only CI job stays green. The counting DPs
emit genuine P-recursive integer sequences, and ``discover_recurrence`` recovers the exact
homogeneous law each satisfies -- the strongest such case being the non-monic central
Delannoy recurrence of the DTW / alignment warping lattice.
"""

from __future__ import annotations

from fractions import Fraction

import numpy as np
import pytest
from omnibias.struct import DAG, ChainTrellis, DTWLattice

discover_recurrence = pytest.importorskip("omnibias.symbolic").discover_recurrence


def _fib_dag_count(n: int) -> int:
    edges: dict[tuple[int, int], float] = {}
    for i in range(n):
        if i + 1 < n:
            edges[(i, i + 1)] = 1.0
        if i + 2 < n:
            edges[(i, i + 2)] = 1.0
    return DAG(n, edges, source=0, sink=n - 1).count_paths()


def test_chain_path_counts_recover_geometric_law() -> None:
    seq = [ChainTrellis(np.zeros((t, 3)), np.zeros((3, 3))).count_paths() for t in range(1, 9)]
    assert seq == [3**t for t in range(1, 9)]
    rel = discover_recurrence(seq, max_order=1, max_index_degree=0)
    assert rel is not None
    assert rel.order == 1 and rel.index_degree == 0
    # 1 * a_n - 3 * a_{n-1} = 0
    assert rel.coefficients == ((Fraction(1),), (Fraction(-3),))
    assert rel.is_satisfied_by([Fraction(x) for x in seq])


def test_fibonacci_dag_recovers_second_order_law() -> None:
    seq = [_fib_dag_count(n) for n in range(2, 13)]
    assert seq == [1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144]
    rel = discover_recurrence(seq, max_order=2, max_index_degree=0)
    assert rel is not None
    assert rel.order == 2 and rel.index_degree == 0
    # a_n - a_{n-1} - a_{n-2} = 0
    assert rel.coefficients == ((Fraction(1),), (Fraction(-1),), (Fraction(-1),))
    assert rel.is_satisfied_by([Fraction(x) for x in seq])


def test_dtw_grid_recovers_central_delannoy_recurrence() -> None:
    # Warping-grid path counts are the central Delannoy numbers 1, 3, 13, 63, 321, ...
    seq = [DTWLattice(k + 1, k + 1).count_paths() for k in range(12)]
    assert seq[:5] == [1, 3, 13, 63, 321]
    rel = discover_recurrence(seq, max_order=2, max_index_degree=1)
    assert rel is not None
    # The genuine, non-monic P-recursive law: n a_n - 3(2n-1) a_{n-1} + (n-1) a_{n-2} = 0,
    # i.e. p_0(n) = n, p_1(n) = 3 - 6 n, p_2(n) = n - 1 (coefficients are (const, n) pairs).
    assert rel.order == 2 and rel.index_degree == 1
    assert rel.coefficients == (
        (Fraction(0), Fraction(1)),
        (Fraction(3), Fraction(-6)),
        (Fraction(-1), Fraction(1)),
    )
    assert rel.is_satisfied_by([Fraction(x) for x in seq])
