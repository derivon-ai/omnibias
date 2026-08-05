# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
"""General matroids: laminar / graphic / transversal / intersection + the p-matroid greedy.

The Rado-Edmonds greedy oracle in the base class is exact for every *single* matroid, so
``max_weight_basis`` must match brute force; the intersection oracle is exact by enumeration.
General matroids expose no soft oracle / group structure.
"""

from __future__ import annotations

import itertools

import numpy as np
import pytest
from omnibias.submodular import (
    Coverage,
    GraphicMatroid,
    LaminarMatroid,
    MatroidIntersection,
    PartitionMatroid,
    TransversalMatroid,
    UniformMatroid,
    brute_force_max,
    p_matroid_greedy,
)
from omnibias.submodular.matroid import Matroid


def _brute_max_weight(m: Matroid, w: np.ndarray) -> float:
    best = -np.inf
    for bits in itertools.product([0, 1], repeat=m.n):
        x = np.array(bits, dtype=float)
        if m.is_independent(x):
            best = max(best, float(w @ x))
    return best


def _brute_rank(m: Matroid) -> int:
    best = 0
    for bits in itertools.product([0, 1], repeat=m.n):
        x = np.array(bits, dtype=float)
        if m.is_independent(x):
            best = max(best, int(x.sum()))
    return best


def _oracle_matches_brute_force(m: Matroid, seed: int, trials: int = 15) -> None:
    rng = np.random.default_rng(seed)
    for _ in range(trials):
        w = rng.standard_normal(m.n)
        y = m.max_weight_basis(w)
        assert m.is_independent(y)
        assert abs(float(w @ y) - _brute_max_weight(m, w)) < 1e-9
    assert m.rank() == _brute_rank(m)  # greedy maximal == maximum (the matroid property)


# ---- LaminarMatroid ---------------------------------------------------------------------


def test_laminar_independence_rank_and_oracle() -> None:
    m = LaminarMatroid([[0, 1, 2, 3, 4, 5], [0, 1, 2], [3, 4]], [4, 2, 1], n=6)
    assert m.n == 6
    assert m.is_independent(np.array([1.0, 1.0, 0.0, 1.0, 0.0, 1.0]))  # big 4, g012 2, g34 1
    assert not m.is_independent(np.array([1.0, 1.0, 1.0, 0.0, 0.0, 0.0]))  # g012 over cap
    assert m.rank() == 4
    _oracle_matches_brute_force(m, seed=0)


def test_laminar_rejects_non_laminar_family() -> None:
    with pytest.raises(ValueError, match="laminar"):
        LaminarMatroid([[0, 1], [1, 2]], [1, 1], n=3)  # overlap that is neither disjoint nor nested


def test_laminar_generalizes_partition() -> None:
    # Disjoint constraint sets = a partition matroid; independence must agree.
    lam = LaminarMatroid([[0, 1, 2], [3, 4]], [2, 1], n=5)
    par = PartitionMatroid([[0, 1, 2], [3, 4]], [2, 1])
    rng = np.random.default_rng(0)
    for _ in range(50):
        x = (rng.random(5) < 0.5).astype(float)
        assert lam.is_independent(x) == par.is_independent(x)


# ---- GraphicMatroid ---------------------------------------------------------------------


def test_graphic_forest_independence_and_oracle() -> None:
    # Triangle: any two edges are a forest; all three form a cycle.
    m = GraphicMatroid([(0, 1), (1, 2), (2, 0)], n_vertices=3)
    assert m.n == 3
    assert m.is_independent(np.array([1.0, 1.0, 0.0]))
    assert not m.is_independent(np.array([1.0, 1.0, 1.0]))  # cycle
    assert m.rank() == 2  # spanning tree of 3 vertices
    _oracle_matches_brute_force(m, seed=1)


def test_graphic_max_weight_basis_is_kruskal() -> None:
    m = GraphicMatroid([(0, 1), (1, 2), (2, 0)], n_vertices=3)
    w = np.array([3.0, 2.0, 1.0])  # drop the lightest edge in the cycle
    y = m.max_weight_basis(w)
    assert list(y) == [1.0, 1.0, 0.0]


def test_graphic_rejects_bad_edge() -> None:
    with pytest.raises(ValueError, match="endpoint"):
        GraphicMatroid([(0, 3)], n_vertices=3)


# ---- TransversalMatroid -----------------------------------------------------------------


def test_transversal_matching_independence_and_oracle() -> None:
    # A "path" bipartite graph 0-a, 1-{a,b}, 2-{b,c}, 3-c over 3 resources {a,b,c}.
    m = TransversalMatroid([[0], [0, 1], [1, 2], [2]], n_resources=3)
    assert m.n == 4
    assert m.is_independent(np.array([1.0, 1.0, 1.0, 0.0]))  # 0->a, 1->b, 2->c
    assert m.is_independent(np.array([1.0, 1.0, 0.0, 0.0]))  # 0->a, 1->b matchable
    assert m.rank() == 3  # only 3 resources
    _oracle_matches_brute_force(m, seed=2)


def test_transversal_unmatchable_subset_is_dependent() -> None:
    # Elements 0 and 1 both only reach resource 0 -> cannot both be matched.
    m = TransversalMatroid([[0], [0], [1]], n_resources=2)
    assert not m.is_independent(np.array([1.0, 1.0, 0.0]))
    assert m.is_independent(np.array([1.0, 0.0, 1.0]))


# ---- MatroidIntersection ----------------------------------------------------------------


def _grid_intersection() -> MatroidIntersection:
    # 2x2 grid, elements row-major [ (0,0)=0, (0,1)=1, (1,0)=2, (1,1)=3 ].
    rows = PartitionMatroid([[0, 1], [2, 3]], [1, 1])  # <= 1 per row
    cols = PartitionMatroid([[0, 2], [1, 3]], [1, 1])  # <= 1 per column
    return MatroidIntersection([rows, cols])


def test_intersection_independence_is_conjunction() -> None:
    m = _grid_intersection()
    assert m.n == 4
    assert len(m.matroids) == 2
    assert m.is_independent(np.array([1.0, 0.0, 0.0, 1.0]))  # a diagonal: 1 per row + column
    assert not m.is_independent(np.array([1.0, 1.0, 0.0, 0.0]))  # two in row 0


def test_intersection_max_weight_basis_is_exact() -> None:
    m = _grid_intersection()
    rng = np.random.default_rng(3)
    for _ in range(15):
        w = rng.standard_normal(4)
        y = m.max_weight_basis(w)
        assert m.is_independent(y)
        assert abs(float(w @ y) - _brute_max_weight(m, w)) < 1e-9


def test_intersection_soft_oracle_and_groups_raise() -> None:
    m = _grid_intersection()
    with pytest.raises(NotImplementedError, match="soft oracle"):
        m.soft_basis(np.ones(4), 10.0)
    with pytest.raises(NotImplementedError, match="groups"):
        m.groups()


def test_general_matroid_has_no_soft_oracle() -> None:
    for m in (
        GraphicMatroid([(0, 1), (1, 2)], n_vertices=3),
        LaminarMatroid([[0, 1, 2]], [2], n=3),
        TransversalMatroid([[0], [1]], n_resources=2),
    ):
        with pytest.raises(NotImplementedError):
            m.soft_basis(np.ones(m.n), 10.0)
        with pytest.raises(NotImplementedError):
            m.caps()


def test_p_matroid_greedy_meets_one_over_p_plus_one() -> None:
    # p = 2 matroids -> a-priori 1/(p+1) = 1/3; average across seeds (anti-overfit).
    for seed in range(6):
        rng = np.random.default_rng(seed)
        fn = Coverage((rng.random((8, 4)) < 0.5).astype(float), rng.random(8) + 0.2)
        inter = _grid_intersection()
        sel, val = p_matroid_greedy(fn, inter)
        assert inter.is_independent(np.asarray(sel, dtype=float))
        _, opt = brute_force_max(fn, inter)
        assert val >= opt / 3.0 - 1e-9, f"seed {seed}: {val} < OPT/3 = {opt / 3.0}"
        assert val <= opt + 1e-9


def test_p_matroid_greedy_accepts_a_sequence_and_single_matroid() -> None:
    rng = np.random.default_rng(0)
    fn = Coverage((rng.random((8, 6)) < 0.5).astype(float), rng.random(8) + 0.2)
    # A single matroid (p = 1) reduces to greedy_maximize.
    sel_seq, val_seq = p_matroid_greedy(fn, [UniformMatroid(6, 3)])
    assert int(np.asarray(sel_seq).sum()) <= 3
    assert val_seq > 0.0
