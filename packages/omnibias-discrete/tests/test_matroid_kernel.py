# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
r"""The shared, representation-neutral matroid independence / rank kernel.

Two kinds of guard:

* **spec parity** -- each kernel predicate matches an independent brute-force definition
  of the family (cardinality / per-group capacity / graph acyclicity) over *every* subset
  of a small ground set; and
* **cross-lens unification** -- the two downstream matroid lenses
  (``omnibias.combinatorics`` polytope matroids and ``omnibias.submodular`` greedy-oracle
  matroids) now route independence through this one kernel, so their ``UniformMatroid`` /
  ``PartitionMatroid`` / ``GraphicMatroid`` must agree on *every* subset. These legs skip
  cleanly when a consumer package is not installed.
"""

from __future__ import annotations

from itertools import combinations

import numpy as np
import pytest
from omnibias.discrete.matroid import (
    MatroidCore,
    graphic_independent,
    graphic_rank,
    independent_sets,
    partition_independent,
    partition_rank,
    uniform_independent,
    uniform_rank,
)


def _all_subsets(n: int) -> list[frozenset[int]]:
    return [frozenset(c) for size in range(n + 1) for c in combinations(range(n), size)]


def _indicator(subset: frozenset[int], n: int) -> np.ndarray:
    x = np.zeros(n, dtype=float)
    for i in subset:
        x[i] = 1.0
    return x


# --- spec parity -----------------------------------------------------------------------


def test_uniform_matches_cardinality_spec() -> None:
    n, k = 7, 3
    for s in _all_subsets(n):
        assert uniform_independent(s, k) == (len(s) <= k)
        assert uniform_rank(s, k) == min(len(s), k)


def test_partition_matches_capacity_spec() -> None:
    groups = ((0, 1, 2), (3, 4), (5, 6))
    caps = (2, 1, 2)
    for s in _all_subsets(7):
        ok = all(len(s & frozenset(g)) <= c for g, c in zip(groups, caps, strict=True))
        rank = sum(min(len(s & frozenset(g)), c) for g, c in zip(groups, caps, strict=True))
        assert partition_independent(s, groups, caps) == ok
        assert partition_rank(s, groups, caps) == rank


def _ref_graph(subset: frozenset[int], edges: list[tuple[int, int]], n: int) -> tuple[bool, int]:
    """Independent (union-find-free) acyclicity + rank via component counting (DFS)."""
    adj: list[list[int]] = [[] for _ in range(n)]
    for e in subset:
        u, v = edges[e]
        adj[u].append(v)
        adj[v].append(u)
    seen = [False] * n
    components = 0
    for start in range(n):
        if seen[start]:
            continue
        components += 1
        stack = [start]
        seen[start] = True
        while stack:
            node = stack.pop()
            for nb in adj[node]:
                if not seen[nb]:
                    seen[nb] = True
                    stack.append(nb)
    rank = n - components
    is_forest = len(subset) == rank  # a subgraph is a forest iff |E| == |V| - components
    return is_forest, rank


def test_graphic_matches_acyclicity_spec() -> None:
    edges = [(0, 1), (1, 2), (2, 0), (2, 3), (3, 4)]
    n = 5
    for s in _all_subsets(len(edges)):
        ref_forest, ref_rank = _ref_graph(s, edges, n)
        assert graphic_independent(s, edges, n) == ref_forest
        assert graphic_rank(s, edges, n) == ref_rank


def test_independent_sets_enumerates_via_protocol() -> None:
    class _Uniform:
        ground_size = 4

        def is_independent(self, subset: frozenset[int]) -> bool:
            return uniform_independent(subset, 2)

        def rank(self, subset: frozenset[int]) -> int:
            return uniform_rank(subset, 2)

    m = _Uniform()
    assert isinstance(m, MatroidCore)
    enumerated = set(independent_sets(m))
    expected = {s for s in _all_subsets(4) if len(s) <= 2}
    assert enumerated == expected


# --- cross-lens unification (skip when a consumer is not installed) ---------------------


def test_combinatorics_and_submodular_uniform_agree() -> None:
    comb = pytest.importorskip("omnibias.combinatorics")
    subm = pytest.importorskip("omnibias.submodular")
    n, k = 6, 3
    cm = comb.UniformMatroid(n, k)
    sm = subm.UniformMatroid(n, k)
    for s in _all_subsets(n):
        want = uniform_independent(s, k)
        assert cm.is_independent(s) == want
        assert sm.is_independent(_indicator(s, n)) == want


def test_combinatorics_and_submodular_partition_agree() -> None:
    comb = pytest.importorskip("omnibias.combinatorics")
    subm = pytest.importorskip("omnibias.submodular")
    groups = ((0, 1, 2), (3, 4, 5))
    caps = (2, 1)
    n = 6
    cm = comb.PartitionMatroid(groups, caps)
    sm = subm.PartitionMatroid([list(g) for g in groups], list(caps))
    for s in _all_subsets(n):
        want = partition_independent(s, groups, caps)
        assert cm.is_independent(s) == want
        assert sm.is_independent(_indicator(s, n)) == want


def test_combinatorics_and_submodular_graphic_agree() -> None:
    comb = pytest.importorskip("omnibias.combinatorics")
    subm = pytest.importorskip("omnibias.submodular")
    edges = [(0, 1), (1, 2), (2, 0), (2, 3)]
    n = 4
    cm = comb.GraphicMatroid(n, tuple(edges))
    sm = subm.GraphicMatroid(edges, n_vertices=n)
    for s in _all_subsets(len(edges)):
        want = graphic_independent(s, edges, n)
        assert cm.is_independent(s) == want
        assert sm.is_independent(_indicator(s, len(edges))) == want
