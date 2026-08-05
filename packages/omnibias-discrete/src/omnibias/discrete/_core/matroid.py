# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
r"""Representation-neutral matroid independence / rank kernel.

This is the single canonical definition of *what makes a subset independent* (and its
rank) for the three matroid families the discrete-optimization stack shares --
**uniform**, **partition**, and **graphic** -- expressed on the mathematical
representation of a matroid: a subset is a ``frozenset[int]`` over
``range(ground_size)``.

Two downstream packages put deliberately different *lenses* on matroids and, before
this kernel, each re-derived the same independence rules:

* :mod:`omnibias.combinatorics` -- the **polytope / LP-certification** lens: a matroid
  exposes its rank inequalities so the (integral) independent-set polytope certifies a
  greedy optimum. It is natively ``frozenset``-based, so it delegates here directly.
* :mod:`omnibias.submodular` -- the **optimization** lens: a matroid exposes the exact
  linear-maximization oracle (Rado-Edmonds greedy) and a differentiable soft basis. It
  is natively ``0/1``-vector-based, so it thresholds to the selected index set and
  delegates the *independence question* (only) here.

Both lenses now route the independence/rank question through these functions, so the
two graphic matroids cannot drift apart on acyclicity and the two partition matroids
cannot drift apart on the capacity rule. The union-find acyclicity primitive behind the
graphic family is the sibling :mod:`omnibias.discrete._core.union_find`; the specialized
oracle / polytope surfaces stay in their own packages.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Sequence
from itertools import combinations
from typing import Protocol, runtime_checkable

from omnibias.discrete._core.union_find import UnionFind


@runtime_checkable
class MatroidCore(Protocol):
    r"""The minimal matroid contract on ground set ``range(ground_size)``.

    A matroid is fully determined for the stack's purposes by its ground-set size, its
    independence oracle, and its rank function on ``frozenset[int]`` subsets. Each lens
    adds its specialized surface (rank inequalities / linear oracle) on top of this core.
    """

    @property
    def ground_size(self) -> int: ...

    def is_independent(self, subset: frozenset[int]) -> bool: ...

    def rank(self, subset: frozenset[int]) -> int: ...


def independent_sets(matroid: MatroidCore) -> Iterator[frozenset[int]]:
    r"""Enumerate every independent set of *matroid* (exponential; small ground sets only)."""
    elements = range(matroid.ground_size)
    for size in range(matroid.ground_size + 1):
        for combo in combinations(elements, size):
            subset = frozenset(combo)
            if matroid.is_independent(subset):
                yield subset


# --- uniform matroid U(n, k): independent iff at most k elements are chosen -------------


def uniform_independent(subset: frozenset[int], k: int) -> bool:
    r"""Whether ``subset`` is independent in the uniform matroid ``U(n, k)`` (``|subset| <= k``)."""
    return len(subset) <= k


def uniform_rank(subset: frozenset[int], k: int) -> int:
    r"""Rank of ``subset`` in ``U(n, k)`` (``min(|subset|, k)``)."""
    return min(len(subset), k)


# --- partition matroid: a per-group capacity over a partition of the ground set ---------


def partition_independent(
    subset: frozenset[int], groups: Iterable[Iterable[int]], caps: Iterable[int]
) -> bool:
    r"""Whether ``subset`` obeys every group capacity (``|subset cap group_i| <= caps[i]``)."""
    for group, cap in zip(groups, caps, strict=True):
        if len(subset & frozenset(group)) > cap:
            return False
    return True


def partition_rank(
    subset: frozenset[int], groups: Iterable[Iterable[int]], caps: Iterable[int]
) -> int:
    r"""Rank of ``subset`` in a partition matroid (``sum_i min(|subset cap group_i|, caps[i])``)."""
    return sum(
        min(len(subset & frozenset(group)), cap)
        for group, cap in zip(groups, caps, strict=True)
    )


# --- graphic matroid: the ground set is the edges; independent iff the edges form a forest


def graphic_independent(
    subset: frozenset[int], edges: Sequence[tuple[int, int]], n_nodes: int
) -> bool:
    r"""Whether the edges indexed by ``subset`` form a forest (are acyclic) over ``n_nodes``."""
    uf = UnionFind(n_nodes)
    for e in subset:
        u, v = edges[e]
        if not uf.union(u, v):
            return False
    return True


def graphic_rank(subset: frozenset[int], edges: Sequence[tuple[int, int]], n_nodes: int) -> int:
    r"""Rank of ``subset`` in the graphic matroid (the number of tree edges it contributes)."""
    uf = UnionFind(n_nodes)
    rank = 0
    for e in subset:
        u, v = edges[e]
        if uf.union(u, v):
            rank += 1
    return rank


__all__ = [
    "MatroidCore",
    "graphic_independent",
    "graphic_rank",
    "independent_sets",
    "partition_independent",
    "partition_rank",
    "uniform_independent",
    "uniform_rank",
]
