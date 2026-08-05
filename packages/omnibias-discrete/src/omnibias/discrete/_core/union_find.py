# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
r"""Union-find (disjoint-set) and the forest / acyclicity test it powers.

Incremental cycle detection over an edge stream is the one low-level graph primitive several
discrete-optimization consumers share: a graphic matroid decides a subset of edges is independent
iff it stays acyclic, whether that matroid is expressed through its LP polytope
(:class:`omnibias.combinatorics.GraphicMatroid`) or its greedy oracle
(:class:`omnibias.submodular.GraphicMatroid`). This is the single shared implementation of that
primitive; the two matroid class hierarchies that consume it keep their own (deliberately
different) public surfaces in their own packages.
"""

from __future__ import annotations

from collections.abc import Iterable


class UnionFind:
    r"""Disjoint-set forest with path halving; ``union`` reports whether it merged two trees.

    ``union(a, b)`` returns ``True`` when ``a`` and ``b`` were in different components (so the
    edge ``(a, b)`` is a tree edge) and ``False`` when they were already connected (so the edge
    closes a cycle) -- exactly the signal an incremental forest / matroid-independence check needs.
    """

    def __init__(self, n: int) -> None:
        if n < 0:
            raise ValueError("n must be >= 0")
        self._parent = list(range(n))

    def find(self, a: int) -> int:
        """The representative of ``a``'s component (with path-halving compression)."""
        parent = self._parent
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def union(self, a: int, b: int) -> bool:
        """Merge the components of ``a`` and ``b``; ``True`` iff they were previously disjoint."""
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return False
        self._parent[ra] = rb
        return True


def is_forest(edges: Iterable[tuple[int, int]], n_vertices: int) -> bool:
    r"""Whether ``edges`` over ``n_vertices`` vertices induce a forest (are acyclic).

    Streams the edges through one :class:`UnionFind`; the first edge that closes a cycle short-
    circuits the result to ``False``. Runs in ``O(E alpha(V))``.
    """
    uf = UnionFind(n_vertices)
    return all(uf.union(u, v) for u, v in edges)


__all__ = ["UnionFind", "is_forest"]
