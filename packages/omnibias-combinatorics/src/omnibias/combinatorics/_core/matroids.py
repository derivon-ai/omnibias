# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
r"""Concrete matroids and their (integral) independent-set polytopes.

A matroid ``M = (E, I)`` on a finite ground set ``E`` is the structure on which the
**greedy** algorithm is exactly optimal for max-weight independent set. Its
independent-set polytope

.. math::
    P(M) = \{ x \in [0, 1]^E : \sum_{e \in S} x_e \le \mathrm{rank}(S)
             \ \ \forall S \subseteq E \}

is **integral** (its vertices are exactly the indicator vectors of independent sets),
so the LP relaxation is exact and the ``lp_dual_lower_bound`` certificate is tight.

Three concrete families are provided, each exposing the compact set of rank
inequalities that describe its polytope (small ground sets / graphs):

* :class:`UniformMatroid` ``U(n, k)`` -- independent iff ``|S| <= k`` (one row ``sum x <= k``);
* :class:`PartitionMatroid` -- a cap per group (one row per group);
* :class:`GraphicMatroid` -- forests of a graph (a row per node subset ``T``:
  ``sum_{e in E[T]} x_e <= |T| - 1``).

Greedy is the exact oracle; :meth:`independent_sets` enumerates the vertices for the
small-instance brute-force cross-check.

The independence / rank rules for the three families are **not** re-derived here: they
are the canonical :mod:`omnibias.discrete.matroid` kernel, so this polytope-lens matroid
and the greedy-oracle-lens :mod:`omnibias.submodular` matroid share one definition of
acyclicity (graphic) and one capacity rule (partition). This module adds only the
polytope surface (:meth:`~Matroid.polytope_constraints`) on top of that core.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray
from omnibias.discrete.matroid import (
    MatroidCore,
    independent_sets,
)
from omnibias.discrete.matroid import graphic_independent as _graphic_independent
from omnibias.discrete.matroid import graphic_rank as _graphic_rank
from omnibias.discrete.matroid import partition_independent as _partition_independent
from omnibias.discrete.matroid import partition_rank as _partition_rank
from omnibias.discrete.matroid import uniform_independent as _uniform_independent
from omnibias.discrete.matroid import uniform_rank as _uniform_rank

FloatArray = NDArray[np.float64]


@runtime_checkable
class Matroid(MatroidCore, Protocol):
    r"""A matroid on ground set ``range(ground_size)`` with its LP-polytope surface.

    Extends the shared :class:`omnibias.discrete.matroid.MatroidCore` contract
    (``ground_size`` / ``is_independent(subset)`` / ``rank(subset)``) with
    ``polytope_constraints()``, which returns the rank inequalities ``A_ineq x <= b_ineq``
    (over ``ground_size`` variables) describing the integral independent-set polytope.
    """

    def polytope_constraints(self) -> tuple[FloatArray, FloatArray]: ...


@dataclass(frozen=True)
class UniformMatroid:
    r"""The uniform matroid ``U(n, k)``: independent iff at most ``k`` elements chosen."""

    n: int
    k: int

    def __post_init__(self) -> None:
        if self.n < 1:
            raise ValueError("n must be >= 1")
        if not 0 <= self.k <= self.n:
            raise ValueError(f"k must be in [0, {self.n}]; got {self.k}")

    @property
    def ground_size(self) -> int:
        return int(self.n)

    def is_independent(self, subset: frozenset[int]) -> bool:
        return _uniform_independent(subset, self.k)

    def rank(self, subset: frozenset[int]) -> int:
        return _uniform_rank(subset, self.k)

    def polytope_constraints(self) -> tuple[FloatArray, FloatArray]:
        """The single cardinality row ``sum_e x_e <= k`` (box ``x <= 1`` handled separately)."""
        a = np.ones((1, self.n), dtype=float)
        b = np.array([float(self.k)], dtype=float)
        return a, b


@dataclass(frozen=True)
class PartitionMatroid:
    r"""A partition matroid: ``groups`` partition the ground set, at most ``caps[i]`` per group."""

    groups: tuple[tuple[int, ...], ...]
    caps: tuple[int, ...]

    def __post_init__(self) -> None:
        if len(self.groups) != len(self.caps):
            raise ValueError("groups and caps must have the same length")
        flat = [e for g in self.groups for e in g]
        if sorted(flat) != list(range(len(flat))):
            raise ValueError("groups must partition range(ground_size) (0-based, contiguous)")
        if any(c < 0 for c in self.caps):
            raise ValueError("caps must be nonnegative")

    @property
    def ground_size(self) -> int:
        return sum(len(g) for g in self.groups)

    def _group_of(self, e: int) -> int:
        for gi, g in enumerate(self.groups):
            if e in g:
                return gi
        raise ValueError(f"element {e} is in no group")

    def is_independent(self, subset: frozenset[int]) -> bool:
        return _partition_independent(subset, self.groups, self.caps)

    def rank(self, subset: frozenset[int]) -> int:
        return _partition_rank(subset, self.groups, self.caps)

    def polytope_constraints(self) -> tuple[FloatArray, FloatArray]:
        """One row per group: ``sum_{e in group_i} x_e <= caps[i]``."""
        n = self.ground_size
        rows = []
        rhs = []
        for gi, g in enumerate(self.groups):
            row = np.zeros(n, dtype=float)
            for e in g:
                row[e] = 1.0
            rows.append(row)
            rhs.append(float(self.caps[gi]))
        return np.asarray(rows, dtype=float), np.asarray(rhs, dtype=float)


@dataclass(frozen=True)
class GraphicMatroid:
    r"""The graphic matroid of an undirected graph: independent iff the edge set is a forest.

    ``edges`` are ``(u, v)`` node pairs (``0 <= u, v < n_nodes``); the ground set is the
    edge index set. Max-weight independent set is the max-weight forest (Kruskal). The
    polytope is enumerated over node subsets, so keep ``n_nodes`` small.
    """

    n_nodes: int
    edges: tuple[tuple[int, int], ...]

    def __post_init__(self) -> None:
        if self.n_nodes < 1:
            raise ValueError("n_nodes must be >= 1")
        for u, v in self.edges:
            if not (0 <= u < self.n_nodes and 0 <= v < self.n_nodes):
                raise ValueError(f"edge ({u}, {v}) references a node outside 0..{self.n_nodes - 1}")

    @property
    def ground_size(self) -> int:
        return len(self.edges)

    def is_independent(self, subset: frozenset[int]) -> bool:
        return _graphic_independent(subset, self.edges, self.n_nodes)

    def rank(self, subset: frozenset[int]) -> int:
        return _graphic_rank(subset, self.edges, self.n_nodes)

    def polytope_constraints(self) -> tuple[FloatArray, FloatArray]:
        r"""Forest-polytope rows ``sum_{e in E[T]} x_e <= |T| - 1`` over node subsets ``T``."""
        n_edges = len(self.edges)
        rows: list[FloatArray] = []
        rhs: list[float] = []
        nodes = list(range(self.n_nodes))
        for size in range(2, self.n_nodes + 1):
            for combo in combinations(nodes, size):
                node_set = set(combo)
                row = np.zeros(n_edges, dtype=float)
                any_edge = False
                for e, (u, v) in enumerate(self.edges):
                    if u in node_set and v in node_set:
                        row[e] = 1.0
                        any_edge = True
                if any_edge:
                    rows.append(row)
                    rhs.append(float(size - 1))
        if not rows:  # no edges within any subset (e.g. empty graph)
            return np.zeros((0, n_edges), dtype=float), np.zeros((0,), dtype=float)
        return np.asarray(rows, dtype=float), np.asarray(rhs, dtype=float)


__all__ = [
    "GraphicMatroid",
    "Matroid",
    "PartitionMatroid",
    "UniformMatroid",
    "independent_sets",
]
