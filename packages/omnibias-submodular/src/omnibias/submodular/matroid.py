# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
r"""Matroid constraints and the linear-maximization (LP) oracle for the maximizers.

A matroid over the ground set ``[n]`` gives the feasible *independent sets*. The greedy /
continuous-greedy maximizers need one primitive -- the **linear oracle**
``argmax_{y in P} <w, y>`` over the matroid polytope ``P`` -- and the classical matroid
(Rado-Edmonds) greedy solves it *exactly for any single matroid*: sort by weight descending
and add each element whose weight is positive and that keeps the set independent. So the
base class implements :meth:`~Matroid.max_weight_basis` (and :meth:`~Matroid.rank`,
:meth:`~Matroid.fill_basis`) generically from :meth:`~Matroid.is_independent`, and every
matroid below only has to define independence.

Matroids that ship:

* :class:`UniformMatroid` -- the cardinality constraint ``|S| <= k``;
* :class:`PartitionMatroid` -- per-group capacities ``|S cap group_g| <= cap_g``;
* :class:`LaminarMatroid` -- nested-family capacities ``|S cap A| <= cap_A`` over a laminar
  family (partition matroids are the disjoint special case);
* :class:`GraphicMatroid` -- forests of a graph (independent = acyclic edge subset);
* :class:`TransversalMatroid` -- subsets of one side of a bipartite graph that can be matched;
* :class:`MatroidIntersection` -- the common independent sets of ``p`` matroids (maximized by
  :func:`~omnibias.submodular.p_matroid_greedy`, a-priori ``1/(p+1)``).

Only the **partition-structured** matroids (:class:`UniformMatroid`, :class:`PartitionMatroid`)
expose the optional :meth:`~Matroid.groups` / :meth:`~Matroid.caps` structure and the
differentiable :meth:`~Matroid.soft_basis` -- a ``sigmoid(beta (w - tau))`` selection whose
per-group threshold ``tau`` sits between the ``cap``-th and ``(cap+1)``-th largest weights, so
it hardens onto the hard basis as ``beta -> inf`` (the feasibility / temperature axis, **not**
the founding ``delta -> 0`` bias collapse). General matroids raise ``NotImplementedError`` on
``soft_basis`` / ``groups`` / ``caps`` (no differentiable twin) and are used through the
greedy family + the marginal certificate.

The *independence question* for the families this lens shares with the polytope lens
(:mod:`omnibias.combinatorics`) -- the partition capacity rule and graphic acyclicity --
is delegated to the shared :mod:`omnibias.discrete.matroid` kernel (via
:func:`_selected_indices`), so the two lenses' matroids cannot drift apart on the
mathematics; only the specialized surfaces (the linear-maximization oracle and
differentiable soft basis here, the rank inequalities there) differ.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from omnibias.discrete.matroid import graphic_independent, partition_independent

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]

_MAX_EXACT_INTERSECTION_N = 20


def _sigmoid(z: FloatArray) -> FloatArray:
    # Clamp before exp so a saturated (|beta (w - tau)| large) argument cannot overflow.
    zc = np.clip(np.asarray(z, dtype=float), -500.0, 500.0)
    return np.asarray(1.0 / (1.0 + np.exp(-zc)), dtype=float)


def _selected_indices(x: object) -> frozenset[int]:
    r"""The chosen index set of a ``0/1`` membership vector (the ``> 0.5`` convention).

    The optimization lens works with ``0/1`` vectors; the shared
    :mod:`omnibias.discrete.matroid` independence kernel works with the canonical
    ``frozenset[int]`` subset. This is the one bridge between them.
    """
    xv = np.asarray(x, dtype=float).reshape(-1)
    return frozenset(int(i) for i in np.flatnonzero(xv > 0.5))


class Matroid(ABC):
    """A matroid over ``[n]`` exposing the exact linear-maximization oracle."""

    @property
    @abstractmethod
    def n(self) -> int:
        """Ground-set size."""

    @abstractmethod
    def is_independent(self, x: object, *, tol: float = 1e-9) -> bool:
        """Whether ``x in {0, 1}^n`` is an independent set of the matroid."""

    def groups(self) -> list[IntArray]:
        """The partition groups (partition-structured matroids only; else raises)."""
        raise NotImplementedError(
            f"{type(self).__name__} is not partition-structured (no groups()); the "
            "group/cap oracle is only defined for UniformMatroid / PartitionMatroid"
        )

    def caps(self) -> list[int]:
        """The per-group capacities (partition-structured matroids only; else raises)."""
        raise NotImplementedError(
            f"{type(self).__name__} is not partition-structured (no caps()); the "
            "group/cap oracle is only defined for UniformMatroid / PartitionMatroid"
        )

    def _greedy_basis(self, w: object, *, fill: bool) -> FloatArray:
        r"""Rado-Edmonds greedy: add elements by descending weight while independent.

        With ``fill=False`` only positive-weight elements are considered (the sign-respecting
        ``argmax`` used for the certificate); with ``fill=True`` every element is a candidate
        (a full max-weight basis for continuous greedy). Exact for any single matroid.
        """
        wv = np.asarray(w, dtype=float).reshape(-1)
        y = np.zeros(self.n, dtype=float)
        for local in np.argsort(-wv, kind="stable"):
            i = int(local)
            if not fill and wv[i] <= 0.0:
                continue
            y[i] = 1.0
            if not self.is_independent(y):
                y[i] = 0.0
        return y

    def max_weight_basis(self, w: object) -> FloatArray:
        r"""The hard LP oracle ``argmax_{y in P} <w, y>`` as a ``0/1`` indicator.

        Rado-Edmonds greedy on the positive weights -- exact for any single matroid; ties
        broken by index. Nonpositive weights are never selected, so capacity is not wasted.
        """
        return self._greedy_basis(w, fill=False)

    def fill_basis(self, w: object) -> FloatArray:
        r"""A *full* max-weight basis (:meth:`rank` elements) for continuous greedy.

        Like :meth:`max_weight_basis` but every element is a candidate (sign is ignored), so
        the iterate moves along genuine matroid bases -- what swap rounding merges without
        bias, and continuous greedy needs in the monotone regime.
        """
        return self._greedy_basis(w, fill=True)

    def rank(self) -> int:
        """The matroid rank (size of a basis), via a greedy maximal independent set."""
        return int(np.sum(self._greedy_basis(np.ones(self.n, dtype=float), fill=True)))

    def soft_basis(self, w: object, beta: float) -> FloatArray:
        """A differentiable soft LP oracle (partition-structured matroids only; else raises)."""
        raise NotImplementedError(
            f"{type(self).__name__} has no differentiable soft oracle; the soft LP oracle "
            "(and the relaxation twins) are only defined for partition-structured matroids "
            "(UniformMatroid / PartitionMatroid). Use the greedy family for general matroids."
        )


class _PartitionStructured(Matroid):
    """Shared implementation for the partition-structured matroids (group/cap oracle)."""

    def rank(self) -> int:
        return sum(min(int(c), int(g.size)) for g, c in zip(self.groups(), self.caps(), strict=True))

    def is_independent(self, x: object, *, tol: float = 1e-9) -> bool:
        """Whether ``x in {0, 1}^n`` respects every group capacity.

        The capacity rule itself is the shared
        :func:`omnibias.discrete.matroid.partition_independent` kernel, so this
        optimization-lens matroid and the polytope-lens
        :class:`omnibias.combinatorics.PartitionMatroid` cannot drift apart on it.
        """
        return partition_independent(_selected_indices(x), self.groups(), self.caps())

    def soft_basis(self, w: object, beta: float) -> FloatArray:
        r"""A differentiable soft LP oracle hardening to :meth:`max_weight_basis`.

        For each group of capacity ``cap``, the threshold ``tau`` is the midpoint of the
        ``cap``-th and ``(cap+1)``-th largest weights, so ``sigmoid(beta (w - tau))`` selects
        the top ``cap`` as ``beta -> inf`` (``sum ~ cap``). Groups with ``cap >= |group|``
        select everything.
        """
        wv = np.asarray(w, dtype=float).reshape(-1)
        y = np.zeros(self.n, dtype=float)
        for g, c in zip(self.groups(), self.caps(), strict=True):
            cap = min(int(c), int(g.size))
            wg = wv[g]
            if cap <= 0:
                continue
            if cap >= g.size:
                y[g] = 1.0
                continue
            sorted_desc = np.sort(wg)[::-1]
            tau = 0.5 * (sorted_desc[cap - 1] + sorted_desc[cap])
            y[g] = _sigmoid(beta * (wg - tau))
        return y


@dataclass(frozen=True)
class UniformMatroid(_PartitionStructured):
    r"""The cardinality (uniform) matroid: independent sets are those with ``|S| <= k``."""

    n_ground: int
    k: int

    def __post_init__(self) -> None:
        if self.n_ground < 1:
            raise ValueError("n_ground must be >= 1")
        if not 0 <= self.k <= self.n_ground:
            raise ValueError(f"k must be in [0, {self.n_ground}], got {self.k}")

    @property
    def n(self) -> int:
        return int(self.n_ground)

    def groups(self) -> list[IntArray]:
        return [np.arange(self.n_ground, dtype=np.int64)]

    def caps(self) -> list[int]:
        return [int(self.k)]


@dataclass(frozen=True)
class PartitionMatroid(_PartitionStructured):
    r"""A partition matroid: ``|S cap group_g| <= cap_g`` for a partition of ``[n]``.

    ``groups_`` is a sequence of disjoint index lists that together cover ``[n]``;
    ``caps_`` are the aligned per-group capacities (``0 <= cap_g <= |group_g|``).
    """

    groups_: Sequence[Sequence[int]]
    caps_: Sequence[int]

    def __post_init__(self) -> None:
        groups = [np.asarray(sorted(int(i) for i in g), dtype=np.int64) for g in self.groups_]
        caps = [int(c) for c in self.caps_]
        if len(groups) != len(caps):
            raise ValueError("groups_ and caps_ must have the same length")
        if not groups:
            raise ValueError("need at least one group")
        flat = np.concatenate(groups) if groups else np.array([], dtype=np.int64)
        n = int(flat.size)
        if sorted(flat.tolist()) != list(range(n)):
            raise ValueError("groups_ must be a partition of [0, n) with no gaps or repeats")
        for g, c in zip(groups, caps, strict=True):
            if not 0 <= c <= int(g.size):
                raise ValueError(f"each cap must be in [0, |group|]; got cap {c} for size {g.size}")
        object.__setattr__(self, "_groups", groups)
        object.__setattr__(self, "_caps", caps)
        object.__setattr__(self, "_n", n)

    @property
    def n(self) -> int:
        return int(self._n)  # type: ignore[attr-defined]

    def groups(self) -> list[IntArray]:
        return list(self._groups)  # type: ignore[attr-defined]

    def caps(self) -> list[int]:
        return list(self._caps)  # type: ignore[attr-defined]


class LaminarMatroid(Matroid):
    r"""A laminar matroid: ``|S cap A| <= cap_A`` for every set ``A`` of a laminar family.

    A *laminar* family is a collection of subsets of ``[n]`` in which any two sets are either
    disjoint or nested (one contains the other). Each set ``A`` carries a capacity ``cap_A``;
    ``S`` is independent iff it obeys every capacity. Partition matroids are the disjoint
    special case; the generic greedy oracle is exact (laminar families are matroids).
    """

    def __init__(
        self, constraint_sets: Sequence[Sequence[int]], caps: Sequence[int], n: int
    ) -> None:
        if n < 1:
            raise ValueError("n must be >= 1")
        sets = [np.asarray(sorted({int(i) for i in s}), dtype=np.int64) for s in constraint_sets]
        cap_list = [int(c) for c in caps]
        if len(sets) != len(cap_list):
            raise ValueError("constraint_sets and caps must have the same length")
        for a in sets:
            if a.size and (int(a[0]) < 0 or int(a[-1]) >= n):
                raise ValueError(f"constraint set {a.tolist()} has an index outside [0, {n})")
        for c in cap_list:
            if c < 0:
                raise ValueError("capacities must be nonnegative")
        for i in range(len(sets)):  # laminar: pairwise disjoint or nested
            si = set(sets[i].tolist())
            for j in range(i + 1, len(sets)):
                sj = set(sets[j].tolist())
                inter = si & sj
                if inter and not (inter == si or inter == sj):
                    raise ValueError(
                        "constraint_sets must form a laminar family (pairwise disjoint or nested)"
                    )
        self._n = int(n)
        self._sets: list[IntArray] = sets
        self._caps: list[int] = cap_list

    @property
    def n(self) -> int:
        return self._n

    def is_independent(self, x: object, *, tol: float = 1e-9) -> bool:
        xv = np.asarray(x, dtype=float).reshape(-1)
        for a, c in zip(self._sets, self._caps, strict=True):
            if a.size and float(np.sum(xv[a])) > float(c) + tol:
                return False
        return True


class GraphicMatroid(Matroid):
    r"""The graphic (cycle) matroid of a graph: the ground set is the edges, forests are independent.

    ``edges[i] = (u, v)`` is the ``i``-th edge over ``n_vertices`` vertices; the ground set is
    the ``m`` edges (so ``n = m``). A subset of edges is independent iff it is acyclic (a
    forest). The generic max-weight basis is exactly Kruskal's maximum-weight spanning forest.
    """

    def __init__(self, edges: Sequence[tuple[int, int]], n_vertices: int) -> None:
        if n_vertices < 1:
            raise ValueError("n_vertices must be >= 1")
        edge_list = [(int(u), int(v)) for u, v in edges]
        for u, v in edge_list:
            if not (0 <= u < n_vertices and 0 <= v < n_vertices):
                raise ValueError(f"edge ({u}, {v}) has an endpoint outside [0, {n_vertices})")
        self._edges = edge_list
        self._n_vertices = int(n_vertices)

    @property
    def n(self) -> int:
        return len(self._edges)

    def is_independent(self, x: object, *, tol: float = 1e-9) -> bool:
        # Acyclicity is the shared omnibias.discrete.matroid.graphic_independent kernel
        # (union-find), so this and omnibias.combinatorics.GraphicMatroid agree by
        # construction on which edge subsets are forests.
        return graphic_independent(_selected_indices(x), self._edges, self._n_vertices)


def _matching_size(selected: list[int], neighbors: list[list[int]], n_resources: int) -> int:
    """Maximum bipartite matching of ``selected`` elements into resources (augmenting paths)."""
    match_of: list[int] = [-1] * n_resources  # resource -> element

    def augment(i: int, seen: list[bool]) -> bool:
        for r in neighbors[i]:
            if not seen[r]:
                seen[r] = True
                if match_of[r] == -1 or augment(match_of[r], seen):
                    match_of[r] = i
                    return True
        return False

    matched = 0
    for i in selected:
        if augment(i, [False] * n_resources):
            matched += 1
    return matched


class TransversalMatroid(Matroid):
    r"""A transversal matroid: subsets of one side of a bipartite graph that can be matched.

    ``neighbors[i]`` lists the resource nodes that ground element ``i`` can be matched to
    (over ``n_resources`` resources). A subset ``S`` is independent iff there is a matching
    saturating ``S`` (a *system of distinct representatives*). Transversal matroids are
    matroids, so the generic greedy oracle is exact.
    """

    def __init__(self, neighbors: Sequence[Sequence[int]], n_resources: int) -> None:
        if n_resources < 0:
            raise ValueError("n_resources must be >= 0")
        nbrs = [sorted({int(r) for r in nb}) for nb in neighbors]
        for nb in nbrs:
            for r in nb:
                if not 0 <= r < n_resources:
                    raise ValueError(f"resource {r} out of range for n_resources={n_resources}")
        self._neighbors: list[list[int]] = nbrs
        self._n_resources = int(n_resources)

    @property
    def n(self) -> int:
        return len(self._neighbors)

    def is_independent(self, x: object, *, tol: float = 1e-9) -> bool:
        xv = np.asarray(x, dtype=float).reshape(-1)
        selected = [i for i in range(len(self._neighbors)) if xv[i] > 0.5]
        if not selected:
            return True
        return _matching_size(selected, self._neighbors, self._n_resources) == len(selected)


class MatroidIntersection(Matroid):
    r"""The intersection of ``p`` matroids: ``S`` is independent iff it is in *every* matroid.

    Maximizing a monotone submodular ``f`` over a ``p``-matroid intersection is handled by
    :func:`~omnibias.submodular.p_matroid_greedy` with the a-priori ``1/(p+1)`` guarantee.
    The generic (single-matroid) greedy is *not* an exact linear oracle for an intersection,
    so :meth:`max_weight_basis` instead computes the exact max-weight common independent set by
    enumeration (small ``n``) -- keeping the marginal certificate sound -- and raises above the
    brute-force cap.
    """

    def __init__(self, matroids: Sequence[Matroid]) -> None:
        mats = list(matroids)
        if not mats:
            raise ValueError("need at least one matroid")
        n0 = mats[0].n
        for m in mats:
            if m.n != n0:
                raise ValueError(f"all matroids must share the ground-set size; {m.n} != {n0}")
        self._matroids = mats
        self._n = int(n0)

    @property
    def n(self) -> int:
        return self._n

    @property
    def matroids(self) -> list[Matroid]:
        """The constituent matroids (``p = len(matroids)``)."""
        return list(self._matroids)

    def is_independent(self, x: object, *, tol: float = 1e-9) -> bool:
        return all(m.is_independent(x, tol=tol) for m in self._matroids)

    def max_weight_basis(self, w: object) -> FloatArray:
        r"""Exact max-weight common independent set by enumeration (sound; small ``n`` only)."""
        n = self._n
        if n > _MAX_EXACT_INTERSECTION_N:
            raise NotImplementedError(
                "no poly-time exact weighted matroid-intersection oracle here; "
                f"n={n} exceeds the {_MAX_EXACT_INTERSECTION_N} brute-force cap. Maximize with "
                "p_matroid_greedy (a-priori 1/(p+1)); the marginal certificate needs a single "
                "matroid or a small intersection."
            )
        wv = np.asarray(w, dtype=float).reshape(-1)
        idx = np.arange(1 << n, dtype=np.int64)
        bits = ((idx[:, None] >> np.arange(n)[None, :]) & 1).astype(float)
        values = bits @ wv
        for m in np.argsort(-values, kind="stable"):  # first independent in desc order is the max
            x = bits[int(m)]
            if self.is_independent(x):
                return np.asarray(x, dtype=float)
        return np.zeros(n, dtype=float)


__all__ = [
    "GraphicMatroid",
    "LaminarMatroid",
    "Matroid",
    "MatroidIntersection",
    "PartitionMatroid",
    "TransversalMatroid",
    "UniformMatroid",
]
