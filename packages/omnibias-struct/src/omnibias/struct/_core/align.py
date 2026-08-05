# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Sequence-alignment lattice: Needleman-Wunsch as a DAG + classic-DP / brute oracles.

Global pairwise alignment of ``a`` (length ``n``) and ``b`` (length ``m``) is a max-score
monotonic path through the ``(n+1) x (m+1)`` edit grid. From cell ``(i, j)`` (``i`` symbols
of ``a`` and ``j`` of ``b`` consumed) a path steps:

* **diagonal** ``(i+1, j+1)`` -- align ``a[i]`` with ``b[j]``, score ``sub[a[i], b[j]]``;
* **down** ``(i+1, j)`` -- a gap in ``b`` (consume ``a[i]``), score ``gap``;
* **right** ``(i, j+1)`` -- a gap in ``a`` (consume ``b[j]``), score ``gap``.

This is exactly a longest-path DAG, so the differentiable alignment *reuses the shared
shortest-path substrate* (negate scores -> costs): the ``beta -> inf`` softmax anneals to
the Needleman-Wunsch optimum, the ``delta -> 0`` tower gives closed-form edge marginals, and
the substitution matrix / gap penalty are learnable. :func:`hard_align` is the classic NW
DP; :func:`brute_force_align` enumerates every alignment (the oracle both are pinned to).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from omnibias.struct._core.trellis import DAG

FloatArray = NDArray[np.float64]

# An edge label: ("sub", i, j) aligns a[i] with b[j]; ("gap",) is an indel.
EdgeLabel = tuple[str, int, int]


@dataclass(frozen=True)
class AlignmentLattice:
    r"""The ``(len_a + 1) x (len_b + 1)`` Needleman-Wunsch edit grid (diag / down / right)."""

    len_a: int
    len_b: int

    def __post_init__(self) -> None:
        if self.len_a < 1 or self.len_b < 1:
            raise ValueError(f"sequences must be non-empty, got {self.len_a}, {self.len_b}")

    @property
    def num_nodes(self) -> int:
        """Number of grid cells ``(len_a + 1) (len_b + 1)``."""
        return (self.len_a + 1) * (self.len_b + 1)

    def node(self, i: int, j: int) -> int:
        """Linear (topological) index of grid cell ``(i, j)``."""
        return i * (self.len_b + 1) + j

    def build_dag(self) -> tuple[DAG, dict[tuple[int, int], EdgeLabel]]:
        """Return the alignment :class:`DAG` (placeholder weights) + its edge-label map."""
        n, m = self.len_a, self.len_b
        edges: dict[tuple[int, int], float] = {}
        labels: dict[tuple[int, int], EdgeLabel] = {}
        for i in range(n + 1):
            for j in range(m + 1):
                u = self.node(i, j)
                if i < n and j < m:
                    e = (u, self.node(i + 1, j + 1))
                    edges[e], labels[e] = 0.0, ("sub", i, j)
                if i < n:
                    e = (u, self.node(i + 1, j))
                    edges[e], labels[e] = 0.0, ("gap", i, -1)
                if j < m:
                    e = (u, self.node(i, j + 1))
                    edges[e], labels[e] = 0.0, ("gap", -1, j)
        dag = DAG(self.num_nodes, edges, source=self.node(0, 0), sink=self.node(n, m))
        return dag, labels

    def edge_score(self, label: EdgeLabel, a: NDArray[np.int_], b: NDArray[np.int_], sub: FloatArray, gap: float) -> float:
        """Score of one edge given the substitution matrix and gap penalty."""
        kind, i, j = label
        if kind == "sub":
            return float(sub[int(a[i]), int(b[j])])
        return float(gap)

    def path_score(self, path: object, a: NDArray[np.int_], b: NDArray[np.int_], sub: FloatArray, gap: float) -> float:
        """Total alignment score of a ``source -> sink`` node path."""
        nodes = [int(x) for x in np.asarray(path, dtype=int).reshape(-1)]
        _, labels = self.build_dag()
        return float(sum(self.edge_score(labels[(nodes[k], nodes[k + 1])], a, b, sub, gap) for k in range(len(nodes) - 1)))


def hard_align(a: object, b: object, sub: FloatArray, gap: float) -> float:
    r"""Classic Needleman-Wunsch optimal global-alignment score (max-score DP)."""
    ai = np.asarray(a, dtype=int).reshape(-1)
    bj = np.asarray(b, dtype=int).reshape(-1)
    s = np.asarray(sub, dtype=float)
    n, m = ai.shape[0], bj.shape[0]
    h = np.full((n + 1, m + 1), -np.inf)
    h[0, 0] = 0.0
    for i in range(n + 1):
        for j in range(m + 1):
            if i == 0 and j == 0:
                continue
            best = -np.inf
            if i > 0 and j > 0:
                best = max(best, h[i - 1, j - 1] + s[ai[i - 1], bj[j - 1]])
            if i > 0:
                best = max(best, h[i - 1, j] + gap)
            if j > 0:
                best = max(best, h[i, j - 1] + gap)
            h[i, j] = best
    return float(h[n, m])


def brute_force_align(a: object, b: object, sub: FloatArray, gap: float) -> float:
    r"""Exact optimal alignment score by enumerating every monotonic path (oracle)."""
    ai = np.asarray(a, dtype=int).reshape(-1)
    bj = np.asarray(b, dtype=int).reshape(-1)
    lattice = AlignmentLattice(ai.shape[0], bj.shape[0])
    dag, _ = lattice.build_dag()
    return max(lattice.path_score(p, ai, bj, np.asarray(sub, dtype=float), gap) for p in dag.enumerate_paths())


def brute_force_soft_align(a: object, b: object, sub: FloatArray, gap: float, beta: float) -> float:
    r"""Exact soft alignment ``beta^-1 log sum_paths exp(beta score)`` (global softmax)."""
    if beta <= 0.0:
        raise ValueError(f"beta must be > 0, got {beta}")
    ai = np.asarray(a, dtype=int).reshape(-1)
    bj = np.asarray(b, dtype=int).reshape(-1)
    lattice = AlignmentLattice(ai.shape[0], bj.shape[0])
    dag, _ = lattice.build_dag()
    s = np.asarray(sub, dtype=float)
    scores = np.array([lattice.path_score(p, ai, bj, s, gap) for p in dag.enumerate_paths()])
    mx = float(np.max(scores))
    return mx + math.log(float(np.sum(np.exp(beta * (scores - mx))))) / beta


# ---------------------------------------------------------------------------
# Smith-Waterman local alignment: free start / end on the shared substrate
# ---------------------------------------------------------------------------


def _edge_score(label: EdgeLabel, a: NDArray[np.int_], b: NDArray[np.int_], sub: FloatArray, gap: float) -> float:
    r"""Score of a labelled alignment edge (``sub`` / ``gap`` / free / affine open / extend)."""
    kind, i, j = label
    if kind == "sub":
        return float(sub[int(a[i]), int(b[j])])
    if kind == "gap":
        return float(gap)
    return 0.0  # a "free" 0-edge (local start / end or final-state choice)


def build_local_dag(len_a: int, len_b: int) -> tuple[DAG, dict[tuple[int, int], EdgeLabel]]:
    r"""Smith-Waterman local-alignment DAG: the NW grid plus free ``source -> cell`` and
    ``cell -> sink`` 0-edges (and an empty ``source -> sink`` edge, the score-``0`` floor).

    A longest ``source -> sink`` path therefore starts and ends at *any* cell for free, i.e.
    scores the best local (substring) alignment -- exactly Smith-Waterman -- on the same
    shortest-path substrate as :class:`AlignmentLattice`.
    """
    if len_a < 1 or len_b < 1:
        raise ValueError(f"sequences must be non-empty, got {len_a}, {len_b}")
    n, m = len_a, len_b

    def cell(i: int, j: int) -> int:
        return 1 + i * (m + 1) + j

    source = 0
    sink = 1 + (n + 1) * (m + 1)
    edges: dict[tuple[int, int], float] = {}
    labels: dict[tuple[int, int], EdgeLabel] = {}
    free: EdgeLabel = ("free", -1, -1)
    for i in range(n + 1):
        for j in range(m + 1):
            u = cell(i, j)
            if i < n and j < m:
                e = (u, cell(i + 1, j + 1))
                edges[e], labels[e] = 0.0, ("sub", i, j)
            if i < n:
                e = (u, cell(i + 1, j))
                edges[e], labels[e] = 0.0, ("gap", i, -1)
            if j < m:
                e = (u, cell(i, j + 1))
                edges[e], labels[e] = 0.0, ("gap", -1, j)
            edges[(source, u)], labels[(source, u)] = 0.0, free  # free local start
            edges[(u, sink)], labels[(u, sink)] = 0.0, free  # free local end
    edges[(source, sink)], labels[(source, sink)] = 0.0, free  # empty alignment (score-0 floor)
    dag = DAG(sink + 1, edges, source=source, sink=sink)
    return dag, labels


def hard_local_align(a: object, b: object, sub: FloatArray, gap: float) -> float:
    r"""Classic Smith-Waterman optimal local-alignment score (max-score DP with a ``0`` floor)."""
    ai = np.asarray(a, dtype=int).reshape(-1)
    bj = np.asarray(b, dtype=int).reshape(-1)
    s = np.asarray(sub, dtype=float)
    n, m = ai.shape[0], bj.shape[0]
    h = np.zeros((n + 1, m + 1))
    best = 0.0
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            h[i, j] = max(
                0.0,
                h[i - 1, j - 1] + s[ai[i - 1], bj[j - 1]],
                h[i - 1, j] + gap,
                h[i, j - 1] + gap,
            )
            best = max(best, float(h[i, j]))
    return best


def _local_path_scores(a: object, b: object, sub: FloatArray, gap: float) -> FloatArray:
    ai = np.asarray(a, dtype=int).reshape(-1)
    bj = np.asarray(b, dtype=int).reshape(-1)
    s = np.asarray(sub, dtype=float)
    dag, labels = build_local_dag(ai.shape[0], bj.shape[0])
    out: list[float] = []
    for path in dag.enumerate_paths():
        nodes = [int(x) for x in np.asarray(path, dtype=int).reshape(-1)]
        out.append(
            float(sum(_edge_score(labels[(nodes[k], nodes[k + 1])], ai, bj, s, gap) for k in range(len(nodes) - 1)))
        )
    return np.array(out, dtype=float)


def brute_force_local_align(a: object, b: object, sub: FloatArray, gap: float) -> float:
    r"""Exact local-alignment score by enumerating every augmented ``source -> sink`` path."""
    return float(np.max(_local_path_scores(a, b, sub, gap)))


def brute_force_soft_local_align(a: object, b: object, sub: FloatArray, gap: float, beta: float) -> float:
    r"""Exact soft local alignment ``lse_beta`` over the augmented lattice paths."""
    if beta <= 0.0:
        raise ValueError(f"beta must be > 0, got {beta}")
    scores = _local_path_scores(a, b, sub, gap)
    mx = float(np.max(scores))
    return mx + math.log(float(np.sum(np.exp(beta * (scores - mx))))) / beta


# ---------------------------------------------------------------------------
# Gotoh affine gaps: a 3-state (M / Ix / Iy) lattice
# ---------------------------------------------------------------------------

# An affine edge label: ("sub", i, j) match; ("open",) a gap-open (open + extend);
# ("extend",) a gap-extend; ("free",) a 0 start/final-state edge.
AffineLabel = tuple[str, int, int]


def build_gotoh_dag(len_a: int, len_b: int) -> tuple[DAG, dict[tuple[int, int], AffineLabel]]:
    r"""Gotoh affine-gap DAG over match / x-gap / y-gap states (arity-1, topological).

    States ``M[i, j]`` (match ``a[i-1]`` with ``b[j-1]``), ``Ix[i, j]`` (gap in ``b``,
    consuming ``a[i-1]``), ``Iy[i, j]`` (gap in ``a``). A gap costs ``open + extend`` to start
    (an ``("open",)`` edge, taken from ``M`` *or the other gap state*) and ``extend`` to
    continue (an ``("extend",)`` edge). A longest ``source -> sink`` path is the Gotoh optimum;
    on the shared shortest-path substrate the soft twin / marginals come for free.
    """
    if len_a < 1 or len_b < 1:
        raise ValueError(f"sequences must be non-empty, got {len_a}, {len_b}")
    n, m = len_a, len_b
    nodes: dict[tuple[str, int, int], bool] = {}

    def m_ok(i: int, j: int) -> bool:
        return (i, j) == (0, 0) or (i >= 1 and j >= 1)

    for i in range(n + 1):
        for j in range(m + 1):
            if m_ok(i, j):
                nodes[("M", i, j)] = True
            if i >= 1:
                nodes[("Ix", i, j)] = True
            if j >= 1:
                nodes[("Iy", i, j)] = True
    ordered = sorted(nodes, key=lambda k: (k[1] + k[2], {"M": 0, "Ix": 1, "Iy": 2}[k[0]], k[1]))
    node_id: dict[tuple[str, int, int], int] = {}
    idx = 1  # 0 reserved for the source
    for key in ordered:
        node_id[key] = idx
        idx += 1
    source = 0
    sink = idx
    edges: dict[tuple[int, int], float] = {}
    labels: dict[tuple[int, int], AffineLabel] = {}
    free: AffineLabel = ("free", -1, -1)
    edges[(source, node_id[("M", 0, 0)])] = 0.0
    labels[(source, node_id[("M", 0, 0)])] = free
    for (state, i, j), nid in node_id.items():
        if state == "M":
            if (i, j) == (0, 0):
                continue
            lab: AffineLabel = ("sub", i - 1, j - 1)
            for pred in (("M", i - 1, j - 1), ("Ix", i - 1, j - 1), ("Iy", i - 1, j - 1)):
                if pred in node_id:
                    edges[(node_id[pred], nid)] = 0.0
                    labels[(node_id[pred], nid)] = lab
        elif state == "Ix":
            if ("M", i - 1, j) in node_id:
                edges[(node_id[("M", i - 1, j)], nid)] = 0.0
                labels[(node_id[("M", i - 1, j)], nid)] = ("open", -1, -1)
            if ("Ix", i - 1, j) in node_id:
                edges[(node_id[("Ix", i - 1, j)], nid)] = 0.0
                labels[(node_id[("Ix", i - 1, j)], nid)] = ("extend", -1, -1)
            if ("Iy", i - 1, j) in node_id:
                edges[(node_id[("Iy", i - 1, j)], nid)] = 0.0
                labels[(node_id[("Iy", i - 1, j)], nid)] = ("open", -1, -1)
        else:  # Iy
            if ("M", i, j - 1) in node_id:
                edges[(node_id[("M", i, j - 1)], nid)] = 0.0
                labels[(node_id[("M", i, j - 1)], nid)] = ("open", -1, -1)
            if ("Iy", i, j - 1) in node_id:
                edges[(node_id[("Iy", i, j - 1)], nid)] = 0.0
                labels[(node_id[("Iy", i, j - 1)], nid)] = ("extend", -1, -1)
            if ("Ix", i, j - 1) in node_id:
                edges[(node_id[("Ix", i, j - 1)], nid)] = 0.0
                labels[(node_id[("Ix", i, j - 1)], nid)] = ("open", -1, -1)
    for state in ("M", "Ix", "Iy"):
        key = (state, n, m)
        if key in node_id:
            edges[(node_id[key], sink)] = 0.0
            labels[(node_id[key], sink)] = free
    dag = DAG(sink + 1, edges, source=source, sink=sink)
    return dag, labels


def _affine_edge_score(
    label: AffineLabel, a: NDArray[np.int_], b: NDArray[np.int_], sub: FloatArray, gap_open: float, gap_extend: float
) -> float:
    kind, i, j = label
    if kind == "sub":
        return float(sub[int(a[i]), int(b[j])])
    if kind == "open":
        return float(gap_open + gap_extend)
    if kind == "extend":
        return float(gap_extend)
    return 0.0


def hard_gotoh(a: object, b: object, sub: FloatArray, gap_open: float, gap_extend: float) -> float:
    r"""Classic Gotoh optimal global affine-gap alignment score (3-matrix max DP).

    A gap of length ``L`` costs ``gap_open + L * gap_extend`` (one open per maximal run,
    including runs that immediately follow a gap of the other type).
    """
    ai = np.asarray(a, dtype=int).reshape(-1)
    bj = np.asarray(b, dtype=int).reshape(-1)
    s = np.asarray(sub, dtype=float)
    n, m = ai.shape[0], bj.shape[0]
    neg = -np.inf
    o = gap_open + gap_extend
    mm = np.full((n + 1, m + 1), neg)
    ix = np.full((n + 1, m + 1), neg)
    iy = np.full((n + 1, m + 1), neg)
    mm[0, 0] = 0.0
    for i in range(n + 1):
        for j in range(m + 1):
            if i >= 1 and j >= 1:
                mm[i, j] = s[ai[i - 1], bj[j - 1]] + max(mm[i - 1, j - 1], ix[i - 1, j - 1], iy[i - 1, j - 1])
            if i >= 1:
                ix[i, j] = max(mm[i - 1, j] + o, ix[i - 1, j] + gap_extend, iy[i - 1, j] + o)
            if j >= 1:
                iy[i, j] = max(mm[i, j - 1] + o, iy[i, j - 1] + gap_extend, ix[i, j - 1] + o)
    return float(max(mm[n, m], ix[n, m], iy[n, m]))


def _gotoh_path_scores(
    a: object, b: object, sub: FloatArray, gap_open: float, gap_extend: float
) -> FloatArray:
    ai = np.asarray(a, dtype=int).reshape(-1)
    bj = np.asarray(b, dtype=int).reshape(-1)
    s = np.asarray(sub, dtype=float)
    dag, labels = build_gotoh_dag(ai.shape[0], bj.shape[0])
    out: list[float] = []
    for path in dag.enumerate_paths():
        nodes = [int(x) for x in np.asarray(path, dtype=int).reshape(-1)]
        out.append(
            float(
                sum(
                    _affine_edge_score(labels[(nodes[k], nodes[k + 1])], ai, bj, s, gap_open, gap_extend)
                    for k in range(len(nodes) - 1)
                )
            )
        )
    return np.array(out, dtype=float)


def brute_force_gotoh(a: object, b: object, sub: FloatArray, gap_open: float, gap_extend: float) -> float:
    r"""Exact affine-gap score by enumerating every ``source -> sink`` path of the Gotoh DAG."""
    return float(np.max(_gotoh_path_scores(a, b, sub, gap_open, gap_extend)))


def brute_force_soft_gotoh(
    a: object, b: object, sub: FloatArray, gap_open: float, gap_extend: float, beta: float
) -> float:
    r"""Exact soft affine-gap alignment ``lse_beta`` over the Gotoh-lattice paths."""
    if beta <= 0.0:
        raise ValueError(f"beta must be > 0, got {beta}")
    scores = _gotoh_path_scores(a, b, sub, gap_open, gap_extend)
    mx = float(np.max(scores))
    return mx + math.log(float(np.sum(np.exp(beta * (scores - mx))))) / beta


__all__ = [
    "AffineLabel",
    "AlignmentLattice",
    "EdgeLabel",
    "brute_force_align",
    "brute_force_gotoh",
    "brute_force_local_align",
    "brute_force_soft_align",
    "brute_force_soft_gotoh",
    "brute_force_soft_local_align",
    "build_gotoh_dag",
    "build_local_dag",
    "hard_align",
    "hard_gotoh",
    "hard_local_align",
]
