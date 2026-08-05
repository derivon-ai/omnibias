# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Exact hard dynamic programming (``max`` / ``min`` semiring) and brute-force oracles.

The hard DP is the ``beta -> inf`` limit that the soft (``lse_beta``) backends anneal
towards: :func:`viterbi` (max-plus best path), :func:`shortest_path` (min-plus), and
:func:`ctc_best` (best CTC alignment). Each is validated against a ``brute_force_*``
oracle that enumerates *every* path / alignment -- the ground truth used by the
certificate's ``agrees_with_bruteforce`` self-check on small instances. All pure numpy
(no backend import); intended for tiny instances (the oracles are exponential).
"""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray
from omnibias.struct._core.trellis import DAG, ChainTrellis, CTCLattice

FloatArray = NDArray[np.float64]


def _logsumexp(scores: FloatArray, beta: float) -> float:
    r"""Stable ``beta^-1 log sum_i exp(beta scores_i)`` (returns ``-inf`` if all ``-inf``)."""
    m = float(np.max(scores))
    if not math.isfinite(m):
        return m
    return m + math.log(float(np.sum(np.exp(beta * (scores - m))))) / beta


# ---------------------------------------------------------------------------
# Viterbi (linear-chain, max-plus)
# ---------------------------------------------------------------------------


def viterbi(trellis: ChainTrellis) -> tuple[float, tuple[int, ...]]:
    r"""Best (max-score) state path of a :class:`ChainTrellis` and its score."""
    emissions, transitions, start = trellis.emissions, trellis.transitions, trellis.start
    n_steps, n_states = trellis.n_steps, trellis.n_states
    v = start + emissions[0]
    back = np.zeros((n_steps, n_states), dtype=int)
    for t in range(1, n_steps):
        scores = v[:, None] + transitions  # (S_prev, S)
        best_prev = np.argmax(scores, axis=0)
        back[t] = best_prev
        v = emissions[t] + scores[best_prev, np.arange(n_states)]
    last = int(np.argmax(v))
    value = float(v[last])
    path = [last]
    for t in range(n_steps - 1, 0, -1):
        last = int(back[t, last])
        path.append(last)
    path.reverse()
    return value, tuple(path)


def brute_force_viterbi(trellis: ChainTrellis) -> tuple[float, tuple[int, ...]]:
    r"""Exact best path by enumerating all ``S ** T`` state sequences (oracle)."""
    best_value: float = -math.inf
    best_path: tuple[int, ...] = ()
    for path in trellis.enumerate_paths():
        value = trellis.path_score(path)
        if value > best_value:
            best_value, best_path = value, path
    return best_value, best_path


# ---------------------------------------------------------------------------
# Shortest path (DAG, min-plus)
# ---------------------------------------------------------------------------


def shortest_path(dag: DAG) -> tuple[float, tuple[int, ...]]:
    r"""Minimum-cost ``source -> sink`` path of a :class:`DAG` and its cost."""
    weights = dag.weight_matrix()
    dist = np.full(dag.num_nodes, np.inf)
    dist[dag.source] = 0.0
    back = [-1] * dag.num_nodes
    for v in range(dag.num_nodes):
        if v == dag.source:
            dist[v] = 0.0
            continue
        preds = dag.incoming(v)
        if not preds:
            continue
        vals = [dist[u] + weights[u, v] for u in preds]
        j = int(np.argmin(vals))
        dist[v], back[v] = vals[j], preds[j]
    cost = float(dist[dag.sink])
    path = [dag.sink]
    node = dag.sink
    while node != dag.source and back[node] != -1:
        node = back[node]
        path.append(node)
    path.reverse()
    return cost, tuple(path)


def brute_force_shortest_path(dag: DAG) -> tuple[float, tuple[int, ...]]:
    r"""Exact minimum-cost path by enumerating all ``source -> sink`` paths (oracle)."""
    best_cost: float = math.inf
    best_path: tuple[int, ...] = ()
    for path in dag.enumerate_paths():
        cost = dag.path_cost(path)
        if cost < best_cost:
            best_cost, best_path = cost, path
    return best_cost, best_path


# ---------------------------------------------------------------------------
# CTC (blank-augmented alignment lattice)
# ---------------------------------------------------------------------------


def ctc_best_alignment(lattice: CTCLattice, log_probs: object) -> tuple[float, tuple[int, ...]]:
    r"""Best CTC alignment (max-score traceback) and its score for ``log_probs`` ``(T, C)``.

    Returns ``(value, alignment)`` where ``alignment`` is the length-``T`` class-id
    sequence that attains the best score; ``lattice.collapse(alignment)`` equals
    ``lattice.targets``. The Viterbi analogue of :func:`ctc_best`, exposing the traceback
    (which :func:`ctc_best` discards) for parity with :func:`viterbi` / :func:`shortest_path`.
    """
    lp = np.asarray(log_probs, dtype=float)
    n_steps = lp.shape[0]
    ext = lattice.extended_labels()
    m = int(ext.shape[0])
    f = np.full((n_steps, m), -math.inf)
    back = np.full((n_steps, m), -1, dtype=int)
    f[0, 0] = lp[0, ext[0]]
    if m > 1:
        f[0, 1] = lp[0, ext[1]]
    for t in range(1, n_steps):
        for s in range(m):
            preds = lattice.incoming(s)
            j = max(preds, key=lambda p: f[t - 1, p])
            if f[t - 1, j] == -math.inf:
                continue
            f[t, s] = lp[t, ext[s]] + f[t - 1, j]
            back[t, s] = j
    ends = [m - 1] + ([m - 2] if m >= 2 else [])
    best_end = max(ends, key=lambda s: f[n_steps - 1, s])
    value = float(f[n_steps - 1, best_end])
    positions = [best_end]
    for t in range(n_steps - 1, 0, -1):
        positions.append(int(back[t, positions[-1]]))
    positions.reverse()
    alignment = tuple(int(ext[s]) for s in positions)
    return value, alignment


def ctc_best(lattice: CTCLattice, log_probs: object) -> float:
    r"""Best single-alignment score (max over CTC alignments) for ``log_probs`` ``(T, C)``."""
    return ctc_best_alignment(lattice, log_probs)[0]


def brute_force_ctc(lattice: CTCLattice, log_probs: object) -> float:
    r"""Exact best CTC alignment score by enumerating all ``C ** T`` label sequences."""
    lp = np.asarray(log_probs, dtype=float)
    n_steps, num_classes = lp.shape
    target = tuple(int(y) for y in lattice.targets)
    best = -math.inf
    for idx in range(num_classes**n_steps):
        seq, rem = [], idx
        for _ in range(n_steps):
            seq.append(rem % num_classes)
            rem //= num_classes
        if lattice.collapse(seq) == target:
            score = float(sum(lp[t, seq[t]] for t in range(n_steps)))
            best = max(best, score)
    return best


# ---------------------------------------------------------------------------
# Brute-force soft log-partition (validates the soft-DP recursion)
# ---------------------------------------------------------------------------


def brute_force_partition(
    problem: ChainTrellis | DAG | CTCLattice,
    beta: float,
    log_probs: object | None = None,
) -> float:
    r"""Exact soft value ``beta^-1 log sum_paths exp(beta score)`` over every path.

    The ground truth for the soft-DP recursion. For a :class:`DAG` the "score" is the
    negated cost (max convention), so the soft *cost* returned by
    :func:`omnibias.struct.torch.soft_shortest_path` is its negation. For a
    :class:`CTCLattice` pass ``log_probs`` ``(T, C)``.
    """
    if beta <= 0.0:
        raise ValueError(f"beta must be > 0, got {beta}")
    if isinstance(problem, ChainTrellis):
        scores = np.array([problem.path_score(p) for p in problem.enumerate_paths()])
        return _logsumexp(scores, beta)
    if isinstance(problem, DAG):
        scores = np.array([-problem.path_cost(p) for p in problem.enumerate_paths()])
        return _logsumexp(scores, beta)
    lp = np.asarray(log_probs, dtype=float)
    n_steps, num_classes = lp.shape
    target = tuple(int(y) for y in problem.targets)
    scores_list: list[float] = []
    for idx in range(num_classes**n_steps):
        seq, rem = [], idx
        for _ in range(n_steps):
            seq.append(rem % num_classes)
            rem //= num_classes
        if problem.collapse(seq) == target:
            scores_list.append(float(sum(lp[t, seq[t]] for t in range(n_steps))))
    return _logsumexp(np.array(scores_list), beta)


__all__ = [
    "brute_force_ctc",
    "brute_force_partition",
    "brute_force_shortest_path",
    "brute_force_viterbi",
    "ctc_best",
    "ctc_best_alignment",
    "shortest_path",
    "viterbi",
]
