# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Overflow-free log-space path counting -- the ``log N`` in the gap bound.

:func:`count_paths` returns the *exact* integer number of complete paths, which for a
dense chain is ``S ** T`` -- an astronomically large Python bignum for realistic sizes.
Feeding that to ``log(N) / beta`` works (CPython's ``math.log`` accepts arbitrary-size
ints), but it first materialises the full bignum. :func:`log_num_paths` instead runs the
count DP in the log domain (``logaddexp`` of log-counts), so it returns a finite float
directly -- no bignum, no float overflow to ``inf``, and it degrades to ``-inf`` for an
infeasible problem (zero complete paths) rather than raising.

This is the ``N`` used by :func:`omnibias.struct.logsumexp_gap_bound`; the certificate can
take ``log_num_paths`` in place of ``log(count_paths(...))`` for large instances.
"""

from __future__ import annotations

import math

from omnibias.struct._core.trellis import DAG, ChainTrellis, CTCLattice


def _logaddexp(values: list[float]) -> float:
    r"""Numerically stable ``log sum_i exp(values_i)`` (``-inf`` if all terms are ``-inf``)."""
    finite = [v for v in values if v != -math.inf]
    if not finite:
        return -math.inf
    m = max(finite)
    return m + math.log(sum(math.exp(v - m) for v in finite))


def log_num_paths(problem: ChainTrellis | DAG | CTCLattice, n_steps: int | None = None) -> float:
    r"""``log`` of the exact number of complete paths / alignments, via a log-space DP.

    Finite for arbitrarily large path counts (never overflows to ``inf`` and never
    materialises the integer count). Returns ``-inf`` for an infeasible problem (no
    complete paths -- e.g. a :class:`CTCLattice` whose ``n_steps`` is shorter than its
    label sequence). For a :class:`CTCLattice` the alignment length ``n_steps`` (``= T``)
    is required, matching :func:`omnibias.struct.count_paths`.
    """
    if isinstance(problem, ChainTrellis):
        return problem.n_steps * math.log(problem.n_states)
    if isinstance(problem, DAG):
        succ: dict[int, list[int]] = {}
        for u, v in problem.edges:
            succ.setdefault(u, []).append(v)
        log_counts = [-math.inf] * problem.num_nodes
        log_counts[problem.sink] = 0.0
        for node in range(problem.num_nodes - 1, -1, -1):
            if node == problem.sink:
                continue
            log_counts[node] = _logaddexp([log_counts[v] for v in succ.get(node, [])])
        return log_counts[problem.source]
    if n_steps is None:
        raise ValueError("log_num_paths on a CTCLattice requires n_steps (= T)")
    m = 2 * problem.n_labels + 1
    if n_steps < problem.n_labels:  # too short to emit every label -> infeasible
        return -math.inf
    log_counts = [-math.inf] * m
    log_counts[0] = 0.0
    if m > 1:
        log_counts[1] = 0.0
    for _ in range(1, n_steps):
        nxt = [-math.inf] * m
        for s in range(m):
            nxt[s] = _logaddexp([log_counts[p] for p in problem.incoming(s)])
        log_counts = nxt
    ends = [log_counts[m - 1]] + ([log_counts[m - 2]] if m >= 2 else [])
    return _logaddexp(ends)


__all__ = ["log_num_paths"]
