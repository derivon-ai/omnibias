# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""The closed-form log-sum-exp gap bound ``lse_beta - max <= log(N) / beta``.

For any finite multiset ``a`` of ``N`` scores,

.. math::
    \max_i a_i \;\le\; \mathrm{lse}_\beta(a)
        := \tfrac1\beta \log \sum_i e^{\beta a_i}
        \;\le\; \max_i a_i + \tfrac{\log N}{\beta}.

The left inequality is ``lse_beta >= max`` (the soft value never *under*-estimates the
hard optimum); the right is this module's gap bound. Applied to the ``N`` complete paths
of a DP problem it certifies, in closed form, how far the soft relaxation sits above the
hard optimum -- shrinking to zero as ``beta -> inf``. This is the *temperature-axis*
(``beta``) accuracy statement; it is distinct from, and machined by, the exact
``delta -> 0`` derivative tower that differentiates ``lse_beta``.
"""

from __future__ import annotations

import math

from omnibias.struct._core.trellis import DAG, ChainTrellis, CTCLattice


def logsumexp_gap_bound(num_paths: int, beta: float) -> float:
    r"""Closed-form upper bound on ``lse_beta - max``: ``log(num_paths) / beta``.

    Parameters
    ----------
    num_paths:
        The exact number ``N >= 1`` of complete paths (e.g. from
        :func:`omnibias.struct.count_paths`).
    beta:
        The inverse temperature ``beta > 0``.

    Returns
    -------
    float
        ``log(num_paths) / beta`` (``0`` when ``num_paths == 1``: a single path has no
        gap). A larger ``num_paths`` or smaller ``beta`` only *widens* the certified
        gap -- the bound is never optimistic.
    """
    if num_paths < 1:
        raise ValueError(f"num_paths must be >= 1, got {num_paths}")
    if beta <= 0.0:
        raise ValueError(f"beta must be > 0, got {beta}")
    return math.log(num_paths) / beta


def stepwise_gap_bound(
    problem: ChainTrellis | DAG | CTCLattice,
    beta: float,
    *,
    n_steps: int | None = None,
) -> float:
    r"""N-free sound gap bound: the critical-path sum of ``log(fan-in) / beta``.

    Each ``lse_beta`` reduction over ``k`` terms contributes at most ``log(k) / beta`` of
    slack, and the slacks compose additively along the *deepest* chain of reductions in
    the DP (leaves are exact). Hence

    .. math::
        V_\beta - V^* \;\le\; \max_{\text{paths}}\ \sum_{\text{reductions } r}
            \frac{\log(\mathrm{fanin}_r)}{\beta}.

    **Honesty note (a data-driven finding).** This is *not* tighter than the global
    ``log(N) / beta``. Because the forward path count obeys
    ``log(paths->v) <= log(indeg(v)) + max_{u -> v} log(paths->u)``, the critical-path sum
    provably dominates ``log N``; the two coincide only for series-parallel structure
    (e.g. a dense :class:`ChainTrellis`) and it is strictly *looser* for reconvergent
    DAGs / CTC lattices. Its real use is as an **N-free** certificate computed from fan-in
    alone when the path count is unavailable. :func:`certify_soft_dp` takes
    ``min(global, stepwise)``, so a sealed bound is never worse than ``log(N) / beta``.
    Sound for both the ``max`` (Viterbi / CTC) and ``min`` (shortest-path softmin) senses.
    """
    if beta <= 0.0:
        raise ValueError(f"beta must be > 0, got {beta}")
    if isinstance(problem, ChainTrellis):
        # T reductions of width S along every path (forward T-1 + final).
        return problem.n_steps * math.log(problem.n_states) / beta
    if isinstance(problem, DAG):
        preds: dict[int, list[int]] = {v: problem.incoming(v) for v in range(problem.num_nodes)}
        slack = [-math.inf] * problem.num_nodes
        slack[problem.source] = 0.0
        for v in range(problem.num_nodes):
            if v == problem.source or not preds[v]:
                continue
            best_child = max(slack[u] for u in preds[v])
            if best_child == -math.inf:
                continue
            slack[v] = math.log(len(preds[v])) / beta + best_child
        return max(slack[problem.sink], 0.0)
    if n_steps is None:
        raise ValueError("stepwise_gap_bound on a CTCLattice requires n_steps (= T)")
    m = 2 * problem.n_labels + 1
    if n_steps < problem.n_labels:
        raise ValueError("infeasible CTC instance: n_steps shorter than the label sequence")
    slack_t = [-math.inf] * m
    slack_t[0] = 0.0
    if m > 1:
        slack_t[1] = 0.0
    for _ in range(1, n_steps):
        nxt = [-math.inf] * m
        for s in range(m):
            preds_s = problem.incoming(s)
            best_child = max((slack_t[p] for p in preds_s), default=-math.inf)
            if best_child == -math.inf:
                continue
            nxt[s] = math.log(len(preds_s)) / beta + best_child
        slack_t = nxt
    ends = [slack_t[m - 1]] + ([slack_t[m - 2]] if m >= 2 else [])
    n_ends = sum(1 for e in ends if e != -math.inf)
    best_end = max(ends)
    return best_end + (math.log(n_ends) / beta if n_ends > 1 else 0.0)


__all__ = ["logsumexp_gap_bound", "stepwise_gap_bound"]
