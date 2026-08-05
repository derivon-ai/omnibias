# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
r"""The Go-like MCTS "search track" for omnibias-nphard.

A from-scratch, self-contained UCT Monte-Carlo tree search over a *construction MDP*
(state = partial assignment; action = fix the next variable; reward = ``-energy``), with
pluggable ``prior_fn`` / ``value_fn``. The default prior is the differentiable
relaxation heatmap (an AlphaZero-style neural prior) and the value is a heatmap-guided
greedy completion. Heuristic search -- the found solution is still handed to
:func:`omnibias.nphard.certify_gap` for a sound optimality gap, never an optimality
guarantee.
"""

from __future__ import annotations

from omnibias.nphard.search.mcts import (
    ConstructionMDP,
    MCTSResult,
    heatmap_rollout,
    hungarian_rollout,
    mcts_search,
    mdp_for,
    qap_mdp,
    random_rollout,
    relaxation_prior,
    scheduling_mdp,
    uniform_prior,
)

__all__ = [
    "ConstructionMDP",
    "MCTSResult",
    "heatmap_rollout",
    "hungarian_rollout",
    "mcts_search",
    "mdp_for",
    "qap_mdp",
    "random_rollout",
    "relaxation_prior",
    "scheduling_mdp",
    "uniform_prior",
]
