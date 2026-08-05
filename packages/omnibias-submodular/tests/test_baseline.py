# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
"""Best-in-class baseline: continuous greedy + rounding must beat or match classical greedy."""

from __future__ import annotations

import numpy as np
from omnibias.submodular import (
    ONE_MINUS_INV_E,
    Coverage,
    PartitionMatroid,
    SubmodularProblem,
    UniformMatroid,
    brute_force_max,
    greedy_maximize,
    maximize,
)

_SEEDS = range(8)


def _coverage(seed: int) -> Coverage:
    r = np.random.default_rng(seed)
    return Coverage((r.random((10, 7)) < 0.35).astype(float), r.random(10) + 0.3)


def test_matches_or_beats_greedy_on_cardinality() -> None:
    ours, base = [], []
    for seed in _SEEDS:
        prob = SubmodularProblem(_coverage(seed), UniformMatroid(7, 3))
        sol = maximize(prob, rounding="pipage")
        _, greedy_val = greedy_maximize(prob.function, prob.matroid)
        assert sol.value >= greedy_val - 1e-9, f"seed {seed}: {sol.value} < greedy {greedy_val}"
        ours.append(sol.value)
        base.append(greedy_val)
    assert float(np.mean(ours)) >= float(np.mean(base)) - 1e-9


def test_matches_or_beats_greedy_on_partition() -> None:
    # On a general matroid classical greedy guarantees only 1/2, continuous greedy 1-1/e;
    # continuous greedy + rounding must never lose to greedy across seeds.
    ours, base = [], []
    for seed in _SEEDS:
        prob = SubmodularProblem(_coverage(seed), PartitionMatroid([[0, 1, 2, 3], [4, 5, 6]], [2, 1]))
        sol = maximize(prob, rounding="pipage")
        _, greedy_val = greedy_maximize(prob.function, prob.matroid)
        _, opt = brute_force_max(prob.function, prob.matroid)
        assert sol.value >= greedy_val - 1e-9
        assert sol.value >= ONE_MINUS_INV_E * opt - 1e-9
        ours.append(sol.value)
        base.append(greedy_val)
    assert float(np.mean(ours)) >= float(np.mean(base)) - 1e-9


def test_swap_rounding_also_matches_or_beats_greedy() -> None:
    for seed in _SEEDS:
        prob = SubmodularProblem(_coverage(seed), UniformMatroid(7, 3))
        sol = maximize(prob, rounding="swap", seed=seed)
        _, greedy_val = greedy_maximize(prob.function, prob.matroid)
        assert sol.value >= greedy_val - 1e-9
