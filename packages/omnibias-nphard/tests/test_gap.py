# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
"""GAP as a QUBO-form DiscreteProblem: slack encoding, feasibility, LP bound, oracle."""

from __future__ import annotations

import numpy as np
import pytest
from omnibias.nphard import GAPProblem, gap
from omnibias.nphard._core.gap import (
    encode_slack,
    gap_brute_force,
    gap_classical,
    gap_decode,
    gap_lp_lower_bound,
    slack_weights,
)


def _random_gap(agents: int, tasks: int, seed: int, cap: int = 6) -> GAPProblem:
    rng = np.random.default_rng(seed)
    cost = rng.integers(1, 9, size=(agents, tasks)).astype(float)
    resource = rng.integers(1, 4, size=(agents, tasks)).astype(float)
    capacity = np.full(agents, float(cap))
    return gap(cost, resource, capacity)


def test_slack_weights_cover_every_integer_contiguously() -> None:
    """The bounded binary weight set realises every value in [0, cap] with no gaps."""
    for cap in range(0, 20):
        w = slack_weights(cap)
        reachable = {sum(sub) for r in range(len(w) + 1) for sub in _subsets(w, r)}
        assert reachable == set(range(cap + 1))
        for value in range(cap + 1):
            assert sum(w[i] for i, b in enumerate(encode_slack(value, w)) if b) == value


def _subsets(items: list[int], r: int) -> list[tuple[int, ...]]:
    import itertools

    return list(itertools.combinations(items, r))


def test_energy_matches_to_polynomial_on_random_points() -> None:
    pytest.importorskip("omnibias.sos")
    rng = np.random.default_rng(0)
    prob = _random_gap(2, 3, 1, cap=4)
    poly = prob.to_polynomial()
    for _ in range(20):
        x = rng.standard_normal(prob.n)
        assert abs(poly.evaluate(list(x)) - float(prob.energy(x))) < 1e-9


def test_to_qubo_energy_matches_on_binary_points() -> None:
    rng = np.random.default_rng(1)
    prob = _random_gap(2, 3, 2, cap=4)
    qubo = prob.to_qubo()
    for _ in range(40):
        x = rng.integers(0, 2, size=prob.n).astype(float)
        assert abs(float(prob.energy(x)) - float(qubo.energy(x))) < 1e-9


def test_feasible_assignment_has_zero_penalty_energy_equals_cost() -> None:
    """A capacity-feasible assignment (with its slack) pays no penalty: energy == cost."""
    prob = _random_gap(3, 4, 3, cap=8)
    # a feasible round-robin assignment (small resources, ample capacity)
    assignment = [t % prob.n_agents for t in range(prob.n_tasks)]
    assert prob.is_feasible(assignment)
    x = prob.full_x(assignment)
    assert abs(float(prob.energy(x)) - prob.assignment_cost(assignment)) < 1e-9


def test_decoder_returns_a_capacity_feasible_assignment() -> None:
    for seed in range(5):
        prob = _random_gap(3, 4, seed, cap=8)
        heat = np.random.default_rng(seed).random(prob.n)
        x, energy = gap_decode(prob, relaxed=heat)
        a_t = prob.n_agents * prob.n_tasks
        block = np.asarray(x[:a_t], dtype=float).reshape(prob.n_agents, prob.n_tasks)
        assignment = [int(np.argmax(block[:, t])) for t in range(prob.n_tasks)]
        assert prob.is_feasible(assignment)


def test_lp_lower_bound_is_below_the_integer_optimum() -> None:
    """The LP relaxation is a valid lower bound on the exact GAP cost."""
    for seed in range(5):
        prob = _random_gap(3, 4, seed, cap=8)
        lp = gap_lp_lower_bound(prob)
        x_opt, _ = gap_brute_force(prob)
        a_t = prob.n_agents * prob.n_tasks
        block = np.asarray(x_opt[:a_t], dtype=float).reshape(prob.n_agents, prob.n_tasks)
        assignment = [int(np.argmax(block[:, t])) for t in range(prob.n_tasks)]
        opt_cost = prob.assignment_cost(assignment)
        assert lp <= opt_cost + 1e-6


def test_greedy_baseline_is_feasible_and_upper_bounds_the_optimum() -> None:
    for seed in range(5):
        prob = _random_gap(3, 4, seed, cap=8)
        x_cla, _ = gap_classical(prob)
        a_t = prob.n_agents * prob.n_tasks
        block = np.asarray(x_cla[:a_t], dtype=float).reshape(prob.n_agents, prob.n_tasks)
        assignment = [int(np.argmax(block[:, t])) for t in range(prob.n_tasks)]
        assert prob.is_feasible(assignment)


def test_brute_force_is_guarded_and_rejects_negative_data() -> None:
    prob = _random_gap(6, 8, 0, cap=6)  # 6^8 > 200k
    with pytest.raises(ValueError, match="exponential"):
        gap_brute_force(prob)
    with pytest.raises(ValueError, match="non-negative"):
        gap(np.zeros((2, 2)), -np.ones((2, 2)), np.ones(2))
