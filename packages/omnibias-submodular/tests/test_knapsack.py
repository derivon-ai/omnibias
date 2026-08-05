# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
"""Knapsack constraint: feasibility, Sviridenko's (1-1/e), and the fractional-knapsack sandwich."""

from __future__ import annotations

import numpy as np
import pytest
from omnibias.submodular import (
    ONE_MINUS_INV_E,
    Coverage,
    FacilityLocation,
    KnapsackConstraint,
    brute_force_max_knapsack,
    budgeted,
    certify_knapsack_gap,
    cost_benefit_greedy,
    knapsack_maximize,
)


def _coverage(seed: int) -> Coverage:
    rng = np.random.default_rng(seed)
    return Coverage((rng.random((10, 8)) < 0.35).astype(float), rng.random(10) + 0.3)


def _facility(seed: int) -> FacilityLocation:
    rng = np.random.default_rng(seed)
    return FacilityLocation(rng.random((9, 8)), rng.random(9) + 0.3)


def _constraint(seed: int, n: int = 8) -> KnapsackConstraint:
    rng = np.random.default_rng(1000 + seed)
    costs = rng.uniform(0.5, 2.0, size=n)
    return KnapsackConstraint(costs, budget=4.0)


def test_constraint_feasibility_and_cost() -> None:
    c = KnapsackConstraint(np.array([1.0, 2.0, 3.0]), budget=4.0)
    assert c.n == 3
    assert c.total_cost([1, 1, 0]) == 3.0
    assert c.is_feasible([1, 1, 0])
    assert not c.is_feasible([0, 1, 1])  # cost 5 > 4


def test_constraint_rejects_bad_inputs() -> None:
    with pytest.raises(ValueError, match="nonnegative"):
        KnapsackConstraint(np.array([1.0, -1.0]), budget=1.0)
    with pytest.raises(ValueError, match="budget"):
        KnapsackConstraint(np.array([1.0]), budget=-1.0)


def test_cost_benefit_greedy_is_feasible() -> None:
    for seed in range(6):
        fn, con = _coverage(seed), _constraint(seed)
        sel, _ = cost_benefit_greedy(fn, con)
        assert con.is_feasible(np.asarray(sel, dtype=float))


def test_knapsack_maximize_is_feasible_and_meets_one_minus_inv_e() -> None:
    for maker in (_coverage, _facility):
        for seed in range(6):
            fn, con = maker(seed), _constraint(seed)
            sel, val = knapsack_maximize(fn, con)
            assert con.is_feasible(np.asarray(sel, dtype=float))
            _, opt = brute_force_max_knapsack(fn, con)
            assert val >= ONE_MINUS_INV_E * opt - 1e-9, f"{maker.__name__} seed {seed}"
            assert val <= opt + 1e-9  # never beats the exact optimum


def test_knapsack_maximize_beats_or_matches_cost_benefit() -> None:
    # Partial enumeration can only help: the size-0/1 sweep dominates the pathological
    # single-heavy-item case that defeats pure ratio greedy.
    for seed in range(6):
        fn, con = _coverage(seed), _constraint(seed)
        _, sv = knapsack_maximize(fn, con)
        _, cb = cost_benefit_greedy(fn, con)
        assert sv >= cb - 1e-9


def test_certify_knapsack_gap_sandwich_is_sound() -> None:
    for maker in (_coverage, _facility):
        for seed in range(6):
            fn, con = maker(seed), _constraint(seed)
            sel, val = knapsack_maximize(fn, con)
            cert = certify_knapsack_gap(fn, con, sel)
            _, opt = brute_force_max_knapsack(fn, con)
            assert cert.value <= opt + 1e-9
            assert opt <= cert.upper_bound + 1e-9  # U(S) >= OPT (fractional-knapsack bound)
            assert cert.internal_consistent
            assert cert.method == "knapsack-fractional"
            assert abs(cert.value - val) < 1e-9


def test_certify_knapsack_gap_rejects_infeasible_selection() -> None:
    fn, con = _coverage(0), KnapsackConstraint(np.ones(8), budget=2.0)
    with pytest.raises(ValueError, match="feasible"):
        certify_knapsack_gap(fn, con, [1, 1, 1, 0, 0, 0, 0, 0])  # cost 3 > 2


def test_budgeted_frontend_matches_knapsack_maximize() -> None:
    fn = _coverage(2)
    costs = _constraint(2).costs
    sol = budgeted(fn, costs, budget=4.0)
    sel, val = knapsack_maximize(fn, KnapsackConstraint(costs, 4.0))
    assert sol.selection == sel
    assert abs(sol.value - val) < 1e-9


def test_brute_force_knapsack_is_capped() -> None:
    fn = Coverage(np.eye(21))
    con = KnapsackConstraint(np.ones(21), budget=3.0)
    with pytest.raises(ValueError, match="exponential"):
        brute_force_max_knapsack(fn, con)
