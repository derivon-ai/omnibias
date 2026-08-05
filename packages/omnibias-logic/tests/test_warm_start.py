# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
"""The relaxation warm start changes search order only -- the exact count is invariant."""

from __future__ import annotations

import numpy as np
import pytest
from omnibias.logic import count, count_models_exact, model_count
from omnibias.logic.model_count.route import _relaxation_branch_order


def _random_cnf(seed: int) -> tuple[list[list[int]], int]:
    rng = np.random.default_rng(seed)
    n = int(rng.integers(3, 8))
    m = int(rng.integers(2, 10))
    clauses = []
    for _ in range(m):
        k = int(rng.integers(1, 4))
        variables = rng.choice(np.arange(1, n + 1), size=min(k, n), replace=False)
        signs = rng.choice([-1, 1], size=len(variables))
        clauses.append([int(s * v) for s, v in zip(signs, variables, strict=True)])
    return clauses, n


def test_explicit_branch_order_does_not_change_the_exact_count() -> None:
    for seed in range(30):
        clauses, n = _random_cnf(seed)
        mc = model_count(clauses, n_vars=n)
        default = count_models_exact(mc)
        rng = np.random.default_rng(9000 + seed)
        for _ in range(3):
            order = list(rng.permutation(np.arange(1, n + 1)).astype(int))
            assert count_models_exact(mc, branch_order=order) == default


def test_weighted_count_is_also_branch_order_invariant() -> None:
    rng = np.random.default_rng(3)
    clauses, n = _random_cnf(4)
    weights = np.round(rng.uniform(0.25, 2.5, size=(n, 2)), 3)
    mc = model_count(clauses, weights=weights, n_vars=n)
    base = count_models_exact(mc)
    reversed_order = list(range(n, 0, -1))
    assert count_models_exact(mc, branch_order=reversed_order) == base


def test_router_warm_start_matches_the_default_order() -> None:
    # warm_start is backend-guarded: with or without torch/jax the sound count is identical.
    for seed in range(12):
        clauses, n = _random_cnf(seed)
        mc = model_count(clauses, n_vars=n)
        plain = count(mc, mode="dpll")
        warmed = count(mc, mode="dpll", warm_start=True)
        assert warmed.value == plain.value


def test_relaxation_branch_order_is_a_permutation_or_none() -> None:
    mc = model_count([[1, 2], [-2, 3], [1, -3]], n_vars=3)
    order = _relaxation_branch_order(mc)
    if order is None:  # no torch / jax backend in this environment
        pytest.skip("no tensor backend available for the relaxation warm start")
    assert sorted(order) == [1, 2, 3]
