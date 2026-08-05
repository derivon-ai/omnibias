# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
"""Exact sound counters (XOR / DPLL / treewidth) agree with the enumeration oracle."""

from __future__ import annotations

import numpy as np
import pytest
from omnibias.logic import (
    count_enclosure,
    count_models_exact,
    exact_model_count,
    model_count,
    treewidth_model_count,
    xor_model_count,
)
from omnibias.logic.model_count.exact import CountBudgetExceeded
from omnibias.logic.model_count.treewidth import TreewidthTooLarge
from omnibias.logic.model_count.xor import XORClause, detect_xor_system


def _random_cnf(seed: int, *, max_n: int = 7) -> tuple[list[list[int]], int]:
    rng = np.random.default_rng(seed)
    n = int(rng.integers(3, max_n))
    m = int(rng.integers(2, 9))
    clauses = []
    for _ in range(m):
        k = int(rng.integers(1, 4))
        variables = rng.choice(np.arange(1, n + 1), size=min(k, n), replace=False)
        signs = rng.choice([-1, 1], size=len(variables))
        clauses.append([int(s * v) for s, v in zip(signs, variables, strict=True)])
    return clauses, n


def _brute_xor(xors: list[XORClause], n: int) -> int:
    count = 0
    for mask in range(1 << n):
        bits = [(mask >> i) & 1 for i in range(n)]
        if all(sum(bits[v - 1] for v in xc.variables) % 2 == xc.parity for xc in xors):
            count += 1
    return count


def _xor_to_cnf(xors: list[XORClause]) -> list[list[int]]:
    """Standard CNF encoding of a parity system (each XOR -> 2^{k-1} clauses)."""
    clauses: list[list[int]] = []
    for xc in xors:
        variables = list(xc.variables)
        k = len(variables)
        for pattern in range(1 << k):
            signs = [(pattern >> i) & 1 for i in range(k)]  # 1 -> negative literal
            if sum(signs) % 2 == (1 - xc.parity):
                clauses.append([-variables[i] if signs[i] else variables[i] for i in range(k)])
    return clauses


def test_dpll_and_treewidth_match_oracle_across_seeds() -> None:
    for seed in range(40):
        clauses, n = _random_cnf(seed)
        mc = model_count(clauses, n_vars=n)
        exact = exact_model_count(mc)
        dpll = count_models_exact(mc)
        tw, width = treewidth_model_count(mc, max_width=64)
        assert isinstance(dpll, int)
        assert isinstance(width, int) and width >= 0
        assert float(dpll) == exact
        assert float(tw) == exact


def test_weighted_dpll_and_treewidth_match_oracle() -> None:
    for seed in range(25):
        rng = np.random.default_rng(500 + seed)
        clauses, n = _random_cnf(seed)
        weights = np.round(rng.uniform(0.25, 3.0, size=(n, 2)), 3)
        mc = model_count(clauses, weights=weights, n_vars=n)
        exact = exact_model_count(mc)
        assert float(count_models_exact(mc)) == pytest.approx(exact)
        tw, _ = treewidth_model_count(mc, max_width=64)
        assert float(tw) == pytest.approx(exact)


def test_exact_methods_sit_inside_the_certified_enclosure() -> None:
    for seed in range(30):
        clauses, n = _random_cnf(seed)
        mc = model_count(clauses, n_vars=n)
        enc = count_enclosure(mc, order=2)
        dpll = count_models_exact(mc)
        tw, _ = treewidth_model_count(mc, max_width=64)
        assert enc.contains(float(dpll))
        assert enc.contains(float(tw))


def test_xor_count_matches_brute_force_enumeration() -> None:
    for seed in range(60):
        rng = np.random.default_rng(seed)
        n = int(rng.integers(1, 9))
        xors = []
        for _ in range(int(rng.integers(0, 6))):
            k = int(rng.integers(1, n + 1))
            variables = tuple(sorted(rng.choice(np.arange(1, n + 1), size=k, replace=False)))
            xors.append(XORClause(variables=tuple(int(v) for v in variables), parity=int(rng.integers(0, 2))))
        assert xor_model_count(xors, n) == _brute_xor(xors, n)


def test_detect_xor_system_round_trips_to_the_exact_count() -> None:
    for seed in range(40):
        rng = np.random.default_rng(2000 + seed)
        n = int(rng.integers(1, 7))
        used: set[tuple[int, ...]] = set()
        xors = []
        for _ in range(int(rng.integers(1, 4))):
            k = int(rng.integers(1, min(3, n) + 1))
            variables = tuple(sorted(int(v) for v in rng.choice(np.arange(1, n + 1), size=k, replace=False)))
            if variables in used:
                continue
            used.add(variables)
            xors.append(XORClause(variables=variables, parity=int(rng.integers(0, 2))))
        clauses = _xor_to_cnf(xors)
        if not clauses:
            continue
        mc = model_count(clauses, n_vars=n)
        detected = detect_xor_system(mc)
        assert detected is not None, (seed, xors, clauses)
        assert xor_model_count(detected, n) == int(exact_model_count(mc))


def test_detect_xor_system_returns_none_on_a_plain_or_clause() -> None:
    # a single 2-variable OR is not a parity constraint (needs 2 clauses over those vars)
    assert detect_xor_system(model_count([[1, 2]], n_vars=2)) is None


def test_treewidth_raises_above_max_width_then_succeeds() -> None:
    mc = model_count([[1, 2, 3, 4, 5]], n_vars=5)  # primal graph is a 5-clique, width 4
    with pytest.raises(TreewidthTooLarge):
        treewidth_model_count(mc, max_width=2)
    count, width = treewidth_model_count(mc, max_width=8)
    assert width == 4
    assert count == int(exact_model_count(mc)) == 31  # only 00000 falsifies


def test_dpll_budget_raises_on_a_branching_instance() -> None:
    rng = np.random.default_rng(7)
    n = 20
    clauses = [
        [int(s * v) for s, v in zip(rng.choice([-1, 1], 3), rng.choice(np.arange(1, n + 1), 3, replace=False), strict=True)]
        for _ in range(int(4.3 * n))
    ]
    mc = model_count(clauses, n_vars=n)
    with pytest.raises(CountBudgetExceeded):
        count_models_exact(mc, node_budget=0)  # any branch node exceeds the budget


def test_xor_model_count_validates_inputs() -> None:
    with pytest.raises(ValueError, match="n_vars"):
        xor_model_count([XORClause((1,), 0)], 0)
    with pytest.raises(ValueError, match="outside"):
        xor_model_count([XORClause((5,), 0)], 3)
