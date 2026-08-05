# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
"""The certified count enclosure sandwiches the exact oracle (across seeds)."""

from __future__ import annotations

import numpy as np
import pytest
from omnibias.logic import count_enclosure, exact_model_count, model_count


def _random_cnf(seed: int) -> tuple[list[list[int]], int]:
    rng = np.random.default_rng(seed)
    n = int(rng.integers(3, 7))
    m = int(rng.integers(2, 9))
    clauses = []
    for _ in range(m):
        k = int(rng.integers(1, 4))
        variables = rng.choice(np.arange(1, n + 1), size=min(k, n), replace=False)
        signs = rng.choice([-1, 1], size=len(variables))
        clauses.append([int(s * v) for s, v in zip(signs, variables, strict=True)])
    return clauses, n


def test_sandwich_holds_across_seeds_and_orders() -> None:
    for seed in range(30):
        clauses, n = _random_cnf(seed)
        mc = model_count(clauses, n_vars=n)
        exact = exact_model_count(mc)
        for order in (0, 1, 2, 3):
            enc = count_enclosure(mc, order=order)
            assert enc.is_sound
            assert enc.contains(exact), (seed, order, enc.lower, exact, enc.upper)
            assert 0.0 <= enc.lower <= enc.upper <= enc.total + 1e-9


def test_full_order_is_tight_and_exact() -> None:
    for seed in range(30):
        clauses, n = _random_cnf(seed)
        mc = model_count(clauses, n_vars=n)
        exact = exact_model_count(mc)
        enc = count_enclosure(mc, order=len(clauses) + 1)
        assert enc.tight
        assert enc.lower == pytest.approx(exact)
        assert enc.upper == pytest.approx(exact)


def test_order_zero_is_the_trivial_box() -> None:
    mc = model_count([[1, -2], [2, 3]], n_vars=3)
    enc = count_enclosure(mc, order=0)
    assert enc.lower == 0.0
    assert enc.upper == enc.total == 8.0
    assert enc.method == "trivial"


def test_enclosure_tightens_monotonically_with_order() -> None:
    # a shared-variable instance where the order matters: [4,8] -> [4,5] -> [5,5].
    mc = model_count([[1, 2], [1, 3]], n_vars=3)
    encs = [count_enclosure(mc, order=o) for o in range(0, 5)]
    for lo, hi in zip(encs, encs[1:], strict=False):
        assert hi.lower >= lo.lower - 1e-12  # lower bound never regresses
        assert hi.upper <= lo.upper + 1e-12  # upper bound never regresses
    assert (encs[3].lower, encs[3].upper) == (5.0, 5.0)


def test_weighted_enclosure_is_sound_and_tight_at_full_order() -> None:
    for seed in range(15):
        rng = np.random.default_rng(1000 + seed)
        clauses, n = _random_cnf(seed)
        # random nonnegative rational-ish weights
        weights = np.round(rng.uniform(0.25, 3.0, size=(n, 2)), 3)
        mc = model_count(clauses, weights=weights, n_vars=n)
        exact = exact_model_count(mc)
        for order in (1, 2, 3):
            enc = count_enclosure(mc, order=order)
            assert enc.weighted
            assert enc.lower - 1e-9 <= exact <= enc.upper + 1e-9
        full = count_enclosure(mc, order=len(clauses) + 1)
        assert full.tight
        assert full.lower <= exact + 1e-9 <= full.upper + 2e-9
        assert full.upper == pytest.approx(exact, rel=1e-9, abs=1e-9)


def test_witness_strengthens_the_lower_bound() -> None:
    # (x0) and (x1) over 3 vars: S1 = Z0 so the order-1 Bonferroni lower clamps to 0,
    # but the two models {110, 111} are sound witnesses -> lower rises to 2.
    mc = model_count([[1], [2]], n_vars=3)
    assert exact_model_count(mc) == 2.0
    plain = count_enclosure(mc, order=1)
    assert plain.lower == 0.0
    models = np.array([[1.0, 1.0, 0.0], [1.0, 1.0, 1.0]])
    with_witness = count_enclosure(mc, order=1, witnesses=models)
    assert with_witness.lower == 2.0
    assert "witness" in with_witness.method
    assert with_witness.contains(2.0)


def test_witness_ignores_non_models_and_duplicates() -> None:
    mc = model_count([[1], [2]], n_vars=3)
    # (0,0,0) is not a model; (1,1,0) repeated should count once.
    witnesses = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 0.0], [1.0, 1.0, 0.0]])
    enc = count_enclosure(mc, order=1, witnesses=witnesses)
    assert enc.lower == 1.0


def test_order_must_be_nonnegative() -> None:
    mc = model_count([[1, 2]], n_vars=2)
    with pytest.raises(ValueError, match="order"):
        count_enclosure(mc, order=-1)
