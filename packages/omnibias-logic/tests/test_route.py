# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
"""The sound router picks the expected method and every result matches the oracle."""

from __future__ import annotations

import numpy as np
import pytest
from omnibias.logic import CountResult, count, exact_model_count, model_count
from omnibias.logic.model_count.treewidth import TreewidthTooLarge


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


def _hard_instance(seed: int = 11, n: int = 18) -> object:
    rng = np.random.default_rng(seed)
    clauses = [
        [int(s * v) for s, v in zip(rng.choice([-1, 1], 3), rng.choice(np.arange(1, n + 1), 3, replace=False), strict=True)]
        for _ in range(int(4.3 * n))
    ]
    return model_count(clauses, n_vars=n)


def test_router_picks_the_affine_path_for_an_xor_system() -> None:
    # (x1 = x2) as CNF, x3 free -> 4 models; router must recognise the parity structure.
    result = count(model_count([[-1, 2], [1, -2]], n_vars=3))
    assert result.method == "affine_gf2"
    assert result.guarantee == "exact"
    assert result.value == 4
    assert result.is_exact and result.is_sound


def test_router_picks_treewidth_for_a_small_cnf() -> None:
    result = count(model_count([[1, 2], [2, 3]], n_vars=3))
    assert result.method == "treewidth_dp"
    assert result.guarantee == "exact"
    assert result.value == 5


def test_router_falls_back_to_a_certified_enclosure_under_budget() -> None:
    mc = _hard_instance()
    result = count(mc, max_width=1, node_budget=1)
    assert result.guarantee == "certified_enclosure"
    assert result.value is None
    assert result.contains(exact_model_count(mc))  # type: ignore[arg-type]
    assert result.width > 0


def test_router_matches_the_oracle_across_seeds() -> None:
    for seed in range(30):
        clauses, n = _random_cnf(seed)
        mc = model_count(clauses, n_vars=n)
        exact = exact_model_count(mc)
        result = count(mc)
        assert result.is_sound
        if result.is_exact:
            assert float(result.value) == exact  # type: ignore[arg-type]
        assert result.contains(exact)


def test_router_weighted_uses_a_weighted_exact_method() -> None:
    weights = np.array([[0.3, 0.7], [0.4, 0.6], [1.0, 1.0]])
    mc = model_count([[1, 2], [-2, 3]], weights=weights, n_vars=3)
    result = count(mc)
    assert result.guarantee == "exact"
    assert float(result.value) == pytest.approx(exact_model_count(mc))  # type: ignore[arg-type]


def test_explicit_modes() -> None:
    mc = model_count([[1, 2], [2, 3]], n_vars=3)
    assert count(mc, mode="dpll").method == "dpll"
    assert count(mc, mode="treewidth").method == "treewidth_dp"
    assert count(mc, mode="enclosure").guarantee == "certified_enclosure"
    # mode='xor' rejects a non-affine CNF and a weighted instance
    with pytest.raises(ValueError, match="affine"):
        count(mc, mode="xor")
    weighted = model_count([[-1, 2], [1, -2]], weights=np.array([[1.0, 1.0]] * 2), n_vars=2)
    with pytest.raises(ValueError, match="weighted"):
        count(weighted, mode="xor")


def test_explicit_treewidth_mode_propagates_too_large() -> None:
    mc = model_count([[1, 2, 3, 4, 5]], n_vars=5)
    with pytest.raises(TreewidthTooLarge):
        count(mc, mode="treewidth", max_width=2)


def test_invalid_mode_raises() -> None:
    with pytest.raises(ValueError, match="mode"):
        count(model_count([[1]], n_vars=1), mode="magic")


def test_count_result_is_a_frozen_tagged_dataclass() -> None:
    result = count(model_count([[1, 2], [2, 3]], n_vars=3))
    assert isinstance(result, CountResult)
    assert result.guarantee in ("exact", "certified_enclosure")
    with pytest.raises(AttributeError):
        result.value = 1  # type: ignore[misc]
