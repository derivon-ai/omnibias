# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Partition-of-unity properties of the numpy reference weights."""

from __future__ import annotations

import numpy as np
import pytest
from omnibias.partition import (
    PartitionConfig,
    gate_activations,
    hard_assignment,
    hard_weights,
    hardened_rules,
    init_params,
    partition_weights,
    region_rule,
)


@pytest.mark.parametrize("depth", [1, 2, 3])
@pytest.mark.parametrize("split_kind", ["oblique", "axis", "sparse"])
def test_weights_are_a_partition_of_unity(depth: int, split_kind: str) -> None:
    cfg = PartitionConfig(n_features=4, depth=depth, split_kind=split_kind, seed=1)
    params = init_params(cfg, rng=0)
    X = np.random.default_rng(2).standard_normal((37, 4))
    W = partition_weights(params, X, beta=3.0)
    assert W.shape == (37, cfg.n_regions)
    assert np.all(W >= 0.0)  # non-negative
    assert np.allclose(W.sum(axis=1), 1.0, atol=1e-12)  # rows sum to one


def test_axis_init_is_single_feature() -> None:
    cfg = PartitionConfig(n_features=5, depth=3, split_kind="axis", seed=7)
    params = init_params(cfg)
    for j in range(cfg.depth):
        assert np.count_nonzero(params.W[j]) == 1  # each gate reads one feature


def test_hardening_matches_hard_weights() -> None:
    cfg = PartitionConfig(n_features=3, depth=2, seed=3)
    params = init_params(cfg, rng=1)
    X = np.random.default_rng(4).standard_normal((50, 3))
    soft = partition_weights(params, X, beta=400.0)
    hard = hard_weights(params, X)
    # away from the split surface the soft POU collapses onto the hard one-hot
    z = np.abs(X @ params.W.T - params.t[None, :])
    interior = np.all(z > 0.05, axis=1)
    assert np.max(np.abs(soft[interior] - hard[interior])) < 1e-3


def test_hard_assignment_selects_max_weight_region_when_sharp() -> None:
    cfg = PartitionConfig(n_features=3, depth=2, seed=5)
    params = init_params(cfg, rng=2)
    X = np.random.default_rng(6).standard_normal((40, 3))
    z = np.abs(X @ params.W.T - params.t[None, :])
    interior = np.all(z > 0.05, axis=1)
    soft = partition_weights(params, X, beta=200.0)
    idx = hard_assignment(params, X)
    assert np.array_equal(np.argmax(soft[interior], axis=1), idx[interior])


def test_gate_activations_shape_and_range() -> None:
    cfg = PartitionConfig(n_features=2, depth=2, seed=0)
    params = init_params(cfg)
    g = gate_activations(params, np.zeros((5, 2)), beta=1.0)
    assert g.shape == (5, 2)
    assert np.all((g > 0.0) & (g < 1.0))


def test_hardened_rules_are_readable() -> None:
    cfg = PartitionConfig(n_features=3, depth=2, split_kind="axis", seed=1)
    params = init_params(cfg)
    rules = hardened_rules(params)
    assert len(rules) == cfg.depth
    assert all("x[" in r for r in rules)  # single-feature threshold form
    clause = region_rule(params, region=cfg.n_regions - 1)
    assert "AND" in clause
