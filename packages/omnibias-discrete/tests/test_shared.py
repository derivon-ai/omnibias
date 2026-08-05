# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
"""The backend-neutral helpers shared by several consumers: union-find / forest test and
the oracle-agnostic decision-focused (SPO+ / normalized-regret) core."""

from __future__ import annotations

import numpy as np
import pytest
from omnibias.discrete import (
    UnionFind,
    is_forest,
    mean_normalized_regret,
    spo_plus_subgradient,
)


def test_union_find_reports_tree_vs_cycle_edges() -> None:
    uf = UnionFind(4)
    assert uf.union(0, 1) is True
    assert uf.union(1, 2) is True
    assert uf.union(0, 2) is False  # 0 and 2 are already connected -> cycle edge
    assert uf.find(0) == uf.find(2)
    assert uf.find(3) == 3  # untouched singleton


def test_is_forest_path_and_triangle() -> None:
    assert is_forest([(0, 1), (1, 2), (2, 3)], 4) is True  # a path is acyclic
    assert is_forest([(0, 1), (1, 2), (2, 0)], 3) is False  # a triangle is not
    assert is_forest([], 3) is True  # the empty edge set is a forest


def _argmin_onehot(cost: np.ndarray) -> np.ndarray:
    x = np.zeros_like(cost)
    x[int(np.argmin(cost))] = 1.0
    return x


def test_spo_plus_subgradient_matches_hand_computation() -> None:
    pred = np.array([[1.0, 2.0, 3.0]])
    true = np.array([[3.0, 2.0, 1.0]])
    # x*(true) selects index 2; x*(2 pred - true) = x*([-1, 2, 5]) selects index 0.
    grad = spo_plus_subgradient(pred, true, _argmin_onehot)
    np.testing.assert_allclose(grad, [[-2.0, 0.0, 2.0]])


def test_spo_plus_subgradient_is_zero_when_pred_equals_true() -> None:
    rng = np.random.default_rng(0)
    c = rng.standard_normal((4, 5))  # 2 pred - true == c, so both oracle calls agree
    np.testing.assert_allclose(spo_plus_subgradient(c, c, _argmin_onehot), 0.0)


def test_mean_normalized_regret_zero_at_optimum() -> None:
    opt = np.array([2.0, 4.0, 6.0])
    true = np.stack([opt, np.zeros(3)], axis=1)  # true[b, 0] encodes the per-instance optimum
    pred = np.zeros((3, 2))
    result = mean_normalized_regret(pred, true, opt, lambda _p, t: float(t[0]))
    assert result == pytest.approx(0.0)


def test_mean_normalized_regret_normalizes_mean_excess() -> None:
    opt = np.array([2.0, 4.0])
    excess = np.array([0.5, 1.5])
    true = np.stack([opt + excess, np.zeros(2)], axis=1)
    pred = np.zeros((2, 2))
    result = mean_normalized_regret(pred, true, opt, lambda _p, t: float(t[0]))
    assert result == pytest.approx(float(np.mean(excess) / np.mean(opt)))
