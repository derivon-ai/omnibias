# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
r"""Newton boosting reduces the loss stage over stage and captures interactions."""

from __future__ import annotations

import numpy as np
import pytest
from omnibias.tab import SoftTreeConfig

torch = pytest.importorskip("torch")

from omnibias.tab.torch import fit_boosted  # noqa: E402


def test_boosting_reduces_loss_over_stages() -> None:
    rng = np.random.default_rng(0)
    X = rng.standard_normal((300, 8))
    w = np.array([1.2, -0.9, 0.7, 0, 0, 0, 0, 0])
    y = ((X @ w + 0.2 * rng.standard_normal(300)) > 0).astype(np.float64)
    tr, va = slice(0, 150), slice(150, 300)

    cfg = SoftTreeConfig(n_features=8, n_trees=1, depth=2, task="binary", beta_final=6.0, seed=1)
    model, res = fit_boosted(
        X[tr], y[tr], cfg, n_stages=12, learning_rate=0.3, inner_steps=25,
        val=(X[va], y[va]),
    )
    # loss decreases from the first stage to the last (a monotone-ish descent)
    assert res.history[-1] < res.history[0]
    assert res.history[-1] < res.history[len(res.history) // 2] + 1e-6
    # the boosted model generalises well above chance on held-out
    assert res.val_metric is not None and res.val_metric > 0.75


def test_boosting_learns_xor_interaction() -> None:
    rng = np.random.default_rng(3)
    X = rng.standard_normal((240, 4))
    y = ((X[:, 0] > 0) ^ (X[:, 1] > 0)).astype(np.float64)
    cfg = SoftTreeConfig(n_features=4, n_trees=1, depth=2, task="binary", beta_final=8.0, seed=2)
    model, res = fit_boosted(X, y, cfg, n_stages=18, learning_rate=0.4, inner_steps=35)
    acc = float(np.mean(model.predict(X) == y))
    assert acc > 0.8


def test_boosting_regression_runs() -> None:
    rng = np.random.default_rng(5)
    X = rng.standard_normal((200, 6))
    y = np.sin(X[:, 0]) + 0.5 * X[:, 1] * X[:, 2]
    cfg = SoftTreeConfig(
        n_features=6, n_trees=1, depth=2, task="regression", n_outputs=1, beta_final=6.0, seed=4
    )
    model, res = fit_boosted(X, y, cfg, n_stages=12, learning_rate=0.3, inner_steps=25)
    assert res.history[-1] < res.history[0]
