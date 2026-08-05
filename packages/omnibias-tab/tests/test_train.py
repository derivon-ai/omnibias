# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
r"""The exact second-order trainer beats the first-order (Adam) baseline on held-out data."""

from __future__ import annotations

import numpy as np
import pytest
from omnibias.tab import SoftTreeConfig

torch = pytest.importorskip("torch")

from omnibias.tab.torch import SoftTreeEnsemble, fit_first_order, fit_second_order  # noqa: E402


def _binary_problem(seed: int = 0, n: int = 300, d: int = 10, noise: float = 0.3):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, d))
    w = np.zeros(d)
    w[:3] = [1.5, -1.0, 0.8]
    logits = X @ w + noise * rng.standard_normal(n)
    y = (logits > 0).astype(np.float64)
    tr, va = slice(0, n // 2), slice(n // 2, n)
    return (X[tr], y[tr]), (X[va], y[va])


def test_second_order_beats_first_order_heldout() -> None:
    r"""Controlled matched-budget comparison on a signal-rich, low-noise problem.

    With a short, equal step budget and matched regularisation the exact-Hessian optimiser
    both drives the training objective far lower *and* -- because Adam has not yet
    converged and the label noise is small so there is nothing to overfit -- reaches a
    strictly better held-out accuracy. (On noisier data better optimisation can trade off
    against generalisation; that regime is what the LightGBM benchmark measures, not this
    optimisation-efficiency invariant.)"""
    (Xtr, ytr), (Xva, yva) = _binary_problem(n=320, d=12, noise=0.1)
    cfg = SoftTreeConfig(n_features=12, n_trees=16, depth=1, task="binary", beta_final=3.0, seed=1, leaf_l2=1e-2)

    torch.manual_seed(0)
    m2 = SoftTreeEnsemble(cfg)
    r2 = fit_second_order(m2, Xtr, ytr, optimizer="trust_region", steps=15, weight_l2=1e-3, anneal=False, val=(Xva, yva))

    torch.manual_seed(0)
    m1 = SoftTreeEnsemble(cfg)
    r1 = fit_first_order(m1, Xtr, ytr, lr=0.05, steps=15, weight_l2=1e-3, anneal=False, val=(Xva, yva))

    # exact curvature optimises the same objective far better per equal step budget ...
    assert r2.train_loss < r1.train_loss
    # ... and (Adam underfits, noise is low) reaches strictly better held-out accuracy.
    assert r2.val_metric is not None and r1.val_metric is not None
    assert r2.val_metric > r1.val_metric


def test_cubic_newton_reduces_loss() -> None:
    (Xtr, ytr), _ = _binary_problem(seed=2)
    cfg = SoftTreeConfig(n_features=10, n_trees=8, depth=1, task="binary", beta_final=4.0, seed=3)
    torch.manual_seed(0)
    model = SoftTreeEnsemble(cfg)
    r = fit_second_order(model, Xtr, ytr, optimizer="cubic", steps=20, anneal=False)
    assert r.train_loss < r.history[0]


def test_kfac_natural_gradient_trains_additive() -> None:
    (Xtr, ytr), (Xva, yva) = _binary_problem(seed=4)
    cfg = SoftTreeConfig(n_features=10, n_trees=12, depth=1, task="binary", beta_final=4.0, seed=5)
    torch.manual_seed(0)
    model = SoftTreeEnsemble(cfg)
    r = fit_second_order(model, Xtr, ytr, optimizer="kfac", steps=40, val=(Xva, yva))
    assert r.train_loss < r.history[0]
    assert r.val_metric is not None and r.val_metric > 0.6


def test_second_order_trains_multiplicative_depth2() -> None:
    # an XOR-like target that a depth-1 additive model cannot fit but a depth-2 tree can
    rng = np.random.default_rng(11)
    X = rng.standard_normal((220, 4))
    y = ((X[:, 0] > 0) ^ (X[:, 1] > 0)).astype(np.float64)
    cfg = SoftTreeConfig(n_features=4, n_trees=8, depth=2, task="binary", beta_final=6.0, seed=7)
    torch.manual_seed(0)
    model = SoftTreeEnsemble(cfg)
    r = fit_second_order(model, X, y, optimizer="trust_region", steps=30, anneal=True)
    acc = float(np.mean(model.predict(X) == y))
    assert r.train_loss < r.history[0]
    assert acc > 0.75  # captures the interaction the additive tier cannot


def test_regression_second_order_runs() -> None:
    rng = np.random.default_rng(9)
    X = rng.standard_normal((200, 6))
    y = X[:, 0] * 1.2 - 0.7 * X[:, 1] + 0.1 * rng.standard_normal(200)
    cfg = SoftTreeConfig(n_features=6, n_trees=8, depth=1, task="regression", n_outputs=1, beta_final=4.0, seed=2)
    torch.manual_seed(0)
    model = SoftTreeEnsemble(cfg)
    r = fit_second_order(model, X, y, optimizer="trust_region", steps=25, anneal=False)
    assert r.train_loss < r.history[0]
