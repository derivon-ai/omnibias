# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
r"""TabPOU trainer, preprocessor, residual, and G1-scale axis-AND smoke."""

from __future__ import annotations

import numpy as np
import pytest
from omnibias.tab import make_axis_rule
from omnibias.tab.pou import TabPOUConfig, TabPreprocessor

torch = pytest.importorskip("torch")

from omnibias.tab.pou import TabMResidual, fit_tabpou  # noqa: E402
from omnibias.tab.pou.ensemble import LinearBatchEnsemble  # noqa: E402


def test_preprocessor_onehots_low_card_and_scales_numeric() -> None:
    rng = np.random.default_rng(0)
    X = np.column_stack(
        [rng.standard_normal(50), np.array([0.0, 1.0, 2.0] * 16 + [0.0, 1.0])]
    )
    prep = TabPreprocessor(onehot_max_card=8).fit(X)
    Xt = prep.transform(X)
    assert prep.cat_cols == (1,)
    assert Xt.shape[1] == 1 + 3
    assert np.max(np.abs(Xt[:, 1:].sum(axis=1) - 1.0)) < 1e-12


def test_batch_ensemble_mean_shape() -> None:
    layer = LinearBatchEnsemble(5, 3, ensemble_k=4)
    x = torch.randn(7, 5, dtype=torch.float64)
    y = layer(x)
    assert y.shape == (7, 4, 3)
    head = TabMResidual(5, 2, ensemble_k=4, hidden=8)
    out = head(x)
    assert out.shape == (7, 2)


def test_fit_tabpou_reduces_loss_on_sine() -> None:
    rng = np.random.default_rng(1)
    X = rng.standard_normal((120, 4))
    y = np.sin(X[:, 0]) + 0.5 * X[:, 1]
    cfg = TabPOUConfig(
        n_features=4, task="regression", n_outputs=1, depth=2, n_stages=8,
        n_quantiles=8, n_bins=4, use_embed=False, robust_scale=True,
        onehot_max_card=1, seed=1, patience=None,
    )
    model, res = fit_tabpou(X, y, cfg)
    assert res.history[-1] < res.history[0]
    pred = model.predict(X)
    assert pred.shape == (120,)
    assert np.isfinite(pred).all()


def test_fit_tabpou_axis_and_beats_majority() -> None:
    X, y, _ = make_axis_rule(n_samples=400, n_features=8, seed=2)
    cfg = TabPOUConfig(
        n_features=8, task="binary", n_outputs=1, depth=2, n_stages=10,
        n_quantiles=12, use_embed=False, robust_scale=False, onehot_max_card=1,
        seed=2, patience=None, learning_rate=0.5, beta_final=8.0,
    )
    model, _ = fit_tabpou(X, y, cfg)
    acc = float(np.mean(model.predict(X) == y))
    majority = float(max(y.mean(), 1.0 - y.mean()))
    assert acc > majority + 0.05


def test_fit_tabpou_residual_runs() -> None:
    rng = np.random.default_rng(4)
    X = rng.standard_normal((80, 3))
    y = X[:, 0] ** 2 - 0.3 * X[:, 1]
    cfg = TabPOUConfig(
        n_features=3, task="regression", depth=1, n_stages=4, n_quantiles=6,
        use_embed=False, use_residual=True, residual_k=4, residual_hidden=8,
        residual_steps=15, onehot_max_card=1, seed=4, patience=None,
    )
    model, res = fit_tabpou(X, y, cfg)
    assert res.used_residual
    assert model.residual is not None
    assert model.residual.n_regions == 2
    assert np.isfinite(model.predict(X)).all()
