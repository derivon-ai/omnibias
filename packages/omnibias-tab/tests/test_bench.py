# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
r"""The benchmark harness + the yes-if / not-worse honesty of the LightGBM comparison."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("sklearn")

from omnibias.tab.bench import (  # noqa: E402
    HeadToHead,
    load_dataset,
    score_predictions,
    train_test_split,
)


def test_score_predictions_higher_is_better() -> None:
    # regression: primary is -rmse (a perfect fit -> 0)
    y = np.array([1.0, 2.0, 3.0])
    perfect = score_predictions(y, y.copy(), None, "regression")
    worse = score_predictions(y, y + 1.0, None, "regression")
    assert perfect["primary"] == pytest.approx(0.0)
    assert perfect["primary"] > worse["primary"]  # higher is better for every task

    # binary: primary is accuracy in [0, 1]
    yb = np.array([0.0, 1.0, 1.0, 0.0])
    s = score_predictions(yb, np.array([0.0, 1.0, 1.0, 0.0]), np.array([0.1, 0.9, 0.8, 0.2]), "binary")
    assert s["accuracy"] == pytest.approx(1.0) and s["primary"] == pytest.approx(1.0)
    assert 0.0 <= s["auc"] <= 1.0


def test_load_and_split_is_standardized_and_stratified() -> None:
    ds = load_dataset("breast_cancer", max_rows=200, seed=0)
    assert ds.task == "binary" and ds.X.shape[0] == 200
    Xtr, Xte, ytr, yte = train_test_split(ds, seed=0)
    # standardization is fit on train -> train columns are ~zero-mean / unit-var
    assert np.allclose(Xtr.mean(axis=0), 0.0, atol=1e-9)
    assert np.allclose(Xtr.std(axis=0), 1.0, atol=1e-6)
    # the test split must be disjoint in size and use the same scaler (not re-fit)
    assert Xte.shape[0] == 50


def test_head_to_head_not_worse_uses_seed_noise() -> None:
    r"""The honesty invariant: ``not_worse`` is ``mean_tab >= mean_lgbm - seed_noise``.

    A hard ``>`` on a single seed would be a brittle, over-claimed gate; the acceptance is
    "within the baseline's own across-seed noise" (or better) -- exactly the
    empirical-validation discipline.
    """
    h = HeadToHead(dataset="synthetic", task="binary", seeds=[0, 1, 2])
    h.lgbm = [{"primary": 0.90}, {"primary": 0.94}, {"primary": 0.92}]  # mean 0.92, std ~0.016
    # tab slightly below the baseline mean but inside its seed noise -> still "not worse"
    h.tab = [{"primary": 0.91}, {"primary": 0.905}, {"primary": 0.915}]  # mean ~0.910
    assert h.mean("tab") < h.mean("lgbm")
    assert h.not_worse is True
    # tab far below the baseline, outside the noise band -> honestly NOT not-worse
    h.tab = [{"primary": 0.70}, {"primary": 0.72}, {"primary": 0.71}]
    assert h.not_worse is False


def test_tiny_head_to_head_runs_end_to_end() -> None:
    r"""A CPU-tiny end-to-end head-to-head (boosted tab vs LightGBM) smoke."""
    pytest.importorskip("torch")
    pytest.importorskip("lightgbm")
    from omnibias.tab.bench import TabConfig, head_to_head

    cfg = TabConfig(method="boost", n_stages=8, learning_rate=0.3, depth=1,
                    inner_steps=10, inner_lr=0.08, beta_final=6.0)
    h = head_to_head("breast_cancer", seeds=2, tab_cfg=cfg, max_rows=150)
    s = h.summary()
    assert 0.0 <= s["tab_mean_primary"] <= 1.0 and 0.0 <= s["lgbm_mean_primary"] <= 1.0
    assert isinstance(s["not_worse"], bool)
