# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
r"""The benchmark harness + the yes-if / not-worse honesty of the LightGBM comparison."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("sklearn")

from omnibias.tab.bench import (  # noqa: E402
    ARRANGEMENT_PUBLIC_MAX_ROWS,
    ARRANGEMENT_PUBLIC_SUITE,
    NOISE_PUBLIC_MAX_ROWS,
    NOISE_PUBLIC_SUITE,
    HeadToHead,
    load_dataset,
    score_predictions,
    train_test_split,
    train_val_test_split,
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


def test_arrangement_public_suite_is_eight_binary_names() -> None:
    assert len(ARRANGEMENT_PUBLIC_SUITE) == 8
    assert ARRANGEMENT_PUBLIC_SUITE[0] == "breast_cancer"
    assert set(ARRANGEMENT_PUBLIC_MAX_ROWS) == set(ARRANGEMENT_PUBLIC_SUITE)


@pytest.mark.parametrize("name", ARRANGEMENT_PUBLIC_SUITE)
def test_arrangement_public_loader_binary_or_skip(name: str) -> None:
    """Every suite name loads as binary offline, or skips cleanly on OpenML failure."""
    max_rows = ARRANGEMENT_PUBLIC_MAX_ROWS.get(name)
    # Keep OpenML fetches small in CI when the cache is warm.
    row_cap = 500 if max_rows is None else min(500, max_rows)
    try:
        ds = load_dataset(name, max_rows=row_cap, seed=0)
    except RuntimeError as exc:
        if name == "breast_cancer":
            raise
        pytest.skip(f"OpenML unavailable for {name}: {exc}")
    assert ds.task == "binary"
    assert ds.n_outputs == 1
    assert ds.X.ndim == 2 and ds.y.ndim == 1
    assert ds.X.shape[0] == ds.y.shape[0] <= row_cap
    assert set(np.unique(ds.y).tolist()).issubset({0.0, 1.0})


def test_train_val_test_split_60_20_20() -> None:
    ds = load_dataset("breast_cancer", max_rows=200, seed=0)
    split = train_val_test_split(ds, seed=0, train_frac=0.6, val_frac=0.2)
    n = ds.X.shape[0]
    assert split["Xtr"].shape[0] + split["Xva"].shape[0] + split["Xte"].shape[0] == n
    assert abs(split["Xtr"].shape[0] / n - 0.6) < 0.05
    assert np.allclose(split["Xtr"].mean(axis=0), 0.0, atol=1e-9)


# --------------------------------------------------------------------------- #
# Theory 05-03 G4: public regression suite + tuned CatBoost/RealMLP/TabM.    #
# --------------------------------------------------------------------------- #


def test_noise_public_suite_has_at_least_six_names() -> None:
    assert len(NOISE_PUBLIC_SUITE) >= 6
    assert len(set(NOISE_PUBLIC_SUITE)) == len(NOISE_PUBLIC_SUITE)  # no duplicates
    assert set(NOISE_PUBLIC_MAX_ROWS) == set(NOISE_PUBLIC_SUITE)


@pytest.mark.parametrize("name", NOISE_PUBLIC_SUITE)
def test_noise_public_loader_regression_or_skip(name: str) -> None:
    """Every public regression name loads as a genuinely continuous target, or skips
    cleanly on OpenML failure (matching test_arrangement_public_loader_binary_or_skip)."""
    max_rows = NOISE_PUBLIC_MAX_ROWS.get(name)
    row_cap = 400 if max_rows is None else min(400, max_rows)
    try:
        ds = load_dataset(name, max_rows=row_cap, seed=0)
    except RuntimeError as exc:
        pytest.skip(f"OpenML unavailable for {name}: {exc}")
    assert ds.task == "regression"
    assert ds.n_outputs == 1
    assert ds.X.ndim == 2 and ds.y.ndim == 1
    assert ds.X.shape[0] == ds.y.shape[0] <= row_cap
    assert np.all(np.isfinite(ds.X)) and np.all(np.isfinite(ds.y))
    # Not the binarized-target OpenML trap noted above _OPENML_REGRESSION (a {0,1}-only
    # target); wine_quality's integer 3-9 rating is still a legitimate, if quantized,
    # regression target and must not trip this.
    assert len(np.unique(ds.y)) > 2


def test_fit_predict_catboost_regression_and_binary() -> None:
    pytest.importorskip("catboost")
    from omnibias.tab.bench import fit_predict_catboost

    rng = np.random.default_rng(0)
    X = rng.standard_normal((120, 4))
    y_reg = np.sin(X[:, 0]) + 0.3 * X[:, 1]
    pred, prob = fit_predict_catboost(X[:80], y_reg[:80], X[80:], task="regression", n_outputs=1, iterations=30)
    assert pred.shape == (40,) and prob is None
    rmse = float(np.sqrt(np.mean((pred - y_reg[80:]) ** 2)))
    assert rmse < float(np.std(y_reg))  # better than a constant-mean-ish baseline

    y_bin = (X[:, 0] > 0).astype(np.float64)
    pred_b, prob_b = fit_predict_catboost(X[:80], y_bin[:80], X[80:], task="binary", n_outputs=1, iterations=30)
    assert set(np.unique(pred_b).tolist()).issubset({0.0, 1.0})
    assert prob_b is not None and prob_b.shape == (40,)


def test_fit_predict_realmlp_regression_runs() -> None:
    pytest.importorskip("pytabkit")
    from omnibias.tab.bench import fit_predict_realmlp

    rng = np.random.default_rng(1)
    X = rng.standard_normal((80, 4))
    y = np.sin(X[:, 0]) + 0.3 * X[:, 1] * X[:, 2]
    pred, prob = fit_predict_realmlp(X[:60], y[:60], X[60:], seed=0, n_epochs=8)
    assert pred.shape == (20,) and prob is None
    assert np.all(np.isfinite(pred))


def test_fit_predict_tabm_regression_runs() -> None:
    pytest.importorskip("pytabkit")
    from omnibias.tab.bench import fit_predict_tabm

    rng = np.random.default_rng(2)
    X = rng.standard_normal((80, 4))
    y = np.sin(X[:, 0]) + 0.3 * X[:, 1] * X[:, 2]
    pred, prob = fit_predict_tabm(X[:60], y[:60], X[60:], seed=0, n_epochs=5)
    assert pred.shape == (20,) and prob is None
    assert np.all(np.isfinite(pred))
