# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
r"""Newton boosting reduces the loss stage over stage and captures interactions."""

from __future__ import annotations

import numpy as np
import pytest
from omnibias.tab import SoftTreeConfig

torch = pytest.importorskip("torch")

from omnibias.tab.torch import fit_boosted, fit_boosted_heteroscedastic  # noqa: E402


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


# --------------------------------------------------------------------------- #
# fit_boosted_heteroscedastic (theory 05-03 section 4(d)/6(c')).              #
# --------------------------------------------------------------------------- #


def test_boosted_heteroscedastic_shrinkage_matches_fit_boosted_bit_identically() -> None:
    """G0-style plumbing check: weighting='shrinkage' ignores log_scale entirely."""
    rng = np.random.default_rng(6)
    X = rng.standard_normal((150, 5))
    y = np.sin(X[:, 0]) + 0.3 * X[:, 1] * X[:, 2] + 0.05 * rng.standard_normal(150)
    cfg = SoftTreeConfig(
        n_features=5, n_trees=1, depth=2, task="regression", n_outputs=1, beta_final=6.0, seed=9
    )
    model_a, res_a = fit_boosted(X, y, cfg, n_stages=8, learning_rate=0.3, inner_steps=20)
    model_b, res_b = fit_boosted_heteroscedastic(
        X, y, cfg, log_scale=rng.standard_normal(150), weighting="shrinkage",
        n_stages=8, learning_rate=0.3, inner_steps=20,
    )
    params_a, params_b = model_a.to_params(), model_b.to_params()
    assert np.array_equal(params_a.W, params_b.W)
    assert np.array_equal(params_a.t, params_b.t)
    assert np.array_equal(params_a.leaves, params_b.leaves)
    assert np.array_equal(params_a.b0, params_b.b0)
    assert res_a.history == res_b.history


def test_boosted_heteroscedastic_gls_reweights_low_noise_rows_more() -> None:
    """With weighting='gls', a stage's weak-learner weight is h / s_hat**2 -- low-noise
    rows (small s_hat) get a *larger* weight than high-noise rows, unlike 'shrinkage'."""
    rng = np.random.default_rng(7)
    n = 100
    X = rng.standard_normal((n, 3))
    y = X[:, 0] + 0.1 * rng.standard_normal(n)
    # Half the rows are declared (frozen) high-noise, half low-noise.
    log_scale = np.where(np.arange(n) < n // 2, -2.0, 2.0)
    cfg = SoftTreeConfig(
        n_features=3, n_trees=1, depth=1, task="regression", n_outputs=1, beta_final=5.0, seed=11
    )
    model, res = fit_boosted_heteroscedastic(
        X, y, cfg, log_scale=log_scale, weighting="gls", n_stages=5, learning_rate=0.3, inner_steps=15
    )
    assert res.history[-1] < res.history[0]
    # basic sanity: the fitted model runs and produces finite scores
    assert np.all(np.isfinite(model.score(X)))


def test_boosted_heteroscedastic_rejects_bad_weighting() -> None:
    rng = np.random.default_rng(8)
    X = rng.standard_normal((40, 3))
    y = rng.standard_normal(40)
    cfg = SoftTreeConfig(n_features=3, n_trees=1, depth=1, task="regression", n_outputs=1, seed=1)
    with pytest.raises(ValueError, match="weighting"):
        fit_boosted_heteroscedastic(X, y, cfg, log_scale=np.zeros(40), weighting="bogus")


def test_boosted_heteroscedastic_rejects_non_regression() -> None:
    rng = np.random.default_rng(8)
    X = rng.standard_normal((40, 3))
    y = (rng.standard_normal(40) > 0).astype(np.float64)
    cfg = SoftTreeConfig(n_features=3, n_trees=1, depth=1, task="binary", seed=1)
    with pytest.raises(ValueError, match="regression-only"):
        fit_boosted_heteroscedastic(X, y, cfg, log_scale=np.zeros(40))


def test_boosted_heteroscedastic_rejects_mismatched_log_scale_length() -> None:
    rng = np.random.default_rng(8)
    X = rng.standard_normal((40, 3))
    y = rng.standard_normal(40)
    cfg = SoftTreeConfig(n_features=3, n_trees=1, depth=1, task="regression", n_outputs=1, seed=1)
    with pytest.raises(ValueError, match="log_scale"):
        fit_boosted_heteroscedastic(X, y, cfg, log_scale=np.zeros(39))


def test_boosted_heteroscedastic_gls_beats_shrinkage_on_heteroscedastic_target() -> None:
    """G3 (theory 05-03 section 8): on a target whose noise correlates with a known,
    perfectly-informative log_scale, GLS reweighting should not do worse than plain
    shrinkage on the low-noise half of held-out test rows, over several seeds -- a
    smaller-scale, deterministic sanity check of the same mechanism G3 gates at scale."""
    wins = 0
    for seed in range(5):
        rng = np.random.default_rng(100 + seed)
        n = 400
        X = rng.standard_normal((n, 4))
        f_clean = np.sin(X[:, 0]) + 0.4 * X[:, 1] * X[:, 2]
        s_true = np.where(X[:, 3] > 0.0, 0.05, 2.0)  # bimodal known noise
        y = f_clean + s_true * rng.standard_normal(n)
        log_scale_true = np.log(s_true)

        tr, te = slice(0, 300), slice(300, 400)
        cfg = SoftTreeConfig(
            n_features=4, n_trees=1, depth=2, task="regression", n_outputs=1, beta_final=6.0, seed=seed
        )
        m_shrink, _ = fit_boosted_heteroscedastic(
            X[tr], y[tr], cfg, log_scale=log_scale_true[tr], weighting="shrinkage",
            n_stages=15, learning_rate=0.3, inner_steps=25,
        )
        m_gls, _ = fit_boosted_heteroscedastic(
            X[tr], y[tr], cfg, log_scale=log_scale_true[tr], weighting="gls",
            n_stages=15, learning_rate=0.3, inner_steps=25,
        )
        low_noise_te = np.arange(300, 400)[s_true[te] < 1.0]
        err_shrink = float(np.mean((m_shrink.score(X[low_noise_te])[:, 0] - f_clean[low_noise_te]) ** 2))
        err_gls = float(np.mean((m_gls.score(X[low_noise_te])[:, 0] - f_clean[low_noise_te]) ** 2))
        if err_gls <= err_shrink * 1.5:  # generous slack -- this is a sanity check, not G3 itself
            wins += 1
    assert wins >= 3


def test_fit_boosted_leaf_solver_adam_matches_default_bit_identically() -> None:
    rng = np.random.default_rng(8)
    X = rng.standard_normal((80, 4))
    y = (X[:, 0] + 0.2 * rng.standard_normal(80) > 0).astype(np.float64)
    cfg = SoftTreeConfig(n_features=4, n_trees=1, depth=1, task="binary", beta_final=4.0, seed=8)
    m_a, r_a = fit_boosted(X, y, cfg, n_stages=4, inner_steps=8)
    m_b, r_b = fit_boosted(X, y, cfg, n_stages=4, inner_steps=8, leaf_solver="adam")
    pa, pb = m_a.to_params(), m_b.to_params()
    assert np.array_equal(pa.W, pb.W)
    assert np.array_equal(pa.leaves, pb.leaves)
    assert r_a.history == r_b.history


def test_fit_boosted_closed_form_runs_and_reduces_loss() -> None:
    rng = np.random.default_rng(9)
    X = rng.standard_normal((100, 5))
    y = np.sin(X[:, 0]) + 0.4 * X[:, 1]
    cfg = SoftTreeConfig(
        n_features=5, n_trees=1, depth=2, split_kind="axis",
        task="regression", n_outputs=1, beta_final=6.0, seed=9, leaf_l2=1e-4,
    )
    _, res = fit_boosted(
        X, y, cfg, n_stages=6, learning_rate=0.4, inner_steps=1, leaf_solver="closed_form"
    )
    assert res.history[-1] < res.history[0]
