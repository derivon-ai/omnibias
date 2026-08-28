# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
r"""Theory 05-05: joint TabPOU polish, grouped splits, residual, G5."""

from __future__ import annotations

import numpy as np
import pytest
from omnibias.tab import SoftTreeConfig, certify_tab_gap, init_params, make_axis_rule
from omnibias.tab._core.forward import leaf_memberships
from omnibias.tab._core.leaves import newton_leaf_loss
from omnibias.tab.pou import TabPOUConfig, token_feature_groups
from omnibias.tab.pou.grow import fit_axis_tree

torch = pytest.importorskip("torch")

from omnibias.tab.pou.joint import (  # noqa: E402
    TabPOUJointConfig,
    fit_tabpou_joint,
    polish_thresholds,
)
from omnibias.tab.torch.model import SoftTreeEnsemble  # noqa: E402


def test_token_feature_groups_concat_raw() -> None:
    groups = token_feature_groups(3, n_bins=2, concat_raw=True, use_embed=True)
    assert len(groups) == 3
    # 3 bins per feature (n_bins+1) then raw offset 9.
    assert list(groups[0]) == [0, 1, 2, 9]
    assert list(groups[2]) == [6, 7, 8, 11]


def test_polish_thresholds_never_worse_on_stump() -> None:
    rng = np.random.default_rng(0)
    cfg = SoftTreeConfig(
        n_features=2, n_trees=1, depth=1, split_kind="axis",
        task="regression", n_outputs=1, seed=0, beta_final=6.0, leaf_l2=1e-4,
    )
    params = init_params(cfg, rng)
    X = rng.standard_normal((40, 2))
    r = rng.standard_normal((40, 1))
    h = np.ones((40, 1))
    P0 = leaf_memberships(params, X, cfg.beta_final)
    loss0 = newton_leaf_loss(P0, r, h, params.leaves)
    model = SoftTreeEnsemble(cfg, params)
    ratio = polish_thresholds(model, X, r, h, n_sweeps=3)
    assert ratio <= 1.0 + 1e-9
    P1 = leaf_memberships(model.to_params(), X, cfg.beta_final)
    loss1 = newton_leaf_loss(P1, r, h, model.to_params().leaves)
    assert loss1 <= loss0 + 1e-9


def test_fit_tabpou_joint_reduces_or_finite_on_sine() -> None:
    rng = np.random.default_rng(2)
    X = rng.standard_normal((60, 3))
    y = np.sin(X[:, 0]) + 0.2 * X[:, 1]
    base = TabPOUConfig(
        n_features=3, task="regression", depth=1, n_stages=4, n_quantiles=6,
        n_bins=4, use_embed=False, onehot_max_card=1, seed=2, patience=None,
        use_residual=False,
    )
    cfg = TabPOUJointConfig(base=base, polish_thresholds=True, polish_sweeps=2)
    model, res = fit_tabpou_joint(X, y, cfg)
    pred = model.predict(X)
    assert pred.shape == (60,)
    assert np.isfinite(pred).all()
    assert res.polished
    assert res.polish_loss_ratio <= 1.0 + 1e-8


def test_fit_tabpou_joint_g5_sound() -> None:
    X, y, _ = make_axis_rule(n_samples=120, n_features=8, seed=1)
    base = TabPOUConfig(
        n_features=8, task="binary", depth=1, n_stages=4, n_quantiles=6,
        use_embed=False, robust_scale=False, onehot_max_card=1, seed=1, patience=None,
    )
    cfg = TabPOUJointConfig(base=base, polish_thresholds=True, binarize_eval=False)
    model, _ = fit_tabpou_joint(X, y, cfg)
    cert = certify_tab_gap(model.to_params(), model.transform(X[:60]))
    assert cert.is_sound


def test_pairwise_grow_runs_on_axis_and() -> None:
    X, y, _ = make_axis_rule(n_samples=80, n_features=8, seed=4)
    residual = y.reshape(-1, 1) - 0.5
    weight = np.ones((80, 1))
    params = fit_axis_tree(
        X, residual, weight, depth=2, beta=8.0, leaf_l2=1e-4,
        n_quantiles=6, colsample=1.0, rng=np.random.default_rng(4),
        n_trees=1, pairwise=True,
    )
    assert params.W.shape == (1, 2, 8)
    assert int(np.count_nonzero(np.abs(params.W[0, 0]) > 1e-15)) == 1
    assert int(np.count_nonzero(np.abs(params.W[0, 1]) > 1e-15)) == 1


def test_embed_freeze_matches_train_when_joint_steps_zero() -> None:
    rng = np.random.default_rng(5)
    X = rng.standard_normal((40, 3))
    y = X[:, 0] - 0.2 * X[:, 1]
    base = TabPOUConfig(
        n_features=3, task="regression", depth=1, n_stages=3, n_quantiles=5,
        n_bins=3, use_embed=True, concat_raw=True, onehot_max_card=1, seed=5,
        patience=None, robust_scale=False,
    )
    frozen, _ = fit_tabpou_joint(X, y, TabPOUJointConfig(base=base, train_embed=False, joint_steps=0))
    trained, _ = fit_tabpou_joint(X, y, TabPOUJointConfig(base=base, train_embed=True, joint_steps=0))
    a = frozen.score(X)
    b = trained.score(X)
    assert np.max(np.abs(a - b)) < 1e-12


def test_grouped_splits_runs_with_band_tokens() -> None:
    rng = np.random.default_rng(6)
    X = rng.standard_normal((36, 2))
    y = np.sin(X[:, 0])
    base = TabPOUConfig(
        n_features=2, task="regression", depth=1, n_stages=3, n_quantiles=5,
        n_bins=3, use_embed=True, concat_raw=True, onehot_max_card=1, seed=6,
        patience=None, robust_scale=False,
    )
    model, res = fit_tabpou_joint(
        X, y, TabPOUJointConfig(base=base, grouped_splits=True, joint_steps=0)
    )
    assert res.grouped_splits
    assert np.isfinite(model.predict(X)).all()


def test_joint_residual_flag_attaches_head() -> None:
    rng = np.random.default_rng(3)
    X = rng.standard_normal((50, 2))
    y = X[:, 0] ** 2
    base = TabPOUConfig(
        n_features=2, task="regression", depth=1, n_stages=3, n_quantiles=5,
        use_embed=False, onehot_max_card=1, seed=3, patience=None,
        residual_k=2, residual_hidden=4, residual_steps=8,
    )
    cfg = TabPOUJointConfig(base=base, joint_residual=True, joint_steps=6)
    model, res = fit_tabpou_joint(X, y, cfg)
    assert res.used_residual
    assert model.residual is not None
    assert np.isfinite(model.predict(X)).all()
