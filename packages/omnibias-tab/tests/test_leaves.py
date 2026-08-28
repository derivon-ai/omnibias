# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
r"""G0 identity: closed-form Newton leaves vs Adam-on-leaves; axis one-hot init."""

from __future__ import annotations

import numpy as np
import pytest
from omnibias.tab import SoftTreeConfig, init_params
from omnibias.tab._core.forward import leaf_memberships
from omnibias.tab._core.leaves import closed_form_leaves, newton_leaf_loss


def test_closed_form_leaves_matches_lstsq_on_stump() -> None:
    rng = np.random.default_rng(0)
    n, L, k = 40, 2, 1
    P = rng.random((n, 1, L))
    P = P / P.sum(axis=-1, keepdims=True)
    residual = rng.standard_normal((n, k))
    weight = np.ones((n, k))
    leaves = closed_form_leaves(P, residual, weight, leaf_l2=0.0)
    # Independent lstsq on the same design.
    A = P[:, 0, :]
    ell, *_ = np.linalg.lstsq(A, residual[:, 0], rcond=None)
    assert np.max(np.abs(leaves[0, :, 0] - ell)) < 1e-8


def test_g0_closed_form_beats_or_matches_long_adam_on_frozen_gates() -> None:
    torch = pytest.importorskip("torch")
    from omnibias.tab.torch.model import SoftTreeEnsemble

    cfg = SoftTreeConfig(
        n_features=4, n_trees=1, depth=2, task="regression", n_outputs=1, seed=3, beta_final=6.0
    )
    rng = np.random.default_rng(3)
    X = rng.standard_normal((80, 4))
    residual = rng.standard_normal((80, 1))
    weight = np.full((80, 1), 2.0)
    params = init_params(cfg, rng)
    P = leaf_memberships(params, X, cfg.beta_final)
    leaves_cf = closed_form_leaves(P, residual, weight, leaf_l2=cfg.leaf_l2)
    loss_cf = newton_leaf_loss(P, residual, weight, leaves_cf)

    model = SoftTreeEnsemble(cfg, params)
    model.set_beta(cfg.beta_final)
    model.W.requires_grad_(False)
    model.t.requires_grad_(False)
    model.b0.requires_grad_(False)
    Xt = torch.as_tensor(X)
    rt = torch.as_tensor(residual)
    wt = torch.as_tensor(weight)
    opt = torch.optim.Adam([model.leaves], lr=0.05)
    for _ in range(400):
        opt.zero_grad(set_to_none=True)
        F = model(Xt)
        loss = (wt * (F - rt) ** 2).mean()
        loss.backward()
        opt.step()
    leaves_ad = model.to_params().leaves
    loss_ad = newton_leaf_loss(P, residual, weight, leaves_ad)
    assert loss_cf / max(loss_ad, 1e-30) <= 1.01


def test_axis_init_is_one_hot_per_gate() -> None:
    cfg = SoftTreeConfig(
        n_features=5, n_trees=3, depth=2, split_kind="axis", task="binary", seed=7
    )
    p = init_params(cfg, 7)
    for m in range(3):
        for j in range(2):
            row = p.W[m, j]
            assert np.count_nonzero(np.abs(row) > 1e-15) == 1
            assert float(row.max()) == 1.0


def test_split_kind_rejects_unknown() -> None:
    with pytest.raises(ValueError, match="split_kind"):
        SoftTreeConfig(n_features=2, split_kind="diagonal")
