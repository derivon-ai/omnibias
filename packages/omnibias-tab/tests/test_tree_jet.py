# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
"""Soft-tree jets: mlp_jet depth-1 parity, extract_tree_jet Leibniz product, torch/jax/AD."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from omnibias.tab import SoftTreeConfig, init_params
from omnibias.tab.torch.jet import (
    extract_arrangement_jet,
    extract_tree_jet,
    extract_tree_jet_directional,
    sequential_mlp_jet,
)
from omnibias.tab.torch.model import SoftTreeEnsemble
from omnibias.torch.jet import jet_to_tower


def test_depth1_mlp_jet_matches_autodiff() -> None:
    cfg = SoftTreeConfig(n_features=3, n_trees=5, depth=1, task="regression", seed=2)
    model = SoftTreeEnsemble(cfg)
    beta = 3.0
    seq = model.to_additive_sequential(beta)
    x0 = torch.tensor([0.2, -0.1, 0.4], dtype=torch.float64)
    v = torch.tensor([1.0, 0.3, -0.2], dtype=torch.float64)
    order = 2
    jet = sequential_mlp_jet(seq, x0, v, order)
    tower = jet_to_tower(jet)[:, 0]

    x0 = x0.detach().requires_grad_(True)
    y = seq(x0.unsqueeze(0)).reshape(-1)[0]
    g = torch.autograd.grad(y, x0, create_graph=True)[0]
    dy = (g * v).sum()
    h = torch.autograd.grad(dy, x0, create_graph=True)[0]
    d2y = (h * v).sum()
    assert abs(float((tower[0] - y).detach())) < 1e-8
    assert abs(float((tower[1] - dy).detach())) < 1e-8
    assert abs(float((tower[2] - d2y).detach())) < 1e-8


def test_extract_tree_jet_matches_forward_and_autodiff() -> None:
    cfg = SoftTreeConfig(n_features=2, n_trees=2, depth=2, task="regression", seed=4)
    p = init_params(cfg, 4)
    model = SoftTreeEnsemble(cfg, p)
    rng = np.random.default_rng(3)
    X = rng.standard_normal((8, 2))
    beta = 5.0
    tj = extract_tree_jet(p, X, beta=beta, max_order=1)
    Xt = torch.as_tensor(X, dtype=torch.float64)
    y = model(Xt, beta=beta)[:, 0]
    assert np.max(np.abs(tj.value() - y.detach().numpy())) < 1e-8
    # FD on first partial wrt x0
    eps = 1e-5
    Xp = X.copy()
    Xm = X.copy()
    Xp[:, 0] += eps
    Xm[:, 0] -= eps
    with torch.no_grad():
        fp = model(torch.as_tensor(Xp, dtype=torch.float64), beta=beta)[:, 0].numpy()
        fm = model(torch.as_tensor(Xm, dtype=torch.float64), beta=beta)[:, 0].numpy()
    fd = (fp - fm) / (2 * eps)
    assert np.max(np.abs(tj.partial((1, 0)) - fd)) < 5e-4
    Xt = torch.as_tensor(X, dtype=torch.float64)
    Xt.requires_grad_(True)
    y_sum = model(Xt, beta=beta)[:, 0].sum()
    g = torch.autograd.grad(y_sum, Xt)[0]
    assert np.max(np.abs(tj.partial((1, 0)) - g[:, 0].detach().numpy())) < 1e-6


def test_extract_tree_jet_torch_jax_parity() -> None:
    pytest.importorskip("jax")
    from omnibias.tab.jax.jet import extract_tree_jet as extract_jax

    cfg = SoftTreeConfig(n_features=2, n_trees=1, depth=2, task="regression", seed=7)
    p = init_params(cfg, 7)
    rng = np.random.default_rng(1)
    X = rng.standard_normal((6, 2))
    beta = 4.0
    tt = extract_tree_jet(p, X, beta=beta, max_order=1)
    tj = extract_jax(p, X, beta=beta, max_order=1)
    assert np.max(np.abs(tt.value() - tj.value())) < 1e-9
    assert np.max(np.abs(tt.partial((1, 0)) - tj.partial((1, 0)))) < 1e-9
    assert np.max(np.abs(tt.partial((0, 1)) - tj.partial((0, 1)))) < 1e-9


def test_extract_arrangement_jet_matches_classifier() -> None:
    from omnibias.tab.torch.arrangement import ArrangementClassifier

    rng = np.random.default_rng(2)
    X = rng.standard_normal((7, 3))
    W = rng.standard_normal((2, 3)) * 0.4
    t = rng.standard_normal((2,)) * 0.1
    cell = rng.standard_normal((4, 1))
    beta = 3.0
    model = ArrangementClassifier(3, 2, beta=beta)
    with torch.no_grad():
        model.W.copy_(torch.as_tensor(W, dtype=torch.float64))
        model.t.copy_(torch.as_tensor(t, dtype=torch.float64))
        model.cell_logits.copy_(torch.as_tensor(cell, dtype=torch.float64))
    tj = extract_arrangement_jet(W, t, cell, X, beta=beta, max_order=0)
    pred = model(torch.as_tensor(X, dtype=torch.float64)).detach().numpy()[:, 0]
    assert np.max(np.abs(tj.value() - pred)) < 1e-8


def test_directional_jet_finite_difference() -> None:
    cfg = SoftTreeConfig(n_features=2, n_trees=1, depth=2, task="regression", seed=5)
    p = init_params(cfg, 5)
    model = SoftTreeEnsemble(cfg, p)
    X = np.array([[0.1, -0.2], [0.3, 0.4]])
    v = np.array([1.0, 0.0])
    beta = 2.5
    tower = extract_tree_jet_directional(p, X, v, beta=beta, max_order=1)
    eps = 1e-5
    with torch.no_grad():
        f0 = model(torch.as_tensor(X, dtype=torch.float64), beta=beta)[:, 0].numpy()
        fp = model(torch.as_tensor(X + eps * v, dtype=torch.float64), beta=beta)[:, 0].numpy()
        fm = model(torch.as_tensor(X - eps * v, dtype=torch.float64), beta=beta)[:, 0].numpy()
    assert np.max(np.abs(tower[0] - f0)) < 1e-8
    assert np.max(np.abs(tower[1] - (fp - fm) / (2 * eps))) < 5e-4


def test_depth1_extract_tree_jet_mlp_matches_sequential_and_leibniz() -> None:
    cfg = SoftTreeConfig(n_features=3, n_trees=4, depth=1, task="regression", seed=3)
    p = init_params(cfg, 3)
    model = SoftTreeEnsemble(cfg, p)
    beta = 2.5
    rng = np.random.default_rng(4)
    X = rng.standard_normal((5, 3))
    tj = extract_tree_jet(p, X, beta=beta, max_order=1)
    Xt = torch.as_tensor(X, dtype=torch.float64)
    y = model(Xt, beta=beta)[:, 0]
    assert np.max(np.abs(tj.value() - y.detach().numpy())) < 1e-8
    Xt.requires_grad_(True)
    y_sum = model(Xt, beta=beta)[:, 0].sum()
    g = torch.autograd.grad(y_sum, Xt)[0]
    assert np.max(np.abs(tj.partial((1, 0, 0)) - g[:, 0].detach().numpy())) < 1e-6
    assert np.max(np.abs(tj.partial((0, 1, 0)) - g[:, 1].detach().numpy())) < 1e-6
    assert np.max(np.abs(tj.partial((0, 0, 1)) - g[:, 2].detach().numpy())) < 1e-6

    x0 = torch.as_tensor(X[0], dtype=torch.float64)
    v = torch.tensor([0.4, -0.2, 0.7], dtype=torch.float64)
    seq = model.to_additive_sequential(beta)
    tower = jet_to_tower(sequential_mlp_jet(seq, x0, v, 1))[:, 0]
    dy = (
        float(tj.partial((1, 0, 0))[0]) * float(v[0])
        + float(tj.partial((0, 1, 0))[0]) * float(v[1])
        + float(tj.partial((0, 0, 1))[0]) * float(v[2])
    )
    assert abs(float(tj.value()[0]) - float(tower[0].detach())) < 1e-8
    assert abs(dy - float(tower[1].detach())) < 1e-8

    from omnibias.tab.torch.jet import _as_arrays, _tree_jet_one, _tree_jet_one_product

    W, t, leaves, b0 = _as_arrays(p)
    mlp = _tree_jet_one(W, t, leaves, b0, x0, beta, 1)
    leib = _tree_jet_one_product(W, t, leaves, b0, x0, beta, 1)
    assert torch.allclose(mlp, leib, atol=1e-8, rtol=1e-8)

    pytest.importorskip("jax")
    from omnibias.tab.jax.jet import extract_tree_jet as extract_jax

    tj_j = extract_jax(p, X, beta=beta, max_order=1)
    assert np.max(np.abs(tj.value() - tj_j.value())) < 1e-9
    assert np.max(np.abs(tj.partial((1, 0, 0)) - tj_j.partial((1, 0, 0)))) < 1e-9


def test_depth1_extract_tree_jet_directional_mlp_matches_sequential() -> None:
    cfg = SoftTreeConfig(n_features=3, n_trees=4, depth=1, task="regression", seed=3)
    p = init_params(cfg, 3)
    model = SoftTreeEnsemble(cfg, p)
    beta = 2.5
    rng = np.random.default_rng(5)
    X = rng.standard_normal((4, 3))
    v = np.array([0.4, -0.2, 0.7])
    tower = extract_tree_jet_directional(p, X, v, beta=beta, max_order=1)
    x0 = torch.as_tensor(X[0], dtype=torch.float64)
    vt = torch.as_tensor(v, dtype=torch.float64)
    seq = model.to_additive_sequential(beta)
    seq_tower = jet_to_tower(sequential_mlp_jet(seq, x0, vt, 1))[:, 0]
    assert abs(float(tower[0, 0]) - float(seq_tower[0].detach())) < 1e-8
    assert abs(float(tower[1, 0]) - float(seq_tower[1].detach())) < 1e-8

    Xt = torch.as_tensor(X, dtype=torch.float64)
    Xt.requires_grad_(True)
    y = model(Xt, beta=beta)[:, 0]
    assert np.max(np.abs(tower[0] - y.detach().numpy())) < 1e-8
    g = torch.autograd.grad(y.sum(), Xt)[0]
    directional = (g * vt).sum(dim=-1).detach().numpy()
    assert np.max(np.abs(tower[1] - directional)) < 1e-6

    from omnibias.tab.torch.jet import (
        _as_arrays,
        _tree_jet_dir_one,
        _tree_jet_dir_one_product,
    )

    W, t, leaves, b0 = _as_arrays(p)
    mlp = _tree_jet_dir_one(W, t, leaves, b0, x0, vt, beta, 1)
    leib = _tree_jet_dir_one_product(W, t, leaves, b0, x0, vt, beta, 1)
    assert torch.allclose(mlp, leib, atol=1e-8, rtol=1e-8)

    pytest.importorskip("jax")
    from omnibias.tab.jax.jet import extract_tree_jet_directional as extract_jax_dir

    tower_j = extract_jax_dir(p, X, v, beta=beta, max_order=1)
    assert np.max(np.abs(tower - tower_j)) < 1e-9
