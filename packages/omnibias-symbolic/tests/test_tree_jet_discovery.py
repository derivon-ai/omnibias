# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
"""Soft-tree jets: NeuralJetDiscoverer recovers dy ≈ y and a two-regime per-leaf gate."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from omnibias.tab import SoftTreeConfig
from omnibias.tab.torch.jet import extract_tree_jet
from omnibias.tab.torch.model import SoftTreeEnsemble
from omnibias.tab.torch.train import fit_second_order


def test_depth1_tree_exp_jet_recovers_dy_approx_y() -> None:
    pytest.importorskip("omnibias.symbolic")
    from omnibias.symbolic.discovery import JetBundle, NeuralJetDiscoverer

    xmin, xmax = -0.6, 0.6
    n = 200
    x = np.linspace(xmin, xmax, n)
    y = np.exp(x)
    cfg = SoftTreeConfig(
        n_features=1,
        n_trees=24,
        depth=1,
        task="regression",
        n_outputs=1,
        beta_final=2.5,
        seed=0,
        leaf_l2=1e-6,
    )
    torch.manual_seed(0)
    model = SoftTreeEnsemble(cfg)
    fit_second_order(
        model,
        x.reshape(-1, 1),
        y,
        steps=150,
        anneal=False,
        leaf_l2=1e-6,
        weight_l2=1e-6,
    )
    pred = model.score(x.reshape(-1, 1))[:, 0]
    fit_rmse = float(np.sqrt(np.mean((pred - y) ** 2)))
    assert fit_rmse < 0.08

    tj = extract_tree_jet(model.to_params(), x.reshape(-1, 1), beta=2.5, max_order=1)
    jets = np.stack([tj.value(), tj.partial((1,))], axis=1)
    idx = np.arange(n)
    tr, va, te = idx[0::3], idx[1::3], idx[2::3]
    disc = NeuralJetDiscoverer(max_library_degree=1)
    result = disc.discover(
        JetBundle(x=x[tr], jets=jets[tr]),
        JetBundle(x=x[va], jets=jets[va]),
        JetBundle(x=x[te], jets=jets[te]),
        candidate_lhs_orders=(1,),
    )
    assert result.test_rmse < 0.05
    names = list(result.equation.term_names)
    assert "y" in names
    c_y = float(result.equation.coefficients[names.index("y")])
    assert abs(result.equation.intercept) < 0.15
    assert abs(c_y - 1.0) < 0.25


def test_neural_jet_discoverer_per_hard_leaf() -> None:
    pytest.importorskip("omnibias.symbolic")
    from omnibias.partition._core.weights import hard_assignment
    from omnibias.symbolic.discovery import JetBundle, NeuralJetDiscoverer
    from omnibias.tab import SoftTreeConfig, tree_params
    from omnibias.tab.torch.jet import extract_tree_jet

    xmin, xmax = -0.8, 0.8
    n = 300
    x = np.linspace(xmin, xmax, n)
    y = np.where(x < 0.0, np.exp(x), np.exp(-2.0 * x))
    cfg = SoftTreeConfig(
        n_features=1,
        n_trees=16,
        depth=2,
        task="regression",
        n_outputs=1,
        beta_final=2.5,
        seed=0,
        leaf_l2=1e-6,
    )
    torch.manual_seed(0)
    model = SoftTreeEnsemble(cfg)
    fit_second_order(
        model,
        x.reshape(-1, 1),
        y,
        steps=120,
        anneal=False,
        leaf_l2=1e-6,
        weight_l2=1e-6,
    )
    pred = model.score(x.reshape(-1, 1), beta=2.5)[:, 0]
    assert float(np.sqrt(np.mean((pred - y) ** 2))) < 0.08

    p = model.to_params()
    tj = extract_tree_jet(p, x.reshape(-1, 1), beta=2.5, max_order=1)
    val = tj.value()
    dval = tj.partial((1,))
    jets = np.stack([val, dval], axis=1)
    disc = NeuralJetDiscoverer(
        max_library_degree=1, include_x=False, complexity_weight=0.05
    )
    found_same_tree = False
    xx = x.reshape(-1, 1)
    n_trees = int(p.W.shape[0])
    for m in range(n_trees):
        idx = hard_assignment(tree_params(p.W[m], p.t[m]), xx)
        far_left = False
        far_right = False
        far_leaves: list[tuple[np.ndarray, float]] = []
        for region in np.unique(idx):
            far = (idx == region) & (np.abs(x) > 0.2)
            if int(far.sum()) < 24:
                continue
            med = float(np.median(x[far]))
            if med < -0.1:
                far_left = True
                far_leaves.append((far, med))
            elif med > 0.1:
                far_right = True
                far_leaves.append((far, med))
        if not (far_left and far_right):
            continue
        tree_left = False
        tree_right = False
        for far, _med in far_leaves:
            jb = jets[far]
            ident_left = float(np.sqrt(np.mean((jb[:, 1] - jb[:, 0]) ** 2)))
            ident_right = float(np.sqrt(np.mean((jb[:, 1] + 2.0 * jb[:, 0]) ** 2)))
            if ident_left >= 0.05 and ident_right >= 0.05:
                continue
            if float(np.std(jb[:, 0])) < 0.08:
                continue
            xs = x[far]
            nn_ = xs.size
            tr, va, te = np.arange(nn_)[0::3], np.arange(nn_)[1::3], np.arange(nn_)[2::3]
            result = disc.discover(
                JetBundle(x=xs[tr], jets=jb[tr]),
                JetBundle(x=xs[va], jets=jb[va]),
                JetBundle(x=xs[te], jets=jb[te]),
                candidate_lhs_orders=(1,),
            )
            names = list(result.equation.term_names)
            assert "y" in names
            c_y = float(result.equation.coefficients[names.index("y")])
            assert result.test_rmse < 0.1
            if ident_left < 0.05:
                assert abs(c_y - 1.0) < 0.35
                tree_left = True
            if ident_right < 0.05:
                assert abs(c_y + 2.0) < 0.4
                tree_right = True
        if tree_left and tree_right:
            found_same_tree = True
    assert found_same_tree
