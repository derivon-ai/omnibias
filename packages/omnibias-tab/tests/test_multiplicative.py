# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
r"""The multiplicative (depth>=2) oblivious soft-tree tier: soft-AND routing + interactions."""

from __future__ import annotations

import numpy as np
import pytest
from omnibias.tab import SoftTreeConfig, TabParams, forward_np, init_params, leaf_memberships


@pytest.mark.parametrize("depth", [1, 2, 3])
@pytest.mark.parametrize("beta", [0.7, 3.0, 20.0])
def test_memberships_form_a_distribution(depth: int, beta: float) -> None:
    cfg = SoftTreeConfig(n_features=5, n_trees=6, depth=depth, task="binary")
    p = init_params(cfg, 0)
    rng = np.random.default_rng(1)
    X = rng.standard_normal((25, 5))
    P = leaf_memberships(p, X, beta)  # (n, T, L)
    assert P.shape == (25, 6, 1 << depth)
    assert np.all(P >= 0.0)
    # each tree routes each sample as a probability distribution over its leaves
    sums = P.sum(axis=-1)
    assert np.max(np.abs(sums - 1.0)) < 1e-10


def test_depth2_tree_represents_xor_exactly() -> None:
    """A single depth-2 oblivious soft tree represents XOR of two features natively."""
    cfg = SoftTreeConfig(n_features=2, n_trees=1, depth=2, task="binary", seed=0)
    W = np.array([[[1.0, 0.0], [0.0, 1.0]]])  # gate0 reads x0, gate1 reads x1
    t = np.zeros((1, 2))
    # leaf bit j = gate j; XOR true when exactly one gate fires -> leaves 1 and 2
    leaves = np.array([[[-5.0], [5.0], [5.0], [-5.0]]])  # (1, 4, 1)
    b0 = np.zeros(1)
    p = TabParams(cfg, W, t, leaves, b0)

    # points well inside the four quadrants (gates saturated at high beta)
    pts = np.array([[1.0, 1.0], [1.0, -1.0], [-1.0, 1.0], [-1.0, -1.0]])
    xor = np.array([0.0, 1.0, 1.0, 0.0])
    F = forward_np(p, pts, beta=50.0)
    pred = (F[:, 0] > 0.0).astype(np.float64)
    assert np.array_equal(pred, xor)


def test_numpy_torch_parity_depth3() -> None:
    torch = pytest.importorskip("torch")
    from omnibias.tab.torch.model import SoftTreeEnsemble

    cfg = SoftTreeConfig(n_features=5, n_trees=4, depth=3, task="multiclass", n_outputs=3, seed=2)
    p = init_params(cfg, 2, leaf_scale=0.5)
    rng = np.random.default_rng(3)
    X = rng.standard_normal((16, 5))
    beta = 4.0
    F_np = forward_np(p, X, beta)
    F_torch = SoftTreeEnsemble(cfg, p).score(X, beta=beta)
    assert np.max(np.abs(F_np - F_torch)) < 1e-9
    del torch
