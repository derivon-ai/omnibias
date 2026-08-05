# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
r"""Bit-identical (``~1e-9``) forward parity: numpy reference vs torch vs jax."""

from __future__ import annotations

import numpy as np
import pytest
from omnibias.tab import SoftTreeConfig, forward_np, hard_forward_np, init_params

torch = pytest.importorskip("torch")
jnp = pytest.importorskip("jax.numpy")


def _params(depth: int, n_outputs: int, task: str, seed: int = 0):
    cfg = SoftTreeConfig(
        n_features=6, n_trees=8, depth=depth, task=task, n_outputs=n_outputs, seed=seed
    )
    rng = np.random.default_rng(seed)
    p = init_params(cfg, rng, leaf_scale=0.5)
    # non-trivial thresholds so gates are not all centred at 0
    p.t = rng.standard_normal(p.t.shape) * 0.5
    return cfg, p


@pytest.mark.parametrize("depth,n_outputs,task", [(1, 1, "binary"), (2, 1, "binary"), (3, 3, "multiclass"), (2, 2, "regression")])
@pytest.mark.parametrize("beta", [0.5, 3.0, 12.0])
def test_forward_parity_numpy_torch_jax(depth: int, n_outputs: int, task: str, beta: float) -> None:
    from omnibias.tab.jax.model import forward as jax_forward
    from omnibias.tab.torch.model import SoftTreeEnsemble

    cfg, p = _params(depth, n_outputs, task)
    rng = np.random.default_rng(depth * 100 + n_outputs)
    X = rng.standard_normal((20, cfg.n_features))

    F_np = forward_np(p, X, beta)

    model = SoftTreeEnsemble(cfg, p)
    F_torch = model.score(X, beta=beta)

    F_jax = np.asarray(jax_forward(p, jnp.asarray(X), beta))

    assert F_np.shape == (20, n_outputs)
    assert np.max(np.abs(F_np - F_torch)) < 1e-9
    assert np.max(np.abs(F_np - F_jax)) < 1e-9


def test_soft_converges_to_hard_as_beta_grows() -> None:
    """For points with a margin, ``sigmoid(beta z) -> step(z)`` so soft -> hard scores."""
    cfg, p = _params(depth=2, n_outputs=1, task="binary")
    rng = np.random.default_rng(7)
    X = rng.standard_normal((32, cfg.n_features))
    # keep only points whose every gate is well away from its boundary
    z = np.einsum("nd,mjd->nmj", X, p.W) - p.t[None, :, :]
    keep = np.min(np.abs(z).reshape(X.shape[0], -1), axis=1) > 0.15
    Xk = X[keep]
    hard = hard_forward_np(p, Xk)
    diff_small = np.max(np.abs(forward_np(p, Xk, 200.0) - hard))
    diff_large = np.max(np.abs(forward_np(p, Xk, 5.0) - hard))
    assert diff_small < diff_large
    assert diff_small < 1e-3


def test_depth1_is_additive_sum_of_sigmoids() -> None:
    """depth-1 forward equals the closed additive form b0 + sum_m g_m (leaf1 - leaf0)."""
    cfg, p = _params(depth=1, n_outputs=1, task="binary")
    rng = np.random.default_rng(3)
    X = rng.standard_normal((16, cfg.n_features))
    beta = 2.5
    from omnibias.tab._core.forward import sigmoid_np

    z = np.einsum("nd,mjd->nmj", X, p.W)[:, :, 0] - p.t[None, :, 0]
    g = sigmoid_np(beta * z)  # (n, T)
    u = p.leaves[:, 1, 0] - p.leaves[:, 0, 0]  # (T,)
    const = p.b0[0] + np.sum(p.leaves[:, 0, 0])
    F_additive = const + g @ u
    assert np.max(np.abs(F_additive - forward_np(p, X, beta)[:, 0])) < 1e-10
