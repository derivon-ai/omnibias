# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
r"""Numpy / torch / jax parity for axis forests and TabPOU band tokens."""

from __future__ import annotations

import numpy as np
import pytest
from omnibias.tab import SoftTreeConfig, forward_np, init_params
from omnibias.tab.torch.embed import BandFeatureEmbedder

torch = pytest.importorskip("torch")
jnp = pytest.importorskip("jax.numpy")


@pytest.mark.parametrize("beta", [0.5, 8.0])
def test_axis_forward_parity_numpy_torch_jax(beta: float) -> None:
    from omnibias.tab.jax.model import forward as jax_forward
    from omnibias.tab.torch.model import SoftTreeEnsemble

    cfg = SoftTreeConfig(
        n_features=5, n_trees=4, depth=2, split_kind="axis",
        task="regression", n_outputs=1, seed=11,
    )
    rng = np.random.default_rng(11)
    p = init_params(cfg, rng)
    p.t = rng.standard_normal(p.t.shape) * 0.4
    X = rng.standard_normal((16, 5))
    F_np = forward_np(p, X, beta)
    model = SoftTreeEnsemble(cfg, p)
    F_torch = model.score(X, beta=beta)
    F_jax = np.asarray(jax_forward(p, jnp.asarray(X), beta))
    assert np.max(np.abs(F_np - F_torch)) < 1e-9
    assert np.max(np.abs(F_np - F_jax)) < 1e-9


def test_pou_token_tree_parity_numpy_torch_jax() -> None:
    from omnibias.tab.jax.model import forward_arrays
    from omnibias.tab.jax.pou import pou_forward_arrays
    from omnibias.tab.torch.model import SoftTreeEnsemble

    rng = np.random.default_rng(5)
    X = rng.standard_normal((12, 3))
    n_bins = 4
    embedder = BandFeatureEmbedder(
        n_features=3, n_bins=n_bins, init="quantile", X_ref=X,
        learnable_beta=False, learnable_thresholds=False, role="band",
    )
    t = embedder.thresholds().detach().cpu().numpy()
    beta_e = float(embedder.beta)
    with torch.no_grad():
        E = embedder(torch.as_tensor(X, dtype=torch.float64)).cpu().numpy()
    tok = np.concatenate([E, X], axis=1)
    cfg = SoftTreeConfig(
        n_features=tok.shape[1], n_trees=2, depth=2, split_kind="axis",
        task="regression", n_outputs=1, seed=5,
    )
    p = init_params(cfg, 5)
    F_np = forward_np(p, tok, 6.0)
    F_torch = SoftTreeEnsemble(cfg, p).score(tok, beta=6.0)
    F_jax_tok = np.asarray(forward_arrays(
        jnp.asarray(p.W), jnp.asarray(p.t), jnp.asarray(p.leaves), jnp.asarray(p.b0),
        jnp.asarray(tok), 6.0, 2,
    ))
    F_jax_pou = np.asarray(pou_forward_arrays(
        jnp.asarray(X), jnp.asarray(t), beta_e,
        jnp.asarray(p.W), jnp.asarray(p.t), jnp.asarray(p.leaves), jnp.asarray(p.b0),
        6.0, 2, role="band", concat_raw=True, use_embed=True,
    ))
    assert np.max(np.abs(F_np - F_torch)) < 1e-9
    assert np.max(np.abs(F_np - F_jax_tok)) < 1e-9
    assert np.max(np.abs(F_np - F_jax_pou)) < 1e-9
