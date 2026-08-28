# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
r"""Theory 05-03: bit-identical (``~1e-9``) embed forward parity: numpy vs torch vs jax."""

from __future__ import annotations

import numpy as np
import pytest
from omnibias.core.band_embed import band_embed, integral_embed

torch = pytest.importorskip("torch")
jnp = pytest.importorskip("jax.numpy")


@pytest.mark.parametrize("role", ["band", "integral"])
@pytest.mark.parametrize("n_features,n_bins", [(1, 1), (3, 3), (5, 7)])
@pytest.mark.parametrize("beta", [0.3, 1.0, 7.5])
def test_embed_forward_parity_numpy_torch_jax(n_features: int, n_bins: int, beta: float, role: str) -> None:
    from omnibias.tab.jax.embed import band_feature_embed
    from omnibias.tab.torch.embed import BandFeatureEmbedder

    rng = np.random.default_rng(n_features * 10 + n_bins)
    thresholds = np.sort(rng.uniform(-2.0, 2.0, size=(n_features, n_bins)), axis=-1)
    X = rng.standard_normal((16, n_features))
    core_fn = band_embed if role == "band" else integral_embed

    F_np = np.stack([core_fn(X[:, k], thresholds[k], beta=beta) for k in range(n_features)], axis=1).reshape(
        16, n_features * (n_bins + 1)
    )

    model = BandFeatureEmbedder(n_features, n_bins, role=role, thresholds_init=thresholds, learnable_beta=False)
    model.set_beta(beta)
    with torch.no_grad():
        F_torch = model(torch.as_tensor(X, dtype=torch.float64)).numpy()

    F_jax = np.asarray(band_feature_embed(jnp.asarray(X), jnp.asarray(thresholds), beta, role=role))

    assert F_np.shape == (16, n_features * (n_bins + 1))
    assert np.max(np.abs(F_np - F_torch)) < 1e-9
    assert np.max(np.abs(F_np - F_jax)) < 1e-9


def test_embed_forward_parity_matches_trained_torch_thresholds() -> None:
    """After a few gradient steps (non-trivial thresholds), all three backends still agree."""
    from omnibias.tab.jax.embed import band_feature_embed
    from omnibias.tab.torch.embed import BandFeatureEmbedder

    torch.manual_seed(3)
    emb = BandFeatureEmbedder(n_features=3, n_bins=4, init="uniform")
    X = torch.randn(40, 3, dtype=torch.float64)
    opt = torch.optim.Adam(emb.parameters(), lr=0.1)
    for _ in range(10):
        opt.zero_grad()
        loss = (emb(X) ** 2).sum()
        loss.backward()
        opt.step()

    t = emb.thresholds().detach().numpy()
    beta = emb.beta
    X_np = X.numpy()

    F_np = np.stack([band_embed(X_np[:, k], t[k], beta=beta) for k in range(3)], axis=1).reshape(40, 15)
    with torch.no_grad():
        F_torch = emb(X).numpy()
    F_jax = np.asarray(band_feature_embed(jnp.asarray(X_np), jnp.asarray(t), beta))

    assert np.max(np.abs(F_np - F_torch)) < 1e-9
    assert np.max(np.abs(F_np - F_jax)) < 1e-9
