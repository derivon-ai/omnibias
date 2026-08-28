# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
r"""Theory 05-03: BandFeatureEmbedder + local_target_consistency_loss (torch)."""

from __future__ import annotations

import numpy as np
import pytest
from omnibias.core.band_embed import band_embed, integral_embed, quantile_thresholds

torch = pytest.importorskip("torch")

from omnibias.tab.torch.embed import (  # noqa: E402, I001
    BandFeatureEmbedder,
    local_target_consistency_loss,
)


def test_forward_shape_and_sum_to_one() -> None:
    emb = BandFeatureEmbedder(n_features=3, n_bins=4, init="uniform")
    X = torch.randn(10, 3, dtype=torch.float64)
    out = emb(X)
    assert out.shape == (10, 3 * 5)
    per_feature = out.reshape(10, 3, 5)
    sums = per_feature.sum(dim=-1)
    assert torch.allclose(sums, torch.ones_like(sums), atol=1e-10)


def test_forward_matches_core_band_embed() -> None:
    """The torch reimplementation reproduces the numpy reference bit-for-bit (~1e-9)."""
    n_features, n_bins = 4, 5
    thresholds_init = np.sort(np.random.default_rng(0).uniform(-2.0, 2.0, size=(n_features, n_bins)), axis=-1)
    emb = BandFeatureEmbedder(n_features, n_bins, beta_init=2.3, thresholds_init=thresholds_init)
    X_np = np.random.default_rng(1).standard_normal((20, n_features))
    X = torch.as_tensor(X_np, dtype=torch.float64)
    with torch.no_grad():
        got = emb(X).reshape(20, n_features, n_bins + 1).numpy()
    expected = np.stack(
        [band_embed(X_np[:, k], thresholds_init[k], beta=2.3) for k in range(n_features)], axis=1
    )
    assert np.max(np.abs(got - expected)) < 1e-9


def test_forward_integral_role_matches_core_integral_embed() -> None:
    n_features, n_bins = 3, 4
    thresholds_init = np.sort(np.random.default_rng(2).uniform(-2.0, 2.0, size=(n_features, n_bins)), axis=-1)
    emb = BandFeatureEmbedder(n_features, n_bins, beta_init=1.7, role="integral", thresholds_init=thresholds_init)
    X_np = np.random.default_rng(3).standard_normal((15, n_features))
    X = torch.as_tensor(X_np, dtype=torch.float64)
    with torch.no_grad():
        got = emb(X).reshape(15, n_features, n_bins + 1).numpy()
    expected = np.stack(
        [integral_embed(X_np[:, k], thresholds_init[k], beta=1.7) for k in range(n_features)], axis=1
    )
    assert np.max(np.abs(got - expected)) < 1e-9


def test_integral_role_is_derivative_of_band_role() -> None:
    """d/dX embedder_integral(X) == beta * embedder_band(X), sharing the same thresholds/beta."""
    n_features, n_bins, beta = 2, 3, 2.0
    thresholds_init = np.sort(np.random.default_rng(4).uniform(-1.5, 1.5, size=(n_features, n_bins)), axis=-1)
    band = BandFeatureEmbedder(n_features, n_bins, beta_init=beta, role="band", thresholds_init=thresholds_init)
    integ = BandFeatureEmbedder(n_features, n_bins, beta_init=beta, role="integral", thresholds_init=thresholds_init)
    X = torch.randn(10, n_features, dtype=torch.float64)
    h = 1e-6
    with torch.no_grad():
        fd = (integ(X + h) - integ(X - h)) / (2 * h)
        analytic = beta * band(X)
    assert torch.max(torch.abs(fd - analytic)) < 1e-6


def test_thresholds_stay_strictly_increasing_after_gradient_step() -> None:
    emb = BandFeatureEmbedder(n_features=2, n_bins=3, init="uniform")
    opt = torch.optim.SGD(emb.parameters(), lr=0.5)
    X = torch.randn(30, 2, dtype=torch.float64)
    for _ in range(20):
        opt.zero_grad()
        out = emb(X)
        loss = (out**2).sum()
        loss.backward()
        opt.step()
    t = emb.thresholds()
    assert torch.all(torch.diff(t, dim=-1) > 0.0)


def test_quantile_init_matches_core_helper() -> None:
    rng = np.random.default_rng(2)
    X_ref = rng.exponential(size=(500, 3))
    emb = BandFeatureEmbedder(n_features=3, n_bins=6, init="quantile", X_ref=X_ref)
    t = emb.thresholds().detach().numpy()
    for k in range(3):
        expected = quantile_thresholds(X_ref[:, k], n_bins=7)
        assert np.allclose(t[k], expected, atol=1e-8)


def test_quantile_init_requires_x_ref() -> None:
    with pytest.raises(ValueError, match="X_ref"):
        BandFeatureEmbedder(n_features=2, n_bins=4, init="quantile")


def test_learnable_thresholds_false_freezes_parameters() -> None:
    emb = BandFeatureEmbedder(n_features=2, n_bins=3, init="uniform", learnable_thresholds=False)
    names = {name for name, _ in emb.named_parameters()}
    assert "t_first" not in names
    assert "_raw_gaps" not in names
    assert "_beta" in names  # beta is still learnable by default


def test_hardens_as_beta_grows() -> None:
    emb = BandFeatureEmbedder(n_features=1, n_bins=1, thresholds_init=np.array([[0.0]]), learnable_beta=False)
    inside = torch.tensor([[0.5]], dtype=torch.float64)
    emb.set_beta(1.0)
    soft = emb(inside)[0, 1].item()
    emb.set_beta(300.0)
    sharp = emb(inside)[0, 1].item()
    assert abs(sharp - 1.0) < 1e-6
    assert abs(sharp - 1.0) < abs(soft - 1.0)


def test_rejects_invalid_n_bins() -> None:
    with pytest.raises(ValueError, match="n_bins"):
        BandFeatureEmbedder(n_features=2, n_bins=0, init="uniform")


def test_rejects_invalid_role() -> None:
    with pytest.raises(ValueError, match="role"):
        BandFeatureEmbedder(n_features=2, n_bins=3, init="uniform", role="bogus")


def test_rejects_invalid_init() -> None:
    with pytest.raises(ValueError, match="init"):
        BandFeatureEmbedder(n_features=2, n_bins=3, init="bogus")


def test_rejects_mismatched_thresholds_init_shape() -> None:
    with pytest.raises(ValueError, match="shape"):
        BandFeatureEmbedder(n_features=2, n_bins=4, thresholds_init=np.zeros((2, 2)))


def test_rejects_non_increasing_thresholds_init() -> None:
    with pytest.raises(ValueError, match="strictly increasing"):
        BandFeatureEmbedder(n_features=1, n_bins=2, thresholds_init=np.array([[1.0, 0.5]]))


def test_forward_rejects_wrong_feature_dim() -> None:
    emb = BandFeatureEmbedder(n_features=3, n_bins=4, init="uniform")
    with pytest.raises(ValueError, match="n_features"):
        emb(torch.randn(5, 2, dtype=torch.float64))


def test_local_target_consistency_loss_jvp_matches_full_jacobian() -> None:
    """The single-JVP shortcut must equal the brute-force full-Jacobian contraction."""
    emb = BandFeatureEmbedder(n_features=3, n_bins=3, init="uniform")
    x = torch.randn(5, 3, dtype=torch.float64, requires_grad=True)
    df_dx = torch.randn(5, 3, dtype=torch.float64)

    loss = local_target_consistency_loss(emb, x, df_dx=df_dx)

    # brute-force: per-row full Jacobian via autograd.functional.jacobian, then contract
    def embed_row(row: torch.Tensor) -> torch.Tensor:
        return emb(row.unsqueeze(0)).squeeze(0)

    nums = []
    dens = []
    for i in range(5):
        jac = torch.autograd.functional.jacobian(embed_row, x[i].detach())  # (D, 3)
        jv = jac @ df_dx[i]
        nums.append(float((jv**2).sum()))
        dens.append(float((df_dx[i] ** 2).sum()))
    eps = 1e-8
    expected = float(np.mean([np.log(d + eps) - np.log(n + eps) for n, d in zip(nums, dens, strict=True)]))
    assert abs(float(loss.item()) - expected) < 1e-6


def test_local_target_consistency_loss_rejects_shape_mismatch() -> None:
    emb = BandFeatureEmbedder(n_features=3, n_bins=3, init="uniform")
    x = torch.randn(5, 3, dtype=torch.float64)
    df_dx = torch.randn(5, 2, dtype=torch.float64)
    with pytest.raises(ValueError, match="share a shape"):
        local_target_consistency_loss(emb, x, df_dx=df_dx)


def test_local_target_consistency_loss_decreases_under_gradient_descent() -> None:
    """Training the embedder's thresholds/beta to maximize R should reduce the surrogate loss."""
    torch.manual_seed(0)
    emb = BandFeatureEmbedder(n_features=2, n_bins=4, init="uniform")
    x = torch.randn(64, 2, dtype=torch.float64)
    true_grad_direction = torch.randn(2, dtype=torch.float64)
    df_dx = true_grad_direction.expand(64, 2).clone()

    opt = torch.optim.Adam(emb.parameters(), lr=0.05)
    first_loss = None
    last_loss = None
    for step in range(40):
        opt.zero_grad()
        loss = local_target_consistency_loss(emb, x, df_dx=df_dx)
        if step == 0:
            first_loss = float(loss.item())
        loss.backward()
        opt.step()
        last_loss = float(loss.item())
    assert first_loss is not None and last_loss is not None
    assert last_loss < first_loss
