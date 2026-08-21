# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Multi-stage correction smoke tests."""

from __future__ import annotations

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402
from omnibias.pinn.jax.discovery import multistage as ms  # noqa: E402


def test_multistage_reduces_simple_residual() -> None:
    y = jnp.linspace(-1.0, 1.0, 64)
    stage1 = jnp.zeros_like(y)

    def residual_fn(theta: jnp.ndarray) -> jnp.ndarray:
        # target theta = sin(4 pi y); residual is theta - target
        return theta - jnp.sin(4.0 * jnp.pi * y)

    out = ms.refine_with_multistage(
        y=y,
        stage1_theta=stage1,
        residual_fn=residual_fn,
        cfg=ms.MultiStageConfig(hidden=24, n_fourier=12, eps=1.0, steps=80, lr=2e-2, seed=0),
    )
    r0 = float(jnp.sqrt(jnp.mean(residual_fn(stage1) ** 2)))
    r1 = float(jnp.sqrt(jnp.mean(residual_fn(jnp.asarray(out["theta"])) ** 2)))
    assert r1 < r0
    assert out["info"]["final_loss"] <= out["info"]["initial_loss"] + 1e-9


def test_compose_profiles() -> None:
    a = np.array([1.0, 2.0])
    b = np.array([0.5, -0.5])
    np.testing.assert_allclose(ms.compose_profiles(a, b, eps=0.1), [1.05, 1.95])


def test_wang_linearized_residual_matches_exact_affine_d() -> None:
    y = jnp.linspace(-1.0, 1.0, 16)
    stage1 = 0.2 * y
    corr = jnp.sin(y)

    def residual_fn(theta: jnp.ndarray) -> jnp.ndarray:
        return 2.0 * theta + 0.3

    r0 = residual_fn(stage1)
    r_lin = ms.wang_linearized_residual(
        residual_fn,
        stage1,
        corr,
        r0=r0,
        eps=0.01,
        fd_eps=1e-6,
        linearized=True,
    )
    expected = r0 + 0.01 * 2.0 * corr
    np.testing.assert_allclose(np.asarray(r_lin), np.asarray(expected), atol=1e-8)
    proxy = corr + r0 / 0.01
    assert float(jnp.max(jnp.abs(r_lin - proxy))) > 1.0


def test_wang_linearized_gn_label_and_reduces_affine() -> None:
    y = jnp.linspace(-1.0, 1.0, 32)
    stage1 = 0.25 * jnp.ones_like(y)

    def residual_fn(theta: jnp.ndarray) -> jnp.ndarray:
        return 2.0 * theta

    out = ms.refine_with_multistage(
        y=y,
        stage1_theta=stage1,
        residual_fn=residual_fn,
        cfg=ms.MultiStageConfig(
            hidden=8,
            n_fourier=4,
            eps=1.0,
            steps=8,
            seed=0,
            optimizer="martens_grosse",
        ),
    )
    assert out["info"]["optimizer"] == "wang_linearized_gn"
    assert out["info"]["final_loss"] < out["info"]["initial_loss"]


def test_stage2_even_hat_is_even() -> None:
    cfg = ms.deepmind_multistage_config(hidden=6, n_fourier=4, steps=1)
    params = ms.init_stage2_params(cfg)
    y = jnp.linspace(-5.0, 5.0, 41, dtype=jnp.float64)
    hat = ms.stage2_even_hat(params, y, lam=0.6057)
    raw = ms.stage2_correction(params, y)
    assert float(jnp.max(jnp.abs(hat - hat[::-1]))) < 1e-14
    assert float(jnp.max(jnp.abs(raw - raw[::-1]))) > 1e-6
    assert cfg.optimizer == "martens_grosse"
    assert cfg.even_q is True


def test_identity_readout_is_exact_zero() -> None:
    cfg = ms.deepmind_multistage_config(hidden=6, n_fourier=4, identity_readout=True)
    params = ms.init_stage2_params(cfg)
    y = jnp.linspace(-3.0, 3.0, 21, dtype=jnp.float64)
    hat = ms.stage2_field(params, y, cfg)
    assert float(jnp.max(jnp.abs(hat))) < 1e-15


def test_train_stage2_even_q_correction_is_even() -> None:
    y = jnp.linspace(-2.0, 2.0, 32, dtype=jnp.float64)
    stage1 = jnp.ones_like(y)

    def residual_fn(theta: jnp.ndarray) -> jnp.ndarray:
        return theta - 1.0

    cfg = ms.deepmind_multistage_config(
        hidden=4, n_fourier=3, steps=2, identity_readout=True, even_q=True
    )
    out = ms.refine_with_multistage(
        y=y, stage1_theta=stage1, residual_fn=residual_fn, cfg=cfg
    )
    corr = np.asarray(out["correction"])
    assert float(np.max(np.abs(corr - corr[::-1]))) < 1e-12


def test_gradient_normalize_residual_downweights_peak() -> None:
    r = jnp.asarray([0.01, 1.0, 0.01], dtype=jnp.float64)
    core = jnp.asarray([0.0, 3.0, 0.0], dtype=jnp.float64)
    rn = ms.gradient_normalize_residual(r, core, alpha=2.0, eps=1e-8)
    assert float(jnp.abs(rn[1])) < float(jnp.abs(r[1]))
    assert float(jnp.abs(rn[1]) / (jnp.mean(jnp.abs(rn)) + 1e-30)) < float(
        jnp.abs(r[1]) / (jnp.mean(jnp.abs(r)) + 1e-30)
    )
