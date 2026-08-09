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
