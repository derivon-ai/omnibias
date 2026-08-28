# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Gaussian Slater log|psi| on the sampling contract + closed-form Laplacian."""

from __future__ import annotations

import pytest

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402
from omnibias.ferminet.antisymmetric import (  # noqa: E402
    gaussian_slater_closed_form_laplacian,
    gaussian_slater_log_abs_psi,
)
from omnibias.ferminet.sampling import autodiff_grad_and_laplacian, metropolis_sample  # noqa: E402


def test_gaussian_slater_log_abs_and_closed_form_lap() -> None:
    centers = jnp.array([[0.0], [1.0]])
    alphas = jnp.array([0.5, 0.5])
    params = {"centers": centers, "alphas": alphas}
    position = jnp.array([0.2, 0.8])
    logabs = gaussian_slater_log_abs_psi(params, position)
    assert jnp.isfinite(logabs)
    closed = gaussian_slater_closed_form_laplacian(params, position)
    _, autodiff_lap = autodiff_grad_and_laplacian(
        gaussian_slater_log_abs_psi, params, position
    )
    assert float(jnp.abs(closed - autodiff_lap)) < 1e-8


def test_sampler_accepts_slater_log_abs_psi() -> None:
    centers = jnp.array([[0.0], [1.2]])
    alphas = jnp.array([0.4, 0.4])
    params = {"centers": centers, "alphas": alphas}
    key = jax.random.PRNGKey(0)
    init = jnp.array([[0.1, 1.0], [0.3, 0.9]])
    result = metropolis_sample(
        gaussian_slater_log_abs_psi,
        params,
        init,
        key,
        n_steps=4,
        step_size=0.05,
    )
    assert result.positions.shape == init.shape
