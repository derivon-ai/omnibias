# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Spectrum + dissipative CCF smoke tests."""

from __future__ import annotations

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402
from omnibias.pinn.jax.discovery import spectrum as spec  # noqa: E402
from omnibias.pinn.jax.equations import ccf_dissipative as diss  # noqa: E402


def test_unstable_count_on_identity_jac() -> None:
    # residual = theta - target; Jacobian = I -> all eigenvalues 1
    target = jnp.zeros(8)

    def r_fn(th: jnp.ndarray) -> jnp.ndarray:
        return th - target

    cert = spec.certified_ccf_unstable_mode_count(
        residual_fn=r_fn, theta=jnp.zeros(8), claimed_order=8
    )
    assert cert["honesty"]["unproven_claim"] is False
    assert cert["measured_unstable_count"] == 8
    assert cert["count_matches_claim"] is True


def test_sealed_hardy_galerkin_smoke() -> None:
    out = spec.sealed_ccf_unstable_mode_count(
        coeffs=[1.0, -0.1],
        scales=[0.8, 1.5],
        lam=0.6,
        claimed_order=0,
    )
    assert out["schema_version"] == "ccf-hardy-unstable-mode-count-1"
    assert out["honesty"]["navier_stokes_proof_claim"] is False


def test_dissipative_alpha_zero_matches_inviscid() -> None:
    y = -np.pi + 2 * np.pi * np.arange(64) / 64
    theta = np.cos(2 * y)
    theta_y = -2 * np.sin(2 * y)
    r0 = diss.ccf_dissipative_residual_samples(
        jnp.asarray(y), jnp.asarray(theta), jnp.asarray(theta_y), 0.5, 0.0
    )
    from omnibias.pinn.jax.equations.cordoba_cordoba_fontelos import ccf_residual_samples

    r_inv = ccf_residual_samples(
        jnp.asarray(y), jnp.asarray(theta), jnp.asarray(theta_y), 0.5
    )
    np.testing.assert_allclose(np.asarray(r0), np.asarray(r_inv), atol=1e-12)


def test_dissipative_alpha_changes_residual() -> None:
    y = -np.pi + 2 * np.pi * np.arange(64) / 64
    theta = np.cos(2 * y)
    theta_y = -2 * np.sin(2 * y)
    r0 = np.asarray(
        diss.ccf_dissipative_residual_samples(
            jnp.asarray(y), jnp.asarray(theta), jnp.asarray(theta_y), 0.5, 0.0
        )
    )
    r1 = np.asarray(
        diss.ccf_dissipative_residual_samples(
            jnp.asarray(y), jnp.asarray(theta), jnp.asarray(theta_y), 0.5, 0.5
        )
    )
    assert float(np.max(np.abs(r1 - r0))) > 1e-6
