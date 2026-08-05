# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Tests for latent-state ODE discovery from a single observed coordinate."""

from __future__ import annotations

import numpy as np
import pytest
from omnibias.symbolic.latent import (
    LinearAutoencoder,
    MLPAutoencoder,
    discover_latent_ode,
    finite_difference_derivative,
    takens_embedding,
)


def _harmonic_series(*, omega: float, dt: float, t_max: float) -> np.ndarray:
    """Observed coordinate ``z1(t) = cos(omega t)`` of ``z1'=z2, z2'=-omega^2 z1``."""
    t = np.arange(0.0, t_max, dt)
    return np.cos(omega * t)


def _damped_series(*, omega: float, gamma: float, dt: float, t_max: float) -> np.ndarray:
    """Observed coordinate of a damped oscillator ``z1'=z2, z2'=-omega^2 z1-2 gamma z2``."""
    t = np.arange(0.0, t_max, dt)
    omega_d = np.sqrt(omega * omega - gamma * gamma)
    return np.exp(-gamma * t) * np.cos(omega_d * t)


# --------------------------------------------------------------------------- #
# Takens embedding & finite differences                                       #
# --------------------------------------------------------------------------- #
def test_takens_embedding_shape_and_alignment() -> None:
    s = np.arange(10.0)
    emb = takens_embedding(s, dim=3, delay=2)
    # span = (3-1)*2 = 4, so 10 - 4 = 6 rows
    assert emb.shape == (6, 3)
    # row 0 is aligned to t = 4: [s[4], s[2], s[0]]
    assert np.allclose(emb[0], [4.0, 2.0, 0.0])
    # last row aligned to t = 9: [s[9], s[7], s[5]]
    assert np.allclose(emb[-1], [9.0, 7.0, 5.0])


def test_takens_embedding_validates_arguments() -> None:
    with pytest.raises(ValueError):
        takens_embedding(np.arange(5.0), dim=0)
    with pytest.raises(ValueError):
        takens_embedding(np.arange(5.0), dim=2, delay=0)
    with pytest.raises(ValueError):
        takens_embedding(np.arange(3.0), dim=4, delay=2)  # too short


def test_finite_difference_matches_analytic_derivative() -> None:
    dt = 1e-3
    t = np.arange(0.0, 2.0, dt)
    values = np.sin(t)
    idx, deriv = finite_difference_derivative(values, dt)
    assert idx[0] == 1 and idx[-1] == values.shape[0] - 2
    assert np.allclose(deriv, np.cos(t[idx]), atol=1e-5)


# --------------------------------------------------------------------------- #
# autoencoders                                                                 #
# --------------------------------------------------------------------------- #
def test_linear_autoencoder_is_exact_on_low_rank_data() -> None:
    rng = np.random.default_rng(0)
    basis = rng.standard_normal((2, 5))
    coords = rng.standard_normal((200, 2))
    data = coords @ basis + 3.0  # lies on a 2-D affine subspace of R^5
    ae = LinearAutoencoder.fit(data, 2)
    assert ae.reconstruction_rmse(data) < 1e-10
    # encode/decode roundtrip recovers the data
    assert np.allclose(ae.decode(ae.encode(data)), data, atol=1e-9)


def test_linear_autoencoder_validates_latent_dim() -> None:
    data = np.zeros((10, 3))
    with pytest.raises(ValueError):
        LinearAutoencoder.fit(data, 0)
    with pytest.raises(ValueError):
        LinearAutoencoder.fit(data, 4)


def test_mlp_autoencoder_training_reduces_reconstruction_error() -> None:
    series = _damped_series(omega=2.0, gamma=0.15, dt=0.01, t_max=20.0)
    emb = takens_embedding(series, dim=5, delay=6)
    ae = MLPAutoencoder.fit(emb, 2, epochs=1500, lr=0.05, seed=0)
    assert ae.losses[-1] < 0.25 * ae.losses[0]
    assert ae.reconstruction_rmse(emb) < 0.2


# --------------------------------------------------------------------------- #
# headline: latent ODE recovery from one observed coordinate                   #
# --------------------------------------------------------------------------- #
def test_discover_latent_ode_recovers_harmonic_frequency() -> None:
    omega = 1.7
    dt = 0.01
    series = _harmonic_series(omega=omega, dt=dt, t_max=40.0)
    result = discover_latent_ode(
        series, dt=dt, latent_dim=2, embedding_dim=4, delay=5, max_degree=2
    )
    # delay embedding lives on a 2-D subspace -> near-exact reconstruction
    assert result.reconstruction_rmse < 1e-8
    assert result.latent_dim == 2
    # eigenvalues are +- i*omega (undamped): imaginary parts ~ omega, real ~ 0
    imag = np.sort(np.abs(result.eigenvalues.imag))
    assert imag[-1] == pytest.approx(omega, abs=2e-2)
    assert np.max(np.abs(result.eigenvalues.real)) < 5e-2
    assert "diffeomorphism" in result.note


def test_discover_latent_ode_recovers_damped_growth_rate() -> None:
    omega, gamma = 2.0, 0.15
    dt = 0.005
    series = _damped_series(omega=omega, gamma=gamma, dt=dt, t_max=30.0)
    result = discover_latent_ode(
        series, dt=dt, latent_dim=2, embedding_dim=5, delay=8, max_degree=2
    )
    omega_d = np.sqrt(omega * omega - gamma * gamma)
    eig = result.eigenvalues
    # both eigenvalues share the decay rate -gamma and frequency omega_d
    assert np.allclose(np.sort(eig.real), [-gamma, -gamma], atol=3e-2)
    assert np.max(np.abs(eig.imag)) == pytest.approx(omega_d, abs=3e-2)


def test_discover_latent_ode_exposes_component_laws() -> None:
    series = _harmonic_series(omega=1.0, dt=0.01, t_max=30.0)
    result = discover_latent_ode(series, dt=0.01, latent_dim=2, embedding_dim=4, delay=6)
    assert len(result.component_formulas) == 2
    assert len(result.component_results) == 2
    # each component law is a string mentioning a time derivative LHS
    assert all("_t =" in f for f in result.component_formulas)
