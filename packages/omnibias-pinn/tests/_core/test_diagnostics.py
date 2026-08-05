# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Unit tests for the backend-agnostic diagnostics in
:mod:`omnibias.pinn._core.diagnostics`."""

from __future__ import annotations

import numpy as np
import pytest
from omnibias.pinn._core.diagnostics import (
    forecast_horizon,
    power_spectrum_per_d,
    relative_l2_per_time,
    spectral_fidelity,
)


def test_relative_l2_zero_on_identical_inputs():
    rng = np.random.default_rng(0)
    u = rng.standard_normal((4, 16))
    rel = relative_l2_per_time(u, u)
    assert rel.shape == (4,)
    np.testing.assert_allclose(rel, 0.0, atol=1e-12)


def test_relative_l2_known_value_constant_offset():
    """``u_pred = u_ref + c`` -> rel_l2 = c / sqrt(<u_ref^2>) (in mean)."""
    rng = np.random.default_rng(1)
    u_ref = rng.standard_normal((4, 32))
    c = 0.5
    u_pred = u_ref + c
    rel = relative_l2_per_time(u_pred, u_ref)
    expected = c / np.sqrt(np.mean(u_ref ** 2, axis=-1))
    np.testing.assert_allclose(rel, expected, rtol=1e-12, atol=1e-12)


def test_relative_l2_2d_spatial():
    rng = np.random.default_rng(2)
    u = rng.standard_normal((3, 8, 8))
    v = u + 0.1
    rel = relative_l2_per_time(u, v)
    assert rel.shape == (3,)
    assert np.all(rel > 0)


def test_relative_l2_shape_mismatch_raises():
    a = np.zeros((4, 8))
    b = np.zeros((4, 16))
    with pytest.raises(ValueError, match="shape mismatch"):
        relative_l2_per_time(a, b)


def test_forecast_horizon_returns_first_crossing():
    times = np.linspace(0.0, 10.0, 11)
    rel = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.51, 0.6, 0.7, 0.8, 0.9, 1.0])
    h = forecast_horizon(times, rel, threshold=0.5)
    assert h == pytest.approx(5.0)


def test_forecast_horizon_returns_t_max_if_never_crossed():
    times = np.linspace(0.0, 10.0, 11)
    rel = np.full(11, 0.1)
    h = forecast_horizon(times, rel, threshold=0.5)
    assert h == pytest.approx(10.0)


def test_forecast_horizon_returns_t_max_at_threshold_exactly():
    times = np.linspace(0.0, 10.0, 11)
    rel = np.full(11, 0.5)  # never strictly exceeds
    h = forecast_horizon(times, rel, threshold=0.5)
    assert h == pytest.approx(10.0)


def test_power_spectrum_per_d_shape_1d():
    rng = np.random.default_rng(3)
    u = rng.standard_normal((4, 32))
    P = power_spectrum_per_d(u, L=2.0 * np.pi)
    assert P.ndim == 1
    assert len(P) >= 16  # at least N//2 bins


def test_power_spectrum_per_d_shape_2d():
    rng = np.random.default_rng(4)
    u = rng.standard_normal((3, 16, 16))
    P = power_spectrum_per_d(u, L=2.0 * np.pi)
    assert P.ndim == 1


def test_spectral_fidelity_zero_on_identical_inputs():
    rng = np.random.default_rng(5)
    u = rng.standard_normal((4, 32))
    fid = spectral_fidelity(u, u, L=2.0 * np.pi)
    assert fid == pytest.approx(0.0, abs=1e-12)


def test_spectral_fidelity_positive_on_perturbed():
    rng = np.random.default_rng(6)
    u = rng.standard_normal((3, 32))
    v = u + rng.standard_normal((3, 32)) * 0.5
    fid = spectral_fidelity(u, v, L=2.0 * np.pi)
    assert fid > 0


def test_spectral_fidelity_truncates_to_n_modes():
    """Truncating to a few low-frequency modes can either drop or
    keep the relative-L^2 -- but it must remain finite and nonnegative."""
    rng = np.random.default_rng(7)
    u = rng.standard_normal((3, 32))
    v = u + rng.standard_normal((3, 32)) * 0.1
    for n in (2, 4, 8, None):
        fid = spectral_fidelity(u, v, L=2.0 * np.pi, n_modes=n)
        assert np.isfinite(fid)
        assert fid >= 0


def test_spectral_fidelity_n_modes_smaller_uses_only_those_bins():
    """When ``n_modes < bin count``, output uses only the first n bins
    of both spectra. We verify by constructing a case where high-freq
    differences are explicitly excluded."""
    # Two signals that agree on low frequencies but differ on high.
    n_t, N = 1, 32
    L = 2.0 * np.pi
    x = np.linspace(0.0, L, N, endpoint=False)
    u = np.sin(x).reshape(n_t, N)         # mode k=1 only
    v = u + 0.5 * np.cos(8 * x).reshape(n_t, N)  # adds mode k=8
    fid_full = spectral_fidelity(u, v, L=L)
    fid_low = spectral_fidelity(u, v, L=L, n_modes=4)  # excludes k=8
    assert fid_low < fid_full


def test_spectral_fidelity_2d_finite():
    rng = np.random.default_rng(8)
    u = rng.standard_normal((2, 16, 16))
    v = u + rng.standard_normal((2, 16, 16)) * 0.1
    fid = spectral_fidelity(u, v, L=2.0 * np.pi)
    assert np.isfinite(fid)
    assert fid > 0
