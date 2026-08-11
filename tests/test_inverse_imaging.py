# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Regression tests for the 05-01 G7 localization-scaling falsifier."""

from __future__ import annotations

import numpy as np
import pytest
from inverse_imaging import (
    L2SQ_SIGMA3,
    L2SQ_SIGMA4,
    TAU_STAR,
    _l2_sq_from_coeffs,
    field_u,
    localize_batch,
    mollifier_peak_response,
    polish_peak,
    predicted_sd_rprime_discrete,
    response_from_y,
    response_kernel,
    sample_grid,
    sigma_n,
    validate_regime,
)
from omnibias.core.polynomials import sigmoid_polynomial_coeffs


def test_l2_constants_from_core_coefficients() -> None:
    assert sigmoid_polynomial_coeffs(3)[0] == 0.0
    assert abs(_l2_sq_from_coeffs(3) - L2SQ_SIGMA3) / L2SQ_SIGMA3 < 1e-10
    assert abs(_l2_sq_from_coeffs(4) - L2SQ_SIGMA4) / L2SQ_SIGMA4 < 1e-10


@pytest.mark.parametrize("n", [3, 4])
def test_noiseless_response_matches_mollifier_peak(n: int) -> None:
    m = n - 2
    J = 50.0
    alpha = 40.0
    x = sample_grid(2000)
    u = field_u(x, m=m, J=J)
    taus = np.linspace(TAU_STAR - 3.0 / alpha, TAU_STAR + 3.0 / alpha, 61)
    K = response_kernel(x, taus, n=n, alpha=alpha)
    r = response_from_y(u, K)
    expected = mollifier_peak_response(taus, n=n, alpha=alpha, J=J)
    # Peak region agreement; edges of a finite domain pick up O(e^{-alpha d}).
    mid = slice(10, -10)
    rel = np.max(np.abs(r[mid] - expected[mid])) / np.max(np.abs(expected[mid]))
    assert rel < 5e-3, f"n={n}: rel={rel}"


@pytest.mark.parametrize("n,m", [(3, 1), (4, 2)])
def test_channel_selectivity(n: int, m: int) -> None:
    """Matched channel shows a mollifier peak at tau*; the lower channel does not."""
    J = 50.0
    alpha = 40.0
    x = sample_grid(2000)
    u = field_u(x, m=m, J=J)
    taus = np.linspace(0.2, 0.6, 401)
    K_n = response_kernel(x, taus, n=n, alpha=alpha)
    r_n = response_from_y(u, K_n)
    peak_idx = int(np.argmax(np.abs(r_n)))
    assert abs(float(taus[peak_idx]) - TAU_STAR) < 2.0 / alpha
    expected_height = J * alpha * 0.25  # |J alpha sigma'(0)|
    assert abs(abs(r_n[peak_idx]) - expected_height) / expected_height < 0.05

    # Channel n-1 sees a smoothed jump (no delta), so |r| at tau* is far below
    # the matched mollifier peak height.
    K_lo = response_kernel(x, taus, n=n - 1, alpha=alpha)
    r_lo = response_from_y(u, K_lo)
    at_star = float(np.interp(TAU_STAR, taus, np.abs(r_lo)))
    assert at_star < 0.25 * expected_height


@pytest.mark.parametrize("n", [3, 4])
def test_newton_polish_stationarity_and_curvature_sign(n: int) -> None:
    m = n - 2
    J = 50.0
    alpha = 40.0
    x = sample_grid(2000)
    u = field_u(x, m=m, J=J)
    tau_hat, rp, rpp = polish_peak(u, x, n=n, alpha=alpha, tau_init=TAU_STAR + 0.01)
    assert abs(rp) < 1e-8
    assert abs(tau_hat - TAU_STAR) < 1e-5
    # r''(tau*) = (-1)^(n-1) J alpha^3 sigma'''(0); sigma'''(0) < 0.
    expected_sign = -((-1.0) ** (n - 1))  # because sigma'''(0) is negative
    assert np.sign(rpp) == np.sign(expected_sign * J)


def test_discrete_sd_rprime_matches_monte_carlo() -> None:
    n = 3
    alpha = 40.0
    s = 0.05
    x = sample_grid(2000)
    pred = predicted_sd_rprime_discrete(x, n=n, alpha=alpha, s=s)
    rng = np.random.default_rng(0)
    R = 400
    Y = rng.normal(0.0, s, size=(R, x.size))  # pure noise; mean-zero response
    Nf = float(x.size)
    z = alpha * (x - TAU_STAR)
    weights = -(alpha ** (n + 1)) * sigma_n(z, n + 1) / Nf
    rprimes = Y @ weights
    emp = float(np.std(rprimes, ddof=1))
    se = emp / np.sqrt(2.0 * (R - 1))
    assert abs(pred - emp) <= 3.0 * se


def test_regime_guard_raises_on_invalid_config() -> None:
    with pytest.raises(RuntimeError, match="INVALID EXPERIMENT"):
        validate_regime(
            n=4,
            s=1.0,  # far too large
            J=1.0,
            N=100,
            alphas=np.array([20.0, 320.0]),
        )


def test_localize_batch_captures_clean_signal() -> None:
    n = 3
    alpha = 40.0
    x = sample_grid(2000)
    u = field_u(x, m=1, J=50.0)
    Y = np.tile(u, (8, 1))
    loc = localize_batch(Y, x, n=n, alpha=alpha)
    assert loc["n_captured"] == loc["n_total"]
    assert abs(loc["empirical_mean"] - TAU_STAR) < 1e-4
