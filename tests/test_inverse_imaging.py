# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Regression tests for the 05-01 G7 localization-scaling falsifier."""

from __future__ import annotations

import numpy as np
import pytest
from inverse_imaging import (
    ALPHA_MAX,
    BOUNDARY_RATIO_MAX,
    JUMP,
    L2SQ_SIGMA3,
    L2SQ_SIGMA4,
    N_SAMPLES,
    ORDER_CONFIG,
    TAU_STAR,
    _l2_sq_from_coeffs,
    _s_from_regime,
    boundary_contamination_ratio,
    field_u,
    global_localize,
    localize_batch,
    mollifier_peak_response,
    polish_peak,
    polish_peak_batch,
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


def test_derived_s_matches_recorded_constants() -> None:
    assert ORDER_CONFIG[3]["s"] == pytest.approx(0.05)
    assert ORDER_CONFIG[4]["s"] == pytest.approx(1e-4)
    assert _s_from_regime(3, J=JUMP, N=N_SAMPLES, alpha_max=ALPHA_MAX) == 0.05
    assert _s_from_regime(4, J=JUMP, N=N_SAMPLES, alpha_max=ALPHA_MAX) == 1e-4


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


@pytest.mark.parametrize("n", [3, 4])
def test_polish_peak_batch_matches_scalar(n: int) -> None:
    m = n - 2
    J = 50.0
    alpha = 40.0
    x = sample_grid(2000)
    u = field_u(x, m=m, J=J)
    rng = np.random.default_rng(0)
    Y = u[None, :] + rng.normal(0.0, 0.01, size=(5, x.size))
    inits = TAU_STAR + rng.normal(0.0, 0.005, size=5)
    clamp = (TAU_STAR - 5.0 / alpha, TAU_STAR + 5.0 / alpha)
    batch_tau, batch_rp, batch_rpp = polish_peak_batch(
        Y, x, n=n, alpha=alpha, tau_init=inits, clamp=clamp
    )
    for i in range(5):
        tau, rp, rpp = polish_peak(
            Y[i], x, n=n, alpha=alpha, tau_init=float(inits[i]), clamp=clamp
        )
        # Batched vs scalar Newton share the same clamped steps; reduction-order
        # float noise can leave the final r' at different ~1e-7 residuals, so
        # gate on tau and stationarity rather than bit-identical derivatives.
        assert abs(batch_tau[i] - tau) < 1e-10
        assert abs(batch_rp[i]) < 1e-6
        assert abs(rp) < 1e-6
        assert abs(batch_rpp[i] - rpp) <= 1e-5 + 1e-8 * abs(rpp)


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


def test_boundary_ratio_guard_raises_when_tau_near_edge() -> None:
    """Moving tau* too close to the right boundary must trip the derived guard."""
    with pytest.raises(RuntimeError, match="boundary contamination"):
        validate_regime(
            n=4,
            s=1e-4,
            J=JUMP,
            N=N_SAMPLES,
            alphas=np.array([20.0, 320.0]),
            tau_star=0.95,
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
    assert loc["seeded"] is True


def test_global_argmax_fails_for_n4_at_large_alpha() -> None:
    """Locks the boundary-artifact failure so the claim cannot be upgraded."""
    x = sample_grid()
    # Matched fields: n = m + 2.
    u3 = field_u(x, m=1, J=JUMP)
    u4 = field_u(x, m=2, J=JUMP)
    Y3 = np.tile(u3, (8, 1))
    Y4 = np.tile(u4, (8, 1))
    # n=3: global search still finds the true peak.
    g3 = global_localize(Y3, x, n=3, alpha=320.0)
    assert g3["capture_rate"] == 1.0
    # n=4 at large alpha: boundary artifact wins.
    g4 = global_localize(Y4, x, n=4, alpha=80.0)
    assert g4["capture_rate"] == 0.0
    # And the contamination ratio at the design point stays below the gate.
    assert (
        boundary_contamination_ratio(n=4, alpha=20.0, J=JUMP, m=2)
        <= BOUNDARY_RATIO_MAX
    )
