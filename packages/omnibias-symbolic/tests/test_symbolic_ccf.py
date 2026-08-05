# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Numpy-only symbolic validation of CCF self-similar candidates."""

from __future__ import annotations

import numpy as np
from omnibias.symbolic.ccf import (
    assess_ccf_candidate,
    ccf_self_similar_residual,
    periodic_hilbert,
    recover_ccf_scaling_law,
    verify_cap_bundle,
    verify_ccf_residual,
)


def _grid(n: int) -> np.ndarray:
    return -np.pi + 2.0 * np.pi * np.arange(n) / n


def _manufactured(y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Even periodic reference profile + its analytic derivative."""
    theta = np.exp(np.cos(y) - 1.0) + 0.25 * np.cos(2.0 * y)
    theta_y = -np.sin(y) * np.exp(np.cos(y) - 1.0) - 0.5 * np.sin(2.0 * y)
    return theta, theta_y


def test_periodic_hilbert_conventions() -> None:
    y = _grid(128)
    np.testing.assert_allclose(periodic_hilbert(np.cos(3 * y)), np.sin(3 * y), atol=1e-10)
    np.testing.assert_allclose(periodic_hilbert(np.sin(3 * y)), -np.cos(3 * y), atol=1e-10)
    np.testing.assert_allclose(periodic_hilbert(np.ones_like(y)), 0.0, atol=1e-12)


def test_residual_matches_hand_closed_form() -> None:
    y = _grid(96)
    lam = 0.6057
    theta = np.cos(2 * y)
    theta_y = -2.0 * np.sin(2 * y)
    expected = (1 + lam) * y * theta_y - lam * theta - 2.0 * np.sin(2 * y) ** 2
    got = ccf_self_similar_residual(y, theta, theta_y, lam)
    np.testing.assert_allclose(got, expected, atol=1e-10)


def test_recover_lambda_from_manufactured_forcing_exact() -> None:
    # Build (theta*, theta*', g) that satisfy the forced relation exactly,
    # then recover lambda from the discovered law.
    y = _grid(256)
    lam_star = 0.5
    theta, theta_y = _manufactured(y)
    forcing = ccf_self_similar_residual(y, theta, theta_y, lam_star)
    out = recover_ccf_scaling_law(y, theta, theta_y, forcing=forcing)
    assert abs(out["lambda_recovered"] - lam_star) < 1e-8
    assert out["advection_consistency_abs"] < 1e-8
    assert out["fit_rmse"] < 1e-9


def test_verify_cap_bundle_roundtrip() -> None:
    y = _grid(128)
    lam = 0.42
    theta, theta_y = _manufactured(y)
    residual = ccf_self_similar_residual(y, theta, theta_y, lam)
    bundle = {
        "validation_inputs": {
            "y": y.tolist(),
            "theta": theta.tolist(),
            "theta_y": theta_y.tolist(),
            "lambda": lam,
            "form": "transport",
            "velocity_sign": 1.0,
        },
        "residual_samples": residual.tolist(),
    }
    report = verify_cap_bundle(bundle)
    assert report["residual_samples_match"]
    assert report["agreement_max_abs_diff"] < 1e-9


def test_assess_candidate_does_not_overclaim() -> None:
    y = _grid(64)
    theta, theta_y = _manufactured(y)
    out = assess_ccf_candidate(y, theta, theta_y, 0.5)
    assert out["exact_solution_claim"] is False
    assert out["navier_stokes_proof_claim"] is False
    assert "residual" in out and "scaling_law" in out


def test_verify_ccf_residual_with_forcing_is_zero_on_manufactured() -> None:
    y = _grid(192)
    lam = 0.5
    theta, theta_y = _manufactured(y)
    forcing = ccf_self_similar_residual(y, theta, theta_y, lam)
    metrics = verify_ccf_residual(y, theta, theta_y, lam, forcing=forcing)
    assert metrics["max_abs_residual"] < 1e-12
