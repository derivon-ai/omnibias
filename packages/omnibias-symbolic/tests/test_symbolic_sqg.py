# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Numpy-only symbolic replay of the 2-D SQG steady-vortex certificate."""

from __future__ import annotations

import numpy as np
import pytest
from omnibias.symbolic.sqg import (
    sqg_grad_theta_sup_sample,
    sqg_grid_residuals,
    sqg_radial_fields,
    sqg_selfsimilar_l2_quantities,
    verify_sqg_linearized_coercivity_attempt,
    verify_sqg_selfsimilar_blowup_attempt,
    verify_sqg_steady_vortex,
)

_COEFFS = [1.0, 0.4, -0.2]
_SCALES = [0.6, 1.3, 2.1]


def _reference_cert(coeffs: list[float], scales: list[float], r_trunc: float) -> dict:
    """A minimal certificate dict driven entirely by the numpy reductions.

    Built without importing ``omnibias.pinn.certified`` so the test exercises the replay's own
    closed forms; the reported sups deliberately *upper-bound* the samples.
    """
    r = np.linspace(0.0, r_trunc, 40001)
    f = sqg_radial_fields(r, np.asarray(coeffs), np.asarray(scales))
    tp = 2.0 * np.pi
    vel = float(np.max(f["r_abs_s"]) / tp)
    temp = float(np.max(f["abs_theta_num"]) / tp)
    strain = float(np.sqrt(np.max(f["strain_sq_num"])) / tp)
    return {
        "coeffs": coeffs,
        "scales": scales,
        "far_field_trunc": r_trunc,
        "total_temperature": float(sum(coeffs)),
        "steady_residual_certified_sup": 0.0,
        "velocity_sup": vel * 1.001,
        "temperature_sup": temp * 1.001,
        "strain_sup": strain * 1.001,
        "grid_points": 9,
        "honesty": {"exact_steady_state": True},
    }


def test_radial_fields_match_direct_formula() -> None:
    # spot check r * S at a single radius against an explicit loop
    r0 = 1.7
    f = sqg_radial_fields(np.asarray([r0]), np.asarray(_COEFFS), np.asarray(_SCALES))
    s0 = sum(c * (r0 * r0 + a * a) ** -1.5 for c, a in zip(_COEFFS, _SCALES, strict=True))
    assert abs(float(f["r_abs_s"][0]) - abs(r0 * s0)) < 1e-12


def test_grid_residuals_confirm_exact_steady_state() -> None:
    grid = sqg_grid_residuals(np.asarray(_COEFFS), np.asarray(_SCALES), 5.2, 13)
    assert grid["steady_residual_grid_max"] < 1e-10
    assert grid["divergence_grid_max"] < 1e-10
    assert grid["riesz_perp_identity_grid_max"] < 1e-10


def test_replay_matches_reference_certificate() -> None:
    cert = _reference_cert(_COEFFS, _SCALES, 5.2)
    rep = verify_sqg_steady_vortex(cert)
    assert rep["replay_match"] is True
    assert rep["sup_dominates_samples"] is True
    assert rep["verdict_match"] is True
    assert rep["unproven_claim"] is False


def test_replay_catches_understated_sup() -> None:
    cert = _reference_cert(_COEFFS, _SCALES, 5.2)
    cert["temperature_sup"] = 1e-9  # forged: far below the true peak
    rep = verify_sqg_steady_vortex(cert)
    assert rep["sup_dominates_samples"] is False
    assert rep["replay_match"] is False


# --------------------------------------------------------------------------- #
# Self-similar obstruction replay                                              #
# --------------------------------------------------------------------------- #
def test_selfsimilar_l2_quantities_satisfy_nogo_identity() -> None:
    """Independent quadrature: <F,Theta> = -||Theta||^2 and ||F|| >= ||Theta|| > 0."""
    q = sqg_selfsimilar_l2_quantities(np.asarray(_COEFFS), np.asarray(_SCALES))
    l2_sq = q["profile_l2_norm_sq"]
    ip = q["selfsimilar_residual_inner_product"]
    f_sq = q["selfsimilar_residual_l2_sq"]
    closed = sum(
        ci * cj / (2.0 * np.pi * (ai + aj) ** 2)
        for ci, ai in zip(_COEFFS, _SCALES, strict=True)
        for cj, aj in zip(_COEFFS, _SCALES, strict=True)
    )
    assert abs(l2_sq - closed) <= 1e-4 * closed
    assert abs(ip + l2_sq) <= 1e-4 * l2_sq  # the no-go identity
    assert f_sq**0.5 >= l2_sq**0.5 > 0.0  # the obstruction lower bound


def _selfsimilar_reference_cert(coeffs: list[float], scales: list[float]) -> dict:
    """Minimal obstruction certificate driven by the numpy closed forms (no omnibias.pinn.certified)."""
    l2_sq = sum(
        ci * cj / (2.0 * np.pi * (ai + aj) ** 2)
        for ci, ai in zip(coeffs, scales, strict=True)
        for cj, aj in zip(coeffs, scales, strict=True)
    )
    return {
        "coeffs": coeffs,
        "scales": scales,
        "far_field_trunc": 2.0 * max(scales) + 1.0,
        "grid_points": 9,
        "profile_l2_norm_sq": [l2_sq * (1 - 1e-12), l2_sq * (1 + 1e-12)],
        "selfsimilar_residual_inner_product": [-l2_sq * (1 + 1e-12), -l2_sq * (1 - 1e-12)],
        "selfsimilar_residual_l2_lower_bound": l2_sq**0.5,
        "l2_self_similar_drift_energy_coefficient": 1.0,
        "exact_selfsimilar_profile_exists": False,
        "honesty": {"blowup_claim": False, "unproven_claim": False, "three_d_claim": False},
    }


def test_selfsimilar_replay_matches_reference_certificate() -> None:
    cert = _selfsimilar_reference_cert(_COEFFS, _SCALES)
    rep = verify_sqg_selfsimilar_blowup_attempt(cert)
    assert rep["replay_match"] is True
    assert rep["nogo_identity_holds"] is True
    assert rep["obstruction_holds"] is True
    assert rep["drift_coefficient_match"] is True
    assert rep["unproven_claim"] is False


def test_selfsimilar_replay_catches_forged_norm() -> None:
    cert = _selfsimilar_reference_cert(_COEFFS, _SCALES)
    cert["profile_l2_norm_sq"] = [1e-6, 2e-6]  # forged understated norm
    rep = verify_sqg_selfsimilar_blowup_attempt(cert)
    assert rep["profile_l2_norm_sq_match"] is False
    assert rep["replay_match"] is False


def test_selfsimilar_replay_catches_profile_exists_overclaim() -> None:
    cert = _selfsimilar_reference_cert(_COEFFS, _SCALES)
    cert["exact_selfsimilar_profile_exists"] = True  # forged overclaim
    rep = verify_sqg_selfsimilar_blowup_attempt(cert)
    assert rep["honesty_consistent"] is False
    assert rep["replay_match"] is False


# --------------------------------------------------------------------------- #
# Linearized L^2 coercivity replay                                             #
# --------------------------------------------------------------------------- #
def test_grad_theta_sup_sample_matches_single_blob_peak() -> None:
    # Single unit blob a: G(r) = a * r (r^2+a^2)^{-5/2} peaks at r = a/2, value
    # (1/2)(5/4)^{-5/2} a^{-3}, so ||grad theta||_inf = (3/2pi)(1/2)(5/4)^{-5/2} a^{-3}.
    a = 0.8
    sampled = sqg_grad_theta_sup_sample(np.asarray([1.0]), np.asarray([a]))
    peak = (3.0 / (2.0 * np.pi)) * 0.5 * (1.25) ** -2.5 * a**-3
    assert sampled == pytest.approx(peak, rel=1e-3)


def _coercivity_reference_cert(coeffs: list[float], scales: list[float]) -> dict:
    """Minimal coercivity certificate driven by the numpy sample (no omnibias.pinn.certified)."""
    grad = sqg_grad_theta_sup_sample(np.asarray(coeffs), np.asarray(scales))
    grad_reported = grad * 1.001  # a valid (slightly conservative) certified sup
    gap = 1.0 - grad_reported
    block = 0.5 * (2.0 - (4.0 * grad_reported * grad_reported) ** 0.5)
    return {
        "coeffs": coeffs,
        "scales": scales,
        "drift_self_adjoint_coefficient": 1.0,
        "riesz_isometry_constant": 1.0,
        "grad_theta_sup": grad_reported,
        "stretching_coupling_bound": grad_reported,
        "l2_coercivity_gap_lower": gap,
        "l2_coercive": gap > 0.0,
        "block_operator_gap": {"gap_lower": block},
        "honesty": {
            "blowup_claim": False,
            "unproven_claim": False,
            "three_d_claim": False,
            "stability_claim": False,
        },
    }


def test_coercivity_replay_matches_reference_certificate() -> None:
    cert = _coercivity_reference_cert(_COEFFS, _SCALES)
    rep = verify_sqg_linearized_coercivity_attempt(cert)
    assert rep["replay_match"] is True
    assert rep["grad_sup_dominates_samples"] is True
    assert rep["l2_gap_match"] is True
    assert rep["block_gap_match"] is True
    assert rep["unproven_claim"] is False


def test_coercivity_replay_catches_forged_grad_sup() -> None:
    cert = _coercivity_reference_cert(_COEFFS, _SCALES)
    cert["grad_theta_sup"] = 1e-9  # forged: below the true sup
    rep = verify_sqg_linearized_coercivity_attempt(cert)
    assert rep["grad_sup_dominates_samples"] is False
    assert rep["replay_match"] is False
