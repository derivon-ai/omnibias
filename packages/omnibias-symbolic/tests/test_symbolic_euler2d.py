# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Numpy-only symbolic replay of the 2-D Euler steady-vortex certificate."""

from __future__ import annotations

import numpy as np
from omnibias.symbolic.euler2d import (
    euler2d_grid_residuals,
    euler2d_radial_fields,
    verify_euler2d_steady_vortex,
)

_COEFFS = [1.0, 0.4, -0.2]
_SCALES = [0.6, 1.3, 2.1]


def _reference_cert(coeffs: list[float], scales: list[float], r_trunc: float) -> dict:
    """A minimal certificate dict driven entirely by the numpy reductions.

    Built without importing ``omnibias.pinn.certified`` so the test exercises the replay's own
    closed forms; the reported sups deliberately *upper-bound* the samples.
    """
    r = np.linspace(0.0, r_trunc, 40001)
    f = euler2d_radial_fields(r, np.asarray(coeffs), np.asarray(scales))
    vel = float(np.max(f["r_abs_q"]) / (2.0 * np.pi))
    vort = float(np.max(f["abs_omega_p"]) / np.pi)
    strain = float(np.sqrt(np.max(f["strain_sq_num"]) / (2.0 * np.pi**2)))
    return {
        "coeffs": coeffs,
        "scales": scales,
        "far_field_trunc": r_trunc,
        "circulation": float(sum(coeffs)),
        "steady_residual_certified_sup": 0.0,
        "velocity_sup": vel * 1.001,
        "vorticity_sup": vort * 1.001,
        "strain_sup": strain * 1.001,
        "grid_points": 9,
        "honesty": {"exact_steady_state": True},
    }


def test_radial_fields_match_direct_formula() -> None:
    # spot check r * Q at a single radius against an explicit loop
    r0 = 1.7
    f = euler2d_radial_fields(np.asarray([r0]), np.asarray(_COEFFS), np.asarray(_SCALES))
    q0 = sum(c / (r0 * r0 + a * a) for c, a in zip(_COEFFS, _SCALES, strict=True))
    assert abs(float(f["r_abs_q"][0]) - abs(r0 * q0)) < 1e-12


def test_grid_residuals_confirm_exact_steady_state() -> None:
    grid = euler2d_grid_residuals(np.asarray(_COEFFS), np.asarray(_SCALES), 5.2, 13)
    assert grid["steady_residual_grid_max"] < 1e-10
    assert grid["divergence_grid_max"] < 1e-10
    assert grid["riesz_trace_identity_grid_max"] < 1e-10


def test_replay_matches_reference_certificate() -> None:
    cert = _reference_cert(_COEFFS, _SCALES, 5.2)
    rep = verify_euler2d_steady_vortex(cert)
    assert rep["replay_match"] is True
    assert rep["sup_dominates_samples"] is True
    assert rep["verdict_match"] is True
    assert rep["unproven_claim"] is False


def test_replay_catches_understated_sup() -> None:
    cert = _reference_cert(_COEFFS, _SCALES, 5.2)
    cert["vorticity_sup"] = 1e-9  # forged: far below the true peak
    rep = verify_euler2d_steady_vortex(cert)
    assert rep["sup_dominates_samples"] is False
    assert rep["replay_match"] is False
