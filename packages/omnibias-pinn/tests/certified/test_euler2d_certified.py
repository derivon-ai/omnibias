# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Certified 2-D Euler steady-vortex certified-evidence tests (Riesz/Leray substrate)."""

from __future__ import annotations

import json
import math

import pytest
from omnibias.pinn.certified import (
    EULER2D_VORTEX_SCHEMA_VERSION,
    certified_euler2d_steady_vortex,
    certified_euler2d_steady_vortex_schema_errors,
)

_COEFFS = [1.0, 0.4, -0.2]
_SCALES = [0.6, 1.3, 2.1]


def _radial_sampled_sups(
    cs: list[float], as_: list[float], r_hi: float, n: int = 20001
) -> tuple[float, float, float]:
    """Dense (in-test, independent) radial sups of |u|, |omega|, ||grad u||_F."""
    vel = vort = strain = 0.0
    for t in range(n + 1):
        r = r_hi * t / n
        d = [r * r + a * a for a in as_]
        q = sum(c / di for c, di in zip(cs, d, strict=True))
        omega_p = sum(c * a * a / (di * di) for c, a, di in zip(cs, as_, d, strict=True))
        w = sum(c / (di * di) for c, di in zip(cs, d, strict=True))
        vel = max(vel, abs(r * q) / (2.0 * math.pi))
        vort = max(vort, abs(omega_p) / math.pi)
        strain = max(strain, math.sqrt((omega_p**2 + r**4 * w**2) / (2.0 * math.pi**2)))
    return vel, vort, strain


def test_euler2d_steady_vortex_is_exact_steady_state() -> None:
    """u . grad omega == 0 exactly (perpendicularity); grid re-confirms ~1e-15."""
    cert = certified_euler2d_steady_vortex(coeffs=_COEFFS, scales=_SCALES)
    assert cert["schema_version"] == EULER2D_VORTEX_SCHEMA_VERSION
    assert certified_euler2d_steady_vortex_schema_errors(cert) == []
    assert cert["steady_residual_certified_sup"] == 0.0
    assert cert["honesty"]["exact_steady_state"] is True
    # the substrate's own independent component evaluation also vanishes to rounding
    assert cert["steady_residual_grid_max"] < 1e-12


def test_euler2d_substrate_identities_hold_on_grid() -> None:
    """div u = 0, R11 omega + R22 omega = -omega, and Leray div-free all ~machine zero."""
    cert = certified_euler2d_steady_vortex(coeffs=_COEFFS, scales=_SCALES, grid_points=11)
    assert cert["divergence_grid_max"] < 1e-10
    assert cert["riesz_trace_identity_grid_max"] < 1e-10
    assert cert["leray_divergence_grid_max"] < 1e-10


def test_euler2d_norm_sups_are_finite_and_dominate_dense_samples() -> None:
    """Anti-faking: the certified whole-plane sups dominate a dense direct sampling."""
    cert = certified_euler2d_steady_vortex(coeffs=_COEFFS, scales=_SCALES)
    for key in ("velocity_sup", "vorticity_sup", "strain_sup"):
        assert math.isfinite(cert[key]) and cert[key] > 0.0
    vel_s, vort_s, strain_s = _radial_sampled_sups(_COEFFS, _SCALES, cert["far_field_trunc"])
    assert vel_s <= cert["velocity_sup"] + 1e-12
    assert vort_s <= cert["vorticity_sup"] + 1e-12
    assert strain_s <= cert["strain_sup"] + 1e-12


def test_euler2d_single_blob_matches_analytic_peaks() -> None:
    """Single unit blob: ||u||=1/(4 pi a), ||omega||=1/(pi a^2) -- certified sup >= and tight."""
    a = 0.8
    cert = certified_euler2d_steady_vortex(coeffs=[1.0], scales=[a])
    vel_peak = 1.0 / (4.0 * math.pi * a)  # max of r/(2 pi (r^2+a^2)) at r=a
    vort_peak = 1.0 / (math.pi * a * a)  # omega(0)
    assert cert["velocity_sup"] >= vel_peak - 1e-12
    assert cert["velocity_sup"] <= vel_peak * 1.01
    assert cert["vorticity_sup"] >= vort_peak - 1e-12
    assert cert["vorticity_sup"] <= vort_peak * 1.01


def test_euler2d_circulation_and_energy_finiteness() -> None:
    """Circulation = sum c_i (unit-mass blobs); energy finite only if it vanishes."""
    cert = certified_euler2d_steady_vortex(coeffs=_COEFFS, scales=_SCALES)
    assert cert["circulation"] == pytest.approx(sum(_COEFFS))
    lo, hi = cert["circulation_enclosure"]
    assert lo <= sum(_COEFFS) <= hi
    assert cert["kinetic_energy_finite"] is False  # net circulation 1.2 != 0
    neutral = certified_euler2d_steady_vortex(coeffs=[1.0, -1.0], scales=[0.6, 1.3])
    assert neutral["circulation"] == pytest.approx(0.0)
    assert neutral["kinetic_energy_finite"] is True


def test_euler2d_honesty_is_not_overclaiming() -> None:
    """2-D Euler steady state: no global-regularity / 3-D / blow-up / SQG claims; SQG gap is recorded."""
    cert = certified_euler2d_steady_vortex(coeffs=_COEFFS, scales=_SCALES)
    h = cert["honesty"]
    assert h["unproven_claim"] is False
    assert h["three_d_claim"] is False
    assert h["blowup_claim"] is False
    assert h["sqg_claim"] is False
    assert h["two_dimensional_euler"] is True
    assert h["whole_plane_certified"] is True
    assert any("sqg" in ob for ob in cert["open_obligations"])


def test_euler2d_schema_rejects_forged_nonzero_residual() -> None:
    """A forged exact-steady cert with a nonzero residual sup is rejected."""
    cert = certified_euler2d_steady_vortex(coeffs=_COEFFS, scales=_SCALES)
    forged = json.loads(json.dumps(cert))
    forged["steady_residual_certified_sup"] = 0.5
    errors = certified_euler2d_steady_vortex_schema_errors(forged)
    assert any("steady_residual_certified_sup" in e for e in errors)


def test_euler2d_json_and_provenance_are_deterministic() -> None:
    """Same inputs -> identical sha256; certificate is JSON-native (lists not tuples)."""
    a = certified_euler2d_steady_vortex(coeffs=_COEFFS, scales=_SCALES)
    b = certified_euler2d_steady_vortex(coeffs=_COEFFS, scales=_SCALES)
    assert a["provenance"]["sha256"] == b["provenance"]["sha256"]
    restored = json.loads(json.dumps(a))
    assert restored["provenance"]["sha256"] == a["provenance"]["sha256"]
    assert restored["open_obligations"] == a["open_obligations"]


def test_euler2d_input_validation() -> None:
    with pytest.raises(ValueError):
        certified_euler2d_steady_vortex(coeffs=[], scales=[])
    with pytest.raises(ValueError):
        certified_euler2d_steady_vortex(coeffs=[1.0, 0.5], scales=[0.6])
    with pytest.raises(ValueError):
        certified_euler2d_steady_vortex(coeffs=[1.0], scales=[0.0])
    with pytest.raises(ValueError):
        certified_euler2d_steady_vortex(coeffs=[1.0], scales=[1.0], far_field_trunc=0.5)


def test_euler2d_symbolic_replay_matches() -> None:
    """The numpy-only twin agrees and confirms the certified sups dominate samples."""
    from omnibias.symbolic import verify_euler2d_steady_vortex

    cert = certified_euler2d_steady_vortex(coeffs=_COEFFS, scales=_SCALES)
    rep = verify_euler2d_steady_vortex(cert)
    assert rep["replay_match"] is True
    assert rep["sup_dominates_samples"] is True
    assert rep["steady_residual_is_zero"] is True
    assert rep["identities_hold"] is True
    assert rep["circulation_match"] is True
    assert rep["unproven_claim"] is False


def test_euler2d_symbolic_replay_catches_forged_sup() -> None:
    """Replaying a certificate with an understated velocity sup must fail (anti-faking)."""
    from omnibias.symbolic import verify_euler2d_steady_vortex

    cert = certified_euler2d_steady_vortex(coeffs=_COEFFS, scales=_SCALES)
    forged = json.loads(json.dumps(cert))
    forged["velocity_sup"] = 1e-6  # impossibly small
    rep = verify_euler2d_steady_vortex(forged)
    assert rep["sup_dominates_samples"] is False
    assert rep["replay_match"] is False
