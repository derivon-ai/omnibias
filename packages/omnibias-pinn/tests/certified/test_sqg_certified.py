# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Certified 2-D SQG steady-vortex certified-evidence tests (single-Riesz/Poisson substrate)."""

from __future__ import annotations

import json
import math

import pytest
from omnibias.pinn.certified import (
    SQG_COERCIVITY_SCHEMA_VERSION,
    SQG_SELFSIMILAR_SCHEMA_VERSION,
    SQG_VORTEX_SCHEMA_VERSION,
    certified_sqg_linearized_coercivity_attempt,
    certified_sqg_linearized_coercivity_attempt_schema_errors,
    certified_sqg_selfsimilar_blowup_attempt,
    certified_sqg_selfsimilar_blowup_attempt_schema_errors,
    certified_sqg_steady_vortex,
    certified_sqg_steady_vortex_schema_errors,
)

_COEFFS = [1.0, 0.4, -0.2]
_SCALES = [0.6, 1.3, 2.1]


def _radial_sampled_sups(
    cs: list[float], as_: list[float], r_hi: float, n: int = 20001
) -> tuple[float, float, float]:
    """Dense (in-test, independent) radial sups of |u|, |theta|, ||grad u||_F."""
    tp = 2.0 * math.pi
    vel = temp = strain = 0.0
    for t in range(n + 1):
        r = r_hi * t / n
        d = [r * r + a * a for a in as_]
        s = sum(c * di**-1.5 for c, di in zip(cs, d, strict=True))
        tt = sum(c * di**-2.5 for c, di in zip(cs, d, strict=True))
        theta = sum(c * a * di**-1.5 for c, a, di in zip(cs, as_, d, strict=True))
        vel = max(vel, abs(r * s) / tp)
        temp = max(temp, abs(theta) / tp)
        strain = max(strain, math.sqrt(abs(2 * s * s - 6 * r * r * s * tt + 9 * r**4 * tt * tt)) / tp)
    return vel, temp, strain


def test_sqg_steady_vortex_is_exact_steady_state() -> None:
    """u . grad theta == 0 exactly (perpendicularity); grid re-confirms ~1e-15."""
    cert = certified_sqg_steady_vortex(coeffs=_COEFFS, scales=_SCALES)
    assert cert["schema_version"] == SQG_VORTEX_SCHEMA_VERSION
    assert certified_sqg_steady_vortex_schema_errors(cert) == []
    assert cert["steady_residual_certified_sup"] == 0.0
    assert cert["honesty"]["exact_steady_state"] is True
    assert cert["steady_residual_grid_max"] < 1e-12


def test_sqg_substrate_identities_hold_on_grid() -> None:
    """div u = 0 and u = R^perp theta both hold to machine zero on the grid."""
    cert = certified_sqg_steady_vortex(coeffs=_COEFFS, scales=_SCALES, grid_points=11)
    assert cert["divergence_grid_max"] < 1e-10
    assert cert["riesz_perp_identity_grid_max"] < 1e-10


def test_sqg_norm_sups_are_finite_and_dominate_dense_samples() -> None:
    """Anti-faking: the certified whole-plane sups dominate a dense direct sampling."""
    cert = certified_sqg_steady_vortex(coeffs=_COEFFS, scales=_SCALES)
    for key in ("velocity_sup", "temperature_sup", "strain_sup"):
        assert math.isfinite(cert[key]) and cert[key] > 0.0
    vel_s, temp_s, strain_s = _radial_sampled_sups(_COEFFS, _SCALES, cert["far_field_trunc"])
    assert vel_s <= cert["velocity_sup"] + 1e-12
    assert temp_s <= cert["temperature_sup"] + 1e-12
    assert strain_s <= cert["strain_sup"] + 1e-12


def test_sqg_single_blob_matches_analytic_peaks() -> None:
    """Single unit blob: ||theta||=1/(2 pi a^2), ||u||=1/(3 sqrt3 pi a^2) -- tight."""
    a = 0.8
    cert = certified_sqg_steady_vortex(coeffs=[1.0], scales=[a])
    temp_peak = 1.0 / (2.0 * math.pi * a * a)  # theta(0)
    vel_peak = 1.0 / (3.0 * math.sqrt(3.0) * math.pi * a * a)  # max of r/(2 pi (r^2+a^2)^{3/2})
    assert cert["temperature_sup"] >= temp_peak - 1e-12
    assert cert["temperature_sup"] <= temp_peak * 1.01
    assert cert["velocity_sup"] >= vel_peak - 1e-12
    assert cert["velocity_sup"] <= vel_peak * 1.01


def test_sqg_total_temperature_and_energy() -> None:
    """Total temperature = sum c_i (unit-mass Poisson blobs); SQG energy always finite."""
    cert = certified_sqg_steady_vortex(coeffs=_COEFFS, scales=_SCALES)
    assert cert["total_temperature"] == pytest.approx(sum(_COEFFS))
    lo, hi = cert["total_temperature_enclosure"]
    assert lo <= sum(_COEFFS) <= hi
    # SQG velocity decays like 1/r^2, so int |u|^2 converges for any coefficients.
    assert cert["kinetic_energy_finite"] is True


def test_sqg_honesty_is_not_overclaiming() -> None:
    """Genuine SQG (single Riesz closed form) but a steady state: no blow-up/global-regularity claim."""
    cert = certified_sqg_steady_vortex(coeffs=_COEFFS, scales=_SCALES)
    h = cert["honesty"]
    assert h["unproven_claim"] is False
    assert h["three_d_claim"] is False
    assert h["blowup_claim"] is False
    assert h["two_dimensional_sqg"] is True
    assert h["sqg_velocity_closed_form"] is True
    assert h["whole_plane_certified"] is True
    # the famous open SQG singularity problem is recorded, not claimed
    assert any("singularity" in ob for ob in cert["open_obligations"])


def test_sqg_schema_rejects_forged_nonzero_residual() -> None:
    """A forged exact-steady cert with a nonzero residual sup is rejected."""
    cert = certified_sqg_steady_vortex(coeffs=_COEFFS, scales=_SCALES)
    forged = json.loads(json.dumps(cert))
    forged["steady_residual_certified_sup"] = 0.5
    errors = certified_sqg_steady_vortex_schema_errors(forged)
    assert any("steady_residual_certified_sup" in e for e in errors)


def test_sqg_schema_rejects_blowup_overclaim() -> None:
    """Flipping blowup_claim to True must be rejected by the schema validator."""
    cert = certified_sqg_steady_vortex(coeffs=_COEFFS, scales=_SCALES)
    forged = json.loads(json.dumps(cert))
    forged["honesty"]["blowup_claim"] = True
    errors = certified_sqg_steady_vortex_schema_errors(forged)
    assert any("blowup_claim" in e for e in errors)


def test_sqg_json_and_provenance_are_deterministic() -> None:
    """Same inputs -> identical sha256; certificate is JSON-native (lists not tuples)."""
    a = certified_sqg_steady_vortex(coeffs=_COEFFS, scales=_SCALES)
    b = certified_sqg_steady_vortex(coeffs=_COEFFS, scales=_SCALES)
    assert a["provenance"]["sha256"] == b["provenance"]["sha256"]
    restored = json.loads(json.dumps(a))
    assert restored["provenance"]["sha256"] == a["provenance"]["sha256"]
    assert restored["open_obligations"] == a["open_obligations"]


def test_sqg_input_validation() -> None:
    with pytest.raises(ValueError):
        certified_sqg_steady_vortex(coeffs=[], scales=[])
    with pytest.raises(ValueError):
        certified_sqg_steady_vortex(coeffs=[1.0, 0.5], scales=[0.6])
    with pytest.raises(ValueError):
        certified_sqg_steady_vortex(coeffs=[1.0], scales=[0.0])
    with pytest.raises(ValueError):
        certified_sqg_steady_vortex(coeffs=[1.0], scales=[1.0], far_field_trunc=0.5)


def test_sqg_symbolic_replay_matches() -> None:
    """The numpy-only twin agrees and confirms the certified sups dominate samples."""
    from omnibias.symbolic import verify_sqg_steady_vortex

    cert = certified_sqg_steady_vortex(coeffs=_COEFFS, scales=_SCALES)
    rep = verify_sqg_steady_vortex(cert)
    assert rep["replay_match"] is True
    assert rep["sup_dominates_samples"] is True
    assert rep["steady_residual_is_zero"] is True
    assert rep["identities_hold"] is True
    assert rep["total_temperature_match"] is True
    assert rep["unproven_claim"] is False


def test_sqg_symbolic_replay_catches_forged_sup() -> None:
    """Replaying a certificate with an understated velocity sup must fail (anti-faking)."""
    from omnibias.symbolic import verify_sqg_steady_vortex

    cert = certified_sqg_steady_vortex(coeffs=_COEFFS, scales=_SCALES)
    forged = json.loads(json.dumps(cert))
    forged["velocity_sup"] = 1e-6  # impossibly small
    rep = verify_sqg_steady_vortex(forged)
    assert rep["sup_dominates_samples"] is False
    assert rep["replay_match"] is False


# --------------------------------------------------------------------------- #
# Self-similar blow-up: the certified obstruction + conditional pipeline       #
# --------------------------------------------------------------------------- #
def _closed_form_l2_sq(cs: list[float], as_: list[float]) -> float:
    """Independent in-test closed form sum_ij c_i c_j / (2 pi (a_i+a_j)^2)."""
    return sum(
        ci * cj / (2.0 * math.pi * (ai + aj) ** 2)
        for ci, ai in zip(cs, as_, strict=True)
        for cj, aj in zip(cs, as_, strict=True)
    )


def test_sqg_selfsimilar_certifies_no_exact_profile() -> None:
    """The certificate proves no localized exact self-similar profile exists."""
    cert = certified_sqg_selfsimilar_blowup_attempt(coeffs=_COEFFS, scales=_SCALES)
    assert cert["schema_version"] == SQG_SELFSIMILAR_SCHEMA_VERSION
    assert certified_sqg_selfsimilar_blowup_attempt_schema_errors(cert) == []
    assert cert["exact_selfsimilar_profile_exists"] is False
    assert cert["exact_selfsimilar_obstruction_certified"] is True
    # residual is bounded below by a strictly positive number ||Theta||_2
    assert cert["selfsimilar_residual_l2_lower_bound"] > 0.0
    lo, hi = cert["profile_l2_norm"]
    assert lo <= cert["selfsimilar_residual_l2_lower_bound"] <= hi


def test_sqg_selfsimilar_nogo_identity_and_l2_norm_closed_form() -> None:
    """<F,Theta> = -||Theta||^2 and ||Theta||^2 matches the independent closed form."""
    cert = certified_sqg_selfsimilar_blowup_attempt(coeffs=_COEFFS, scales=_SCALES)
    l2_lo, l2_hi = cert["profile_l2_norm_sq"]
    closed = _closed_form_l2_sq(_COEFFS, _SCALES)
    assert l2_lo <= closed <= l2_hi
    # the inner product encloses -||Theta||^2 (strictly negative -> no zero residual)
    ip_lo, ip_hi = cert["selfsimilar_residual_inner_product"]
    assert ip_hi < 0.0
    assert ip_lo <= -closed <= ip_hi


def test_sqg_selfsimilar_drift_is_destabilizing() -> None:
    """div V = 2 and the L^2 drift energy coefficient is +1 (destabilizing)."""
    cert = certified_sqg_selfsimilar_blowup_attempt(coeffs=_COEFFS, scales=_SCALES)
    assert cert["divergence_of_selfsimilar_drift"] == 2.0
    assert cert["l2_self_similar_drift_energy_coefficient"] == 1.0
    assert cert["l2_drift_is_destabilizing"] is True
    # F = y . grad theta confirmed: (R^perp theta) . grad theta == 0 on the grid
    assert cert["perpendicularity_grid_max"] < 1e-12


def test_sqg_selfsimilar_honesty_records_open_lemma() -> None:
    """No blow-up / global-regularity / 3-D claim; the conditional pipeline names one open lemma."""
    cert = certified_sqg_selfsimilar_blowup_attempt(coeffs=_COEFFS, scales=_SCALES)
    h = cert["honesty"]
    assert h["unproven_claim"] is False
    assert h["three_d_claim"] is False
    assert h["blowup_claim"] is False
    assert h["localized_selfsimilar_obstruction_certified"] is True
    assert h["conditional_blowup_pending_infinite_tail_coercivity"] is True
    assert any("coercivity" in ob for ob in cert["open_obligations"])
    assert any("cordoba" in ob for ob in cert["open_obligations"])


def test_sqg_selfsimilar_schema_rejects_profile_exists_overclaim() -> None:
    """Flipping exact_selfsimilar_profile_exists to True must be rejected."""
    cert = certified_sqg_selfsimilar_blowup_attempt(coeffs=_COEFFS, scales=_SCALES)
    forged = json.loads(json.dumps(cert))
    forged["exact_selfsimilar_profile_exists"] = True
    errors = certified_sqg_selfsimilar_blowup_attempt_schema_errors(forged)
    assert any("exact_selfsimilar_profile_exists" in e for e in errors)


def test_sqg_selfsimilar_schema_rejects_blowup_and_positive_residual() -> None:
    """blowup_claim=True and a non-negative inner product are both rejected."""
    cert = certified_sqg_selfsimilar_blowup_attempt(coeffs=_COEFFS, scales=_SCALES)
    forged = json.loads(json.dumps(cert))
    forged["honesty"]["blowup_claim"] = True
    forged["selfsimilar_residual_inner_product"] = [0.0, 1.0]
    errors = certified_sqg_selfsimilar_blowup_attempt_schema_errors(forged)
    assert any("blowup_claim" in e for e in errors)
    assert any("selfsimilar_residual_inner_product" in e for e in errors)


def test_sqg_selfsimilar_input_validation() -> None:
    with pytest.raises(ValueError):
        certified_sqg_selfsimilar_blowup_attempt(coeffs=[], scales=[])
    with pytest.raises(ValueError):
        certified_sqg_selfsimilar_blowup_attempt(coeffs=[1.0, 0.5], scales=[0.6])
    with pytest.raises(ValueError):
        certified_sqg_selfsimilar_blowup_attempt(coeffs=[1.0], scales=[0.0])
    with pytest.raises(ValueError):
        certified_sqg_selfsimilar_blowup_attempt(coeffs=[0.0, 0.0], scales=[0.6, 1.3])


def test_sqg_selfsimilar_deterministic_sha() -> None:
    a = certified_sqg_selfsimilar_blowup_attempt(coeffs=_COEFFS, scales=_SCALES)
    b = certified_sqg_selfsimilar_blowup_attempt(coeffs=_COEFFS, scales=_SCALES)
    assert a["provenance"]["sha256"] == b["provenance"]["sha256"]
    restored = json.loads(json.dumps(a))
    assert restored["provenance"]["sha256"] == a["provenance"]["sha256"]


def test_sqg_selfsimilar_symbolic_replay_matches() -> None:
    """The numpy quadrature twin independently confirms the no-go + obstruction."""
    from omnibias.symbolic import verify_sqg_selfsimilar_blowup_attempt

    cert = certified_sqg_selfsimilar_blowup_attempt(coeffs=_COEFFS, scales=_SCALES)
    rep = verify_sqg_selfsimilar_blowup_attempt(cert)
    assert rep["replay_match"] is True
    assert rep["nogo_identity_holds"] is True
    assert rep["obstruction_holds"] is True
    assert rep["profile_l2_norm_sq_match"] is True
    assert rep["drift_coefficient_match"] is True
    assert rep["perpendicularity_holds"] is True
    assert rep["unproven_claim"] is False


def test_sqg_selfsimilar_replay_catches_forged_norm() -> None:
    """A forged (understated) profile L^2 norm is caught by the quadrature twin."""
    from omnibias.symbolic import verify_sqg_selfsimilar_blowup_attempt

    cert = certified_sqg_selfsimilar_blowup_attempt(coeffs=_COEFFS, scales=_SCALES)
    forged = json.loads(json.dumps(cert))
    forged["profile_l2_norm_sq"] = [1e-6, 2e-6]  # impossibly small
    rep = verify_sqg_selfsimilar_blowup_attempt(forged)
    assert rep["profile_l2_norm_sq_match"] is False
    assert rep["replay_match"] is False


# --------------------------------------------------------------------------- #
# Conditional L^2 coercivity of the linearized rescaled operator               #
# --------------------------------------------------------------------------- #
def _grad_theta_sup_sampled(cs: list[float], as_: list[float], r_hi: float, n: int = 20001) -> float:
    """In-test dense sample of ||grad theta||_inf = (3/2pi) max_r |r sum c_i a_i D_i^{-5/2}|."""
    best = 0.0
    for t in range(n + 1):
        r = r_hi * t / n
        inner = sum(c * a * (r * r + a * a) ** -2.5 for c, a in zip(cs, as_, strict=True))
        best = max(best, (3.0 / (2.0 * math.pi)) * abs(r * inner))
    return best


def test_sqg_coercivity_l2_gap_is_one_minus_grad_sup() -> None:
    """gap = 1 - ||grad theta||_inf, with the +1 drift and Riesz isometry as exact facts."""
    cert = certified_sqg_linearized_coercivity_attempt(coeffs=_COEFFS, scales=_SCALES)
    assert cert["schema_version"] == SQG_COERCIVITY_SCHEMA_VERSION
    assert certified_sqg_linearized_coercivity_attempt_schema_errors(cert) == []
    assert cert["drift_self_adjoint_coefficient"] == 1.0
    assert cert["riesz_isometry_constant"] == 1.0
    assert cert["l2_coercivity_gap_lower"] == pytest.approx(
        1.0 - cert["grad_theta_sup"], abs=1e-9
    )


def test_sqg_coercivity_block_engine_reproduces_weyl_bound() -> None:
    """The general finite-section+tail engine (a=d=1, b=coupling) matches the Weyl gap."""
    cert = certified_sqg_linearized_coercivity_attempt(coeffs=_COEFFS, scales=_SCALES)
    block = cert["block_operator_gap"]
    assert block["gap_lower"] == pytest.approx(cert["l2_coercivity_gap_lower"], abs=1e-9)
    assert block["tail_is_hypothesis"] is True
    assert block["threshold_tail_gap"] == pytest.approx(cert["grad_theta_sup"] ** 2, rel=1e-9)


def test_sqg_coercivity_grad_sup_dominates_dense_samples() -> None:
    """Anti-faking: the certified whole-plane ||grad theta||_inf dominates a dense sample."""
    cert = certified_sqg_linearized_coercivity_attempt(coeffs=_COEFFS, scales=_SCALES)
    sampled = _grad_theta_sup_sampled(_COEFFS, _SCALES, cert["far_field_trunc"])
    assert sampled <= cert["grad_theta_sup"] + 1e-12


def test_sqg_coercivity_large_amplitude_is_not_l2_coercive() -> None:
    """A large-gradient background has ||grad theta||_inf > 1, so the L^2 gap is negative."""
    cert = certified_sqg_linearized_coercivity_attempt(coeffs=[2.0], scales=[0.5])
    assert cert["grad_theta_sup"] > 1.0
    assert cert["l2_coercive"] is False
    assert cert["l2_coercivity_gap_lower"] < 0.0


def test_sqg_coercivity_honesty_is_a_diagnostic_not_stability() -> None:
    cert = certified_sqg_linearized_coercivity_attempt(coeffs=_COEFFS, scales=_SCALES)
    h = cert["honesty"]
    assert h["unproven_claim"] is False
    assert h["three_d_claim"] is False
    assert h["blowup_claim"] is False
    assert h["stability_claim"] is False
    assert h["l2_linear_diagnostic_only"] is True
    assert h["background_is_exact_profile"] is False
    assert len(cert["what_this_does_not_prove"]) >= 3
    assert any("weighted" in ob for ob in cert["open_obligations"])


def test_sqg_coercivity_schema_rejects_stability_overclaim() -> None:
    cert = certified_sqg_linearized_coercivity_attempt(coeffs=_COEFFS, scales=_SCALES)
    forged = json.loads(json.dumps(cert))
    forged["honesty"]["stability_claim"] = True
    errors = certified_sqg_linearized_coercivity_attempt_schema_errors(forged)
    assert any("stability_claim" in e for e in errors)


def test_sqg_coercivity_schema_rejects_inconsistent_gap_flag() -> None:
    cert = certified_sqg_linearized_coercivity_attempt(coeffs=_COEFFS, scales=_SCALES)
    forged = json.loads(json.dumps(cert))
    forged["l2_coercive"] = not forged["l2_coercive"]  # contradict the gap sign
    errors = certified_sqg_linearized_coercivity_attempt_schema_errors(forged)
    assert any("l2_coercive" in e for e in errors)


def test_sqg_coercivity_deterministic_sha() -> None:
    a = certified_sqg_linearized_coercivity_attempt(coeffs=_COEFFS, scales=_SCALES)
    b = certified_sqg_linearized_coercivity_attempt(coeffs=_COEFFS, scales=_SCALES)
    assert a["provenance"]["sha256"] == b["provenance"]["sha256"]


def test_sqg_coercivity_input_validation() -> None:
    with pytest.raises(ValueError):
        certified_sqg_linearized_coercivity_attempt(coeffs=[], scales=[])
    with pytest.raises(ValueError):
        certified_sqg_linearized_coercivity_attempt(coeffs=[1.0, 0.5], scales=[0.6])
    with pytest.raises(ValueError):
        certified_sqg_linearized_coercivity_attempt(coeffs=[1.0], scales=[0.0])
    with pytest.raises(ValueError):
        certified_sqg_linearized_coercivity_attempt(coeffs=[1.0], scales=[1.0], far_field_trunc=0.5)


def test_sqg_coercivity_symbolic_replay_matches() -> None:
    from omnibias.symbolic import verify_sqg_linearized_coercivity_attempt

    cert = certified_sqg_linearized_coercivity_attempt(coeffs=_COEFFS, scales=_SCALES)
    rep = verify_sqg_linearized_coercivity_attempt(cert)
    assert rep["replay_match"] is True
    assert rep["grad_sup_dominates_samples"] is True
    assert rep["l2_gap_match"] is True
    assert rep["block_gap_match"] is True
    assert rep["unproven_claim"] is False


def test_sqg_coercivity_replay_catches_forged_grad_sup() -> None:
    from omnibias.symbolic import verify_sqg_linearized_coercivity_attempt

    cert = certified_sqg_linearized_coercivity_attempt(coeffs=_COEFFS, scales=_SCALES)
    forged = json.loads(json.dumps(cert))
    forged["grad_theta_sup"] = 1e-6  # impossibly small -> sample beats it
    rep = verify_sqg_linearized_coercivity_attempt(forged)
    assert rep["grad_sup_dominates_samples"] is False
    assert rep["replay_match"] is False
