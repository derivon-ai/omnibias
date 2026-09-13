# SPDX-License-Identifier: Apache-2.0
"""A global smooth theta quasimode with an explicit actual-vacuum error.

The quaternion polynomial, Haar moments and spectral-projection proof are in
docs/api/gauge-theta-quasimode.md. Rational replay certifies their numerical
budget and the canonical absolute excited-energy source, not the analytic proof.
"""

from __future__ import annotations

from fractions import Fraction as Q
from math import factorial
from typing import Any

from omnibias.core.proof.certificate import make_certificate, verify_certificate_digest
from omnibias.geometry.gauge.transfer.static_sources import _rational

_KAPPA_MAX = Q(1, 64)


def _theta_source(coupling: Q) -> tuple[dict[str, Any], bool, Q | None]:
    from omnibias.geometry.gauge.transfer.theta_weak_blocks import (
        replay_su2_theta_weak_block_certificate,
        su2_theta_weak_block_gaps,
    )

    result = su2_theta_weak_block_gaps(coupling)
    source = result.get("certificate", {})
    if not isinstance(source, dict):
        return {}, False, None
    try:
        p = source["payload"]
        excited = Q(p["arithmetic"]["first_reduced_excitation_lower"])
        verified = bool(
            replay_su2_theta_weak_block_certificate(source)
            and p["type"] == "su2_theta_weak_block_gaps_v1"
            and Q(p["inputs"]["kappa"]) == coupling
            and p["status"] == "PASS"
            and p["actual_full_reduced_gap_verified_in_written_analysis"] is True
            and p["actual_theta_vacuum_identified_in_written_analysis"] is True
            and excited >= Q(32, 5)
        )
    except (KeyError, TypeError, ValueError, ZeroDivisionError):
        excited, verified = None, False
    return source, verified, excited


def su2_theta_quasimode_error(kappa: int | Q = _KAPPA_MAX) -> dict[str, Any]:
    """Bound the actual normalized theta vacuum by a smooth global trial.

    For 0<kappa<=1/64, the residual is at most 83*kappa in L2(Haar),
    the positive normalized vacuum differs from the trial by at most
    332*kappa in L2, and their densities differ by at most 664*kappa in L1.
    The bounds need not be informative near the upper endpoint. Positive
    inputs outside the proved interval return INCONCLUSIVE, not nonexistence.
    """
    coupling = _rational(kappa, "kappa")
    if coupling <= 0:
        raise ValueError("kappa must be strictly positive")
    source, source_ok, excited = _theta_source(coupling)
    moments = {m: Q(512, 3) * factorial(m + 2) * Q(9, 16) ** (m + 3) for m in (2, 3, 4)}
    residual_squared = Q(25, 4) * moments[2] + 5 * moments[3] + moments[4]
    gates = {
        "requested_kappa_in_proved_interval": coupling <= _KAPPA_MAX,
        "canonical_absolute_excited_energy_source": source_ok,
        "phase_lower_at_least_8_9": Q(8, 9) ** 2 <= Q(4, 5),
        "phase_upper_at_most_6_5": Q(4, 3) <= Q(6, 5) ** 2,
        "haar_normalization_lower": Q(1152 * 22, 7) <= 4096,
        "residual_coefficient_below_83_squared": residual_squared < 83**2,
        "positive_phase_projection_factor_below_4": 2 * Q(5, 2) ** 2 < 4**2,
    }
    passed = all(gates.values())
    arithmetic = {
        "proved_kappa_upper": str(_KAPPA_MAX),
        "phase_lower_per_action": "8/9",
        "phase_upper_per_action": "6/5",
        "trial_normalization_lower_coefficient": "1/4096",
        "trial_normalization_lower_power": 3,
        "local_residual_action_coefficient": "5/2",
        "local_residual_squared_action_coefficient": "1",
        "normalized_action_moment_coefficients": {str(m): str(v) for m, v in moments.items()},
        "residual_l2_squared_coefficient_upper": str(residual_squared),
        "residual_l2_coefficient_upper": "83",
        "oscillator_energy": "(3/2)*(sqrt(3)+sqrt(5))",
        "oscillator_energy_upper": "6",
        "source_first_reduced_excitation_lower": str(excited) if excited is not None else None,
        "excited_distance_from_oscillator_lower": "2/5" if passed else None,
        "vacuum_l2_coefficient_upper": "332",
        "vacuum_density_l1_coefficient_upper": "664",
        "residual_l2_upper": str(83 * coupling) if passed else None,
        "vacuum_l2_upper": str(332 * coupling) if passed else None,
        "vacuum_density_l1_upper": str(664 * coupling) if passed else None,
        "gates": gates,
    }
    scopes = {
        "actual_quasimode_error_verified_in_written_analysis": passed,
        "actual_vacuum_l2_error_verified_in_written_analysis": passed,
        "actual_vacuum_density_l1_error_verified_in_written_analysis": passed,
        "canonical_source_replayed": source_ok,
        "all_reduced_representations_included": passed,
        "global_smooth_trial_verified_in_written_analysis": passed,
        "density_error_is_normalized_kernel_error": False,
        "explicit_maximal_correlation_threshold_verified": False,
        "unrestricted_original_seven_link_gap_verified": False,
        "ambient_exterior_uniformity_verified": False,
        "uniform_in_volume_claim": False,
        "all_scale_refinement_claim": False,
        "continuum_claim": False,
        "yang_mills_claim": False,
        "yang_mills_mass_gap_claim": False,
        "analytic_proof_formally_verified": False,
        "theorem_prover_verified": False,
        "mathlib_verified": False,
    }
    payload = {
        "type": "su2_theta_quasimode_error_v1",
        "status": "PASS" if passed else "INCONCLUSIVE",
        "inputs": {"kappa": str(coupling)},
        "theta_weak_source_certificate": source,
        "model": "seven original unit-weight links and two adjacent square plaquettes",
        "reduced_operator": "H=kappa*(3*(Cx+Cy)+C_simultaneous_left)/2+2*S/kappa",
        "normalization": "C_fund=3/4; normalized product Haar on SU2^2; S=4-2*x0-2*y0",
        "phase": "F=(1/sqrt(3)+1/sqrt(5))*S+2*(1/sqrt(5)-1/sqrt(3))*dot(X,Y)",
        "trial": "v=exp(-F/kappa)/sqrt(integral(exp(-2*F/kappa),Haar^2))",
        "vacuum": "the unique positive L2-normalized actual reduced theta ground state",
        "norms": "L2 wavefunction and unhalved L1 density norms; same bounds under unitary exponential-coordinate rescaling",
        "source_role": "absolute first full-reduced excited energy >=32/5, not an assumed trial-vacuum overlap",
        "arithmetic": arithmetic,
        **scopes,
    }
    certificate = make_certificate(
        claim="explicit smooth theta quasimode residual and actual normalized vacuum error",
        payload=payload,
        meta={
            "analytic_implication": "docs/api/gauge-theta-quasimode.md",
            "transcend_backend": "not_used",
            "proof_register": "written quaternion/Haar/spectral projection proof and exact rational canonical replay",
        },
    )
    return {**payload, "certificate": certificate}


def replay_su2_theta_quasimode_certificate(certificate: dict[str, Any]) -> bool:
    """Rebuild every field and the nested absolute-energy certificate."""
    try:
        if not isinstance(certificate, dict) or not verify_certificate_digest(certificate):
            return False
        payload = certificate["payload"]
        if payload["type"] != "su2_theta_quasimode_error_v1":
            return False
        expected = su2_theta_quasimode_error(Q(payload["inputs"]["kappa"]))
        return bool(certificate == expected["certificate"])
    except (KeyError, TypeError, ValueError, ZeroDivisionError, OverflowError):
        return False


__all__ = [
    "replay_su2_theta_quasimode_certificate",
    "su2_theta_quasimode_error",
]
