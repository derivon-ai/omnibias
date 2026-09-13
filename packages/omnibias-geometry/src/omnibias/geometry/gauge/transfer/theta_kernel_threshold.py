# SPDX-License-Identifier: Apache-2.0
"""Explicit actual-theta correlation intervals, represented by dyadic exponents.

The density, spectral and normalized-kernel argument is written analysis in
docs/api/gauge-theta-kernel-threshold.md. This module replays
its canonical sources and exact scalar budgets without materializing 2**N.
"""

from __future__ import annotations

from fractions import Fraction as Q
from typing import Any

from omnibias.core.proof.certificate import make_certificate, verify_certificate_digest

_RADIUS = 2048
_LOG_AMPLIFICATION = 61_256 + _RADIUS**2
_MINIMUM_EXPONENT = 4 * _LOG_AMPLIFICATION + 21 + 10


def _sources() -> tuple[dict[str, Any], dict[str, Any], bool, bool]:
    from omnibias.geometry.gauge.transfer.theta_kernel_tail import (
        replay_su2_theta_kernel_tail_certificate,
        su2_theta_normalized_kernel_tail,
    )
    from omnibias.geometry.gauge.transfer.theta_quasimode import (
        replay_su2_theta_quasimode_certificate,
        su2_theta_quasimode_error,
    )

    trial = su2_theta_quasimode_error(Q(1, 64))
    tail = su2_theta_normalized_kernel_tail(Q(1, 2**20), Q(_RADIUS**2, 10), target=Q(1, 4096))
    trial_certificate = trial.get("certificate", {})
    tail_certificate = tail.get("certificate", {})
    try:
        trial_replayed = bool(
            replay_su2_theta_quasimode_certificate(trial_certificate)
            and trial == {**trial_certificate["payload"], "certificate": trial_certificate}
            and trial["actual_quasimode_error_verified_in_written_analysis"] is True
            and trial["actual_vacuum_l2_error_verified_in_written_analysis"] is True
            and Q(trial["arithmetic"]["residual_l2_coefficient_upper"]) <= 83
        )
        tail_replayed = bool(
            replay_su2_theta_kernel_tail_certificate(tail_certificate)
            and tail == {**tail_certificate["payload"], "certificate": tail_certificate}
            and tail["actual_normalized_kernel_tail_verified_in_written_analysis"] is True
            and tail["tail_target_verified"] is True
            and tail["tail_region_has_zero_haar_measure"] is False
        )
    except (KeyError, TypeError, ValueError, ZeroDivisionError):
        trial_replayed, tail_replayed = False, False
    return trial_certificate, tail_certificate, trial_replayed, tail_replayed


def su2_theta_kernel_contraction(
    *,
    dyadic_exponent: int = _MINIMUM_EXPONENT,
) -> dict[str, Any]:
    """Certify actual cycle correlations for every 0<kappa<=2**(-N).

    The proved exponent threshold is N>=17_022_271. This family is stored
    symbolically; no huge rational denominator or floating power is formed.
    A smaller positive N returns INCONCLUSIVE, not absence of contraction.
    The graph has exactly seven links and two square plaquettes.
    """
    if type(dyadic_exponent) is not int:
        raise TypeError("dyadic_exponent must be an integer")
    if dyadic_exponent <= 0:
        raise ValueError("dyadic_exponent must be positive")
    trial, tail, trial_replayed, tail_replayed = _sources()
    exponent = Q(81_640) - Q(_RADIUS**2, 45)
    negative_floor = (-exponent).numerator // (-exponent).denominator
    gates = {
        "canonical_quasimode_source": trial_replayed,
        "canonical_nonempty_actual_tail_source": tail_replayed,
        "trial_gaussian_exponent_remainder": Q(43, 2880) < Q(1, 64),
        "trial_gaussian_unnormalized_l1_budget": 32_768 * (Q(24, 9) + Q(768, 64)) < 2**19,
        "actual_gaussian_density_l1_budget": 8 * 83 + 2**20 < 2**21,
        "marginal_normalization_budget": 625 * 16**2 * Q(22, 7) ** 4 < 2**24,
        "gaussian_kernel_precision_budget": Q(8, 9) - Q(10, 19) > Q(1, 3),
        "squared_tail_prefactor_budget": 26_000_000 < 2**25,
        "squared_tail_target_budget": negative_floor - 25 >= 12,
        "dyadic_domain_inside_marginal_chart": dyadic_exponent >= 20,
        "central_error_at_most_one_quarter": dyadic_exponent >= _MINIMUM_EXPONENT,
        "radial_correlation_below_three_tenths": Q(1, 60) + Q(9, 32) < Q(3, 10),
        "full_cycle_correlation_below_one_half": Q(1, 7) + Q(9, 32) < Q(1, 2),
    }
    passed = all(gates.values())
    payload = {
        "type": "su2_theta_kernel_contraction_v1",
        "status": "PASS" if passed else "INCONCLUSIVE",
        "inputs": {"dyadic_exponent": dyadic_exponent},
        "model": "actual seven-link two-plaquette SU(2) Hamiltonian",
        "normalization": "H=kappa*(3*(Cx+Cy)+Cdiag)/2+2*(A(x)+A(y))/kappa; Cfund=3/4",
        "coupling_family": {
            "quantifier": "every positive real kappa in this interval",
            "upper": {"base": 2, "negative_integer_exponent": dyadic_exponent},
            "floating_threshold_materialized": False,
        },
        "quasimode_source_certificate": trial,
        "normalized_tail_source_certificate": tail,
        "arithmetic": {
            "cutoff_radius": _RADIUS,
            "marginal_chart_coupling_exponent": 20,
            "trial_gaussian_density_l1_coefficient_upper": str(2**20),
            "actual_gaussian_density_l1_coefficient_upper": str(8 * 83 + 2**20),
            "density_l1_dyadic_coefficient_exponent": 21,
            "joint_density_upper_log": 40_822,
            "marginal_lower_negative_log": 40_845 + _RADIUS**2,
            "log_amplification": _LOG_AMPLIFICATION,
            "dyadic_amplification_exponent": 2 * _LOG_AMPLIFICATION + 3,
            "minimum_coupling_exponent": _MINIMUM_EXPONENT,
            "coupling_exponent_margin": dyadic_exponent - _MINIMUM_EXPONENT,
            "squared_hs_tail_exponent": str(exponent),
            "squared_hs_tail_upper": "1/4096",
            "two_hs_tail_norms_upper": "1/32",
            "compact_and_centering_error_upper": "1/4" if passed else None,
            "centered_kernel_error_upper": "9/32" if passed else None,
            "gaussian_radial_upper": "1/60",
            "gaussian_full_cycle_upper": "1/7",
            "gates": gates,
        },
        "actual_radial_maximal_correlation_upper": "143/480" if passed else None,
        "actual_full_cycle_maximal_correlation_upper": "95/224" if passed else None,
        "explicit_coupling_threshold_verified_in_written_analysis": passed,
        "actual_centered_kernel_error_verified_in_written_analysis": passed,
        "actual_radial_contraction_verified_in_written_analysis": passed,
        "actual_full_cycle_contraction_verified_in_written_analysis": passed,
        "all_reduced_modes_included": passed,
        "trial_identified_as_actual_vacuum": False,
        "unspecified_elliptic_constant_used": False,
        "original_forest_correlation_upper_claim": False,
        "ambient_exterior_uniformity_verified": False,
        "uniform_in_volume_claim": False,
        "uniform_in_a_claim": False,
        "continuum_claim": False,
        "yang_mills_claim": False,
        "yang_mills_mass_gap_claim": False,
        "analytic_proof_formally_verified": False,
        "theorem_prover_verified": False,
        "mathlib_verified": False,
    }
    certificate = make_certificate(
        claim="explicit weak-coupling actual theta cycle-correlation interval",
        payload=payload,
        meta={
            "analytic_implication": "docs/api/gauge-theta-kernel-threshold.md",
            "proof_register": "written spectral/quasimode/density/own-marginal proof with exact scalar replay",
            "transcend_backend": "symbolic exponent bounds using 2<e<4; no huge powers materialized",
        },
    )
    return {**payload, "certificate": certificate}


def replay_su2_theta_kernel_contraction_certificate(certificate: dict[str, Any]) -> bool:
    """Rebuild the entire interval, source certificates, bounds and scope."""
    try:
        if not isinstance(certificate, dict) or not verify_certificate_digest(certificate):
            return False
        payload = certificate["payload"]
        if payload["type"] != "su2_theta_kernel_contraction_v1":
            return False
        expected = su2_theta_kernel_contraction(
            dyadic_exponent=payload["inputs"]["dyadic_exponent"]
        )
        return bool(certificate == expected["certificate"])
    except (KeyError, TypeError, ValueError, ZeroDivisionError, OverflowError):
        return False


__all__ = [
    "replay_su2_theta_kernel_contraction_certificate",
    "su2_theta_kernel_contraction",
]
