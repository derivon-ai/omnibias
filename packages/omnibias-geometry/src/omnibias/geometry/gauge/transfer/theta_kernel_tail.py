# SPDX-License-Identifier: Apache-2.0
"""Actual weak-theta normalized-kernel tails, with symbolic exponential bounds.

The original-link Bochner estimate, global barriers, and marginal denominators
are proved in docs/api/gauge-theta-kernel-tail.md. No correlation upper bound
or Gaussian approximation is inferred from the tail theorem alone.
"""

from __future__ import annotations

from fractions import Fraction as Q
from typing import Any

from omnibias.core.proof.certificate import make_certificate, verify_certificate_digest
from omnibias.geometry.gauge.transfer.static_sources import _rational

_KAPPA_MAX = Q(1, 64)
_TAIL_PREFACTOR = Q(26_000_000)


def _theta_energy_source(coupling: Q) -> tuple[dict[str, Any], bool, Q | None]:
    from omnibias.geometry.gauge.transfer.theta_weak_blocks import (
        replay_su2_theta_weak_block_certificate,
        su2_theta_weak_block_gaps,
    )

    row = su2_theta_weak_block_gaps(coupling)
    certificate = row.get("certificate", {})
    if not isinstance(certificate, dict):
        return {}, False, None
    replayed = replay_su2_theta_weak_block_certificate(certificate)
    try:
        payload = certificate["payload"]
        ground = Q(payload["arithmetic"]["ground_energy_upper"])
        valid = bool(
            replayed
            and payload["type"] == "su2_theta_weak_block_gaps_v1"
            and Q(payload["inputs"]["kappa"]) == coupling
            and payload["actual_theta_vacuum_identified_in_written_analysis"] is True
            and 0 <= ground <= 6
        )
    except (KeyError, TypeError, ValueError, ZeroDivisionError):
        ground, valid = None, False
    return certificate, valid, ground


def _ceil_log_two(value: Q) -> int:
    """Least integer n with 2**n >= value, without expanding a large power."""
    numerator, denominator = value.numerator, value.denominator
    estimate = numerator.bit_length() - denominator.bit_length()
    sufficient = (
        denominator << estimate >= numerator
        if estimate >= 0
        else denominator >= numerator << (-estimate)
    )
    return estimate if sufficient else estimate + 1


def _dyadic_exponent(exponent: Q) -> int:
    """exp(exponent) <= 2**(-result), using 2 < e < 4."""
    if exponent <= 0:
        negative = -exponent
        return negative.numerator // negative.denominator
    ceiling = -(-exponent.numerator // exponent.denominator)
    return -2 * ceiling


def su2_theta_normalized_kernel_tail(
    kappa: int | Q,
    radius: int | Q,
    *,
    target: int | Q | None = None,
) -> dict[str, Any]:
    """Bound the actual squared Hilbert--Schmidt tail on F0 >= radius*kappa.

    Here F0=8*(2-cos(theta_x/2)-cos(theta_y/2)), and the normalized
    joint kernel is rho/sqrt(rho_x*rho_y), relative to product Haar.
    For 0<kappa<=1/64 its tail integral is at most
    26_000_000*exp(81640-2*radius/9). This is a full reduced SU2^2 theorem.

    An optional positive rational target is compared to an outward factored
    dyadic upper bound. Its gate is distinct from the analytic theorem's PASS.
    Couplings outside the proved interval return INCONCLUSIVE. No floating
    exponential or enormous power of two is materialized.
    """
    coupling, cutoff = _rational(kappa, "kappa"), _rational(radius, "radius")
    requested = None if target is None else _rational(target, "target")
    if coupling <= 0 or cutoff <= 0 or (requested is not None and requested <= 0):
        raise ValueError("kappa, radius and any target must be strictly positive")
    source, source_valid, ground = _theta_energy_source(coupling)
    exponent = Q(81_640) - 2 * cutoff / 9
    dyadic = _dyadic_exponent(exponent)
    empty = cutoff * coupling >= 16
    gates = {
        "canonical_actual_theta_ground_energy_source": source_valid,
        "coupling_in_proved_interval": coupling <= _KAPPA_MAX,
        "ground_energy_at_most_six": ground is not None and ground <= 6,
        "bochner_contradiction": Q(64, 21) > Q(649, 2048),
        "core_path_log_ratio_below_10000": Q(8192 * 22, 21) < 10_000,
        "supersolution_outside_core": Q(2, 5) * 256 - Q(414, 5) > 0,
        "subsolution_outside_core": Q(36, 5) - Q(4, 25) * 256 < 0,
        "smoothed_action_ratio": Q(1) / (1 + Q(2, 16)) == Q(8, 9),
        "normalized_kernel_decay": 4 * Q(4, 5) * Q(8, 9) - 2 * Q(6, 5) == Q(4, 9),
        "marginal_haar_integral_lower": Q(175, 32076) > Q(1, 200),
        "tail_haar_prefactor": 40_000 * Q(9801, 392) ** 2 < _TAIL_PREFACTOR,
    }
    passed = all(gates.values())
    threshold = None if requested is None else _ceil_log_two(_TAIL_PREFACTOR / requested)
    target_passed = bool(passed and threshold is not None and (empty or dyadic >= threshold))
    symbolic = {
        "prefactor": str(_TAIL_PREFACTOR),
        "exponent": str(exponent),
        "meaning": "prefactor * exp(exponent); squared Hilbert-Schmidt tail",
    }
    arithmetic = {
        "kappa": str(coupling),
        "proved_kappa_upper": str(_KAPPA_MAX),
        "radius": str(cutoff),
        "unscaled_tail_threshold": str(cutoff * coupling),
        "maximum_F0": "16",
        "ground_energy_upper": str(ground) if ground is not None else None,
        "cutoff_support_scale": "1024",
        "core_action_scale": "256",
        "bochner_scaled_gradient_cutoff": "16384",
        "core_gradient_coefficient": "512/3",
        "core_log_ratio_to_identity_upper": "10000",
        "core_log_oscillation_upper": "20000",
        "barrier_regularization": "1/16",
        "upper_barrier_action_coefficient": "4/5",
        "lower_barrier_action_coefficient": "6/5",
        "lower_barrier_log_prefactor": "-10000",
        "upper_barrier_log_prefactor": "10410",
        "barrier_ratio_log_upper": "20410",
        "normalized_kernel_decay_coefficient": "4/9",
        "squared_hs_tail_prefactor": str(_TAIL_PREFACTOR),
        "squared_hs_tail_exponent": str(exponent),
        "dyadic_upper_negative_integer_exponent": dyadic,
        "target_required_dyadic_exponent": threshold,
        "gates": gates,
    }
    payload = {
        "type": "su2_theta_normalized_kernel_tail_v1",
        "status": "PASS" if passed else "INCONCLUSIVE",
        "inputs": {
            "kappa": str(coupling),
            "radius": str(cutoff),
            "target": str(requested) if requested is not None else None,
        },
        "theta_ground_energy_source_certificate": source,
        "model": "seven original unit-weight links, two adjacent square plaquettes",
        "normalization": "H=kappa*Ctheta/2+2*(A(x)+A(y))/kappa; Ctheta=3*(Cx+Cy)+Cdiag, C_fund=3/4",
        "vacuum": "actual unique positive ground state on reduced L2(SU2^2), with its physical seven-link lift",
        "kernel": "k(x,y)=rho(x,y)/sqrt(rho_x(x)*rho_y(y)), rho=Phi^2 with normalized Haar marginals",
        "tail_region": "F0(x,y)>=radius*kappa; F0=8*(2-cos(theta_x/2)-cos(theta_y/2)), theta in[0,pi]",
        "tail_measure": "product normalized Haar; the integral of |k|^2, not its square root",
        "squared_hs_tail_upper": symbolic if passed else None,
        "effective_squared_hs_tail_upper": ({"exact_rational": "0"} if empty else symbolic)
        if passed
        else None,
        "factored_dyadic_squared_hs_tail_upper": {
            "prefactor": str(_TAIL_PREFACTOR),
            "base": 2,
            "negative_integer_exponent": dyadic,
            "meaning": "prefactor * base**(-negative_integer_exponent); never materialized",
        }
        if passed
        else None,
        "tail_region_has_zero_haar_measure": empty,
        "tail_target_status": "NOT_REQUESTED"
        if requested is None
        else "PASS"
        if target_passed
        else "INCONCLUSIVE",
        "tail_target_verified": target_passed,
        "arithmetic": arithmetic,
        "actual_positive_vacuum_identified_in_written_analysis": source_valid,
        "actual_normalized_kernel_tail_verified_in_written_analysis": passed,
        "uniform_rescaled_hs_tail_verified_in_written_analysis": passed,
        "radial_marginal_kernel_tail_verified_in_written_analysis": passed,
        "all_reduced_modes_included": passed,
        "source_gap_used_as_premise": False,
        "gaussian_kernel_comparison_verified": False,
        "maximal_correlation_upper_verified": False,
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
    certificate = make_certificate(
        claim="actual weak-theta normalized joint-kernel squared-Hilbert-Schmidt tail",
        payload=payload,
        meta={
            "analytic_implication": "docs/api/gauge-theta-kernel-tail.md",
            "proof_register": "written original-link Bochner/barrier/marginal-denominator proof; exact rational replay",
            "transcend_backend": "symbolic exponential and outward factored dyadic bound",
        },
    )
    return {**payload, "certificate": certificate}


def replay_su2_theta_kernel_tail_certificate(certificate: dict[str, Any]) -> bool:
    """Canonically rebuild source, gates, region, positive representation and scope."""
    try:
        if not isinstance(certificate, dict) or not verify_certificate_digest(certificate):
            return False
        payload = certificate["payload"]
        if payload["type"] != "su2_theta_normalized_kernel_tail_v1":
            return False
        inputs = payload["inputs"]
        expected = su2_theta_normalized_kernel_tail(
            Q(inputs["kappa"]),
            Q(inputs["radius"]),
            target=None if inputs["target"] is None else Q(inputs["target"]),
        )
        return bool(certificate == expected["certificate"])
    except (KeyError, TypeError, ValueError, ZeroDivisionError, OverflowError):
        return False


__all__ = [
    "replay_su2_theta_kernel_tail_certificate",
    "su2_theta_normalized_kernel_tail",
]
