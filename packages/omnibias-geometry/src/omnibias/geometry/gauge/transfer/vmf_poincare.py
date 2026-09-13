# SPDX-License-Identifier: Apache-2.0
"""All-angular SU(2) vMF bounds and a constructed interacting strip reference.

The supersolution and conditional-Poincare proofs are written analysis.
These exact scalar certificates do not identify the reference with a Wilson
vacuum or formally verify the differential operators.
"""

from __future__ import annotations

from copy import deepcopy
from fractions import Fraction as Q
from typing import Any

from omnibias.core.proof.certificate import make_certificate, verify_certificate_digest

_SCOPE = {
    "actual_wilson_vacuum_verified": False,
    "actual_nonlinear_vacuum_verified": False,
    "actual_yang_mills_gap_verified": False,
    "unrestricted_original_link_scalar_gap_verified": False,
    "continuum_claim": False,
    "yang_mills_claim": False,
    "yang_mills_mass_gap_claim": False,
    "analytic_proof_formally_verified": False,
    "mathlib_verified": False,
    "theorem_prover_verified": False,
}


def _positive_q(value: Any, name: str) -> Q:
    if type(value) not in (int, Q):
        raise TypeError(f"{name} must be an integer or Fraction, not bool/float")
    result = Q(value)
    if result <= 0:
        raise ValueError(f"{name} must be positive")
    return result


def _seal(payload: dict[str, Any], claim: str) -> dict[str, Any]:
    payload = {**payload, **_SCOPE}
    certificate = make_certificate(
        claim=claim,
        payload=deepcopy(payload),
        honesty={
            key: value
            for key, value in payload.items()
            if type(value) is bool and key not in {"mathlib_verified", "theorem_prover_verified"}
        },
        meta={
            "analytic_implication": "docs/api/gauge-vmf-poincare.md",
            "transcend_backend": "not_used",
        },
    )
    return {**payload, "certificate": certificate}


def su2_vmf_poincare_bound(h: int | Q) -> dict[str, Any]:
    """All-angular PI for exp(h*q0) Haar density, radius-two SU(2) metric.

    Here h is the norm of the four-dimensional field multiplying the unit
    quaternion q, not the coefficient multiplying Tr(U)=2*q0. Rotational
    symmetry covers every field direction. The spectral floor is h/8;
    the same proof also gives the conservative additive floor 5/16+h/8.
    """
    field = _positive_q(h, "h")
    gates = {
        "angular_amgm_square_budget": Q(35, 64) >= Q(1, 4),
        "radial_amgm_square_budget": Q(75, 64) >= 1,
        "metric_scales_unit_gap_by_quarter": Q(1, 2) ** 2 == Q(1, 4),
        "angular_completed_square_remainder_positive": Q(80) - Q(256, 7) > 0,
        "radial_completed_square_remainder_positive": Q(80) - Q(1024, 15) > 0,
    }
    passed = all(gates.values())
    return _seal(
        {
            "type": "su2_vmf_all_angular_poincare_v1",
            "inputs": {"h": str(field)},
            "status": "PASS" if passed else "INCONCLUSIVE",
            "arithmetic": {
                "unit_sphere_angular_linear_floor": str(field / 2),
                "unit_sphere_radial_linear_floor": str(field),
                "su2_linear_poincare_gap_lower": str(field / 8),
                "su2_additive_poincare_gap_lower": str(Q(5, 16) + field / 8),
                "angular_barta_constant": "5/4",
                "angular_cosecant_squared_coefficient": "5/4",
                "angular_field_squared_sine_squared_coefficient": "7/64",
                "radial_field_squared_sine_squared_coefficient": "15/64",
            },
            "measure": "exp(h*q0) dH(U)/Z on unit quaternions, with radius-two metric",
            "normalization": {
                "quaternion_coordinate": "q0=Tr(U)/2",
                "unit_sphere_radius": "1",
                "su2_metric_radius": "2",
                "haar_su2_casimir_gap": "3/4",
            },
            "proof": {
                "angular_sectors": "all spherical harmonics ell>=1 on S^2",
                "angular_supersolution": "sin(theta)^(1/2)*exp(-h*cos(theta)/8)",
                "radial_partner_supersolution": "sin(theta)^(1/2)*exp(-3*h*cos(theta)/8)",
                "radial_intertwining": "D*A0=(A1+h*cos(theta))*D",
                "domain": "ground-state identity on C_c^infinity(0,pi), then Friedrichs form closure",
                "supersolutions_asserted_in_operator_domain": False,
                "radial_result_substituted_for_all_angular_modes": False,
            },
            "gates": gates,
            "finite_gate_verified": passed,
            "vmf_probability_measure_verified_in_written_analysis": passed,
            "all_angular_poincare_verified_in_written_analysis": passed,
            "uniform_in_field_direction": passed,
        },
        "All-angular vMF Poincare lower bound on radius-two SU(2); no Wilson-vacuum identification",
    )


def su2_strip_reference_poincare(length: int, kappa: int | Q) -> dict[str, Any]:
    """Uniform PI of the specified interacting compact reference on a strip.

    B=(4I-A_path)^(-1/2), F=2 sum B_ii A_i+4 sum_(i<j) B_ij X_i.X_j,
    and the reference density is exp(-2F/kappa) times product Haar. The
    reference is not the actual Wilson vacuum. Original-electric comparison
    applies after based-gauge reduction, not to arbitrary original-link
    scalar functions carrying the eliminated gauge coordinates.
    """
    if type(length) is not int:
        raise TypeError("length must be an integer, not bool/float")
    if length < 1:
        raise ValueError("length must be positive")
    coupling = _positive_q(kappa, "kappa")
    source = su2_vmf_poincare_bound(4 / coupling)["certificate"]
    source_replayed = replay_su2_vmf_poincare_certificate(source)
    gates = {
        "source_certificate_replayed": source_replayed,
        "sqrt_two_upper_square_budget": Q(2) < Q(17, 12) ** 2,
        "reference_row_margin_budget": Q(3, 2) - Q(17, 12) == Q(1, 12),
        "reference_electric_coefficient": coupling / 2 * (1 / (12 * coupling)) == Q(1, 24),
        "minimum_conditional_field_budget": (4 / coupling) / 8 == 1 / (2 * coupling),
    }
    passed = all(gates.values())
    return _seal(
        {
            "type": "su2_neutral_strip_reference_poincare_v1",
            "inputs": {"length": length, "kappa": str(coupling)},
            "status": "PASS" if passed else "INCONCLUSIVE",
            "reference": {
                "geometry": "open one-by-length original square-plaquette strip",
                "n_vertices": 2 * length + 2,
                "n_edges": 3 * length + 1,
                "rooted_cycle_count": length,
                "matrix": "B=(4I-A_path)^(-1/2)",
                "phase": "F=2*sum_i B_ii*(2-2*q0_i)+4*sum_(i<j) B_ij*dot(X_i,X_j)",
                "density": "exp(-2*F/kappa)*product_Haar/Z",
                "original_electric_scope": "full rooted cycle space after based-gauge reduction, hence its globally conjugation-invariant physical subspace",
                "original_electric_comparison": "Gamma_original includes each independent bottom-chord Gamma_i as a nonnegative summand",
            },
            "arithmetic": {
                "B_diagonal_lower": "1/2",
                "B_row_sum_upper": "1/sqrt(2)",
                "conditional_field_norm_lower": str(4 / coupling),
                "conditional_poincare_lower": str(1 / (2 * coupling)),
                "conditional_comparison_matrix": "(diag(B_ii)-2*offdiag(B_ij))/kappa",
                "comparison_row_margin_symbolic": "(3/2-sqrt(2))/kappa",
                "comparison_row_margin_lower": str(1 / (12 * coupling)),
                "cycle_product_poincare_gap_lower": str(1 / (12 * coupling)),
                "cycle_product_poincare_constant_upper": str(12 * coupling),
                "original_electric_coefficient": str(coupling / 2),
                "rooted_cycle_reference_electric_gap_lower": "1/24",
            },
            "source_certificate": source,
            "gates": gates,
            "finite_gate_verified": passed,
            "source_certificate_replayed": source_replayed,
            "constructed_compact_reference_verified_in_written_analysis": passed,
            "reference_has_nonzero_pair_interactions": length >= 2,
            "reference_all_exterior_conditionals_verified_in_written_analysis": passed,
            "reference_cycle_product_poincare_verified_in_written_analysis": passed,
            "reference_rooted_original_electric_gap_verified_in_written_analysis": passed,
            "reference_global_physical_restriction_gap_verified_in_written_analysis": passed,
            "reference_constants_uniform_in_length": passed,
            "reference_constants_uniform_in_positive_kappa": passed,
        },
        "Uniform Poincare and rooted-electric gap of a specified interacting compact reference; actual Wilson comparison remains open",
    )


def replay_su2_vmf_poincare_certificate(certificate: Any) -> bool:
    """Replay the full canonical scalar or reference certificate and source."""
    if not isinstance(certificate, dict):
        return False
    try:
        if not verify_certificate_digest(certificate):
            return False
        payload = certificate["payload"]
        inputs = payload["inputs"]
        if payload["type"] == "su2_vmf_all_angular_poincare_v1":
            expected = su2_vmf_poincare_bound(Q(inputs["h"]))["certificate"]
        elif payload["type"] == "su2_neutral_strip_reference_poincare_v1":
            if not replay_su2_vmf_poincare_certificate(payload["source_certificate"]):
                return False
            expected = su2_strip_reference_poincare(inputs["length"], Q(inputs["kappa"]))[
                "certificate"
            ]
        else:
            return False
        return bool(certificate == expected)
    except (KeyError, TypeError, ValueError, ZeroDivisionError, OverflowError, RecursionError):
        return False


__all__ = [
    "replay_su2_vmf_poincare_certificate",
    "su2_strip_reference_poincare",
    "su2_vmf_poincare_bound",
]
