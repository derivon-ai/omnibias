# SPDX-License-Identifier: Apache-2.0
"""Quantitative open-box lower density for the actual periodic SU(2) corner tilt.

The exact budget implements the original-link, comb-tree and IMS proof in
docs/api/gauge-periodic-corner-box.md. It is a written analytic
theorem with canonical rational replay, not a formal proof or a mass-gap claim.
"""

from __future__ import annotations

from fractions import Fraction as Q
from typing import Any

from omnibias.core.proof.certificate import make_certificate, verify_certificate_digest


def _integer(value: object, name: str, lower: int, upper: int) -> int:
    if type(value) is not int:
        raise TypeError(f"{name} must be an exact integer")
    if not lower <= value <= upper:
        raise ValueError(f"{name} must lie in [{lower}, {upper}]")
    return value


def _ceil_log2(value: int) -> int:
    """Least j with value <= 2**j, for a positive exact integer."""
    if type(value) is not int or value <= 0:
        raise ValueError("a positive integer is required")
    return (value - 1).bit_length()


def su2_periodic_corner_box_lower(
    box_side: int = 8192, *, kappa_exponent: int | None = None
) -> dict[str, Any]:
    """Certify a periodic lower density on 0<kappa<=2**(-p), N>=box_side.

    The result is uniform in the translation-averaged tilt t in [0,1/4].
    Open-box vertex sides range from box_side to 2*box_side-1. Every internal
    original link and face is retained, with the proved crossing-cell face
    coefficient 2-t. No nonlinear remainder or reference inverse is an input.

    Input limits bound representation cost, not the periodic volume N.
    Exponents are positive multiples of three so the upper endpoint for
    tau=kappa**(1/3) is itself rational. Failed budgets are inconclusive.
    """
    ell = _integer(box_side, "box_side", 3, 2**20)
    volume_min = ell**3
    volume_max = 8 * ell**3
    edge_max = 24 * ell**3
    filling = 96 * ell**4
    metric_constant = 32 * edge_max**2
    potential_constant = 2048 * filling
    radius_bits = _ceil_log2(16384 * (metric_constant + potential_constant))
    radius = Q(1, 2**radius_bits)
    action_cutoff = radius**2 / filling
    raw_exponent = 2 * radius_bits + _ceil_log2(8192 * filling * volume_max)
    default_exponent = 3 * ((raw_exponent + 2) // 3)
    exponent = (
        default_exponent
        if kappa_exponent is None
        else _integer(kappa_exponent, "kappa_exponent", 3, 9999)
    )
    if exponent % 3:
        raise ValueError("kappa_exponent must be a multiple of three")
    kappa = Q(1, 2**exponent)
    tau = Q(1, 2 ** (exponent // 3))
    metric_loss = metric_constant * radius
    potential_loss = potential_constant * radius
    relative_loss = max(metric_loss, potential_loss)
    density_ratio = 1 + 4 * radius**2
    good_factor = (1 - relative_loss) / density_ratio
    bad_floor = action_cutoff / (4 * kappa)
    harmonic_box_cap = 18 * volume_max
    ims_error = 20 * kappa / action_cutoff
    local_error = 18 * (1 - good_factor) + ims_error / volume_min
    surface_error = Q(54, ell)
    total_error = local_error + surface_error
    gates = {
        "six_letter_chart_radius": radius <= Q(1, 32),
        "positive_original_metric_comparison": metric_loss < 1,
        "positive_original_potential_comparison": potential_loss < 1,
        "bad_branch_above_every_box_harmonic_energy": bad_floor >= harmonic_box_cap,
        "nonlinear_density_error_at_most_1_128": local_error <= Q(1, 128),
        "surface_density_error_at_most_1_128": surface_error <= Q(1, 128),
        "total_density_error_at_most_1_64": total_error <= Q(1, 64),
    }
    passed = all(gates.values())
    arithmetic = {
        "minimum_periodic_side": ell,
        "box_vertex_side_interval": [ell, 2 * ell - 1],
        "tilt_interval": ["0", "1/4"],
        "box_vertex_count_lower": volume_min,
        "box_vertex_count_upper": volume_max,
        "box_edge_count_upper": edge_max,
        "individual_chord_filling_faces_upper": 4 * ell,
        "total_chord_action_to_face_action_constant": filling,
        "metric_relative_lipschitz_constant": metric_constant,
        "potential_relative_lipschitz_constant": potential_constant,
        "six_letter_action_remainder_constant": 192,
        "total_potential_remainder_constant": 2048,
        "chart_radius_bits": radius_bits,
        "chart_radius": str(radius),
        "action_cutoff": str(action_cutoff),
        "raw_default_kappa_exponent": raw_exponent,
        "default_kappa_exponent": default_exponent,
        "kappa_exponent": exponent,
        "kappa_upper": str(kappa),
        "tau_upper": str(tau),
        "metric_relative_loss_upper": str(metric_loss),
        "potential_relative_loss_upper": str(potential_loss),
        "haar_density_ratio_upper": str(density_ratio),
        "good_harmonic_factor_lower": str(good_factor),
        "bad_localized_energy_lower": str(bad_floor),
        "all_box_harmonic_energy_upper": harmonic_box_cap,
        "ims_error_per_box_upper": str(ims_error),
        "nonlinear_density_error_upper": str(local_error),
        "surface_density_error_upper": str(surface_error),
        "total_density_error_upper": str(total_error),
        "claimed_density_error_upper": "1/64" if passed else None,
        "quadratic_operator_norm_upper": "51/4",
        "quadratic_rank_loss_per_crossing_cell_upper": 3,
        "crossing_cell_density_upper": str(Q(3, ell)),
        "gates": gates,
    }
    scope = {
        "actual_nonlinear_lower_verified_in_written_analysis": passed,
        "uniform_periodic_volume_lower_density_verified": passed,
        "uniform_on_declared_kappa_and_tilt_intervals_verified": passed,
        "all_spin_original_link_energy_comparison_verified": passed,
        "analytic_proof_formally_verified": False,
        "arbitrary_exterior_conditional_verified": False,
        "physical_gap_claim": False,
        "infinite_volume_reconstruction_claim": False,
        "all_scale_refinement_claim": False,
        "continuum_claim": False,
        "yang_mills_mass_gap_claim": False,
        "theorem_prover_verified": False,
        "mathlib_verified": False,
    }
    payload = {
        "type": "su2_periodic_corner_box_lower_v1",
        "inputs": {"box_side": ell, "kappa_exponent": kappa_exponent},
        "status": "PASS" if passed else "INCONCLUSIVE",
        "arithmetic": arithmetic,
        **scope,
    }
    certificate = make_certificate(
        claim="actual periodic SU2 corner-tilt lower energy density on the declared finite-volume family",
        payload=payload,
        meta={
            "transcend_backend": "not_used",
            "analytic_implication": "docs/api/gauge-periodic-corner-box.md",
            "normalization": "H_t=kappa/2 sum_original_edges C_e + sum_x[(2+t/2)S_x-(t/2)A(D_x)]/kappa; C_fund=3/4",
            "boundary_rule": "retain complete tilted corner cells and every internal face of crossing cells with coefficient 2-t; discard crossing links",
            "reference": "identity-background periodic quadratic trace; zero frequencies contribute zero",
            "no_toron_minimum_premise": True,
            "proof_register": "written original-link comb-tree and IMS theorem with exact rational replay",
        },
    )
    return {**payload, "certificate": certificate}


def replay_su2_periodic_corner_box_lower_certificate(certificate: dict[str, Any]) -> bool:
    """Replay both outcomes; an inconclusive replay never earns the theorem."""
    try:
        if not isinstance(certificate, dict) or not verify_certificate_digest(certificate):
            return False
        payload = certificate["payload"]
        if payload["type"] != "su2_periodic_corner_box_lower_v1":
            return False
        inputs = payload["inputs"]
        expected = su2_periodic_corner_box_lower(
            inputs["box_side"], kappa_exponent=inputs["kappa_exponent"]
        )
        return bool(certificate == expected["certificate"])
    except (KeyError, TypeError, ValueError, OverflowError, ZeroDivisionError):
        return False


__all__ = [
    "replay_su2_periodic_corner_box_lower_certificate",
    "su2_periodic_corner_box_lower",
]
