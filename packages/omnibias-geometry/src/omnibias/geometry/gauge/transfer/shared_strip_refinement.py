# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Actual-vacuum refinement on a four-plaquette, thirteen-edge SU(2) strip.

The two coarse rectangles share an electric edge. Their actual joint marginal
is retained. Exact rational gates bound the whole physical complement, its
relative coupling, and joint Fourier density/log norms without a spin cutoff.
"""

from __future__ import annotations

from fractions import Fraction as Q
from typing import Any

from omnibias.core.proof.certificate import make_certificate, verify_certificate_digest
from omnibias.geometry.gauge.transfer.invariant_vacuum_fourier import (
    invariant_vacuum_fourier_bounds,
)
from omnibias.geometry.gauge.transfer.static_sources import _integer, _rational
from omnibias.geometry.gauge.transfer.theta_vacuum_refinement import _sqrt_upper

# Bottom vertices 0..4, top vertices 5..9. Edge order B1..B4,T1..T4,V0..V4.
_EDGES = (
    (0, 1), (1, 2), (2, 3), (3, 4),
    (5, 6), (6, 7), (7, 8), (8, 9),
    (0, 5), (1, 6), (2, 7), (3, 8), (4, 9),
)
_PLAQUETTES = ((1, 10, -5, -9), (2, 11, -6, -10),
               (3, 12, -7, -11), (4, 13, -8, -12))


def su2_shared_strip_refinement(
    kappa: int | Q,
    *,
    correction_radius: int | Q,
    exponent_steps: int = 8,
    sqrt_bits: int = 64,
) -> dict[str, Any]:
    """Seal actual shared-strip embedding, conditional estimates and norm bounds.

The physical-gap status and joint-log-ball status are separate sufficient
gates. Neither uses the source's earlier gap or factorization conclusions.
The induced coarse density is jointly conjugation invariant; independence of
the two rectangles and a closed Wilson dictionary are not assumed.
"""
    coupling = _rational(kappa, "kappa")
    radius = _rational(correction_radius, "correction_radius")
    steps = _integer(exponent_steps, "exponent_steps")
    bits = _integer(sqrt_bits, "sqrt_bits")
    if coupling <= 0 or radius <= 0 or steps < 1 or bits < 0:
        raise ValueError("require positive kappa, radius, exponent_steps and nonnegative sqrt_bits")
    source = invariant_vacuum_fourier_bounds(
        10, _EDGES, group="su2", kappa=coupling, correction_radius=radius,
        plaquettes=_PLAQUETTES, weighted_incidence_cap=2,
    )
    source_witness = source["witness"]
    source_arithmetic = source_witness["arithmetic"]
    fixed_point = bool(source_witness["structural_caps_verified"]
                       and source_arithmetic["fixed_point_verified"])
    alpha, g = coupling / 2, 4 / coupling**2
    marginal_linear = 16 * (g + radius) / 3
    marginal_quadratic = 16 * g**2 / 9 + 26 * radius / 3
    marginal_oscillation = min(marginal_linear, marginal_quadratic)
    fiber_oscillation = (32 * g + 16 * radius) / 3
    exponential_domain = max(marginal_oscillation, fiber_oscillation) < steps
    conditional = fixed_point and exponential_domain
    mixed_hessian = 7 * g / 6 + 4 * radius / 3
    marginal_exp: Q | None = None
    fiber_exp: Q | None = None
    fiber_poincare: Q | None = None
    compression: Q | None = None
    fiber: Q | None = None
    covariance: Q | None = None
    relative_square: Q | None = None
    discriminant: Q | None = None
    sqrt_bound: Q | None = None
    candidate_gap: Q | None = None
    gap: Q | None = None
    if conditional:
        marginal_exp = (1 - marginal_oscillation / steps)**steps
        fiber_exp = (1 - fiber_oscillation / steps)**steps
        fiber_poincare = Q(3, 4) * fiber_exp
        compression = alpha * Q(9, 2) * marginal_exp
        fiber = alpha * Q(11, 5) * fiber_poincare
        covariance = mixed_hessian**2 / fiber_poincare
        relative_square = 4 * alpha * covariance / 5
        discriminant = (compression - fiber)**2 + 4 * compression * relative_square
        sqrt_bound = _sqrt_upper(discriminant, bits)
        candidate_gap = (compression + fiber - sqrt_bound) / 2
        if relative_square < fiber and candidate_gap > 0:
            gap = candidate_gap
    passed = gap is not None

    # A_s uses (1+sum doubled spins)**s times the coefficient trace norm.
    # The zero-Haar-mean log representative gives the true normalizer Z>=1.
    source_a0 = 16 * g / 3 + 13 * radius / 3
    source_a2 = 104 * g / 3 + 24 * radius
    joint_exponential_domain = source_a0 < steps
    joint_bounds = fixed_point and joint_exponential_domain
    positive_exp: Q | None = None
    density_a0: Q | None = None
    density_a2: Q | None = None
    own_marginals_a2: Q | None = None
    log_a0: Q | None = None
    log_a2: Q | None = None
    if joint_bounds:
        positive_exp = (1 - source_a0 / steps)**(-steps)
        density_a0 = positive_exp - 1
        density_a2 = source_a2 * (1 + source_a0) * positive_exp
        # For h=rho-1=h_L+h_R+h_mixed the A2 norm adds by disjoint
        # Fourier labels. With x+y+z<=B, z+x*y<=max(B,B**2/4).
        own_marginals_a2 = max(density_a2, density_a2**2 / 4)
        if density_a0 < 1:
            # -log(1-b)<=b/(1-b); no transcendental evaluation is needed.
            log_a0 = density_a0 / (1 - density_a0)
            log_a2 = density_a2 / (1 - density_a0)**2
    log_ball = log_a0 is not None

    failures = []
    if not fixed_point:
        failures.append("actual_vacuum_fixed_point")
    if not exponential_domain:
        failures.append("strict_rational_exponential_domain")
    if conditional and relative_square is not None and fiber is not None:
        if relative_square >= fiber:
            failures.append("relative_form_coercivity")
        elif gap is None:
            failures.append("positive_gap_at_requested_sqrt_precision")
    joint_failures = []
    if not fixed_point:
        joint_failures.append("actual_vacuum_fixed_point")
    if not joint_exponential_domain:
        joint_failures.append("joint_positive_exponential_domain")
    if joint_bounds and not log_ball:
        joint_failures.append("joint_A0_log_ball")

    def encode(value: Q | None) -> str | None:
        return str(value) if value is not None else None

    witness = {
        "inputs": {"kappa": str(coupling), "correction_radius": str(radius),
                   "exponent_steps": steps, "sqrt_bits": bits},
        "graph": {"n_vertices": 10, "edges": [list(edge) for edge in _EDGES],
                  "plaquettes": [list(loop) for loop in _PLAQUETTES],
                  "edge_order": "B1..B4,T1..T4,V0..V4; bottom vertices 0..4, top vertices 5..9",
                  "shared_cell_boundary_edge": 11,
                  "eliminated_internal_edges": [10, 12],
                  "gauge_tree_edges": [1, 2, 3, 4, 5, 6, 7, 8, 11]},
        "normalization": "aH=kappa/2*sum_13_edges(C_e)+2/kappa*(8-sum_4_plaquettes(chi_p))",
        "energy_units": "dimensionless aH",
        "gap_scope": "entire gauge-invariant physical thirteen-edge Hilbert space above the actual fine vacuum",
        "gauss_constraint": "at every vertex",
        "coarse_paths": {"X": [-2, -1, 9, 5, 6], "Y": [3, 4, 13, -8, -7], "Z": [11]},
        "coarse_holonomies": {"W_L": [-2, -1, 9, 5, 6, -11],
                              "W_R": [3, 4, 13, -8, -7, -11]},
        "physical_fiber_holonomies": {"R_L": [-2, 10, 6, -11],
                                      "R_R": [3, 12, -7, -11]},
        "physical_coordinate_scope": "four loop matrices modulo simultaneous conjugation; coarse pair need not be separately central or independent",
        "actual_vacuum": "psi_f=normalized exp((g/3)*sum_4_plaquettes(chi_p)+U), original-edge spin norm N(U)<=r",
        "actual_joint_marginal": "nu=p_*(psi_f^2 dHaar), phi=sqrt(dnu/dHaar_pair)",
        "haar_embedding": "J(Phi)=psi_f*Phi(W_L,W_R)/phi(W_L,W_R)",
        "embedding_scope": "analytic isometry through the constructed actual vacuum and joint marginal; no polynomial evaluator or independence replacement",
        "embedding_maps_actual_vacua": "J phi = psi_f",
        "coarse_electric_metric": "5|p_L|^2+5|p_R|^2+|p_L+p_R|^2; separate left/right frames retained before fiber minimization",
        "source_certificate": source["certificate"],
        "source_use": "structural membership and actual-vacuum fixed point only; source gap and factorization conclusions are not inputs",
        "arithmetic": {
            "g": str(g), "electric_alpha": str(alpha),
            "fixed_point_verified": fixed_point,
            "fixed_point_self_map_slack": source_arithmetic["self_map_slack"],
            "fixed_point_contraction_upper": source_arithmetic["contraction_upper"],
            "marginal_linear_log_oscillation_candidate": str(marginal_linear),
            "marginal_quadratic_log_oscillation_candidate": str(marginal_quadratic),
            "marginal_log_oscillation_upper": str(marginal_oscillation) if fixed_point else None,
            "marginal_bound_method": "quadratic_Haar_cancellation" if marginal_quadratic <= marginal_linear else "two_outer_edge_oscillation",
            "fiber_log_oscillation_upper": str(fiber_oscillation) if fixed_point else None,
            "exponential_domain_verified": exponential_domain,
            "marginal_exp_negative_lower": encode(marginal_exp),
            "fiber_exp_negative_lower": encode(fiber_exp),
            "conditional_pair_poincare_lower": encode(fiber_poincare),
            "coarse_path_kinetic_coefficients": ["5", "5", "1"],
            "coarse_product_gradient_lower_coefficient": "5",
            "physical_vertical_kinetic_coefficient": "11/5",
            "coarse_Haar_physical_gap": "9/2",
            "compression_gap_lower": encode(compression),
            "all_physical_fiber_modes_gap_lower": encode(fiber),
            "conditional_drift_mixed_hessian_upper": str(mixed_hessian) if fixed_point else None,
            "conditional_drift_covariance_upper": encode(covariance),
            "relative_cross_form_beta_squared_upper": encode(relative_square),
            "schur_discriminant": encode(discriminant),
            "schur_sqrt_upper": encode(sqrt_bound),
            "gap_candidate_lower": encode(candidate_gap),
            "physical_gap_lower": encode(gap),
        },
        "joint_fourier": {
            "norm": "A_s=sum_labels (1+sum_i(two_j_i))^s * nuclear_norm(coefficient_matrix)",
            "density_scope": "actual normalized joint coarse density, not separately central character coefficients",
            "normalizer_lower": "1" if fixed_point else None,
            "twice_log_vacuum_A0_upper": str(source_a0) if fixed_point else None,
            "twice_log_vacuum_A2_upper": str(source_a2) if fixed_point else None,
            "positive_exponential_domain_verified": joint_exponential_domain,
            "exp_positive_upper": encode(positive_exp),
            "density_minus_one_A0_upper": encode(density_a0),
            "density_minus_one_A2_upper": encode(density_a2),
            "actual_joint_to_own_marginals_A2_upper": encode(own_marginals_a2),
            "actual_joint_to_own_marginals_enclosed": joint_bounds,
            "own_marginals_scope": "density rho-(integral_R rho)*(integral_L rho) of this actual coarse pair; not a log interaction or potential norm",
            "joint_log_density_A0_upper": encode(log_a0),
            "joint_log_density_A2_upper": encode(log_a2),
            "joint_log_vacuum_A2_upper": encode(log_a2 / 2 if log_a2 is not None else None),
            "actual_joint_density_bounds_verified": joint_bounds,
            "actual_joint_log_ball_verified": log_ball,
            "connected_interaction_A2_bound_verified": False,
            "connected_interaction_A2_upper": None,
            "failed_constraints": joint_failures,
        },
        "failed_constraints": failures,
        "actual_vacuum_embedding_verified": fixed_point,
        "actual_conditional_estimates_verified": conditional,
        "physical_gap_verified": passed,
        "actual_joint_density_bounds_verified": joint_bounds,
        "actual_joint_log_ball_verified": log_ball,
        "actual_joint_to_own_marginals_enclosed": joint_bounds,
        "coarse_model_identification": "actual joint marginal induces the coarse model; no equality to an independent product or prescribed Wilson family",
    }
    certificate = make_certificate(
        claim="actual shared-strip vacuum refinement with conditional physical gap and joint Fourier density/log bounds when the respective gates pass",
        payload={"type": "su2_shared_strip_refinement_v1", "witness": witness},
        honesty={
            "actual_vacuum_embedding_verified": fixed_point,
            "actual_conditional_estimates_verified": conditional,
            "physical_finite_graph_gap_verified": passed,
            "actual_joint_density_bounds_verified": joint_bounds,
            "actual_joint_log_ball_verified": log_ball,
            "actual_joint_to_own_marginals_enclosed": joint_bounds,
            "independent_cell_replacement_verified": False,
            "all_scale_refinement_claim": False,
            "continuum_claim": False, "yang_mills_mass_gap_claim": False,
        },
        meta={"analytic_implication": "docs/api/gauge-shared-strip-refinement.md",
              "transcend_backend": "not_used"},
    )
    return {
        "status": "PASS" if passed else "INCONCLUSIVE",
        "finite_gate_verified": passed,
        "actual_vacuum_embedding_verified": fixed_point,
        "actual_conditional_estimates_verified": conditional,
        "physical_finite_graph_gap_verified": passed,
        "actual_joint_density_bounds_verified": joint_bounds,
        "actual_joint_log_ball_verified": log_ball,
        "actual_joint_to_own_marginals_enclosed": joint_bounds,
        "actual_joint_to_own_marginals_A2_upper": encode(own_marginals_a2),
        "physical_gap_lower": encode(gap),
        "witness": witness, "certificate": certificate,
        "digest_verified": verify_certificate_digest(certificate),
        "independent_cell_replacement_verified": False,
        "connected_interaction_A2_bound_verified": False,
        "all_scale_refinement_claim": False, "coarse_wilson_family_closed": False,
        "infinite_volume_claim": False, "uniform_in_a_claim": False,
        "continuum_claim": False, "yang_mills_claim": False,
        "yang_mills_mass_gap_claim": False,
        "theorem_prover_verified": False, "mathlib_verified": False,
    }


def replay_su2_shared_strip_refinement_certificate(certificate: dict[str, Any]) -> bool:
    """Replay either success or refusal by full canonical recomputation."""
    if not isinstance(certificate, dict):
        return False
    try:
        if not verify_certificate_digest(certificate):
            return False
        payload = certificate["payload"]
        if payload["type"] != "su2_shared_strip_refinement_v1":
            return False
        inputs = payload["witness"]["inputs"]
        result = su2_shared_strip_refinement(
            Q(inputs["kappa"]), correction_radius=Q(inputs["correction_radius"]),
            exponent_steps=inputs["exponent_steps"], sqrt_bits=inputs["sqrt_bits"],
        )
        return bool(result["certificate"] == certificate)
    except (KeyError, TypeError, ValueError, ZeroDivisionError, OverflowError, IndexError):
        return False


__all__ = [
    "replay_su2_shared_strip_refinement_certificate",
    "su2_shared_strip_refinement",
]
