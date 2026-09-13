# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Actual-vacuum conditional estimates on the seven-edge SU(2) theta graph.

The Fourier fixed point constructs the actual fine vacuum. Its true marginal
defines an isometric, vacuum-preserving coarse embedding. Exact rational
estimates then bound compression, every physical fiber mode, and relative
drift coupling. No preceding full-Hamiltonian gap is an input.
"""

from __future__ import annotations

from fractions import Fraction as Q
from math import isqrt
from typing import Any

from omnibias.core.proof.certificate import make_certificate, verify_certificate_digest
from omnibias.geometry.gauge.transfer.invariant_vacuum_fourier import (
    invariant_vacuum_fourier_bounds,
)
from omnibias.geometry.gauge.transfer.static_sources import _integer, _rational

_EDGES = ((0, 2), (2, 3), (3, 1), (0, 4), (4, 5), (5, 1), (0, 1))
_PLAQUETTES = ((1, 2, 3, -7), (4, 5, 6, -7))


def _sqrt_upper(value: Q, bits: int) -> Q:
    """Smallest dyadic multiple of 2**(-bits) at least sqrt(value)."""
    scale = 1 << bits
    radicand = value.numerator * scale**2
    root = isqrt(radicand // value.denominator)
    if root * root * value.denominator < radicand:
        root += 1
    return Q(root, scale)


def su2_theta_vacuum_refinement(
    kappa: int | Q,
    *,
    correction_radius: int | Q,
    exponent_steps: int = 8,
    sqrt_bits: int = 64,
) -> dict[str, Any]:
    """Certify one actual-marginal refinement and its physical finite gap.

    Hamiltonian aH=kappa/2*sum(C_e)+2/kappa*(4-chi_L-chi_R),
    seven unit electric edges, two unit plaquettes, Gauss law at every vertex.
    The returned embedding is analytically specified through the constructed
    true vacuum; a sampled or truncated wavefunction is not substituted.
    Failure means this sufficient criterion did not establish the claim.
    """
    coupling = _rational(kappa, "kappa")
    radius = _rational(correction_radius, "correction_radius")
    steps = _integer(exponent_steps, "exponent_steps")
    bits = _integer(sqrt_bits, "sqrt_bits")
    if coupling <= 0 or radius <= 0 or steps < 1 or bits < 0:
        raise ValueError("require positive kappa, radius, exponent_steps and nonnegative sqrt_bits")
    source = invariant_vacuum_fourier_bounds(
        6, _EDGES, group="su2", kappa=coupling, correction_radius=radius,
        plaquettes=_PLAQUETTES, weighted_incidence_cap=2,
    )
    source_witness = source["witness"]
    source_arithmetic = source_witness["arithmetic"]
    fixed_point = bool(source_witness["structural_caps_verified"]
                       and source_arithmetic["fixed_point_verified"])
    alpha = coupling / 2
    g = 4 / coupling**2
    marginal_linear = 8 * (g + radius) / 3
    marginal_quadratic = 8 * g**2 / 9 + 14 * radius / 3
    marginal_oscillation = min(marginal_linear, marginal_quadratic)
    fiber_oscillation = (16 * g + 8 * radius) / 3
    exponential_domain = max(marginal_oscillation, fiber_oscillation) < steps
    actual_bounds = fixed_point and exponential_domain
    compression: Q | None = None
    fiber: Q | None = None
    loop_poincare: Q | None = None
    covariance_direct: Q | None = None
    covariance_hessian: Q | None = None
    relative_square: Q | None = None
    marginal_exp: Q | None = None
    fiber_exp: Q | None = None
    discriminant: Q | None = None
    sqrt_bound: Q | None = None
    gap: Q | None = None
    candidate_gap: Q | None = None
    if actual_bounds:
        marginal_exp = (1 - marginal_oscillation / steps)**steps
        fiber_exp = (1 - fiber_oscillation / steps)**steps
        loop_poincare = Q(3, 4) * fiber_exp
        compression = alpha * Q(9, 2) * marginal_exp
        fiber = alpha * Q(5, 2) * loop_poincare
        covariance_direct = 4 * (g + radius)**2
        covariance_hessian = (g + 2 * radius / 3)**2 / loop_poincare
        relative_square = 2 * alpha / 3 * min(covariance_direct, covariance_hessian)
        discriminant = (compression - fiber)**2 + 4 * compression * relative_square
        sqrt_bound = _sqrt_upper(discriminant, bits)
        candidate_gap = (compression + fiber - sqrt_bound) / 2
        if relative_square < fiber and candidate_gap > 0:
            gap = candidate_gap
    passed = gap is not None
    failures = []
    if not fixed_point:
        failures.append("actual_vacuum_fixed_point")
    if not exponential_domain:
        failures.append("strict_rational_exponential_domain")
    if actual_bounds and relative_square is not None and fiber is not None:
        if relative_square >= fiber:
            failures.append("relative_form_coercivity")
        elif gap is None:
            failures.append("positive_gap_at_requested_sqrt_precision")

    def encode(value: Q | None) -> str | None:
        return str(value) if value is not None else None

    witness = {
        "inputs": {"kappa": str(coupling), "correction_radius": str(radius),
                   "exponent_steps": steps, "sqrt_bits": bits},
        "graph": {"n_vertices": 6, "edges": [list(edge) for edge in _EDGES],
                  "plaquettes": [list(loop) for loop in _PLAQUETTES]},
        "normalization": "aH=kappa/2*sum(C_e)+2/kappa*(4-chi_L-chi_R)",
        "energy_units": "dimensionless aH",
        "gap_scope": "entire gauge-invariant physical seven-edge Hilbert space above the actual fine vacuum",
        "gauss_constraint": "at every vertex",
        "coarse_holonomy": "W=X*Y^-1, X=U1*U2*U3, Y=U4*U5*U6",
        "physical_fiber_holonomy": "R=U7*Y^-1 after endpoint gauge reduction",
        "actual_vacuum": "psi_f=normalized exp((g/3)*(chi_L+chi_R)+U), spin norm N(U)<=r",
        "actual_marginal": "nu=p_*(psi_f^2 dHaar), psi_nu=sqrt(dnu/dHaar)",
        "haar_embedding": "J(Phi)(U)=psi_f(U)*Phi(p(U))/psi_nu(p(U))",
        "embedding_scope": "analytic isometry defined through the actual constructed vacuum and actual marginal; no numerical marginal evaluator is claimed",
        "embedding_maps_actual_vacua": "J psi_nu = psi_f",
        "source_certificate": source["certificate"],
        "source_use": "structural membership and fixed-point construction only; source neutral-gap and factorization gates are not inputs",
        "arithmetic": {
            "g": str(g), "electric_alpha": str(alpha),
            "fixed_point_verified": fixed_point,
            "fixed_point_self_map_slack": source_arithmetic["self_map_slack"],
            "fixed_point_contraction_upper": source_arithmetic["contraction_upper"],
            "marginal_linear_log_oscillation_candidate": str(marginal_linear),
            "marginal_quadratic_log_oscillation_candidate": str(marginal_quadratic),
            "marginal_log_oscillation_upper": str(marginal_oscillation) if fixed_point else None,
            "marginal_bound_method": "quadratic_Haar_cancellation" if marginal_quadratic <= marginal_linear else "outer_edge_oscillation",
            "fiber_log_oscillation_upper": str(fiber_oscillation) if fixed_point else None,
            "exponential_domain_verified": exponential_domain,
            "marginal_exp_negative_lower": encode(marginal_exp),
            "fiber_exp_negative_lower": encode(fiber_exp),
            "marginal_density_ratio_enclosure": [str(marginal_exp), str(1 / marginal_exp)] if marginal_exp is not None else None,
            "conditional_loop_poincare_lower": encode(loop_poincare),
            "coarse_kinetic_coefficient": "6",
            "physical_vertical_kinetic_coefficient": "5/2",
            "compression_gap_lower": encode(compression),
            "all_physical_fiber_modes_gap_lower": encode(fiber),
            "conditional_drift_covariance_direct_upper": encode(covariance_direct),
            "conditional_drift_covariance_hessian_upper": encode(covariance_hessian),
            "relative_cross_form_beta_squared_upper": encode(relative_square),
            "schur_discriminant": encode(discriminant),
            "schur_sqrt_upper": encode(sqrt_bound),
            "gap_candidate_lower": encode(candidate_gap),
            "physical_gap_lower": encode(gap),
        },
        "failed_constraints": failures,
        "actual_vacuum_embedding_verified": fixed_point,
        "actual_conditional_estimates_verified": actual_bounds,
        "physical_gap_verified": passed,
        "coarse_model_identification": "the induced marginal defines the coarse model; equality to a prescribed Wilson coarse vacuum is unproved",
    }
    certificate = make_certificate(
        claim="actual-vacuum theta refinement with conditional bounds and a physical finite-graph gap when its gates pass",
        payload={"type": "su2_theta_vacuum_refinement_v1", "witness": witness},
        honesty={
            "actual_vacuum_embedding_verified": fixed_point,
            "actual_conditional_estimates_verified": actual_bounds,
            "physical_finite_graph_gap_verified": passed,
            "all_scale_refinement_claim": False,
            "continuum_claim": False, "yang_mills_mass_gap_claim": False,
        },
        meta={"analytic_implication": "docs/api/gauge-theta-vacuum-refinement.md",
              "transcend_backend": "not_used"},
    )
    return {
        "status": "PASS" if passed else "INCONCLUSIVE",
        "finite_gate_verified": passed,
        "actual_vacuum_embedding_verified": fixed_point,
        "actual_conditional_estimates_verified": actual_bounds,
        "physical_finite_graph_gap_verified": passed,
        "physical_gap_lower": str(gap) if gap is not None else None,
        "witness": witness, "certificate": certificate,
        "digest_verified": verify_certificate_digest(certificate),
        "all_scale_refinement_claim": False, "coarse_wilson_family_closed": False,
        "infinite_volume_claim": False, "uniform_in_a_claim": False,
        "continuum_claim": False, "yang_mills_claim": False,
        "yang_mills_mass_gap_claim": False,
        "theorem_prover_verified": False, "mathlib_verified": False,
    }


def replay_su2_theta_vacuum_refinement_certificate(certificate: dict[str, Any]) -> bool:
    """Replay success or refusal by complete canonical recomputation."""
    if not isinstance(certificate, dict):
        return False
    try:
        if not verify_certificate_digest(certificate):
            return False
        payload = certificate["payload"]
        if payload["type"] != "su2_theta_vacuum_refinement_v1":
            return False
        inputs = payload["witness"]["inputs"]
        result = su2_theta_vacuum_refinement(
            Q(inputs["kappa"]), correction_radius=Q(inputs["correction_radius"]),
            exponent_steps=inputs["exponent_steps"], sqrt_bits=inputs["sqrt_bits"],
        )
        return bool(result["certificate"] == certificate)
    except (KeyError, TypeError, ValueError, ZeroDivisionError, OverflowError, IndexError):
        return False


__all__ = [
    "replay_su2_theta_vacuum_refinement_certificate",
    "su2_theta_vacuum_refinement",
]
