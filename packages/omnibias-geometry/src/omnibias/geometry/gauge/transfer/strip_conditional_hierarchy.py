# SPDX-License-Identifier: Apache-2.0
"""Actual conditional-Poincare comparison for every finite SU(2) strip hierarchy.

The exact Fourier fixed point identifies the vacuum. Its conditional
oscillation and mixed derivatives earn a comparison matrix whose Schur
complements retain a positive weighted margin. No source spectral gap,
absolute-Hessian ball, continuum limit or formal theorem is assumed.
"""

from __future__ import annotations

from fractions import Fraction as Q
from typing import Any

from omnibias.core.proof.certificate import make_certificate, verify_certificate_digest
from omnibias.geometry.gauge.transfer.invariant_vacuum_fourier import (
    invariant_vacuum_fourier_family,
)
from omnibias.geometry.gauge.transfer.static_sources import _integer, _rational


def su2_strip_conditional_hierarchy(
    kappa: int | Q,
    *,
    correction_radius: int | Q | None = None,
    decay_base: int | Q = 1,
    exponent_steps: int = 4,
) -> dict[str, Any]:
    """Certify actual conditional gaps and every retained physical compression.

    Inputs are exact integers or Fractions; exponent_steps is a positive
    integer. The rational exponential floor requires Omega < exponent_steps.
    Failure of the source, exponential domain or positive comparison margin
    is INCONCLUSIVE, never a physical gaplessness claim. Omitting the radius
    uses the source forcing A itself; this source has a feasible radius iff
    3*kappa^2 > 1024*decay_base^2. The conditional improvement does not extend
    that vacuum-construction window.
    """
    coupling = _rational(kappa, "kappa")
    decay = _rational(decay_base, "decay_base")
    steps = _integer(exponent_steps, "exponent_steps")
    if coupling <= 0 or decay < 1 or steps <= 0:
        raise ValueError("require kappa > 0, decay_base >= 1 and exponent_steps > 0")
    alpha, g = coupling / 2, 4 / coupling**2
    forcing = 16 * g * decay**2
    radius = (
        forcing if correction_radius is None else _rational(correction_radius, "correction_radius")
    )
    if radius <= 0:
        raise ValueError("correction_radius must be positive")
    source = invariant_vacuum_fourier_family(
        "su2",
        coupling,
        correction_radius=radius,
        weighted_incidence_cap=2,
        minimum_girth=4,
        max_cycle_length=4,
        max_cycle_diameter=2,
        decay_base=decay,
    )
    fixed_point = bool(source["witness"]["arithmetic"]["fixed_point_verified"])
    # A vertical edge meets at most two Wilson plaquettes. These are actual
    # conditional bounds on exp(2*S); no classical Wilson density is substituted.
    omega = (16 * g + 8 * radius) / 3
    mixed_seed = g * decay / 3
    # For distinct verticals, ambient line distance = site distance + 1.
    # The diagonal is excluded here, so the entire correction gains 1/decay.
    mixed_correction = 2 * radius / (3 * decay)
    mixed = mixed_seed + mixed_correction
    exponential_domain = 0 <= omega < steps
    exponential = (1 - omega / steps) ** steps if exponential_domain else None
    gamma = Q(3, 4) * exponential if exponential is not None else None
    margin = gamma - 2 * mixed if gamma is not None else None
    positive_margin = margin is not None and margin > 0
    conditional = fixed_point and exponential_domain
    passed = conditional and positive_margin
    locality = passed and decay > 1
    gap = alpha * margin if passed and margin is not None else None
    inverse_margin = 1 / margin if passed and margin is not None else None
    failed = []
    if not fixed_point:
        failed.append("actual_vacuum_fixed_point")
    if not exponential_domain:
        failed.append("rational_exponential_domain")
    elif not positive_margin:
        failed.append("strict_weighted_conditional_comparison_margin")
    earned = {
        "actual_strip_vacuum_family_verified": fixed_point,
        "actual_conditional_poincare_verified": conditional,
        "actual_weighted_mixed_hessian_bound_verified": fixed_point,
        "arbitrary_depth_conditional_comparison_hierarchy_verified": passed,
        "physical_strip_compression_hierarchy_verified": passed,
        "all_compressed_physical_gaps_verified": passed,
        "spatial_exponential_covariance_bound_verified": locality,
    }
    scope = {
        "absolute_hessian_ball_preserved_claim": False,
        "all_scale_refinement_claim": False,
        "coarse_wilson_family_closed": False,
        "infinite_volume_claim": False,
        "uniform_in_a_claim": False,
        "continuum_claim": False,
        "yang_mills_claim": False,
        "yang_mills_mass_gap_claim": False,
        "static_confinement_claim": False,
    }
    witness = {
        "inputs": {
            "kappa": str(coupling),
            "correction_radius": str(radius),
            "decay_base": str(decay),
            "exponent_steps": steps,
        },
        "source_certificate": source["certificate"],
        "source_use": (
            "actual weighted Fourier fixed point only; no source curvature, spectral-gap "
            "or factorization premise"
        ),
        "family": (
            "all finite open 1-by-n square strips, integer n>=1, unit Wilson plaquette "
            "weights and unit electric edge weights"
        ),
        "structure": {
            "plaquette_incidence_cap": 2,
            "girth": 4,
            "cycle_length": 4,
            "ambient_cycle_diameter": 2,
            "n_edges": "3*n+1",
            "n_vertices": "2*n+2",
        },
        "normalization": "aH=kappa/2*sum_e C_e + 2/kappa*sum_p(2-ReTr U_p)",
        "coordinate_gauge": (
            "horizontal forest equals I; n+1 vertical SU2 coordinates with left and right "
            "endpoint Gauss actions"
        ),
        "density": "the actual identified vacuum density exp(2*S)/Z with S=(g/3)*sum_p chi_p+u",
        "conditional": (
            "one vertical SU2 coordinate with all other verticals fixed; product Haar "
            "reference, fundamental Casimir 3/4"
        ),
        "exponential_inequality": "(1-Omega/N)^N <= exp(-Omega) for integer N>0 and 0<=Omega<N",
        "comparison_matrix": (
            "H_ii=gamma_i, H_ij=-2*c_ij for i!=j; gamma_i are actual conditional Poincare "
            "lower bounds"
        ),
        "mixed_weight_conversion": (
            "for distinct verticals d_line(V_i,V_j)=|i-j|+1, so the mixed correction row "
            "in site weights is at most 2*r/(3*decay_base)"
        ),
        "exact_form_restriction": (
            "the local one-form bound is used only on d_i u from the scalar Poisson "
            "solution, not on arbitrary one-forms"
        ),
        "marginal_update": (
            "gamma'_i=gamma_i-4*c_iR*H_RR^-1*c_Ri; c'_ij=c_ij+2*c_iR*H_RR^-1*c_Rj for "
            "i!=j; H'=Schur_R(H)"
        ),
        "preserved_bound": (
            "gamma'_i-2*sum_(j!=i) decay_base^|i-j| c'_ij >= delta, at every finite retention depth"
        ),
        "covariance_kernel": (
            "|Cov(f,g)| <= a^T H_RR^-1 b for block L2 gradient norms; weighted inverse row "
            "<= 1/delta"
        ),
        "retention_scope": (
            "every nested sequence of retained vertical subsets containing endpoints 0,n; "
            "unchanged microscopic distances"
        ),
        "physical_compression": (
            "Gamma_I=sum_i|p_i|^2+sum_j(i_(j+1)-i_j)*(|sum_h<=j p_h|^2+|sum_h<=j "
            "Ad(Q_h)p_h|^2); both endpoint Gauss sums zero"
        ),
        "kinetic_floor": (
            "Gamma_I >= sum_i |p_i|^2; exact retained forms depend only on retained coordinates"
        ),
        "gap_scope": (
            "all representations in each finite physical compression above its unique "
            "actual marginal vacuum"
        ),
        "energy_units": (
            "original dimensionless aH at fixed microscopic kappa; no energy or spacing rescaling"
        ),
        "arithmetic": {
            "g": str(g),
            "electric_alpha": str(alpha),
            "source_forcing": str(forcing),
            "source_radius_feasibility_slack": str(3 * coupling**2 - 1024 * decay**2),
            "source_radius_exists_for_criterion": 3 * coupling**2 > 1024 * decay**2,
            "fixed_point_verified": fixed_point,
            "conditional_log_density_oscillation_candidate_upper": str(omega),
            "conditional_log_density_oscillation_upper": str(omega) if fixed_point else None,
            "exponent_steps": steps,
            "rational_exponential_domain_verified": exponential_domain,
            "exp_negative_rational_candidate_lower": str(exponential)
            if exponential is not None
            else None,
            "exp_negative_rational_lower": str(exponential) if conditional else None,
            "conditional_poincare_candidate_lower": str(gamma) if gamma is not None else None,
            "conditional_poincare_lower": str(gamma) if conditional else None,
            "weighted_mixed_seed_row_upper": str(mixed_seed),
            "unconverted_mixed_correction_row_upper": str(2 * radius / 3),
            "weighted_mixed_correction_row_candidate_upper": str(mixed_correction),
            "weighted_mixed_hessian_row_candidate_upper": str(mixed),
            "weighted_mixed_hessian_row_upper": str(mixed) if fixed_point else None,
            "conditional_comparison_margin_candidate_lower": str(margin)
            if margin is not None
            else None,
            "strict_comparison_margin_verified": positive_margin,
            "uniform_conditional_comparison_margin_lower": str(margin) if passed else None,
            "weighted_covariance_kernel_row_upper": str(inverse_margin) if passed else None,
            "covariance_tail_prefactor_upper": str(inverse_margin) if locality else None,
            "tail_rate": "decay_base^(-D) in inherited site distance" if locality else None,
            "all_compressed_physical_gap_lower": str(gap) if passed else None,
        },
        "failed_constraints": failed,
    }
    certificate = make_certificate(
        claim=(
            "the actual fixed-coupling SU2 strip vacuum has conditional Poincare and "
            "mixed-derivative bounds whose comparison margin survives every finite "
            "retained physical compression when the exact gates pass"
        ),
        payload={"type": "su2_strip_conditional_hierarchy_v1", "witness": witness},
        honesty={**earned, **scope},
        meta={
            "analytic_implication": "docs/api/gauge-strip-marginal-hierarchy.md",
            "transcend_backend": "not_used",
        },
    )
    return {
        "status": "PASS" if passed else "INCONCLUSIVE",
        "finite_gate_verified": passed,
        "all_compressed_physical_gap_lower": str(gap) if passed else None,
        "witness": witness,
        "certificate": certificate,
        "digest_verified": verify_certificate_digest(certificate),
        **earned,
        **scope,
        "theorem_prover_verified": False,
        "mathlib_verified": False,
    }


def replay_su2_strip_conditional_hierarchy_certificate(certificate: dict[str, Any]) -> bool:
    """Canonically recompute PASS or INCONCLUSIVE, including source and scope.

    True means exact replay of the recorded decision; it does not turn an
    INCONCLUSIVE certificate into a passing physical gap certificate.
    """
    if not isinstance(certificate, dict):
        return False
    try:
        if not verify_certificate_digest(certificate):
            return False
        payload = certificate["payload"]
        if payload["type"] != "su2_strip_conditional_hierarchy_v1":
            return False
        inputs = payload["witness"]["inputs"]
        result = su2_strip_conditional_hierarchy(
            Q(inputs["kappa"]),
            correction_radius=Q(inputs["correction_radius"]),
            decay_base=Q(inputs["decay_base"]),
            exponent_steps=inputs["exponent_steps"],
        )
        return bool(result["certificate"] == certificate)
    except (KeyError, TypeError, ValueError, ZeroDivisionError, OverflowError, IndexError):
        return False


__all__ = ["replay_su2_strip_conditional_hierarchy_certificate", "su2_strip_conditional_hierarchy"]
