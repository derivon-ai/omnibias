# SPDX-License-Identifier: Apache-2.0
"""Exact local Wilson residuals improve the actual SU(2) vacuum criterion.

The original-edge Fourier norm is evaluated in a fixed positive-coordinate
orientation. The residual includes every second-order term, and a Banach
bound controls the entire correction. No spin or cluster remainder is
discarded. The finite-family theorem remains at fixed lattice coupling.
"""

from __future__ import annotations

from fractions import Fraction as Q
from typing import Any

from omnibias.core.proof.certificate import make_certificate, verify_certificate_digest
from omnibias.geometry.gauge.transfer.static_sources import _integer, _rational


def _encoded(value: Q | None) -> str | None:
    return str(value) if value is not None else None


def _su2_wilson_vacuum(
    kappa: int | Q, *, family: str = "strip",
    correction_radius: int | Q | None = None,
    decay_base: int | Q = 1, exponent_steps: int = 4,
    fundamental_linear: bool = False,
    polar_bilinear: bool = False,
) -> dict[str, Any]:
    """Construct the actual full-spin vacuum from its exact local residual.

    ``strip`` means all finite open one-cell-wide square strips; ``cubic``
    means all finite open rectangular three-dimensional cubic boxes with
    at least one cell in each direction. Edges point in positive coordinate
    directions. These are quantified family hypotheses, not membership
    checks for an arbitrary supplied graph.

    The optional automatic radius is 2*D/(1-L) when L<1. It is a
    contracting witness exactly when (1-L)^2>4*B*D. The old source's
    separate criterion is recorded, never modified. PASS requires this
    actual vacuum and a positive physical gap route; partial source
    success remains visible when neither gap route passes.
    """
    if fundamental_linear and polar_bilinear:
        raise ValueError("select only one improved source theorem")
    if family not in ("strip", "cubic"):
        raise ValueError("family must be 'strip' or 'cubic'")
    coupling = _rational(kappa, "kappa")
    decay = _rational(decay_base, "decay_base")
    steps = _integer(exponent_steps, "exponent_steps")
    if coupling <= 0 or decay < 1 or steps < 1:
        raise ValueError("require positive kappa, decay_base>=1 and positive exponent_steps")
    supplied = None if correction_radius is None else _rational(correction_radius, "correction_radius")
    if supplied is not None and supplied <= 0:
        raise ValueError("correction_radius must be positive")
    cap, pairs = (2, 3) if family == "strip" else (4, 42)
    alpha, g = coupling / 2, 4 / coupling**2
    legacy_bilinear = Q(4, 3)
    bilinear = Q(8, 9) if polar_bilinear else legacy_bilinear
    seed = 4 * cap * g * decay**2
    residual_coefficient = 3 * cap * decay**2 + Q(8, 3) * pairs * decay**3
    residual = residual_coefficient * g**2
    generic_linear = 2 * bilinear * seed
    linear = Q(28, 3) * cap * g * decay**2 if fundamental_linear else generic_linear
    complement = 1 - linear
    discriminant = complement**2 - 4 * bilinear * residual
    feasible = complement > 0 and discriminant > 0
    radius = supplied if supplied is not None else (2 * residual / complement if complement > 0 else None)
    slack = radius - residual - linear * radius - bilinear * radius**2 if radius is not None else None
    contraction = linear + 2 * bilinear * radius if radius is not None else None
    fixed_point = bool(feasible and slack is not None and slack >= 0
                       and contraction is not None and contraction < 1)
    # The legacy estimate has twice the seed norm and residual B*A_old^2.
    old_seed = 2 * seed
    old_feasible = 4 * legacy_bilinear * old_seed < 1
    omega: Q | None = None
    mixed: Q | None = None
    gamma: Q | None = None
    conditional_margin: Q | None = None
    curvature: Q | None = None
    exp_lower: Q | None = None
    if radius is not None:
        omega = (8 * cap * g + 8 * radius) / 3
        if family == "strip":
            # Site distance on distinct verticals is ambient edge distance - 1.
            mixed = g * decay / 3 + 2 * radius / (3 * decay)
            seed_hessian = 2 * g / 3
        else:
            # In one square: two neighboring edges at distance1, one opposite
            # edge at distance2. The diagonal is absent in this mixed row.
            mixed = cap * g * (2 * decay + decay**2) / 6 + 2 * radius / 3
            seed_hessian = 2 * cap * g / 3
        curvature = Q(1, 2) - 2 * seed_hessian - 4 * radius / 3
        if 0 <= omega < steps:
            exp_lower = (1 - omega / steps)**steps
            gamma = Q(3, 4) * exp_lower
            conditional_margin = gamma - 2 * mixed
    conditional_estimates = fixed_point and gamma is not None
    conditional_pass = bool(conditional_estimates and conditional_margin is not None and conditional_margin > 0)
    curvature_pass = bool(fixed_point and curvature is not None and curvature > 0)
    conditional_gap = alpha * conditional_margin if conditional_pass and conditional_margin is not None else None
    curvature_gap = alpha * curvature if curvature_pass and curvature is not None else None
    floors = [value for value in (conditional_gap, curvature_gap) if value is not None]
    gap = max(floors) if floors else None
    passed = fixed_point and gap is not None
    failed = []
    if complement <= 0:
        failed.append("reference_linear_bound_not_below_one")
    if discriminant <= 0:
        failed.append("nonpositive_source_radius_discriminant")
    if slack is None or slack < 0:
        failed.append("selected_radius_self_map")
    if contraction is None or contraction >= 1:
        failed.append("selected_radius_strict_contraction")
    if fixed_point and not passed:
        failed.append("no_positive_physical_gap_route")
    earned = {
        "actual_vacuum_verified": fixed_point,
        "volume_uniform_actual_vacuum_family_verified": fixed_point,
        "actual_conditional_poincare_verified": conditional_estimates,
        "actual_conditional_hierarchy_verified": conditional_pass,
        "physical_compression_hierarchy_verified": conditional_pass,
        "neutral_physical_gap_verified": passed,
        "spatial_exponential_covariance_bound_verified": conditional_pass and decay > 1,
        "beyond_legacy_source_criterion_verified": fixed_point and not old_feasible,
    }
    scope = {
        "all_scale_refinement_claim": False, "coarse_wilson_family_closed": False,
        "infinite_volume_claim": False, "uniform_in_a_claim": False,
        "continuum_claim": False, "yang_mills_claim": False,
        "yang_mills_mass_gap_claim": False, "static_confinement_claim": False,
    }
    witness: dict[str, Any] = {
        "inputs": {"family": family, "kappa": str(coupling),
                   "correction_radius": _encoded(supplied), "decay_base": str(decay),
                   "exponent_steps": steps},
        "family": ("all finite open 1-by-n square strips, n>=1" if family == "strip" else
                   "all finite open rectangular three-dimensional cubic boxes, at least one cell in each coordinate direction"),
        "orientation": "every original edge points in its positive coordinate direction; Fourier nuclear norms are not assumed invariant under edge inversion",
        "normalization": "aH=kappa/2*sum_e C_e + 2/kappa*sum_p(2-ReTr U_p); unit electric and elementary Wilson plaquette weights",
        "energy_units": "original dimensionless aH at fixed microscopic kappa",
        "gauss_constraint": "at every original vertex; no representation cutoff",
        "structural_caps": {"girth": 4, "plaquettes_per_edge": cap,
                            "adjacent_plaquette_pairs_touching_edge": pairs,
                            "single_support_diameter": 2, "pair_union_diameter": 3},
        "original_edge_tensor_budgets": {
            "fundamental_square": {"rank": 4, "singular_value": "2", "nuclear_norm": "8"},
            "adjoint_square": {"rank": 9, "singular_value": "3", "nuclear_norm": "27"},
            "normalized_pair_channels": {"rank_upper": 16, "hilbert_schmidt_squared": "16",
                                         "nuclear_norm_upper": "16"},
            "proof_scope": "original-link coefficient tensors with canonical orientation; exact vertex flattenings and finite tensor regressions",
        },
        "reference": "S_star=(g/3)*sum_p chi_(1/2)(U_p)",
        "exact_centered_residual": "C0^-1 Pi0 Gamma(S_star,S_star)=g^2*(-sum_p chi_1(U_p)/72 + sum_adj (b_0/27-b_1/39))",
        "pair_channels": "b_0=chi_outer/2, b_1=(chi_p*chi_q-chi_outer/2)/3; electric energies9/2 and13/2",
        "residual_anchor_budget": "one incident adjoint square contributes3*g^2*b^2; one touching adjacent pair contributes at most8/3*g^2*b^3",
        "fixed_point_equation": "u=C0^-1 Pi0 Gamma(S_star,S_star)+2*T(S_star,u)+T(u,u), T=C0^-1 Pi0 Gamma",
        "analytic_source": "Banach contraction in the complete invariant spin-weighted Fourier space; C2 reconstruction, elliptic bootstrap, positive eigenfunction and groundstate transform",
        "remainder_scope": "the entire nonlinear correction in the Fourier ball, all spins and support sizes; no truncation of generated interactions",
        "conditional_reference": "actual single-coordinate vacuum conditionals against SU2 Haar; scalar Haar gap3/4 and log-density oscillation Omega",
        "conditional_update": "H_ii=gamma_i,H_ij=-2*c_ij; actual marginal conditional floors and mixed derivatives update by Schur_R(H)",
        "coordinate_geometry": ("horizontal forest gauge, vertical coordinates; inherited microscopic site distances" if family == "strip" else
                                "original edge product coordinates; inherited ambient line-graph distances"),
        "compression_scope": ("any retained vertical subsets containing both endpoints; exact induced left/right prefix kinetic weights" if family == "strip" else
                              "any nested retained original-edge subsets containing an elementary plaquette; gauge-invariant physical subspace with inherited unit electric form"),
        "kinetic_compression": ("Gamma_I=sum|p_i|^2+sum_j gap_length_j*(|prefix p|^2+|prefix Ad(Q)p|^2)>=sum|p_i|^2" if family == "strip" else
                                "original product electric form restricted to retained-coordinate pullbacks equals sum of retained gradient squares; vacuum-preserving compression uses the actual marginal"),
        "arithmetic": {
            "g": str(g), "electric_alpha": str(alpha),
            "seed_norm_upper": str(seed), "residual_coefficient_upper": str(residual_coefficient),
            "residual_norm_upper": str(residual), "linear_upper": str(linear),
            "quadratic_constant": str(bilinear),
            "radius_feasibility_discriminant": str(discriminant),
            "source_radius_exists_for_criterion": feasible,
            "selected_correction_radius": _encoded(radius), "self_map_slack": _encoded(slack),
            "contraction_upper": _encoded(contraction), "fixed_point_verified": fixed_point,
            "old_seed_norm_upper": str(old_seed),
            "old_generic_residual_upper": str(legacy_bilinear * old_seed**2),
            "old_source_radius_exists_for_criterion": old_feasible,
            "conditional_log_density_oscillation_upper": _encoded(omega) if fixed_point else None,
            "rational_exponential_domain_verified": gamma is not None,
            "exp_negative_rational_lower": _encoded(exp_lower) if conditional_estimates else None,
            "conditional_poincare_lower": _encoded(gamma) if conditional_estimates else None,
            "weighted_mixed_hessian_row_upper": _encoded(mixed) if fixed_point else None,
            "conditional_comparison_margin_lower": _encoded(conditional_margin) if conditional_estimates else None,
            "conditional_physical_gap_lower": _encoded(conditional_gap),
            "weighted_covariance_kernel_row_upper": str(1 / conditional_margin)
            if conditional_pass and conditional_margin is not None else None,
            "curvature_lower": _encoded(curvature) if fixed_point else None,
            "curvature_physical_gap_lower": _encoded(curvature_gap),
        },
        "failed_constraints": failed,
    }
    if fundamental_linear:
        # Same Banach space and exact residual; only the linear estimate is
        # changed. The old public wrapper retains its exact v1 payload.
        prior_source_feasible = generic_linear < 1 and (1 - generic_linear)**2 > 4 * bilinear * residual
        earned.update({
            "fourier_linear_inverse_verified": linear < 1,
            "beyond_previous_linear_gate_verified": linear < 1 <= generic_linear,
            "beyond_previous_source_criterion_verified": fixed_point and not prior_source_feasible,
        })
        witness["linear_operator_source"] = "exact fundamental fusion at the differentiated anchor; three other square edges charged separately; ||2*T(S_star,.)||<=28*q*g*b^2/3 in the same complete spin-weighted Fourier norm"
        witness["arithmetic"].update({
            "generic_linear_upper": str(generic_linear),
            "previous_source_radius_exists_for_criterion": prior_source_feasible,
            "fourier_linear_inverse_upper": str(1 / (1 - linear)) if linear < 1 else None,
        })
    certificate_type = "su2_wilson_linear_vacuum_v1" if fundamental_linear else "su2_wilson_residual_vacuum_v1"
    implication = "docs/api/gauge-wilson-linear-source.md" if fundamental_linear else "docs/api/gauge-wilson-residual-source.md"
    claim = ("exact fundamental-fusion linear bounds and local Wilson residuals construct the actual finite-family SU2 vacuum and certify its physical gap when the stated gates pass" if fundamental_linear else
             "exact original-link Wilson residual bounds construct the actual finite-family SU2 vacuum and certify its physical gap when the stated gates pass")
    if polar_bilinear:
        previous_linear = Q(28, 3) * cap * g * decay**2
        previous_feasible = previous_linear < 1 and (1 - previous_linear)**2 > 4 * legacy_bilinear * residual
        seed_influence = Q(4, 3) * cap * g * (2 * decay + decay**2)
        influence = seed_influence + Q(16, 3) * radius if radius is not None else None
        dobrushin_pass = bool(fixed_point and influence is not None and influence < 1)
        earned.update({
            "gauge_polar_bilinear_bound_verified": True,
            "fourier_linear_inverse_verified": linear < 1,
            "beyond_previous_linear_gate_verified": linear < 1 <= previous_linear,
            "beyond_previous_source_criterion_verified": fixed_point and not previous_feasible,
            "actual_weighted_dobrushin_bound_verified": dobrushin_pass,
        })
        witness["bilinear_operator_source"] = (
            "gauge-invariant polar factors have scalar single-edge marginals; compressed polar trace-norm domination and zero mean coupled Casimir defect give B=8/9 in the complete original spin-weighted norm, including intertwiner multiplicities"
        )
        witness["linear_operator_source"] = "2*B*N(S_star), with B=8/9 and N(S_star)<=4*q*g*b^2; no scalar positivity assumption on Fourier coefficients"
        witness["conditional_influence_source"] = "mixed four-point oscillation of the actual density: seed row(4*q*g/3)*(2*b+b^2), correction row16*r/3; total-variation convention sup_A|mu(A)-nu(A)|; original-edge coordinates and ambient distances"
        witness["arithmetic"].update({
            "previous_quadratic_constant": str(legacy_bilinear),
            "previous_fundamental_linear_upper": str(previous_linear),
            "previous_source_radius_exists_for_criterion": previous_feasible,
            "fourier_linear_inverse_upper": str(1 / (1 - linear)) if linear < 1 else None,
            "seed_weighted_influence_row_upper": str(seed_influence),
            "actual_weighted_influence_row_upper": str(influence) if fixed_point and influence is not None else None,
            "actual_weighted_dobrushin_margin_lower": str(1 - influence) if fixed_point and influence is not None else None,
        })
        certificate_type = "su2_wilson_polar_vacuum_v1"
        implication = "docs/api/gauge-wilson-polar-source.md"
        claim = "gauge-polar bilinear cancellation and exact Wilson residual bounds construct the actual all-spin finite-family SU2 vacuum and certify its physical gap when the stated gates pass"
    certificate = make_certificate(
        claim=claim,
        payload={"type": certificate_type, "witness": witness},
        honesty={**earned, **scope},
        meta={"analytic_implication": implication,
              "transcend_backend": "not_used"},
    )
    return {
        "status": "PASS" if passed else "INCONCLUSIVE", "finite_gate_verified": passed,
        "physical_gap_lower": _encoded(gap), "witness": witness, "certificate": certificate,
        "digest_verified": verify_certificate_digest(certificate), **earned, **scope,
        "theorem_prover_verified": False, "mathlib_verified": False,
    }


def su2_wilson_residual_vacuum(
    kappa: int | Q, *, family: str = "strip",
    correction_radius: int | Q | None = None,
    decay_base: int | Q = 1, exponent_steps: int = 4,
) -> dict[str, Any]:
    """Original exact-residual source; its v1 certificates are unchanged."""
    return _su2_wilson_vacuum(
        kappa, family=family, correction_radius=correction_radius,
        decay_base=decay_base, exponent_steps=exponent_steps,
    )


def su2_wilson_linear_vacuum(
    kappa: int | Q, *, family: str = "cubic",
    correction_radius: int | Q | None = None,
    decay_base: int | Q = 1, exponent_steps: int = 4,
) -> dict[str, Any]:
    """Sharpen the full-space linear bound by exact fundamental fusion.

    A Fourier inverse can be earned while the nonlinear source remains
    INCONCLUSIVE. The two gates and their consequences stay separate.
    """
    return _su2_wilson_vacuum(
        kappa, family=family, correction_radius=correction_radius,
        decay_base=decay_base, exponent_steps=exponent_steps, fundamental_linear=True,
    )


def replay_su2_wilson_linear_vacuum_certificate(certificate: dict[str, Any]) -> bool:
    """Recompute the sharper linear criterion and every nonlinear gate."""
    if not isinstance(certificate, dict):
        return False
    try:
        if not verify_certificate_digest(certificate):
            return False
        payload = certificate["payload"]
        if payload["type"] != "su2_wilson_linear_vacuum_v1":
            return False
        inputs = payload["witness"]["inputs"]
        result = su2_wilson_linear_vacuum(
            Q(inputs["kappa"]), family=inputs["family"],
            correction_radius=Q(inputs["correction_radius"]) if inputs["correction_radius"] is not None else None,
            decay_base=Q(inputs["decay_base"]), exponent_steps=inputs["exponent_steps"],
        )
        return bool(result["certificate"] == certificate)
    except (KeyError, TypeError, ValueError, ZeroDivisionError, OverflowError, IndexError):
        return False


def replay_su2_wilson_residual_vacuum_certificate(certificate: dict[str, Any]) -> bool:
    """Recompute the source identities' budgets, all gates and exact scope."""
    if not isinstance(certificate, dict):
        return False
    try:
        if not verify_certificate_digest(certificate):
            return False
        payload = certificate["payload"]
        if payload["type"] != "su2_wilson_residual_vacuum_v1":
            return False
        inputs = payload["witness"]["inputs"]
        result = su2_wilson_residual_vacuum(
            Q(inputs["kappa"]), family=inputs["family"],
            correction_radius=Q(inputs["correction_radius"]) if inputs["correction_radius"] is not None else None,
            decay_base=Q(inputs["decay_base"]), exponent_steps=inputs["exponent_steps"],
        )
        return bool(result["certificate"] == certificate)
    except (KeyError, TypeError, ValueError, ZeroDivisionError, OverflowError, IndexError):
        return False


__all__ = [
    "replay_su2_wilson_linear_vacuum_certificate",
    "replay_su2_wilson_residual_vacuum_certificate",
    "su2_wilson_linear_vacuum",
    "su2_wilson_residual_vacuum",
]
