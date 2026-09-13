# SPDX-License-Identifier: Apache-2.0
"""Two actual-vacuum reductions of the physical four-plaquette SU(2) strip.

The true joint marginal, its conditional derivatives and the anisotropic
electric metric are retained. Two relative Schur bounds control all modes
of this fixed graph. Neither the preceding full gap nor an independent
cell model is an input to the nested spectral estimate.
"""

from __future__ import annotations

from fractions import Fraction as Q
from typing import Any

from omnibias.core.proof.certificate import make_certificate, verify_certificate_digest
from omnibias.geometry.gauge.transfer.shared_strip_refinement import (
    su2_shared_strip_refinement,
)
from omnibias.geometry.gauge.transfer.static_sources import _integer, _rational
from omnibias.geometry.gauge.transfer.theta_vacuum_refinement import _sqrt_upper


def _encode(value: Q | None) -> str | None:
    return str(value) if value is not None else None


def _schur(coarse: Q, fiber: Q, relative: Q, bits: int) -> dict[str, Any]:
    discriminant = (coarse - fiber)**2 + 4 * coarse * relative
    radical = _sqrt_upper(discriminant, bits)
    candidate = (coarse + fiber - radical) / 2
    passed = coarse > 0 and fiber > 0 and relative < fiber and candidate > 0
    return {
        "compression_gap_lower": str(coarse),
        "all_physical_fiber_modes_gap_lower": str(fiber),
        "relative_cross_form_beta_squared_upper": str(relative),
        "discriminant": str(discriminant), "sqrt_upper": str(radical),
        "gap_candidate_lower": str(candidate),
        "gap_lower": str(candidate) if passed else None,
        "relative_form_coercivity_verified": relative < fiber,
        "gap_verified": passed,
    }


def su2_shared_strip_two_step(
    kappa: int | Q,
    *,
    correction_radius: int | Q,
    exponent_steps: int = 8,
    sqrt_bits: int = 64,
) -> dict[str, Any]:
    """Seal a four-to-two-to-one actual-marginal reduction, including its gap.

    The source constructs the actual fine vacuum. Its conditional first
    complement and relative cross-form bounds are used without importing
    its old compression gap, full gap or joint logarithm ball. A new
    marginal derivative estimate yields the second Schur input. Each
    stage reports its own refusal rather than promoting a partial success.
    """
    coupling = _rational(kappa, "kappa")
    radius = _rational(correction_radius, "correction_radius")
    steps = _integer(exponent_steps, "exponent_steps")
    bits = _integer(sqrt_bits, "sqrt_bits")
    if coupling <= 0 or radius <= 0 or steps < 1 or bits < 0:
        raise ValueError("require positive kappa, radius, exponent_steps and nonnegative sqrt_bits")
    source = su2_shared_strip_refinement(
        coupling, correction_radius=radius, exponent_steps=steps, sqrt_bits=bits,
    )
    inherited = source["witness"]["arithmetic"]
    fixed_point = bool(inherited["fixed_point_verified"])
    first_conditional = bool(source["actual_conditional_estimates_verified"])
    alpha, g, correction_hessian = coupling / 2, 4 / coupling**2, 2 * radius / 3
    fiber_oscillation = (16 * g + 8 * radius) / 3
    marginal_oscillation = 8 * (g + radius) / 3
    second_exponential_domain = max(fiber_oscillation, marginal_oscillation) < steps
    derivative = fixed_point and first_conditional
    second_conditional = derivative and second_exponential_domain
    first_poincare: Q | None = None
    first_fiber: Q | None = None
    first_relative: Q | None = None
    mixed_covariance: Q | None = None
    marginal_drift_derivative: Q | None = None
    fiber_exp: Q | None = None
    marginal_exp: Q | None = None
    second_poincare: Q | None = None
    second: dict[str, Any] | None = None
    first: dict[str, Any] | None = None
    if derivative:
        first_poincare = Q(inherited["conditional_pair_poincare_lower"])
        first_fiber = Q(inherited["all_physical_fiber_modes_gap_lower"])
        first_relative = Q(inherited["relative_cross_form_beta_squared_upper"])
        mixed_covariance = 2 * (g / 6 + correction_hessian)**2 / first_poincare
        marginal_drift_derivative = 5 * correction_hessian + 10 * mixed_covariance
    if second_conditional:
        assert marginal_drift_derivative is not None
        assert first_fiber is not None and first_relative is not None
        fiber_exp = (1 - fiber_oscillation / steps)**steps
        marginal_exp = (1 - marginal_oscillation / steps)**steps
        second_poincare = Q(3, 4) * fiber_exp
        coarse_floor = Q(15, 2) * alpha * marginal_exp
        fiber_floor = Q(7, 2) * alpha * second_poincare
        relative = 2 * alpha * marginal_drift_derivative**2 / (5 * second_poincare)
        second = _schur(coarse_floor, fiber_floor, relative, bits)
        if second["gap_verified"]:
            first = _schur(Q(second["gap_lower"]), first_fiber, first_relative, bits)
    intermediate_verified = bool(second is not None and second["gap_verified"])
    passed = bool(first is not None and first["gap_verified"])
    failures = []
    if not fixed_point:
        failures.append("actual_vacuum_fixed_point")
    if not first_conditional:
        failures.append("first_actual_conditional_estimates")
    if not second_exponential_domain:
        failures.append("second_strict_rational_exponential_domain")
    if second_conditional and not intermediate_verified:
        failures.append("second_relative_form_or_gap_precision")
    if intermediate_verified and not passed:
        failures.append("composed_relative_form_or_gap_precision")
    witness = {
        "inputs": {"kappa": str(coupling), "correction_radius": str(radius),
                   "exponent_steps": steps, "sqrt_bits": bits},
        "graph": source["witness"]["graph"],
        "normalization": source["witness"]["normalization"],
        "energy_units": "original dimensionless aH at both reductions; no spacing or coupling rescaling",
        "source_certificate": source["certificate"],
        "source_use": "actual vacuum construction and first all-complement/relative-form bounds only; old compression gap, full gap, joint log ball are not premises",
        "coordinate_gauge": "eight horizontal links equal I, five vertical links (X,R_L,Z,R_R,Y), with residual endpoint left/right actions",
        "first_marginal_log_vacuum": "S1(X,Z,Y)=1/2*log integral exp(2*S(X,R_L,Z,R_R,Y)) dR_L dR_R, up to a constant",
        "second_coarse_holonomy": "W=X*Y^-1=(X*Z^-1)*(Y*Z^-1)^-1",
        "second_physical_fiber_holonomy": "R=Z*Y^-1 after endpoint gauge reduction",
        "intermediate_electric_weights": ["5", "5", "1"],
        "outer_loop_electric_weight": "10",
        "second_physical_fiber_kinetic_floor": "7/2",
        "mixed_derivative_identity": "D_Z E[f_v|X,Z,Y]=E[D_Z f_v|X,Z,Y]+2 Cov(f_v,D_Z S|X,Z,Y)",
        "frame_scope": "dp_X and dp_Y depend only on X,Y; their Z,R_L,R_R derivatives vanish; scalar contractions descend under the endpoint gauge",
        "nested_haar_isometries": "J01(Phi)=psi0*Phi/phi1; J12(F)=phi1*F/phi2; J01*J12(F)=psi0*F/phi2",
        "marginal_consistency": "same actual fine density at both stages; iterated Haar integration equals direct outer marginal by Fubini",
        "compression_consistency": "K1=J01^* K0 J01 and K2=J12^* K1 J12 as closed forms; full fine dynamics is not replaced by a compression",
        "arithmetic": {
            "g": str(g), "electric_alpha": str(alpha),
            "fixed_point_verified": fixed_point,
            "first_conditional_estimates_verified": first_conditional,
            "source_correction_hessian_row_upper": str(correction_hessian) if fixed_point else None,
            "first_conditional_pair_poincare_lower": _encode(first_poincare),
            "first_all_physical_fiber_modes_gap_lower": _encode(first_fiber),
            "first_relative_cross_form_beta_squared_upper": _encode(first_relative),
            "conditional_mixed_covariance_upper": _encode(mixed_covariance),
            "marginal_drift_Z_derivative_upper": _encode(marginal_drift_derivative),
            "second_fiber_log_oscillation_upper": str(fiber_oscillation) if fixed_point else None,
            "outer_marginal_log_oscillation_upper": str(marginal_oscillation) if fixed_point else None,
            "second_exponential_domain_verified": second_exponential_domain,
            "second_fiber_exp_negative_lower": _encode(fiber_exp),
            "outer_marginal_exp_negative_lower": _encode(marginal_exp),
            "second_conditional_loop_poincare_lower": _encode(second_poincare),
        },
        "second_step": second,
        "composed_first_step": first,
        "failed_constraints": failures,
        "actual_nested_marginals_verified": fixed_point,
        "actual_marginal_derivative_bound_verified": derivative,
        "second_actual_conditional_estimates_verified": second_conditional,
        "intermediate_physical_gap_verified": intermediate_verified,
        "physical_finite_graph_gap_verified": passed,
        "scope": "one nested finite thirteen-edge SU2 graph, all physical representations; no fixed effective Wilson dictionary",
    }
    earned = {
        "actual_nested_marginals_verified": fixed_point,
        "actual_marginal_derivative_bound_verified": derivative,
        "second_actual_conditional_estimates_verified": second_conditional,
        "intermediate_physical_gap_verified": intermediate_verified,
        "physical_finite_graph_gap_verified": passed,
        "two_step_refinement_verified": passed,
    }
    scope = {
        "independent_cell_replacement_verified": False,
        "all_scale_refinement_claim": False, "coarse_wilson_family_closed": False,
        "infinite_volume_claim": False, "uniform_in_a_claim": False,
        "continuum_claim": False, "yang_mills_claim": False,
        "yang_mills_mass_gap_claim": False,
    }
    certificate = make_certificate(
        claim="two nested actual-vacuum reductions with conditional derivative and physical finite-graph gap bounds when their gates pass",
        payload={"type": "su2_shared_strip_two_step_v1", "witness": witness},
        honesty={**earned, **scope},
        meta={"analytic_implication": "docs/api/gauge-shared-strip-two-step.md",
              "transcend_backend": "not_used"},
    )
    return {
        "status": "PASS" if passed else "INCONCLUSIVE", "finite_gate_verified": passed,
        "intermediate_physical_gap_lower": second["gap_lower"] if second is not None else None,
        "physical_gap_lower": first["gap_lower"] if first is not None else None,
        "witness": witness, "certificate": certificate,
        "digest_verified": verify_certificate_digest(certificate), **earned, **scope,
        "theorem_prover_verified": False, "mathlib_verified": False,
    }


def replay_su2_shared_strip_two_step_certificate(certificate: dict[str, Any]) -> bool:
    """Replay each nested source, derivative estimate, Schur gate and scope."""
    if not isinstance(certificate, dict):
        return False
    try:
        if not verify_certificate_digest(certificate):
            return False
        payload = certificate["payload"]
        if payload["type"] != "su2_shared_strip_two_step_v1":
            return False
        inputs = payload["witness"]["inputs"]
        result = su2_shared_strip_two_step(
            Q(inputs["kappa"]), correction_radius=Q(inputs["correction_radius"]),
            exponent_steps=inputs["exponent_steps"], sqrt_bits=inputs["sqrt_bits"],
        )
        return bool(result["certificate"] == certificate)
    except (KeyError, TypeError, ValueError, ZeroDivisionError, OverflowError, IndexError):
        return False


__all__ = [
    "replay_su2_shared_strip_two_step_certificate",
    "su2_shared_strip_two_step",
]
