# SPDX-License-Identifier: Apache-2.0
"""Actual adjacent-vacuum gaps from a proved coefficient oscillation bound."""

from __future__ import annotations

from fractions import Fraction as Q
from typing import Any

from omnibias.core.proof.certificate import make_certificate, verify_certificate_digest
from omnibias.geometry.gauge.transfer.adjacent_vacuum import (
    replay_su2_adjacent_preconditioned_vacuum_certificate,
)
from omnibias.geometry.gauge.transfer.static_sources import _integer


def _replay_source(certificate: dict[str, Any]) -> bool:
    if not isinstance(certificate, dict):
        return False
    payload = certificate.get("payload")
    if not isinstance(payload, dict):
        return False
    kind = payload.get("type")
    if kind == "su2_adjacent_preconditioned_vacuum_v1":
        return replay_su2_adjacent_preconditioned_vacuum_certificate(certificate)
    if kind == "su2_adjacent_cone_vacuum_v1":
        from omnibias.geometry.gauge.transfer.adjacent_cone_vacuum import (
            replay_su2_adjacent_cone_vacuum_certificate,
        )

        return replay_su2_adjacent_cone_vacuum_certificate(certificate)
    return False


def su2_adjacent_vacuum_gap_comparison(
    source_certificate: dict[str, Any], *, exponent_steps: int = 4,
) -> dict[str, Any]:
    """Compare an actual seven-edge vacuum to a known reference and to Haar.

    Only the two explicitly supported canonical actual-source schemas are
    accepted. The original Fourier radius is inherited, never supplied.
    A canonical source with a failed nonlinear gate yields INCONCLUSIVE.

    exponent_steps is a positive integer minimum. If necessary it is raised
    deterministically to floor(the larger oscillation bound)+1, so both
    rational exponential floors remain strictly positive for every finite
    earned radius. No positive-curvature premise is used.
    """
    requested = _integer(exponent_steps, "exponent_steps")
    if requested < 1:
        raise ValueError("exponent_steps must be a strictly positive integer")
    if not _replay_source(source_certificate):
        raise ValueError("a canonical supported adjacent actual-source certificate is required")
    source = source_certificate["payload"]["witness"]
    arithmetic = source["arithmetic"]
    coupling = Q(source["inputs"]["kappa"])
    g = Q(arithmetic["g"])
    radius = (
        Q(arithmetic["selected_correction_radius"])
        if arithmetic["selected_correction_radius"] is not None else None
    )
    actual = (
        source_certificate["honesty"].get("actual_vacuum_verified") is True
        and arithmetic["fixed_point_verified"] is True
        and radius is not None and radius > 0
    )
    curvature = (
        Q(arithmetic["curvature_lower"]) if arithmetic.get("curvature_lower") is not None else None
    )
    correction_oscillation = Q(13, 18) * radius if radius is not None else None
    neutral_oscillation = (
        8 * g / 3 + correction_oscillation if correction_oscillation is not None else None
    )
    scalar_oscillation = (
        16 * g / 3 + correction_oscillation if correction_oscillation is not None else None
    )
    steps = (
        max(requested, scalar_oscillation.numerator // scalar_oscillation.denominator + 1)
        if actual and scalar_oscillation is not None else None
    )
    neutral_exp = (
        (1 - neutral_oscillation / steps)**steps
        if steps is not None and neutral_oscillation is not None else None
    )
    scalar_exp = (
        (1 - scalar_oscillation / steps)**steps
        if steps is not None and scalar_oscillation is not None else None
    )
    passed = actual and neutral_exp is not None and scalar_exp is not None and scalar_exp > 0
    neutral_gap = 3 * coupling * neutral_exp / 8 if passed and neutral_exp is not None else None
    scalar_gap = 3 * coupling * scalar_exp / 8 if passed and scalar_exp is not None else None
    witness = {
        "inputs": {"exponent_steps": requested},
        "source_certificate": source_certificate,
        "source_type": source_certificate["payload"]["type"],
        "graph_scope": "fixed two adjacent elementary squares; seven original unit electric edges; all six Gauss constraints",
        "normalization": "aH=kappa*C/2+2*(4-chi_p-chi_q)/kappa; g=4/kappa^2",
        "actual_vacuum": "psi0 proportional to exp(S_star+U), S_star=(g/3)*(chi_p+chi_q); N(U)<=r from source",
        "coefficient_basis": "normalized theta b_(a,b,s), doubled admissible labels, |b_(a,b,s)|<=1",
        "coefficient_dual": {
            "weights": ["1/72", "1/72", "11/72"],
            "all_spin_inequality": "E*(a+1)^2*(b+1)^2*(a+b+11*s)>=144 for every nonconstant admissible state",
            "coefficient_l1_over_N_upper": "13/72",
            "coefficient_sharpness_witness": [
                {"state": [1, 0, 1], "coefficient": "1/12"},
                {"state": [0, 1, 1], "coefficient": "1/12"},
                {"state": [1, 1, 0], "coefficient": "1/72"},
            ],
            "oscillation_optimality_claim": False,
        },
        "reference_measure": "nu proportional to exp(2*S_star)*Haar; not the actual vacuum measure",
        "actual_density_comparison": "dmu/dnu=exp(2*U)/nu[exp(2*U)], density ratio <=exp(13*r/18)",
        "neutral_reference_metric": (
            "top and vertical edges form a spanning tree independent of bottom links; "
            "chord reference is product exp((2g/3)*chi); the two original bottom "
            "electric terms equal the two unit chord gradient norms, and five further terms are nonnegative"
        ),
        "neutral_sector": "neutral gauge-invariant physical functions, with the complete original seven-edge Dirichlet form",
        "full_scalar_sector": "all scalar functions on the original seven-link product, before imposing Gauss invariance",
        "reference_poincare": "gamma_neutral(nu)>=3/4*exp(-8*g/3) by single-link comparison and product tensorization",
        "full_scalar_comparison": "original product Haar gap3/4; osc(2*S_star)<=16*g/3",
        "variance_comparison": "Var_mu(f)<=exp(osc(2*U))/gamma_nu * integral Gamma_original(f,f) dmu",
        "exponential_rule": (
            "m=max(requested_steps,floor(Omega_scalar)+1); exp(-Omega)>="
            "(1-Omega/m)^m>0 for both sectors"
        ),
        "vacuum_subtraction": "the true ground energy established by the actual nonlinear source",
        "arithmetic": {
            "kappa": str(coupling), "g": str(g),
            "selected_correction_radius": str(radius) if radius is not None else None,
            "source_nonlinear_fixed_point_verified": actual,
            "inherited_global_curvature_lower": str(curvature) if curvature is not None else None,
            "correction_log_density_oscillation_upper": str(correction_oscillation)
            if correction_oscillation is not None else None,
            "neutral_combined_oscillation_upper": str(neutral_oscillation)
            if neutral_oscillation is not None else None,
            "full_scalar_combined_oscillation_upper": str(scalar_oscillation)
            if scalar_oscillation is not None else None,
            "effective_exponent_steps": steps,
            "neutral_exp_negative_lower": str(neutral_exp) if neutral_exp is not None else None,
            "full_scalar_exp_negative_lower": str(scalar_exp) if scalar_exp is not None else None,
            "neutral_physical_gap_lower": str(neutral_gap) if neutral_gap is not None else None,
            "full_scalar_gap_lower": str(scalar_gap) if scalar_gap is not None else None,
        },
        "failed_constraints": [] if passed else ["canonical_source_has_no_earned_actual_nonlinear_vacuum"],
    }
    earned = {
        "actual_vacuum_verified": actual,
        "neutral_physical_gap_verified": passed,
        "full_scalar_gap_verified": passed,
        "beyond_nonpositive_curvature_verified": passed and curvature is not None and curvature <= 0,
    }
    scope = {
        "positive_global_curvature_required": False,
        "uniform_in_volume_claim": False, "infinite_volume_claim": False,
        "uniform_in_a_claim": False, "all_scale_refinement_claim": False,
        "continuum_claim": False, "yang_mills_claim": False, "yang_mills_mass_gap_claim": False,
    }
    certificate = make_certificate(
        claim="positive neutral and full-scalar spectral gaps from the replayed actual adjacent vacuum and a complete coefficient oscillation bound",
        payload={"type": "su2_adjacent_vacuum_gap_comparison_v1", "witness": witness},
        honesty={**earned, **scope},
        meta={"analytic_implication": "docs/api/gauge-adjacent-gap-comparison.md", "transcend_backend": "not_used"},
    )
    return {
        "status": "PASS" if passed else "INCONCLUSIVE",
        "finite_gate_verified": passed,
        "physical_gap_lower": str(neutral_gap) if neutral_gap is not None else None,
        "full_scalar_gap_lower": str(scalar_gap) if scalar_gap is not None else None,
        "witness": witness, "certificate": certificate,
        "digest_verified": verify_certificate_digest(certificate),
        **earned, **scope, "theorem_prover_verified": False, "mathlib_verified": False,
    }


def replay_su2_adjacent_vacuum_gap_comparison_certificate(certificate: dict[str, Any]) -> bool:
    """Rebuild the coefficient bound, actual source, both sectors, and all scopes."""
    if not isinstance(certificate, dict):
        return False
    try:
        if not verify_certificate_digest(certificate):
            return False
        payload = certificate["payload"]
        if payload["type"] != "su2_adjacent_vacuum_gap_comparison_v1":
            return False
        witness = payload["witness"]
        result = su2_adjacent_vacuum_gap_comparison(
            witness["source_certificate"], exponent_steps=witness["inputs"]["exponent_steps"],
        )
        return bool(result["certificate"] == certificate)
    except (KeyError, TypeError, ValueError, ZeroDivisionError, OverflowError, IndexError):
        return False


__all__ = [
    "replay_su2_adjacent_vacuum_gap_comparison_certificate",
    "su2_adjacent_vacuum_gap_comparison",
]
