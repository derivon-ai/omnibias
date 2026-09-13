# SPDX-License-Identifier: Apache-2.0
"""Actual exterior-uniform block conditionals of canonical cubic SU(2) vacua."""

from __future__ import annotations

from fractions import Fraction as Q
from typing import Any

from omnibias.core.proof.certificate import make_certificate, verify_certificate_digest
from omnibias.geometry.gauge.transfer.static_sources import _integer
from omnibias.geometry.gauge.transfer.wilson_residual_source import (
    replay_su2_wilson_linear_vacuum_certificate,
    replay_su2_wilson_residual_vacuum_certificate,
)


def _replay_source(certificate: dict[str, Any]) -> bool:
    if not isinstance(certificate, dict):
        return False
    payload = certificate.get("payload")
    if not isinstance(payload, dict):
        return False
    kind = payload.get("type")
    if kind == "su2_wilson_residual_vacuum_v1":
        return replay_su2_wilson_residual_vacuum_certificate(certificate)
    if kind == "su2_wilson_linear_vacuum_v1":
        return replay_su2_wilson_linear_vacuum_certificate(certificate)
    if kind == "su2_wilson_polar_vacuum_v1":
        from omnibias.geometry.gauge.transfer.wilson_polar_source import (
            replay_su2_wilson_polar_vacuum_certificate,
        )

        return replay_su2_wilson_polar_vacuum_certificate(certificate)
    return False


def _exp_floor(omega: Q, requested: int) -> tuple[int, Q]:
    steps = max(requested, omega.numerator // omega.denominator + 1)
    return steps, (1 - omega / steps)**steps


def su2_wilson_conditional_block(
    source_certificate: dict[str, Any], *, block_size: int,
    exponent_steps: int = 4,
) -> dict[str, Any]:
    """Certify all actual conditionals on at most block_size original edges.

    The source must be a canonically replayed cubic Wilson vacuum. The
    coupling, local Fourier correction radius and every source premise are
    inherited. Uniformity is over all boxes in that source family and every
    exterior configuration; no isolated-block spectrum is used.

    Direct Haar comparison is positive for every finite earned radius.
    The separate Schur and original-coordinate curvature routes, when
    positive, apply to blocks of every cardinality. Conditional diffusion
    is distinguished from a Hamiltonian with frozen Wilson terms.
    """
    size, requested = _integer(block_size, "block_size"), _integer(exponent_steps, "exponent_steps")
    if size < 1 or requested < 1:
        raise ValueError("block_size and exponent_steps must be strictly positive integers")
    if not _replay_source(source_certificate):
        raise ValueError("a canonical supported Wilson actual-source certificate is required")
    source = source_certificate["payload"]["witness"]
    if source["inputs"]["family"] != "cubic":
        raise ValueError("only cubic original-edge source coordinates are supported")
    a = source["arithmetic"]
    coupling, g = Q(source["inputs"]["kappa"]), Q(a["g"])
    decay = Q(source["inputs"]["decay_base"])
    radius = Q(a["selected_correction_radius"]) if a["selected_correction_radius"] is not None else None
    actual = bool(
        source_certificate["honesty"].get("actual_vacuum_verified") is True
        and source_certificate["honesty"].get("volume_uniform_actual_vacuum_family_verified") is True
        and a["fixed_point_verified"] is True and radius is not None and radius > 0
    )
    cap, energy_floor = 4, Q(3)
    correction_omega = 8 * size * radius / energy_floor if radius is not None else None
    seed_omega = Q(8, 3) * size * cap * g
    omega_one = Q(8, 3) * (cap * g + radius) if radius is not None else None
    omega_block = size * omega_one if omega_one is not None else None
    mixed = cap * g / 2 + 2 * radius / 3 if radius is not None else None
    curvature = Q(1, 2) - 4 * cap * g / 3 - 4 * radius / 3 if radius is not None else None
    direct_steps: int | None = None
    single_steps: int | None = None
    direct_exp: Q | None = None
    single_exp: Q | None = None
    direct_gap: Q | None = None
    single_gap: Q | None = None
    schur_margin: Q | None = None
    schur_gap: Q | None = None
    curvature_gap: Q | None = None
    best: Q | None = None
    cardinality_independent: Q | None = None
    if actual and omega_block is not None and omega_one is not None and mixed is not None:
        direct_steps, direct_exp = _exp_floor(omega_block, requested)
        single_steps, single_exp = _exp_floor(omega_one, requested)
        direct_gap, single_gap = Q(3, 4) * direct_exp, Q(3, 4) * single_exp
        schur_margin = single_gap - 2 * mixed
        schur_gap = schur_margin if schur_margin > 0 else None
        curvature_gap = curvature if curvature is not None and curvature > 0 else None
        uniform_floors = [v for v in (schur_gap, curvature_gap) if v is not None]
        cardinality_independent = max(uniform_floors) if uniform_floors else None
        best = max([direct_gap, *uniform_floors])
    passed = actual and best is not None and best > 0
    witness = {
        "inputs": {"block_size": size, "exponent_steps": requested},
        "source_certificate": source_certificate,
        "source_type": source_certificate["payload"]["type"],
        "family_scope": source["family"],
        "coordinate_scope": "original unit electric edge product; not a gauge-fixed strip chart",
        "block_quantifier": "every nonempty original-edge subset B with |B|<=block_size in every source-family box",
        "exterior_quantifier": "every exterior configuration z in SU2^(edges outside B), before integration or sampling",
        "actual_conditional": "mu_B^z proportional to exp(2*S_star(x_B,z)+2*U(x_B,z))*Haar_B",
        "reference_conditional": "nu_B^z includes all elementary plaquettes touching B, including crossing boundary plaquettes",
        "correction_source": "complete actual invariant Fourier coefficients with N_b(U)<=r, b>=1 and E_nonzero>=3",
        "local_oscillation_lemma": "osc_B(2U|z)<=4*sum_(X touches B)||F_X||1<=8*|B|*r/3",
        "weighted_neighborhood_tail": "coefficients touching B and of diameter>=R have oscillation<=8*|B|*r/(3*b^R), R>=0",
        "plaquette_count_lemma": "number of distinct elementary plaquettes touching B <=4*|B|; each contributes at most8*g/3",
        "direct_comparison": "product Haar_B scalar gap3/4; bounded-density variance comparison uses the same block gradient and its own conditional means",
        "schur_comparison": "actual one-edge conditional floors gamma and mixed row c give H_ii=gamma,H_ij=-2*c_ij; every conditional principal block retains margin gamma-2*c",
        "curvature_comparison": "original-edge Ricci1/2 and Hessian principal-block restriction; no strip-coordinate curvature substitution",
        "energy_scaling": "multiplication by kappa/2 expresses conditional diffusion floors in microscopic aH units",
        "frozen_hamiltonian_identification": False,
        "exponential_rule": "separate m=max(requested_steps,floor(Omega)+1) for single-edge and block floors; no transcendental backend",
        "arithmetic": {
            "kappa": str(coupling), "g": str(g), "decay_base": str(decay),
            "selected_correction_radius": str(radius) if radius is not None else None,
            "source_actual_vacuum_verified": actual,
            "invariant_electric_floor": str(energy_floor), "plaquettes_per_edge_upper": cap,
            "touching_plaquettes_upper": cap * size,
            "block_correction_log_oscillation_upper": str(correction_omega) if correction_omega is not None else None,
            "block_seed_log_oscillation_upper": str(seed_omega),
            "block_log_density_oscillation_upper": str(omega_block) if omega_block is not None else None,
            "single_edge_log_density_oscillation_upper": str(omega_one) if omega_one is not None else None,
            "unweighted_mixed_hessian_row_upper": str(mixed) if mixed is not None else None,
            "original_coordinate_curvature_lower": str(curvature) if curvature is not None else None,
            "block_effective_exponent_steps": direct_steps,
            "single_edge_effective_exponent_steps": single_steps,
            "block_exp_negative_lower": str(direct_exp) if direct_exp is not None else None,
            "single_edge_exp_negative_lower": str(single_exp) if single_exp is not None else None,
            "direct_block_poincare_lower": str(direct_gap) if direct_gap is not None else None,
            "single_edge_poincare_lower": str(single_gap) if single_gap is not None else None,
            "conditional_schur_margin_lower": str(schur_margin) if schur_margin is not None else None,
            "conditional_schur_poincare_lower": str(schur_gap) if schur_gap is not None else None,
            "conditional_curvature_poincare_lower": str(curvature_gap) if curvature_gap is not None else None,
            "all_cardinalities_conditional_poincare_lower": str(cardinality_independent)
            if cardinality_independent is not None else None,
            "conditional_poincare_lower": str(best) if best is not None else None,
            "conditional_energy_units_lower": str(coupling * best / 2) if best is not None else None,
        },
        "failed_constraints": [] if passed else ["canonical_source_has_no_earned_actual_nonlinear_vacuum"],
    }
    earned = {
        "actual_vacuum_verified": actual,
        "actual_block_conditionals_verified": passed,
        "uniform_over_all_exteriors_verified": passed,
        "volume_uniform_conditional_block_verified": passed,
        "all_cardinalities_conditional_gap_verified": cardinality_independent is not None,
        "conditional_schur_gap_verified": schur_gap is not None,
        "conditional_curvature_gap_verified": curvature_gap is not None,
        "spatial_correction_tail_verified": actual and decay > 1,
    }
    scope = {
        "actual_arbitrary_graph_membership_verified": False,
        "isolated_block_gap_substitution": False, "frozen_wilson_hamiltonian_claim": False,
        "all_scale_refinement_claim": False, "coarse_wilson_family_closed": False,
        "infinite_volume_claim": False, "uniform_in_a_claim": False,
        "continuum_claim": False, "yang_mills_claim": False, "yang_mills_mass_gap_claim": False,
    }
    certificate = make_certificate(
        claim="actual exterior-uniform conditional block Poincare floors from a replayed cubic Wilson vacuum and local Fourier oscillation bounds",
        payload={"type": "su2_wilson_conditional_block_v1", "witness": witness},
        honesty={**earned, **scope},
        meta={"analytic_implication": "docs/api/gauge-wilson-conditional-block.md", "transcend_backend": "not_used"},
    )
    return {
        "status": "PASS" if passed else "INCONCLUSIVE",
        "conditional_poincare_lower": str(best) if best is not None else None,
        "conditional_energy_units_lower": str(coupling * best / 2) if best is not None else None,
        "witness": witness, "certificate": certificate,
        "digest_verified": verify_certificate_digest(certificate),
        **earned, **scope, "theorem_prover_verified": False, "mathlib_verified": False,
    }


def replay_su2_wilson_conditional_block_certificate(certificate: dict[str, Any]) -> bool:
    """Replay the complete actual source, all-exterior bounds and all routes."""
    if not isinstance(certificate, dict):
        return False
    try:
        if not verify_certificate_digest(certificate):
            return False
        payload = certificate["payload"]
        if payload["type"] != "su2_wilson_conditional_block_v1":
            return False
        witness = payload["witness"]
        inputs = witness["inputs"]
        result = su2_wilson_conditional_block(
            witness["source_certificate"], block_size=inputs["block_size"],
            exponent_steps=inputs["exponent_steps"],
        )
        return bool(result["certificate"] == certificate)
    except (KeyError, TypeError, ValueError, ZeroDivisionError, OverflowError, IndexError, ArithmeticError):
        return False


__all__ = [
    "replay_su2_wilson_conditional_block_certificate",
    "su2_wilson_conditional_block",
]

