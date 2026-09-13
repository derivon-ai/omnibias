# SPDX-License-Identifier: Apache-2.0
"""Actual theta vacuum using directional inverse weights and spherical products."""
from __future__ import annotations

from fractions import Fraction as Q
from typing import Any

from omnibias.core.proof.certificate import make_certificate, verify_certificate_digest
from omnibias.geometry.gauge.transfer.adjacent_cone_inverse import (
    replay_su2_adjacent_cone_inverse_certificate,
    su2_adjacent_cone_inverse,
)
from omnibias.geometry.gauge.transfer.adjacent_vacuum import (
    replay_su2_adjacent_preconditioned_vacuum_certificate,
    su2_adjacent_preconditioned_vacuum,
)
from omnibias.geometry.gauge.transfer.static_sources import _integer, _rational


def su2_adjacent_cone_vacuum(
    kappa: int | Q, *, cutoff: int = 3, correction_radius: int | Q | None = None,
) -> dict[str, Any]:
    """Recompute the source enclosure, sharper inverse and all-spin nonlinear gate.

    PASS means the actual vacuum is constructed. Its curvature gate is
    independent and can fail; a separate actual-measure comparison can then
    establish a physical spectral floor.
    """
    coupling, size = _rational(kappa, "kappa"), _integer(cutoff, "cutoff")
    supplied = None if correction_radius is None else _rational(correction_radius, "correction_radius")
    if coupling <= 0 or size < 2 or (supplied is not None and supplied <= 0):
        raise ValueError("require kappa>0, cutoff>=2 and positive optional correction_radius")
    # The older source's residual enclosure remains valid even when its
    # sufficient B=4/3 radius criterion fails. It is replayed in full.
    residual = su2_adjacent_preconditioned_vacuum(coupling, cutoff=size)
    inverse = su2_adjacent_cone_inverse(coupling, cutoff=size)
    residual_replayed = replay_su2_adjacent_preconditioned_vacuum_certificate(residual["certificate"])
    inverse_replayed = replay_su2_adjacent_cone_inverse_certificate(inverse["certificate"])
    source = residual["witness"]
    arithmetic = source["arithmetic"]
    parents_match = (source["source_inverse_certificate"] ==
                     inverse["witness"]["source_inverse_certificate"])
    ready = bool(residual_replayed and inverse_replayed and parents_match
                 and residual["preconditioned_residual_verified"]
                 and inverse["full_spin_fourier_reference_inverse_verified"])
    g, quadratic = 4 / coupling**2, Q(8, 9)
    j = Q(inverse["original_N_inverse_upper"]) if ready else None
    epsilon = Q(arithmetic["preconditioned_residual_N_upper"]) if ready else None
    beta = j * quadratic if j is not None else None
    discriminant = 1 - 4 * beta * epsilon if beta is not None and epsilon is not None else None
    radius = supplied
    if radius is None and discriminant is not None and discriminant > 0 and epsilon is not None:
        radius = 2 * epsilon
    slack = radius - epsilon - beta * radius**2 if radius is not None and epsilon is not None and beta is not None else None
    contraction = 2 * beta * radius if beta is not None and radius is not None else None
    actual = bool(ready and slack is not None and slack >= 0
                  and contraction is not None and contraction < 1)
    curvature = Q(1, 2) - 8 * g / 3 - 4 * radius / 3 if radius is not None else None
    curvature_passed = actual and curvature is not None and curvature > 0
    gap = coupling * curvature / 2 if curvature_passed and curvature is not None else None
    witness = {
        "inputs": {"kappa": str(coupling), "cutoff": size,
                   "correction_radius": str(supplied) if supplied is not None else None},
        "graph": source["graph"], "normalization": source["normalization"],
        "reference": source["reference"],
        "residual_source_certificate": residual["certificate"],
        "cone_inverse_certificate": inverse["certificate"],
        "nonlinear_map": "U=L^-1*R_star+L^-1*T(U,U) in the same complete original theta N",
        "spherical_product_lemma": (
            "normalized trivalent spherical products have nonnegative squared overlap coefficients of mass1; "
            "their mean output Casimir is E_in1+E_in2, so E|Gamma|<=2*sum_path length*j*k"
        ),
        "bilinear_bound": "N(T(U,V))<=8*N(U)*N(V)/9, all theta spins and signed or complex coefficients",
        "actual_vacuum_implication": "the complete contracting log-vacuum solution identifies the positive groundstate of the original seven-edge Hamiltonian",
        "curvature_is_separate": True,
        "arithmetic": {
            "g": str(g), "residual_source_replayed": residual_replayed,
            "cone_inverse_replayed": inverse_replayed, "reference_parents_identical": parents_match,
            "preconditioned_residual_N_upper": str(epsilon) if epsilon is not None else None,
            "original_N_inverse_upper": str(j) if j is not None else None,
            "quadratic_constant": str(quadratic),
            "preconditioned_quadratic_upper": str(beta) if beta is not None else None,
            "radius_feasibility_discriminant": str(discriminant) if discriminant is not None else None,
            "selected_correction_radius": str(radius) if radius is not None else None,
            "self_map_slack": str(slack) if slack is not None else None,
            "contraction_upper": str(contraction) if contraction is not None else None,
            "fixed_point_verified": actual, "curvature_lower": str(curvature) if curvature is not None else None,
            "curvature_physical_gap_lower": str(gap) if gap is not None else None,
        },
        "failed_constraints": [] if actual else ["complete_preconditioned_nonlinear_radius"],
    }
    earned = {"actual_vacuum_verified": actual, "all_spin_actual_vacuum_verified": actual,
              "curvature_gap_verified": curvature_passed}
    scope = {"uniform_in_volume_claim": False, "infinite_volume_claim": False,
             "uniform_in_a_claim": False, "all_scale_refinement_claim": False,
             "continuum_claim": False, "yang_mills_claim": False, "yang_mills_mass_gap_claim": False}
    certificate = make_certificate(
        claim="actual full-spin vacuum on the fixed adjacent-square graph from directional dual weights and spherical products",
        payload={"type": "su2_adjacent_cone_vacuum_v1", "witness": witness},
        honesty={**earned, **scope},
        meta={"analytic_implication": "docs/api/gauge-adjacent-cone-vacuum.md", "transcend_backend": "not_used"},
    )
    return {"status": "PASS" if actual else "INCONCLUSIVE",
            "physical_gap_lower": str(gap) if gap is not None else None,
            "witness": witness, "certificate": certificate,
            "digest_verified": verify_certificate_digest(certificate),
            **earned, **scope, "theorem_prover_verified": False, "mathlib_verified": False}


def replay_su2_adjacent_cone_vacuum_certificate(certificate: dict[str, Any]) -> bool:
    if not isinstance(certificate, dict):
        return False
    try:
        if not verify_certificate_digest(certificate):
            return False
        payload = certificate["payload"]
        if payload["type"] != "su2_adjacent_cone_vacuum_v1":
            return False
        inputs = payload["witness"]["inputs"]
        radius = inputs["correction_radius"]
        result = su2_adjacent_cone_vacuum(Q(inputs["kappa"]), cutoff=inputs["cutoff"],
                                         correction_radius=None if radius is None else Q(radius))
        return bool(result["certificate"] == certificate)
    except (KeyError, TypeError, ValueError, ZeroDivisionError, OverflowError, IndexError, ArithmeticError):
        return False


__all__ = [
    "replay_su2_adjacent_cone_vacuum_certificate",
    "su2_adjacent_cone_vacuum",
]
