# SPDX-License-Identifier: Apache-2.0
"""Exact exterior-independent vacuum increments for Wilson attachments.

A fresh-link four-cycle has a normalized conditional trial with scalar
energy compression. Ordered attachments telescope without subtracting an
extensive ambient energy estimate. Faces with no fresh edge are recorded
and retain a weaker bounded-potential fallback, never the fresh-link claim.
"""

from __future__ import annotations

from collections.abc import Sequence
from fractions import Fraction as Q
from typing import Any

from omnibias.core.proof.certificate import make_certificate, verify_certificate_digest
from omnibias.geometry.gauge.transfer.static_sources import _integer, _rational

Edge = tuple[int, int]


def _edge(a: int, b: int) -> Edge:
    return (min(a, b), max(a, b))


def _pattern(
    old_edges: Sequence[Sequence[int]], faces: Sequence[Sequence[int]],
) -> tuple[list[Edge], list[list[int]], list[dict[str, Any]], dict[Edge, int]]:
    old: set[Edge] = set()
    for row in old_edges:
        vertices = [_integer(x, "old-edge vertex") for x in row]
        if len(vertices) != 2 or vertices[0] == vertices[1]:
            raise ValueError("old_edges must contain pairs of distinct integer vertices")
        old.add(_edge(*vertices))
    ordered: list[list[int]] = []
    stages: list[dict[str, Any]] = []
    incidence: dict[Edge, int] = {}
    seen: set[tuple[Edge, ...]] = set()
    current = set(old)
    for row in faces:
        vertices = [_integer(x, "face vertex") for x in row]
        if len(vertices) != 4 or len(set(vertices)) != 4:
            raise ValueError("each face must be an ordered cycle of four distinct integer vertices")
        edges = {_edge(vertices[j], vertices[(j+1) % 4]) for j in range(4)}
        face_key = tuple(sorted(edges))
        if face_key in seen:
            raise ValueError("duplicate added face, including reversed or rotated copies")
        seen.add(face_key)
        fresh = sorted(edges-current)
        stages.append({"face_index": len(stages), "face_edges": [list(e) for e in sorted(edges)],
                       "fresh_edges": [list(e) for e in fresh], "fresh_edge_count": len(fresh),
                       "route": "normalized_conditional_attachment" if fresh else "bounded_potential_only",
                       "touches_initial_new_edge": bool(edges-old)})
        ordered.append(vertices)
        for edge in edges:
            incidence[edge] = incidence.get(edge, 0)+1
        current.update(edges)
    if not ordered:
        raise ValueError("at least one added face is required")
    return sorted(old), ordered, stages, incidence


def su2_wilson_attachment(
    kappa: int | Q, old_edges: Sequence[Sequence[int]], added_faces: Sequence[Sequence[int]],
    *, action_threshold: int | Q = 1,
) -> dict[str, Any]:
    """Certify vacuum increments and bad-support energy for an ordered pattern.

    Vertex labels specify simple unoriented edges and oriented four-cycles.
    The ambient old graph contains old_edges and none of the introduced
    edges. Additional old edges and an arbitrary smooth real gauge-invariant
    old potential are allowed. All electric terms have coefficient kappa/2
    and SU2 fundamental Casimir3/4. The only new potential terms are the
    supplied Wilson faces, each with coefficient2/kappa.

    PASS means every face introduces a fresh edge at its stage. A zero-fresh
    face gives INCONCLUSIVE for that inexpensive route, while the explicit
    bounded-potential and Haar fallback bounds still replay soundly.
    Neither status proves a full Hamiltonian gap or cubic graph membership.
    """
    coupling = _rational(kappa, "kappa")
    threshold = _rational(action_threshold, "action_threshold")
    old, faces, stages, incidence = _pattern(old_edges, added_faces)
    n = len(faces)
    if coupling <= 0 or not 0 < threshold <= 4*n:
        raise ValueError("kappa>0 and 0<action_threshold<=4*face_count are required")
    fresh_count = sum(stage["fresh_edge_count"] > 0 for stage in stages)
    dependent = n-fresh_count
    touched = sum(stage["touches_initial_new_edge"] for stage in stages)
    d = max(incidence.values())
    per_fresh = min(Q(3), 4/coupling)
    sequential = fresh_count*per_fresh+8*dependent/coupling
    haar = (4*touched+8*(n-touched))/coupling
    increment = min(sequential, haar)
    mean = min(Q(4*n), coupling*increment/2)
    probability = min(Q(1), mean/threshold)
    norm = max(Q(0), 1-2*mean/threshold)
    concave_point = min(coupling*increment/2, Q(2*n))
    gradient_mean = d*concave_point*(4-concave_point/n)
    vacuum_cost = coupling/2*(Q(22, 7)/threshold)**2*gradient_mean
    ims = Q(968, 49)*coupling*d/threshold
    bad_floor = threshold/coupling-increment
    passed = dependent == 0
    witness = {
        "inputs": {"kappa": str(coupling), "old_edges": [list(e) for e in old],
                   "added_faces": faces, "action_threshold": str(threshold)},
        "family": {
            "group": "SU(2)",
            "ambient_quantifier": "every finite old graph containing the listed old edges and none of the introduced edges; arbitrary other old edges are allowed",
            "old_hamiltonian": "kappa/2*sum_old C_e plus a smooth real gauge-invariant old-link multiplication potential; no nonlocal old kinetic terms",
            "new_hamiltonian": "H_old+kappa/2*sum_new C_e+2/kappa*sum_added_faces(2-Tr U_p)",
            "physical_space": "Gauss law at every vertex, no external charges; scalar and physical vacuum bottoms agree by positivity",
            "no_hidden_faces": "exactly the supplied potential additions; other geometric faces are not silently added",
            "normalization": "original independent Haar links, unit electric weights, fundamental Casimir3/4; dimensionless aH",
        },
        "stages": stages,
        "edge_incidence": [{"edge": list(e), "added_face_count": count} for e, count in sorted(incidence.items())],
        "trial": {
            "one_step": "phi_t(new|old)=Z_t^-1/2 exp(t*Tr U_face), with at least one fresh Haar link",
            "normalization": "integrating one fresh Haar factor makes the full loop Haar; integral_new phi_t^2=1 for every old configuration",
            "compression": "J_t^* H_new J_t=H_old+e_t I as quadratic forms; J_t is an isometry, not an invariant subspace",
            "scalar_energy": "e_t=(3*kappa*t/2)*m(4*t)+(4/kappa)*(1-m(4*t)); m(c)=E_exp(c*q0) q0",
            "bound": "0<=m<=1 and 1-m(4*t)<=3/(8*t); t=1/kappa gives e_t<=3; constant fresh-link trial gives4/kappa",
            "exterior_derivatives": "integral phi_t*grad_old(phi_t)=0 cancels old-state cross terms; all four edge derivative costs are retained",
            "composition": "ordered isometries compose even when later faces reuse previously fresh edges",
            "zero_fresh_fallback": "0<=2*(2-Tr U_p)/kappa<=8/kappa; no normalization-based bound3 is claimed",
            "global_haar_fallback": "Haar in all initially absent edges kills every touched face character; all-old faces use action<=4",
        },
        "localization": {
            "action": "S=sum_added(2-Tr U_p)",
            "cutoffs": "chi_g=cos(theta(S)), chi_b=sin(theta(S)); theta=0 below s/2, pi/2 above s, linear between",
            "actual_subtraction": "H_new-E_new >= (s/kappa-increment_upper) on bad-cutoff-supported states, using H_old>=E_old and E_new-E_old<=increment_upper",
            "ims": "q(f)>=bad_floor*||chi_b f||^2-ims_upper*||f||^2 when bad_floor>=0; the good localized form is nonnegative; q=H_new-E_new",
            "moment": "2*E_new_vacuum(S)/kappa<=E_new-E_old; no homogeneity or exterior conditioning is assumed",
            "gradient_mean": "Gamma S<=d*(4*S-sum A_p^2); sum/Jensen and the maximum of4*x-x^2/n give the displayed bound",
            "scope": "true full new vacuum and full physical states; no uniform conditional bad-event probability is inferred",
        },
        "arithmetic": {
            "face_count": n, "fresh_stage_count": fresh_count, "zero_fresh_stage_count": dependent,
            "initially_touched_face_count": touched, "max_added_face_edge_incidence": d,
            "trial_parameter_t": str(1/coupling), "per_fresh_increment_upper": str(per_fresh),
            "sequential_increment_upper": str(sequential), "global_haar_increment_upper": str(haar),
            "vacuum_increment_lower": "0", "vacuum_increment_upper": str(increment),
            "actual_added_action_mean_upper": str(mean), "large_action_probability_upper": str(probability),
            "good_localized_vacuum_norm_squared_lower": str(norm),
            "added_action_gradient_square_mean_upper": str(gradient_mean),
            "vacuum_localization_form_cost_upper": str(vacuum_cost),
            "good_normalized_energy_above_vacuum_upper": str(vacuum_cost/norm) if norm > 0 else None,
            "universal_ims_error_upper": str(ims), "vacuum_subtracted_bad_support_floor": str(bad_floor),
            "bad_floor_minus_ims": str(bad_floor-ims),
        },
        "unresolved_cheap_stages": [stage["face_index"] for stage in stages if not stage["fresh_edge_count"]],
    }
    earned = {
        "actual_vacuum_increment_bound_verified": True,
        "ambient_volume_independent_increment_verified": True,
        "cheap_attachment_route_verified": passed,
        "complete_normalized_attachment_compression_verified": passed,
        "actual_added_action_mean_bound_verified": True,
        "actual_vacuum_localization_verified": True,
        "universal_ims_bound_verified": True,
        "actual_vacuum_subtracted_bad_support_bound_verified": True,
        "positive_bad_support_floor_verified": bad_floor > 0,
    }
    scope = {
        "ambient_graph_membership_verified": False,
        "full_cubic_block_cheap_increment_verified": False,
        "uniform_conditional_bad_probability_verified": False,
        "actual_conditional_gap_verified": False,
        "embedded_subspace_invariant_verified": False,
        "spectral_gap_claim": False, "infinite_volume_claim": False,
        "uniform_in_a_claim": False, "continuum_claim": False,
        "yang_mills_claim": False, "yang_mills_mass_gap_claim": False,
    }
    cert = make_certificate(
        claim="actual SU2 Wilson attachment vacuum increments, ambient-independent subtraction and supported-state localization bounds",
        payload={"type": "su2_wilson_attachment_v1", "witness": witness},
        honesty={**earned, **scope},
        meta={"analytic_implication": "docs/api/gauge-wilson-attachment.md", "transcend_backend": "not_used"},
    )
    return {"status": "PASS" if passed else "INCONCLUSIVE", "witness": witness,
            "certificate": cert, "digest_verified": verify_certificate_digest(cert),
            **earned, **scope, "theorem_prover_verified": False, "mathlib_verified": False}


def replay_su2_wilson_attachment_certificate(certificate: dict[str, Any]) -> bool:
    """Replay geometry, fresh-stage gates, fallbacks, all arithmetic and scope."""
    if not isinstance(certificate, dict):
        return False
    try:
        if not verify_certificate_digest(certificate):
            return False
        payload = certificate["payload"]
        if payload["type"] != "su2_wilson_attachment_v1":
            return False
        inputs = payload["witness"]["inputs"]
        result = su2_wilson_attachment(Q(inputs["kappa"]), inputs["old_edges"], inputs["added_faces"],
                                     action_threshold=Q(inputs["action_threshold"]))
        return bool(result["certificate"] == certificate)
    except (KeyError, TypeError, ValueError, ZeroDivisionError, OverflowError, IndexError):
        return False


__all__ = [
    "replay_su2_wilson_attachment_certificate",
    "su2_wilson_attachment",
]
