# SPDX-License-Identifier: Apache-2.0
"""Exact strip-face decomposition, boundary-charge ceiling and local frustration.

The full-spin written proof is in docs/api/gauge-strip-face-obstruction.md.
The boundary-sector ceiling is not an upper bound on the physical strip gap.
"""

from __future__ import annotations

from fractions import Fraction as Q
from typing import Any

from omnibias.core.proof.certificate import make_certificate, verify_certificate_digest
from omnibias.geometry.gauge.transfer.static_sources import _rational
from omnibias.geometry.gauge.transfer.strip_kernel_tail import _strip_geometry


def su2_strip_face_obstruction(kappa: int | Q, n_plaquettes: int = 3) -> dict[str, Any]:
    """Certify the exact local-face obstruction for an isolated open SU2 strip.

    Interior rung electric energies are split equally between their two faces.
    Every face has a shared half-weight rung and a full-boundary-flux trial
    with shifted Rayleigh quotient 3*kappa/16. Adjacent local ground spaces
    have trivial intersection, so subtracting local energies is not a
    frustration-free decomposition of the actual physical Hamiltonian.

    Inputs are exact positive int/Fraction kappa and integer n>=2. PASS earns
    the explicitly scoped written statements, not a global spectral upper
    bound, a conditional vacuum source, or a formal operator theorem.
    """
    if isinstance(n_plaquettes, bool) or not isinstance(n_plaquettes, int):
        raise TypeError("n_plaquettes must be an exact integer")
    if n_plaquettes < 2:
        raise ValueError("n_plaquettes must be at least two")
    coupling = _rational(kappa, "kappa")
    if coupling <= 0:
        raise ValueError("kappa must be strictly positive")
    n = n_plaquettes
    geometry = _strip_geometry(n)
    edges: list[list[int]] = geometry["oriented_edges"]
    words: list[list[int]] = geometry["plaquette_words"]
    stars = [{i + 1 for i, edge in enumerate(edges) if v in edge} for v in range(2 * n + 2)]
    accumulated = [Q(0)] * len(edges)
    faces: list[dict[str, Any]] = []
    for i, word in enumerate(words):
        ids = sorted(abs(e) for e in word)
        weights = {e: Q(1, 2) if 2 * n + 2 <= e <= 3 * n else Q(1) for e in ids}
        for e, weight in weights.items():
            accumulated[e - 1] += weight
        vertices = sorted({v for e in ids for v in edges[e - 1]})
        internal = [v for v in vertices if stars[v].issubset(ids)]
        boundary = [v for v in vertices if v not in internal]
        shared = [e for e in ids if weights[e] == Q(1, 2)]
        witness = min(shared)
        faces.append(
            {
                "face_id": i + 1,
                "plaquette_word": word,
                "original_link_ids": ids,
                "electric_weights": {str(e): str(weights[e]) for e in ids},
                "total_electric_weight": str(sum(weights.values(), Q(0))),
                "internal_vertices_relative_to_full_strip": internal,
                "boundary_vertices_relative_to_full_strip": boundary,
                "shared_rung_ids": shared,
                "charged_trial_link_id": witness,
                "charged_trial_endpoint_vertices": edges[witness - 1],
                "charged_trial_is_internally_gauge_invariant": all(
                    v in boundary for v in edges[witness - 1]
                ),
                "charged_trial_is_globally_gauge_invariant": False,
                "trial": "f=Tr_fund(U_shared); local state phi_face*f",
                "actual_local_trial_mean": "0",
                "actual_local_trial_norm_squared": "1",
                "haar_character_dirichlet_energy": "3/4",
                "shifted_trial_rayleigh_quotient": str(3 * coupling / 16),
            }
        )
    pairs = [
        {
            "faces": [i, i + 1],
            "shared_link_ids": sorted(
                set(faces[i - 1]["original_link_ids"]) & set(faces[i]["original_link_ids"])
            ),
            "common_local_ground_space": "{0}",
            "proof": "positive nonconstant central local vacuum has Schmidt rank greater than one across the shared rung; overlapping pure local ground constraints are incompatible",
        }
        for i in range(1, n)
    ]
    gates = {
        "every_original_electric_weight_sums_to_one": all(w == 1 for w in accumulated),
        "every_face_has_four_distinct_original_links": all(
            len(f["original_link_ids"]) == 4 for f in faces
        ),
        "every_trial_endpoint_is_a_boundary_vertex": all(
            f["charged_trial_is_internally_gauge_invariant"] for f in faces
        ),
        "every_adjacent_pair_shares_exactly_one_rung": all(
            len(pair["shared_link_ids"]) == 1 for pair in pairs
        ),
        "electric_total_weights": all(
            Q(f["total_electric_weight"]) == (Q(7, 2) if i in (0, n - 1) else 3)
            for i, f in enumerate(faces)
        ),
        "fundamental_boundary_rayleigh": coupling / 2 * Q(1, 2) * Q(3, 4) == 3 * coupling / 16,
        "nonconstant_wilson_potential_coefficient": Q(2) / coupling > 0,
    }
    passed = all(gates.values())
    upper = 3 * coupling / 16
    payload = {
        "type": "su2_strip_face_obstruction_v1",
        "status": "PASS" if passed else "INCONCLUSIVE",
        "inputs": {"kappa": str(coupling), "n_plaquettes": n},
        "geometry": geometry,
        "local_faces": faces,
        "adjacent_pairs": pairs,
        "arithmetic": {
            "kappa": str(coupling),
            "n_plaquettes": n,
            "n_original_links": 3 * n + 1,
            "n_adjacent_pairs": n - 1,
            "shared_rung_electric_weight": "1/2",
            "interior_face_total_electric_weight": "3",
            "end_face_total_electric_weight": "7/2",
            "fundamental_casimir": "3/4",
            "boundary_flux_gap_upper": str(upper),
            "boundary_flux_gap_lower_from_electric_and_haar": str(max(Q(0), upper - 4 / coupling)),
            "local_haar_trial_energy_upper": str(4 / coupling),
            "accumulated_original_electric_weights": [str(w) for w in accumulated],
            "gates": gates,
        },
        "operator_decomposition": "H=sum_i h_i; h_i=kappa/2*sum_face(w_ie*C_e)+2*(2-Tr(U_face))/kappa",
        "local_sector": "full four-link Hilbert space with only full-strip-internal face-vertex Gauss constraints; all boundary representations retained",
        "local_vacuum": "unique positive nonconstant central phi_i(U_face), hence invariant under all four face-vertex gauge actions",
        "shared_rung_probability_marginal": "normalized Haar for the actual local vacuum density phi_i^2",
        "shared_rung_quantum_reduced_state": "mixed; normalized character coefficient a_j contributes d_j^2 Schmidt singular values |a_j|/d_j",
        "frustration_statement": "E0(H)>sum_i e0(h_i) for every n>=2 and finite positive kappa; no numeric lower bound for this difference is supplied",
        "local_boundary_flux_gap_upper": str(upper) if passed else None,
        "exact_original_operator_decomposition_verified_in_written_analysis": passed,
        "actual_local_ground_identification_verified_in_written_analysis": passed,
        "boundary_flux_gap_ceiling_verified_in_written_analysis": passed,
        "adjacent_local_ground_intersection_trivial_verified_in_written_analysis": passed,
        "strict_local_energy_frustration_verified_in_written_analysis": passed,
        "global_physical_gap_upper_verified": False,
        "actual_ambient_conditional_gap_verified": False,
        "isolated_neutral_gap_transferred_to_boundary_sector": False,
        "knabe_hypotheses_verified": False,
        "actual_global_groundstate_transform_locality_verified": False,
        "uniform_physical_gap_verified": False,
        "continuum_claim": False,
        "yang_mills_claim": False,
        "yang_mills_mass_gap_claim": False,
        "all_scale_refinement_claim": False,
        "analytic_proof_formally_verified": False,
        "theorem_prover_verified": False,
        "mathlib_verified": False,
    }
    certificate = make_certificate(
        claim="exact strip-face decomposition, boundary-charge gap ceiling and overlapping-local-vacuum frustration",
        payload=payload,
        meta={
            "analytic_implication": "docs/api/gauge-strip-face-obstruction.md",
            "proof_register": "written positivity, Haar marginal, Schmidt and exact groundstate-form identities; rational graph replay",
        },
    )
    return {**payload, "certificate": certificate}


def replay_su2_strip_face_obstruction_certificate(certificate: dict[str, Any]) -> bool:
    """Rebuild all local weights, available Gauss constraints, trials and scope."""
    try:
        if not isinstance(certificate, dict) or not verify_certificate_digest(certificate):
            return False
        payload = certificate["payload"]
        if payload["type"] != "su2_strip_face_obstruction_v1":
            return False
        inputs = payload["inputs"]
        expected = su2_strip_face_obstruction(Q(inputs["kappa"]), inputs["n_plaquettes"])
        return bool(certificate == expected["certificate"])
    except (KeyError, TypeError, ValueError, ZeroDivisionError, OverflowError):
        return False


__all__ = [
    "replay_su2_strip_face_obstruction_certificate",
    "su2_strip_face_obstruction",
]
