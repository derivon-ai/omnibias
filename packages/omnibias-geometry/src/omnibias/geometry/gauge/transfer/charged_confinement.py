# SPDX-License-Identifier: Apache-2.0
"""Actual static-source confinement bounds from verified SU(2) vacuum curvature.

A source-separating center flip makes the charged field conditionally odd
on its edge cut. Conditional curvature supplies a Poincare constant independent
of cut cardinality. Edge-disjoint BFS cuts then give a linear lower bound with
actual vacuum subtraction and all spins included. The structural-family claim
is at fixed strong coupling; no continuum or asymptotic tension limit is claimed.
"""

from __future__ import annotations

from collections import deque
from fractions import Fraction as Q
from typing import Any

from omnibias.core.proof.certificate import make_certificate, verify_certificate_digest
from omnibias.geometry.gauge.transfer.static_sources import Edge, _graph, _integer
from omnibias.geometry.gauge.transfer.vacuum_fourier import replay_su2_vacuum_fourier_certificate


def _upstream(certificate: dict[str, Any], expected_type: str) -> dict[str, Any]:
    if not replay_su2_vacuum_fourier_certificate(certificate):
        raise ValueError("a canonical passing Fourier certificate is required")
    payload = certificate["payload"]
    if payload["type"] != expected_type:
        raise ValueError("the Fourier certificate has the wrong graph/family scope")
    witness: dict[str, Any] = payload["witness"]
    return witness


def _distance_cuts(n: int, graph: tuple[Edge, ...], source: int, target: int) -> dict[str, Any]:
    s, t = _integer(source, "source"), _integer(target, "target")
    if not (0 <= s < n and 0 <= t < n) or s == t:
        raise ValueError("source and target must be distinct vertices in range")
    adjacency: list[list[tuple[int, int]]] = [[] for _ in range(n)]
    for token, (u, v) in enumerate(graph, 1):
        adjacency[u].append((v, token))
        adjacency[v].append((u, -token))
    distances = [-1] * n
    predecessors: list[tuple[int, int] | None] = [None] * n
    distances[s] = 0
    queue = deque([s])
    while queue:
        u = queue.popleft()
        for v, token in adjacency[u]:
            if distances[v] < 0:
                distances[v] = distances[u] + 1
                predecessors[v] = (u, token)
                queue.append(v)
    if any(level < 0 for level in distances):
        raise ValueError("the static confinement graph must be connected")
    path: list[int] = []
    v = t
    while v != s:
        previous = predecessors[v]
        assert previous is not None
        v, token = previous
        path.append(token)
    path.reverse()
    cuts = []
    multiplicity = [0] * len(graph)
    for k in range(distances[t]):
        inside = {v for v, level in enumerate(distances) if level <= k}
        crossing = [i for i, (u, v) in enumerate(graph, 1) if (u in inside) != (v in inside)]
        for token in crossing:
            multiplicity[token - 1] += 1
        parity = (int(s in inside) + int(t in inside)) % 2
        cuts.append(
            {
                "level": k,
                "vertices": sorted(inside),
                "edges": crossing,
                "external_fundamental_source_parity": parity,
                "charged_center_character": -1 if parity else 1,
                "outside_cut_links_fixed_by_center_gauge_transform": True,
            }
        )
    verified = (
        len(path) == distances[t]
        and all(abs(distances[u] - distances[v]) <= 1 for u, v in graph)
        and all(count <= 1 for count in multiplicity)
        and all(cut["edges"] and cut["external_fundamental_source_parity"] == 1 for cut in cuts)
    )
    return {
        "distance_labels": distances,
        "graph_distance": distances[t],
        "path": path,
        "path_two_spins": [int(i + 1 in {abs(token) for token in path}) for i in range(len(graph))],
        "cuts": cuts,
        "edge_cut_multiplicity": multiplicity,
        "cuts_verified": verified,
    }


def _report(witness: dict[str, Any], *, family: bool) -> dict[str, Any]:
    passed = bool(witness["linear_bounds_positive_and_ordered"]) and (
        family or bool(witness["cut_witness"]["cuts_verified"])
    )
    family_passed = passed and (family or bool(witness["upstream_structural_family_verified"]))
    kind = "su2_static_confinement_family_v1" if family else "su2_static_confinement_graph_v1"
    certificate = make_certificate(
        claim="linear all-spin static-source energy bounds from actual SU(2) vacuum conditional curvature",
        payload={"type": kind, "witness": witness},
        honesty={
            "finite_graph_family_confinement": family_passed,
            "yang_mills_claim": False,
            "continuum_claim": False,
            "asymptotic_string_tension_claim": False,
        },
        meta={
            "transcend_backend": "not_used",
            "analytic_implication": "docs/api/gauge-charged-confinement.md",
            "scope": "fixed-coupling finite-graph static sources with actual vacuum subtraction",
        },
    )
    return {
        "status": "PASS" if passed else "INCONCLUSIVE",
        "finite_gate_verified": passed,
        "verification_kind": "EXACT_RATIONAL_WITH_WRITTEN_ANALYTIC_IMPLICATION",
        "witness": witness,
        "certificate": certificate,
        "digest_verified": verify_certificate_digest(certificate),
        "finite_graph_static_confinement": passed and not family,
        "finite_graph_family_confinement": family_passed,
        "infinite_volume_static_potential_claim": False,
        "asymptotic_string_tension_claim": False,
        "string_tension_limit_claim": False,
        "infinite_volume_claim": False,
        "uniform_in_a_claim": False,
        "continuum_claim": False,
        "yang_mills_claim": False,
        "theorem_prover_verified": False,
        "mathlib_verified": False,
        "analytic_implication_formally_verified": False,
    }


def su2_static_confinement_bounds(
    fourier_graph_certificate: dict[str, Any], source: int, target: int
) -> dict[str, Any]:
    """Derive a matched static-energy enclosure on the replayed connected graph.

    No supplied curvature or honesty flag is accepted. Both source vertices
    obey fundamental Gauss law, every other vertex is a singlet, and no matter
    or extra external-source rest energy is added. The upper trial is a
    normalized fundamental transport along the derived shortest path.
    """
    upstream = _upstream(fourier_graph_certificate, "su2_vacuum_fourier_graph_v1")
    n = upstream["n_vertices"]
    graph = _graph(n, [tuple(edge) for edge in upstream["edges"]])
    cut_witness = _distance_cuts(n, graph, source, target)
    coupling = Q(upstream["kappa"])
    radius = Q(upstream["radius"])
    rho = Q(1, 2) - Q(4, 3) * radius
    distance = cut_witness["graph_distance"]
    lower, upper = coupling * rho * distance / 2, 3 * coupling * distance / 8
    witness = {
        "model": "su2_static_fundamental_pair_from_actual_vacuum_fourier_curvature",
        "normalization": upstream["normalization"],
        "energy_units": "dimensionless aH",
        "kappa": str(coupling),
        "source": source,
        "target": target,
        "external_source_representations": "fundamental at source, antifundamental at target",
        "gauss_law": "source covariance and singlet constraint at every other vertex",
        "dynamical_matter": False,
        "source_bare_rest_energy": "0",
        "n_vertices": n,
        "edges": upstream["edges"],
        "fourier_certificate": fourier_graph_certificate,
        "fourier_radius": str(radius),
        "conditional_curvature_lower": str(rho),
        "conditional_cut_poincare_floor": str(rho),
        "conditional_cut_mean_zero_reason": "actual vacuum cut-center invariance and odd external source parity",
        "graph_distance": distance,
        "path": cut_witness["path"],
        "path_trial_pointwise_HS_norm_squared": "1",
        "path_trial_total_casimir": str(Q(3 * len(cut_witness["path"]), 4)),
        "cut_witness": cut_witness,
        "static_energy_enclosure": [str(lower), str(upper)],
        "linear_energy_coefficients": [str(coupling * rho / 2), str(3 * coupling / 8)],
        "linear_bounds_positive_and_ordered": 0 < lower <= upper,
        "all_spins_included": True,
        "vacuum_subtraction": "groundstate Dirichlet transform by the actual positive neutral vacuum",
        "upstream_structural_family_verified": bool(
            upstream["structural_caps_verified"]
            and upstream["family"]["arithmetic"]["fixed_point_verified"]
        ),
    }
    return _report(witness, family=False)


def su2_static_confinement_family(fourier_family_certificate: dict[str, Any]) -> dict[str, Any]:
    """Derive linear static bounds for every connected graph in a Fourier family.

    Source placement is arbitrary at two distinct vertices. This is a theorem
    about finite graphs with fixed coupling and the upstream structural caps;
    it does not assert an infinite-volume or asymptotic string-tension limit.
    """
    upstream = _upstream(fourier_family_certificate, "su2_vacuum_fourier_family_v1")
    coupling, radius = Q(upstream["kappa"]), Q(upstream["radius"])
    rho = Q(1, 2) - Q(4, 3) * radius
    witness = {
        "model": "su2_static_fundamental_pair_from_actual_vacuum_fourier_curvature",
        "normalization": upstream["normalization"],
        "energy_units": "dimensionless aH",
        "kappa": str(coupling),
        "fourier_certificate": fourier_family_certificate,
        "family_class": "every connected graph in the upstream structural family, with arbitrary distinct source vertices",
        "upstream_structural_class": upstream["family_class"],
        "gauss_law": "one fundamental and one antifundamental static source; singlet at every other vertex",
        "dynamical_matter": False,
        "source_bare_rest_energy": "0",
        "electric_weights": "one on every edge",
        "fourier_radius": str(radius),
        "conditional_curvature_lower": str(rho),
        "linear_energy_coefficients": [str(coupling * rho / 2), str(3 * coupling / 8)],
        "linear_bounds_positive_and_ordered": 0 < coupling * rho / 2 <= 3 * coupling / 8,
        "distance_convention": "ordinary undirected graph distance between the source vertices",
        "quantifier": "for every graph in the connected subclass and every pair of distinct source vertices",
        "all_spins_included": True,
        "vacuum_subtraction": "groundstate Dirichlet transform by the actual positive neutral vacuum",
    }
    return _report(witness, family=True)


def replay_su2_static_confinement_certificate(certificate: dict[str, Any]) -> bool:
    """Accept only canonical passing certificates, replaying the full Fourier parent."""
    if not isinstance(certificate, dict):
        return False
    try:
        if not verify_certificate_digest(certificate):
            return False
        payload = certificate["payload"]
        w = payload["witness"]
        if payload["type"] == "su2_static_confinement_graph_v1":
            report = su2_static_confinement_bounds(
                w["fourier_certificate"], w["source"], w["target"]
            )
        elif payload["type"] == "su2_static_confinement_family_v1":
            report = su2_static_confinement_family(w["fourier_certificate"])
        else:
            return False
        return bool(report["finite_gate_verified"] and report["certificate"] == certificate)
    except (KeyError, TypeError, ValueError, ZeroDivisionError, OverflowError, IndexError):
        return False
