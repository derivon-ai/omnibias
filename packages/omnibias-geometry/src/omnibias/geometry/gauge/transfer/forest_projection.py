# SPDX-License-Identifier: Apache-2.0
"""Exact relative-forest transitivity and conditional projection criteria.

This checker verifies graph topology, not a measure or vacuum. See
docs/api/gauge-forest-projection.md for the conditional equivariance proof.
Available gauge transformations are independent at every nonfixed vertex.
Observed edge IDs are positive and one-based; vertex IDs are zero-based.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Sequence
from copy import deepcopy
from typing import Any

from omnibias.core.proof.certificate import make_certificate, verify_certificate_digest

Edge = tuple[int, int]
_KIND = "gauge_forest_projection_v1"
_SCOPE = {
    "actual_vacuum_verified": False,
    "actual_measure_verified": False,
    "input_gauge_invariance_verified": False,
    "physical_gap_verified": False,
    "uniform_in_volume_claim": False,
    "arbitrary_exterior_graph_claim": False,
    "all_scale_refinement_claim": False,
    "continuum_claim": False,
    "yang_mills_claim": False,
    "yang_mills_mass_gap_claim": False,
    "analytic_proof_formally_verified": False,
}


def _integer(value: object, name: str) -> int:
    if type(value) is not int:
        raise TypeError(f"{name} must be an exact Python integer, not a bool or float")
    return value


def _sequence(value: object, name: str) -> Sequence[Any]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise TypeError(f"{name} must be a sequence")
    return value


def _ids(values: Sequence[int], name: str, lower: int, upper: int) -> tuple[int, ...]:
    result = tuple(_integer(value, name) for value in _sequence(values, name))
    if any(not lower <= value <= upper for value in result):
        raise ValueError(f"{name} contains an out-of-range ID")
    if len(set(result)) != len(result):
        raise ValueError(f"{name} must not contain duplicate IDs")
    return tuple(sorted(result))


def _tree_path(
    start: int,
    end: int,
    adjacency: dict[int, list[tuple[int, int]]],
) -> list[int]:
    """Return oriented tree-edge IDs; adjacency stores (neighbor,signedID)."""
    queue = deque([start])
    predecessor: dict[int, tuple[int, int] | None] = {start: None}
    while queue and end not in predecessor:
        vertex = queue.popleft()
        for other, edge in adjacency[vertex]:
            if other not in predecessor:
                predecessor[other] = (vertex, edge)
                queue.append(other)
    path: list[int] = []
    current = end
    while current != start:
        step = predecessor[current]
        assert step is not None
        current, edge = step
        path.append(edge)
    return list(reversed(path))


def gauge_forest_projection(
    n_vertices: int,
    edges: Sequence[Edge],
    observed_edge_ids: Sequence[int],
    *,
    fixed_vertices: Sequence[int] = (),
) -> dict[str, Any]:
    """Check the exact forest/available-gauge transitivity condition.

    PASS requires no observed cycle and at most one fixed vertex in each
    observed connected component. Selfloops and distinct parallel links are
    valid graph data and can supply cycle obstructions. Unobserved cycles
    do not matter. Empty observed sets pass, including the empty graph.

    Under the separately supplied invariance of a probability measure and
    F under all nonfixed vertex gauges, PASS implies E[F|observed]=E[F].
    The checker earns no such measure, source, or Hamiltonian assertion.
    """
    count = _integer(n_vertices, "n_vertices")
    if count < 0:
        raise ValueError("n_vertices must be nonnegative")
    graph: list[Edge] = []
    for index, value in enumerate(_sequence(edges, "edges")):
        edge = _sequence(value, f"edges[{index}]")
        if len(edge) != 2:
            raise ValueError("each edge must have exactly two endpoints")
        left, right = (_integer(endpoint, "edge endpoint") for endpoint in edge)
        if not 0 <= left < count or not 0 <= right < count:
            raise ValueError("edge endpoints must lie in [0,n_vertices)")
        graph.append((left, right))
    observed = _ids(observed_edge_ids, "observed_edge_ids", 1, len(graph))
    fixed = _ids(fixed_vertices, "fixed_vertices", 0, count - 1)
    fixed_set = set(fixed)
    adjacency: dict[int, list[tuple[int, int]]] = {}
    for edge_id in observed:
        left, right = graph[edge_id - 1]
        adjacency.setdefault(left, []).append((right, edge_id))
        adjacency.setdefault(right, []).append((left, -edge_id))
    for neighbors in adjacency.values():
        neighbors.sort(key=lambda pair: (abs(pair[1]), pair[1]))

    remaining = set(adjacency)
    components: list[dict[str, Any]] = []
    obstructions: list[dict[str, Any]] = []
    while remaining:
        first = min(remaining)
        queue = deque([first])
        vertices = {first}
        component_edges: set[int] = set()
        while queue:
            vertex = queue.popleft()
            for other, token in adjacency[vertex]:
                component_edges.add(abs(token))
                if other not in vertices:
                    vertices.add(other)
                    queue.append(other)
        remaining.difference_update(vertices)
        fixed_here = sorted(vertices & fixed_set)
        root = fixed_here[0] if fixed_here else min(vertices)
        queue = deque([root])
        visited = {root}
        tree: dict[int, list[tuple[int, int]]] = {v: [] for v in vertices}
        tree_edges: set[int] = set()
        gauge_order: list[dict[str, int]] = []
        while queue:
            vertex = queue.popleft()
            for other, token in adjacency[vertex]:
                if other not in visited:
                    visited.add(other)
                    queue.append(other)
                    tree_edges.add(abs(token))
                    tree[vertex].append((other, token))
                    tree[other].append((vertex, -token))
                    gauge_order.append(
                        {"parent": vertex, "child": other, "oriented_edge_id": token}
                    )
        chords = sorted(component_edges - tree_edges)
        index = len(components)
        if chords:
            edge_id = chords[0]
            left, right = graph[edge_id - 1]
            obstructions.append(
                {
                    "kind": "observed_cycle",
                    "component": index,
                    "closed_oriented_edge_ids": [edge_id, *_tree_path(right, left, tree)],
                    "su2_haar_control": "trace of cycle holonomy; mean 0, variance 1; observed projection equals this nonconstant observable",
                }
            )
        if len(fixed_here) > 1:
            obstructions.append(
                {
                    "kind": "multiple_fixed_terminals",
                    "component": index,
                    "fixed_endpoints": fixed_here[:2],
                    "oriented_path_edge_ids": _tree_path(fixed_here[0], fixed_here[1], tree),
                    "su2_haar_control": "trace of transport between fixed endpoints; mean 0, variance 1; available gauges cannot change this path holonomy",
                }
            )
        components.append(
            {
                "vertices": sorted(vertices),
                "observed_edge_ids": sorted(component_edges),
                "fixed_vertices": fixed_here,
                "cycle_rank": len(component_edges) - len(vertices) + 1,
                "root": root,
                "spanning_tree_gauge_order": gauge_order,
                "gauge_elimination_compatible_with_fixed_vertices": len(fixed_here) <= 1,
            }
        )

    passed = not obstructions
    payload = {
        "type": _KIND,
        "inputs": {
            "n_vertices": count,
            "edges": [list(edge) for edge in graph],
            "observed_edge_ids": list(observed),
            "fixed_vertices": list(fixed),
        },
        "status": "PASS" if passed else "INCONCLUSIVE",
        "finite_gate_verified": passed,
        "topology_checked": True,
        "observed_forest_verified": all(c["cycle_rank"] == 0 for c in components),
        "fixed_terminal_condition_verified": all(len(c["fixed_vertices"]) <= 1 for c in components),
        "available_gauge_action_transitive": passed,
        "conditional_projection_implication_verified_in_written_analysis": passed,
        "empty_observed_set": not observed,
        "components": components,
        "obstructions": obstructions,
        "premises": [
            "mu is a probability measure on the ambient original-link configuration space",
            "mu and F are invariant under every independent vertex gauge transformation except at fixed_vertices",
            "F is integrable; in the Hilbert-space statement F lies in L2(mu)",
            "the observed projection is conditional expectation in this SAME measure",
            "for an exterior-conditioned application every vertex whose gauge transformation changes frozen exterior data is included in fixed_vertices",
        ],
        "conditional_conclusion": (
            "E_mu[F|observed links]=E_mu[F]; on the centered available-gauge-invariant L2 space the observed projection is zero"
            if passed
            else None
        ),
        "gauge_action": "U_(u,v) -> g_u U_(u,v) g_v^-1; g_v=identity at fixed_vertices",
        "tree_assignment": "set root gauge to identity; for each signed parent-to-child edge t set g_child=g_parent*U_|t|^sign(t)",
        "failed_gate_meaning": "no universal collapse implication from this topology; no assertion that a specified unknown measure has a nonzero angle",
        **_SCOPE,
    }
    honesty = {key: value for key, value in payload.items() if type(value) is bool}
    certificate = make_certificate(
        claim="exact relative-forest gauge transitivity criterion and a conditional projection implication with explicit measure premises",
        payload=deepcopy(payload),
        honesty=honesty,
        meta={
            "transcend_backend": "not_used",
            "analytic_implication": "docs/api/gauge-forest-projection.md",
        },
    )
    return {
        **deepcopy(payload),
        "certificate": certificate,
        "theorem_prover_verified": False,
        "mathlib_verified": False,
    }


def replay_gauge_forest_projection_certificate(certificate: dict[str, Any]) -> bool:
    """Rebuild graph, obstructions, premises, metadata and all honesty flags."""
    if not isinstance(certificate, dict):
        return False
    try:
        if not verify_certificate_digest(certificate):
            return False
        payload = certificate["payload"]
        if payload["type"] != _KIND:
            return False
        inputs = payload["inputs"]
        expected = gauge_forest_projection(
            inputs["n_vertices"],
            inputs["edges"],
            inputs["observed_edge_ids"],
            fixed_vertices=inputs["fixed_vertices"],
        )
        return bool(certificate == expected["certificate"])
    except (KeyError, TypeError, ValueError, IndexError, OverflowError):
        return False


__all__ = ["gauge_forest_projection", "replay_gauge_forest_projection_certificate"]
