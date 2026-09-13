# SPDX-License-Identifier: Apache-2.0
"""Exact graph bounds for a static fundamental SU(2) source pair.

The kinetic-only energy equals a weighted shortest-path cost, with no spin
cutoff. A groundstate-transform argument gives the same cost as an upper
bound for the interacting, vacuum-subtracted energy. The lower bound uses
diamagnetism and, optionally, a finite-volume Haar vacuum trial. These are
written analytic implications, not Lean verification or continuum claims.

The model is dimensionless aH = kappa/2 sum_e w_e C_e
    + 2/kappa sum_p v_p (2 - Tr U_p).
Each supplied plaquette is a simple closed oriented graph cycle. No spatial
dimension or embedding is inferred from the graph. See docs/api/static-sources.md.
"""
from __future__ import annotations

from collections.abc import Sequence
from fractions import Fraction
from typing import Any

from omnibias.core.proof.certificate import make_certificate, verify_certificate_digest

Edge = tuple[int, int]


def _integer(value: int, name: str) -> int:
    if type(value) is not int:
        raise TypeError(f"{name} must be an exact integer, not a bool or float")
    return value


def _rational(value: int | Fraction, name: str) -> Fraction:
    if type(value) is not int and not isinstance(value, Fraction):
        raise TypeError(f"{name} must be an exact integer or Fraction")
    return Fraction(value)


def _graph(n_vertices: int, edges: Sequence[Edge]) -> tuple[Edge, ...]:
    if _integer(n_vertices, "n_vertices") < 2:
        raise ValueError("at least two vertices are required")
    result: list[Edge] = []
    for edge in edges:
        if len(edge) != 2:
            raise ValueError("each edge must have two endpoints")
        u, v = (_integer(x, "edge endpoint") for x in edge)
        if not (0 <= u < n_vertices and 0 <= v < n_vertices) or u == v:
            raise ValueError("endpoints must be distinct vertices in range; self-loops are refused")
        result.append((u, v))
    return tuple(result)


def _cycles(edges: tuple[Edge, ...], plaquettes: Sequence[Sequence[int]]) -> tuple[tuple[int, ...], ...]:
    result: list[tuple[int, ...]] = []
    for cycle in plaquettes:
        tokens = tuple(_integer(x, "oriented edge") for x in cycle)
        if len(tokens) < 2 or len({abs(x) for x in tokens}) != len(tokens):
            raise ValueError("a plaquette must use at least two distinct graph edges")
        oriented: list[Edge] = []
        for token in tokens:
            if token == 0 or abs(token) > len(edges):
                raise ValueError("oriented edges use signed, one-based edge indices")
            edge = edges[abs(token) - 1]
            oriented.append(edge if token > 0 else (edge[1], edge[0]))
        if any(oriented[i][1] != oriented[(i + 1) % len(tokens)][0] for i in range(len(tokens))):
            raise ValueError("plaquette edges must form a closed oriented walk")
        if len({u for u, _ in oriented}) != len(tokens):
            raise ValueError("plaquettes must be simple cycles without repeated vertices")
        result.append(tokens)
    return tuple(result)


def _shortest_path(n: int, edges: tuple[Edge, ...], weights: tuple[Fraction, ...],
                   source: int, target: int) -> tuple[tuple[Fraction, ...], tuple[int, ...]]:
    adjacency: list[list[tuple[int, int, Fraction]]] = [[] for _ in range(n)]
    for i, ((u, v), weight) in enumerate(zip(edges, weights, strict=True), 1):
        adjacency[u].append((v, i, weight))
        adjacency[v].append((u, -i, weight))
    distance: list[Fraction | None] = [None] * n
    predecessor: list[tuple[int, int] | None] = [None] * n
    distance[source] = Fraction(0)
    remaining = set(range(n))
    while remaining:
        reachable = [i for i in remaining if distance[i] is not None]
        if not reachable:
            raise ValueError("the graph must be connected")
        vertex = min(reachable, key=lambda i: (distance[i], i))
        current = distance[vertex]
        assert current is not None
        remaining.remove(vertex)
        for other, token, weight in adjacency[vertex]:
            old = distance[other]
            proposal = current + weight
            if other in remaining and (old is None or proposal < old):
                distance[other] = proposal
                predecessor[other] = (vertex, token)
    path: list[int] = []
    vertex = target
    while vertex != source:
        previous = predecessor[vertex]
        assert previous is not None
        vertex, token = previous
        path.append(token)
    path.reverse()
    return tuple(x for x in distance if x is not None), tuple(path)


def _separating_bridges(n: int, edges: tuple[Edge, ...], source: int,
                        target: int, path: tuple[int, ...]) -> tuple[int, ...]:
    # A source-separating bridge belongs to every source-target path. Test
    # only edges of the already derived shortest path; parallel edges remain
    # distinct. Connectivity after deleting a single edge is exact.
    adjacency: list[list[tuple[int, int]]] = [[] for _ in range(n)]
    for i, (u, v) in enumerate(edges, 1):
        adjacency[u].append((v, i))
        adjacency[v].append((u, i))
    result: list[int] = []
    for removed in sorted(abs(token) for token in path):
        reached, frontier = {source}, [source]
        while frontier:
            u = frontier.pop()
            for v, edge in adjacency[u]:
                if edge != removed and v not in reached:
                    reached.add(v)
                    frontier.append(v)
        if target not in reached:
            result.append(removed)
    return tuple(result)


def su2_static_source_bounds(
    n_vertices: int,
    edges: Sequence[Edge],
    source: int,
    target: int,
    *,
    kappa: int | Fraction = 1,
    edge_weights: Sequence[int | Fraction] | None = None,
    plaquettes: Sequence[Sequence[int]] = (),
    magnetic_weights: Sequence[int | Fraction] | None = None,
) -> dict[str, Any]:
    """Seal exact bounds on the matched charged-minus-vacuum ground energy.

    One fixed fundamental source and one antifundamental source are included
    in Gauss's law; all other vertices are singlets. Weights are rational,
    electric weights strictly positive, magnetic weights nonnegative. Parallel
    edges are distinct. Each cycle uses signed one-based edge indices.

    The pure-electric identity follows from center parity and an attaining
    fundamental path. Source-separating bridges give a volume-independent
    interacting lower bound because no simple plaquette crosses a bridge.
    Otherwise the lower bound can be zero: PASS means a sound energy
    enclosure, not confinement or a mass gap.
    """
    graph = _graph(n_vertices, edges)
    s, t = _integer(source, "source"), _integer(target, "target")
    if not (0 <= s < n_vertices and 0 <= t < n_vertices) or s == t:
        raise ValueError("source and target must be distinct vertices in range")
    coupling = _rational(kappa, "kappa")
    if coupling <= 0:
        raise ValueError("kappa must be strictly positive")
    weights = tuple(_rational(x, "edge weight") for x in (
        [1] * len(graph) if edge_weights is None else edge_weights))
    if len(weights) != len(graph) or any(x <= 0 for x in weights):
        raise ValueError("one strictly positive electric weight per edge is required")
    loops = _cycles(graph, plaquettes)
    magnetic = tuple(_rational(x, "magnetic weight") for x in (
        [1] * len(loops) if magnetic_weights is None else magnetic_weights))
    if len(magnetic) != len(loops) or any(x < 0 for x in magnetic):
        raise ValueError("one nonnegative magnetic weight per plaquette is required")
    distances, path = _shortest_path(n_vertices, graph, weights, s, t)
    distance = distances[t]
    bridges = _separating_bridges(n_vertices, graph, s, t, path)
    bridge_distance = sum((weights[i - 1] for i in bridges), Fraction(0))
    electric_cost = 3 * coupling * distance / 8
    bridge_cost = 3 * coupling * bridge_distance / 8
    haar_vacuum_upper = 4 * sum(magnetic, Fraction(0)) / coupling
    lower = max(bridge_cost, electric_cost - haar_vacuum_upper)
    pure_electric = not any(magnetic)
    path_spins = [int(i + 1 in {abs(token) for token in path}) for i in range(len(graph))]
    witness = {
        "model": "su2_graph_static_fundamental_pair_v1",
        "normalization": "aH=kappa/2*sum(w_e*C_e)+2/kappa*sum(v_p*(2-Tr(U_p)))",
        "n_vertices": n_vertices, "edges": [list(e) for e in graph],
        "source": s, "target": t, "kappa": str(coupling),
        "edge_weights": list(map(str, weights)), "plaquettes": [list(p) for p in loops],
        "magnetic_weights": list(map(str, magnetic)),
        "shortest_distances": list(map(str, distances)), "path": list(path),
        "path_two_spins": path_spins, "weighted_distance": str(distance),
        "source_separating_bridges": list(bridges),
        "weighted_bridge_distance": str(bridge_distance),
        "bridge_energy_lower": str(bridge_cost),
        "pure_electric_path_energy": str(electric_cost),
        "haar_vacuum_upper": str(haar_vacuum_upper),
        "static_energy_enclosure": [str(lower), str(electric_cost)],
        "pure_electric_identity": pure_electric,
        "exact_static_energy": lower == electric_cost,
        "positive_static_energy_lower_bound": lower > 0,
        "volume_independent_upper_bound": True,
        "finite_volume_haar_lower": str(electric_cost - haar_vacuum_upper),
        "haar_lower_depends_on_total_plaquette_weight": not pure_electric,
        "string_tension_claim": False, "continuum_claim": False, "yang_mills_claim": False,
    }
    certificate = make_certificate(
        claim="exact graph bounds on vacuum-subtracted SU(2) static-source energy",
        payload={"type": "su2_static_source_bounds_v1", "witness": witness},
        honesty={"yang_mills_claim": False, "continuum_claim": False,
                 "string_tension_claim": False},
        meta={"scope": "finite graph; all spins; written parity, bridge and groundstate-transform implications",
              "analytic_implication": "docs/api/static-sources.md",
              "transcend_backend": "not_used"},
    )
    return {
        "status": "PASS", "verification_kind": "EXACT_RATIONAL",
        "finite_gate_verified": True, "witness": witness, "certificate": certificate,
        "digest_verified": verify_certificate_digest(certificate),
        "theorem_prover_verified": False, "mathlib_verified": False,
        "analytic_implication_formally_verified": False,
        "string_tension_claim": False, "continuum_claim": False, "yang_mills_claim": False,
    }


def replay_su2_static_source_certificate(certificate: dict[str, Any]) -> bool:
    """Reconstruct the complete canonical model, bounds, path and claim scope."""
    try:
        if not verify_certificate_digest(certificate):
            return False
        payload = certificate["payload"]
        if payload["type"] != "su2_static_source_bounds_v1":
            return False
        w = payload["witness"]
        result = su2_static_source_bounds(
            w["n_vertices"], w["edges"], w["source"], w["target"],
            kappa=Fraction(w["kappa"]),
            edge_weights=[Fraction(x) for x in w["edge_weights"]],
            plaquettes=w["plaquettes"],
            magnetic_weights=[Fraction(x) for x in w["magnetic_weights"]],
        )
        return bool(result["certificate"] == certificate)
    except (KeyError, TypeError, ValueError, ZeroDivisionError, OverflowError, IndexError):
        return False
