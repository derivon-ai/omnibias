# SPDX-License-Identifier: Apache-2.0
"""Exact strong-coupling gates for the actual SU(2) Hamiltonian vacuum.

The Fourier fixed-point proof gives a neutral Hamiltonian gap uniform over
finite graphs with stated local incidence and cycle-diameter bounds. The
optional weighted norm also controls distant Hessian blocks of log psi_0.
These are fixed-coupling finite-volume family theorems, not an infinite-volume
construction, static-source confinement, or a continuum Yang--Mills claim.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Sequence
from fractions import Fraction as Q
from typing import Any

from omnibias.core.proof.certificate import make_certificate, verify_certificate_digest
from omnibias.geometry.gauge.transfer.static_sources import (
    Edge,
    _cycles,
    _graph,
    _integer,
    _rational,
)


def _parameters(
    kappa: int | Q, decay_base: int | Q, radius: int | Q, tail_radius: int
) -> tuple[Q, Q, Q, int]:
    coupling = _rational(kappa, "kappa")
    decay = _rational(decay_base, "decay_base")
    candidate = _rational(radius, "radius")
    distance = _integer(tail_radius, "tail_radius")
    if coupling <= 0 or candidate <= 0:
        raise ValueError("kappa and the candidate fixed-point radius must be positive")
    if decay < 1 or distance < 0:
        raise ValueError("decay_base must be at least one and tail_radius nonnegative")
    return coupling, decay, candidate, distance


def _caps(incidence: int | Q, length: int, diameter: int) -> tuple[Q, int, int]:
    cap = _rational(incidence, "weighted_incidence_cap")
    count = _integer(length, "max_cycle_length")
    span = _integer(diameter, "max_cycle_diameter")
    if cap < 0 or count < 2 or span < 0:
        raise ValueError(
            "require nonnegative incidence/diameter caps and cycle length at least two"
        )
    return cap, count, span


def _cycle_diameters(
    graph: tuple[Edge, ...], loops: tuple[tuple[int, ...], ...]
) -> tuple[int, ...]:
    """Exact support diameters in the ambient ordinary line graph, with shortcuts.

    Vertex incidence implements line-graph BFS without constructing a dense
    clique for high-valence graph vertices. Each simple cycle lies in one
    connected component; no distance across components is requested.
    """
    incidence: dict[int, list[int]] = {}
    for index, edge in enumerate(graph):
        for vertex in edge:
            incidence.setdefault(vertex, []).append(index)
    result = []
    for loop in loops:
        support = {abs(token) - 1 for token in loop}
        diameter = 0
        for start in support:
            distances = {start: 0}
            remaining = support - {start}
            queue = deque([start])
            expanded_vertices: set[int] = set()
            while remaining:
                current = queue.popleft()
                for vertex in graph[current]:
                    if vertex in expanded_vertices:
                        continue
                    expanded_vertices.add(vertex)
                    for other in incidence[vertex]:
                        if other in distances:
                            continue
                        distances[other] = distances[current] + 1
                        remaining.discard(other)
                        queue.append(other)
            diameter = max(diameter, *(distances[i] for i in support))
        result.append(diameter)
    return tuple(result)


def _arithmetic(coupling: Q, forcing: Q, decay: Q, radius: Q, distance: int) -> dict[str, Any]:
    quadratic = Q(16, 3) * radius**2
    self_map = forcing + quadratic <= radius
    contraction = Q(32, 3) * radius
    curvature = Q(1, 2) - Q(4, 3) * radius
    passed = self_map and contraction < 1 and curvature > 0
    hessian = Q(2, 3) * radius
    influence = Q(1936, 147) * radius  # 4*pi^2/3 <= 1936/147, using pi < 22/7.
    factorization = passed and influence < 1
    return {
        "forcing_norm_upper": str(forcing),
        "candidate_radius": str(radius),
        "bilinear_constant": "16/3",
        "quadratic_correction": str(quadratic),
        "self_map_slack": str(radius - forcing - quadratic),
        "self_map_verified": self_map,
        "contraction_upper": str(contraction),
        "strict_contraction_verified": contraction < 1,
        "curvature_candidate_lower": str(curvature),
        "curvature_positive": curvature > 0,
        "fixed_point_verified": passed,
        "actual_log_vacuum_hessian_row_upper": str(hessian) if passed else None,
        "tail_radius": distance,
        "tail_distance_convention": "ambient line-graph distance strictly greater than tail_radius",
        "actual_log_vacuum_hessian_tail_row_upper": str(hessian / decay**distance)
        if passed
        else None,
        "exponential_hessian_tail_verified": passed and decay > 1,
        "actual_tv_influence_row_upper": str(influence) if passed else None,
        "actual_tv_influence_tail_upper": str(influence / decay**distance) if passed else None,
        "factorization_bound_verified": factorization,
        "actual_factorization_constant_upper": str(1 / (1 - influence)) if factorization else None,
        "neutral_gap_lower": str(coupling * curvature / 2) if passed else "0",
        "neutral_gap_verified": passed,
    }


def _family_witness(
    coupling: Q, cap: Q, length: int, diameter: int, decay: Q, radius: Q, distance: int
) -> dict[str, Any]:
    forcing = 4 / coupling**2 * 2 ** (length - 1) * decay**diameter * cap
    return {
        "model": "su2_actual_hamiltonian_vacuum_fourier",
        "normalization": "aH=kappa/2*sum(C_e)+2/kappa*sum(v_p*(2-Tr(U_p)))",
        "measure": "psi_0**2 times product Haar; positive actual Hamiltonian vacuum",
        "energy_units": "dimensionless aH",
        "kappa": str(coupling),
        "g": str(4 / coupling**2),
        "decay_base": str(decay),
        "radius": str(radius),
        "tail_radius": distance,
        "weighted_incidence_cap": str(cap),
        "max_cycle_length": length,
        "max_cycle_diameter": diameter,
        "family_class": "every finite graph with at least one edge, distinct-endpoint edges, simple oriented cycles within the stated length and ambient line-graph diameter caps, nonnegative plaquette weights, and weighted incidence at each edge no greater than the cap",
        "electric_weights": "one on every edge",
        "boundary_scope": "the specified finite graph; no external boundary interaction is silently added",
        "disconnected_components": "the Hamiltonian and positive vacuum factor across components; use componentwise Fourier spaces",
        "fixed_coupling": True,
        "arithmetic": _arithmetic(coupling, forcing, decay, radius, distance),
    }


def _report(witness: dict[str, Any], *, family: bool) -> dict[str, Any]:
    if family:
        passed = bool(witness["arithmetic"]["fixed_point_verified"])
        family_passed = passed
        actual_gap_passed = False
        actual_factorization = False
        family_factorization = bool(witness["arithmetic"]["factorization_bound_verified"])
        kind = "su2_vacuum_fourier_family_v1"
        claim = "uniform neutral Hamiltonian gap and log-vacuum Hessian bounds on the specified finite-graph SU(2) family"
    else:
        caps_passed = bool(witness["structural_caps_verified"])
        actual_gap_passed = bool(witness["arithmetic"]["fixed_point_verified"])
        passed = caps_passed and actual_gap_passed
        family_passed = caps_passed and bool(
            witness["family"]["arithmetic"]["fixed_point_verified"]
        )
        actual_factorization = bool(witness["arithmetic"]["factorization_bound_verified"])
        family_factorization = caps_passed and bool(
            witness["family"]["arithmetic"]["factorization_bound_verified"]
        )
        kind = "su2_vacuum_fourier_graph_v1"
        claim = "actual SU(2) finite-graph vacuum Fourier bounds with independently checked structural family caps"
    certificate = make_certificate(
        claim=claim,
        payload={"type": kind, "witness": witness},
        honesty={
            "yang_mills_claim": False,
            "continuum_claim": False,
            "static_confinement_claim": False,
        },
        meta={
            "transcend_backend": "not_used",
            "analytic_implication": "docs/api/gauge-vacuum-fourier.md",
            "scope": "written all-spin strong-coupling analysis at fixed lattice coupling",
        },
    )
    return {
        "status": "PASS" if passed else "INCONCLUSIVE",
        "finite_gate_verified": passed,
        "verification_kind": "EXACT_RATIONAL_WITH_WRITTEN_ANALYTIC_IMPLICATION",
        "witness": witness,
        "certificate": certificate,
        "digest_verified": verify_certificate_digest(certificate),
        "explicit_graph_neutral_gap_verified": actual_gap_passed,
        "volume_uniform_finite_graph_family_verified": family_passed,
        "volume_uniform_neutral_gap_claim": family_passed,
        "explicit_graph_factorization_bound_verified": actual_factorization,
        "volume_uniform_factorization_bound_verified": family_factorization,
        "infinite_volume_claim": False,
        "uniform_in_a_claim": False,
        "continuum_claim": False,
        "yang_mills_claim": False,
        "static_confinement_claim": False,
        "string_tension_claim": False,
        "theorem_prover_verified": False,
        "mathlib_verified": False,
        "analytic_implication_formally_verified": False,
    }


def su2_vacuum_fourier_family(
    kappa: int | Q,
    weighted_incidence_cap: int | Q,
    *,
    max_cycle_length: int = 4,
    max_cycle_diameter: int = 2,
    decay_base: int | Q = 1,
    radius: int | Q = Q(3, 64),
    tail_radius: int = 0,
) -> dict[str, Any]:
    """Check a uniform theorem for all finite graphs in the stated structural class.

    The incidence cap is a hypothesis defining the quantified class, not a
    claim about an uninspected graph. Use su2_vacuum_fourier_bounds to check
    the hypotheses on a concrete graph. Failure of an inequality is
    INCONCLUSIVE, not evidence that the actual vacuum has no gap.
    """
    coupling, decay, candidate, distance = _parameters(kappa, decay_base, radius, tail_radius)
    cap, length, diameter = _caps(weighted_incidence_cap, max_cycle_length, max_cycle_diameter)
    witness = _family_witness(coupling, cap, length, diameter, decay, candidate, distance)
    return _report(witness, family=True)


def su2_vacuum_fourier_bounds(
    n_vertices: int,
    edges: Sequence[Edge],
    *,
    kappa: int | Q = 64,
    plaquettes: Sequence[Sequence[int]] = (),
    magnetic_weights: Sequence[int | Q] | None = None,
    decay_base: int | Q = 1,
    radius: int | Q = Q(3, 64),
    tail_radius: int = 0,
    weighted_incidence_cap: int | Q | None = None,
    max_cycle_length: int | None = None,
    max_cycle_diameter: int | None = None,
) -> dict[str, Any]:
    """Check actual local forcing and structural-family membership on a finite graph.

    Edges and signed one-based plaquette cycles use the shared graph validator.
    Each omitted cap defaults to the exactly computed graph value. A coarse
    family gate may fail while the stronger direct graph forcing gate passes;
    both verdicts are reported. All electric weights are one.
    """
    graph = _graph(n_vertices, edges)
    if not graph:
        raise ValueError("at least one graph edge is required")
    loops = _cycles(graph, plaquettes)
    magnetic = tuple(
        _rational(x, "magnetic weight")
        for x in ([1] * len(loops) if magnetic_weights is None else magnetic_weights)
    )
    if len(magnetic) != len(loops) or any(weight < 0 for weight in magnetic):
        raise ValueError("one nonnegative magnetic weight per plaquette is required")
    coupling, decay, candidate, distance = _parameters(kappa, decay_base, radius, tail_radius)
    diameters = _cycle_diameters(graph, loops)
    incidence = [Q(0)] * len(graph)
    forcing = [Q(0)] * len(graph)
    g = 4 / coupling**2
    for loop, weight, diameter in zip(loops, magnetic, diameters, strict=True):
        for token in loop:
            index = abs(token) - 1
            incidence[index] += weight
            forcing[index] += g * 2 ** (len(loop) - 1) * decay**diameter * weight
    actual_incidence = max(incidence)
    actual_length = max(map(len, loops), default=2)
    actual_diameter = max(diameters, default=0)
    cap, length, diameter = _caps(
        actual_incidence if weighted_incidence_cap is None else weighted_incidence_cap,
        actual_length if max_cycle_length is None else max_cycle_length,
        actual_diameter if max_cycle_diameter is None else max_cycle_diameter,
    )
    cap_checks = {
        "weighted_incidence": actual_incidence <= cap,
        "cycle_length": actual_length <= length,
        "cycle_diameter": actual_diameter <= diameter,
    }
    witness = {
        "model": "su2_actual_hamiltonian_vacuum_fourier",
        "normalization": "aH=kappa/2*sum(C_e)+2/kappa*sum(v_p*(2-Tr(U_p)))",
        "measure": "psi_0**2 times product Haar; positive actual Hamiltonian vacuum",
        "energy_units": "dimensionless aH",
        "n_vertices": n_vertices,
        "edges": [list(edge) for edge in graph],
        "plaquettes": [list(loop) for loop in loops],
        "magnetic_weights": list(map(str, magnetic)),
        "cycle_ambient_line_graph_diameters": list(diameters),
        "weighted_incidence": list(map(str, incidence)),
        "forcing_by_edge": list(map(str, forcing)),
        "kappa": str(coupling),
        "g": str(g),
        "decay_base": str(decay),
        "radius": str(candidate),
        "tail_radius": distance,
        "structural_cap_checks": cap_checks,
        "structural_caps_verified": all(cap_checks.values()),
        "family": _family_witness(coupling, cap, length, diameter, decay, candidate, distance),
        "arithmetic": _arithmetic(coupling, max(forcing), decay, candidate, distance),
    }
    return _report(witness, family=False)


def replay_su2_vacuum_fourier_certificate(certificate: dict[str, Any]) -> bool:
    """Accept only canonical passing certificates after full recomputation.

    A faithfully stored INCONCLUSIVE report returns False, as does tampering.
    Failed reports remain useful diagnostics but certify none of these bounds.
    """
    if not isinstance(certificate, dict):
        return False
    try:
        if not verify_certificate_digest(certificate):
            return False
        payload = certificate["payload"]
        witness = payload["witness"]
        if payload["type"] == "su2_vacuum_fourier_family_v1":
            report = su2_vacuum_fourier_family(
                Q(witness["kappa"]),
                Q(witness["weighted_incidence_cap"]),
                max_cycle_length=witness["max_cycle_length"],
                max_cycle_diameter=witness["max_cycle_diameter"],
                decay_base=Q(witness["decay_base"]),
                radius=Q(witness["radius"]),
                tail_radius=witness["tail_radius"],
            )
        elif payload["type"] == "su2_vacuum_fourier_graph_v1":
            family = witness["family"]
            report = su2_vacuum_fourier_bounds(
                witness["n_vertices"],
                [tuple(edge) for edge in witness["edges"]],
                kappa=Q(witness["kappa"]),
                plaquettes=witness["plaquettes"],
                magnetic_weights=[Q(x) for x in witness["magnetic_weights"]],
                decay_base=Q(witness["decay_base"]),
                radius=Q(witness["radius"]),
                tail_radius=witness["tail_radius"],
                weighted_incidence_cap=Q(family["weighted_incidence_cap"]),
                max_cycle_length=family["max_cycle_length"],
                max_cycle_diameter=family["max_cycle_diameter"],
            )
        else:
            return False
        return bool(report["finite_gate_verified"] and report["certificate"] == certificate)
    except (KeyError, TypeError, ValueError, ZeroDivisionError, OverflowError, IndexError):
        return False
