# SPDX-License-Identifier: Apache-2.0
"""Gauge-invariant Fourier corrections to an explicit Wilson trial vacuum.

Exact rational gates for written, all-representation strong-coupling
estimates. The graph class has Gauss's law at every vertex and a checked
lower girth bound. No continuum or formal verification is implied.
Existing vacuum_fourier v1 certificates retain their original constants.
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
from omnibias.geometry.gauge.transfer.vacuum_fourier import _caps, _cycle_diameters, _parameters


def _constants(group: str) -> tuple[int, Q, Q, Q, Q]:
    # N, smallest nonzero Casimir, Ricci, Hessian/norm, influence/norm.
    if group == "su2":
        return 2, Q(3, 4), Q(1, 2), Q(2, 3), Q(16, 3)
    if group == "su3":
        return 3, Q(4, 3), Q(3, 4), Q(1), Q(3, 2)
    raise ValueError("group must be 'su2' or 'su3'")


def _girth_bound(value: int) -> int:
    result = _integer(value, "minimum_girth")
    if result < 3:
        raise ValueError("minimum_girth must be at least three on a simple graph")
    return result


def _simple_graph_girth(n: int, edges: tuple[Edge, ...]) -> int | None:
    """Shortest cycle in the ENTIRE electric graph; None means a forest."""
    if len({tuple(sorted(edge)) for edge in edges}) != len(edges):
        raise ValueError("parallel electric edges are outside this simple-graph theorem")
    adjacency: list[list[int]] = [[] for _ in range(n)]
    for u, v in edges:
        adjacency[u].append(v)
        adjacency[v].append(u)
    shortest: int | None = None
    for start in range(n):
        distance = {start: 0}
        parent = {start: -1}
        queue = deque([start])
        while queue:
            u = queue.popleft()
            for v in adjacency[u]:
                if v not in distance:
                    distance[v] = distance[u] + 1
                    parent[v] = u
                    queue.append(v)
                elif parent[u] != v and parent[v] != u:
                    candidate = distance[u] + distance[v] + 1
                    shortest = candidate if shortest is None else min(shortest, candidate)
        if shortest == 3:
            break
    return shortest


def _forcing_multiplier(group: str, length: int) -> int:
    return int(2 ** (length - 1) if group == "su2" else 3**length)


def _arithmetic(
    group: str, coupling: Q, girth: int, forcing: Q, seed_hessian: Q,
    seed_influence: Q, radius: Q, seed_oscillation: Q, gap_method: str,
) -> dict[str, Any]:
    _, casimir, ricci, hessian_factor, influence_factor = _constants(group)
    bilinear = (4 if group == "su2" else 2) / (girth * casimir)
    residual = bilinear * forcing**2
    linear = 2 * bilinear * forcing * radius
    quadratic = bilinear * radius**2
    slack = radius - residual - linear - quadratic
    contraction = 2 * bilinear * (forcing + radius)
    fixed_point = slack >= 0 and contraction < 1
    hessian = seed_hessian + hessian_factor * radius
    curvature = ricci - 2 * hessian
    curvature_passed = fixed_point and curvature > 0
    influence = seed_influence + influence_factor * radius
    factorization = fixed_point and influence < 1
    oscillation = seed_oscillation + (8 if group == "su2" else 4) * radius / (girth * casimir)
    # log(1-z) <= -z for 0 <= z < 1 gives this positive rational
    # lower bound on exp(-oscillation). No transcendental evaluation.
    exponent_steps = int(oscillation) + 1 if factorization else 0
    exponential_lower = (1 - oscillation / exponent_steps) ** exponent_steps if factorization else Q(0)
    curvature_gap = coupling * curvature / 2 if curvature_passed else Q(0)
    conditional_gap = coupling * casimir / 2 * (1 - influence) * exponential_lower if factorization else Q(0)
    gap = curvature_gap if gap_method == "curvature" else conditional_gap
    passed = gap > 0
    return {
        "coefficient_norm": "spin_weighted_N" if group == "su2" else "Casimir_weighted_M",
        "minimum_nonconstant_invariant_energy": str(girth * casimir),
        "bilinear_constant": str(bilinear),
        "seed_norm_upper": str(forcing),
        "seed_hessian_row_upper": str(seed_hessian),
        "seed_tv_influence_row_upper": str(seed_influence),
        "influence_norm_factor": str(influence_factor),
        "influence_candidate_row_upper": str(influence),
        "seed_conditional_log_density_oscillation_upper": str(seed_oscillation),
        "residual_norm_upper": str(residual),
        "correction_radius": str(radius),
        "linear_correction_upper": str(linear),
        "quadratic_correction_upper": str(quadratic),
        "self_map_slack": str(slack),
        "self_map_verified": slack >= 0,
        "contraction_upper": str(contraction),
        "strict_contraction_verified": contraction < 1,
        "fixed_point_verified": fixed_point,
        "hessian_candidate_row_upper": str(hessian),
        "curvature_candidate_lower": str(curvature),
        "curvature_gap_verified": curvature_passed,
        "curvature_gap_lower": str(curvature_gap),
        "conditional_log_density_oscillation_upper": str(oscillation) if fixed_point else None,
        "exp_negative_lower_steps": exponent_steps,
        "exp_negative_rational_lower": str(exponential_lower) if factorization else None,
        "conditional_block_poincare_lower": str(casimir * (1 - influence) * exponential_lower)
        if factorization else None,
        "factorization_gap_verified": factorization,
        "factorization_gap_lower": str(conditional_gap),
        "gap_method": gap_method,
        "actual_log_vacuum_hessian_row_upper": str(hessian) if fixed_point else None,
        "neutral_gap_verified": passed,
        "neutral_gap_lower": str(gap),
        "resolvent_time_in_aH_units": "1",
        "vacuum_resolvent_contraction_upper": str(1 / (1 + gap)) if passed else None,
        "charged_linear_lower_coefficient": str(gap),
        "charged_linear_upper_coefficient": str(coupling * casimir / 2),
        "actual_tv_influence_row_upper": str(influence) if fixed_point else None,
        "factorization_bound_verified": factorization,
        "actual_factorization_constant_upper": str(1 / (1 - influence))
        if factorization else None,
    }


def _family_witness(
    group: str, coupling: Q, cap: Q, girth: int, length: int, diameter: int,
    decay: Q, radius: Q, gap_method: str,
) -> dict[str, Any]:
    n, casimir, _, _, _ = _constants(group)
    g = 4 / coupling**2
    forcing = g * _forcing_multiplier(group, length) * decay**diameter * cap
    seed_hessian = g * cap / (2 * casimir)
    seed_influence = 2 * n * g * cap * (length - 1) / (length * casimir)
    seed_oscillation = 4 * n * g * cap / (girth * casimir)
    return {
        "group": group,
        "normalization": f"aH=kappa/2*sum(C_e)+2/kappa*sum(v_p*({n}-ReTr(U_p)))",
        "energy_units": "dimensionless aH",
        "kappa": str(coupling), "g": str(g),
        "weighted_incidence_cap": str(cap),
        "minimum_girth": girth,
        "max_cycle_length": length, "max_cycle_diameter": diameter,
        "decay_base": str(decay), "correction_radius": str(radius),
        "gap_method": gap_method,
        "candidate": "S_star=C_0^-1*g*sum_p(v_p*ReTr(U_p)); true vacuum is exp(S_star+U)",
        "family_class": "all finite simple electric graphs with at least one edge and girth at least minimum_girth (forests allowed), nonnegative simple-cycle magnetic interactions within the length, ambient line-graph diameter, and weighted edge-incidence caps",
        "electric_weights": "one on every edge",
        "gauss_constraint": "at every vertex; imposed on the Fourier construction of the neutral vacuum",
        "gap_scope": "entire scalar product-group Hilbert space above its unique positive vacuum",
        "charged_scope": "fundamental and conjugate sources at distinct connected vertices; Gauss law at every vertex; no dynamical fundamental matter or flux-absorbing boundary; exact interacting vacuum subtraction",
        "boundary_scope": "only the stated finite graph and interactions",
        "arithmetic": _arithmetic(group, coupling, girth, forcing, seed_hessian,
                                  seed_influence, radius, seed_oscillation, gap_method),
    }


def _report(witness: dict[str, Any], *, family: bool) -> dict[str, Any]:
    membership = family or bool(witness["structural_caps_verified"])
    passed = membership and bool(witness["arithmetic"]["neutral_gap_verified"])
    family_witness = witness if family else witness["family"]
    family_passed = membership and bool(family_witness["arithmetic"]["neutral_gap_verified"])
    kind = "invariant_vacuum_fourier_family_v1" if family else "invariant_vacuum_fourier_graph_v1"
    certificate = make_certificate(
        claim="gauge-invariant local Fourier correction constructs the actual full-spin finite-graph vacuum and bounds its neutral gap and fundamental static-source energy",
        payload={"type": kind, "witness": witness},
        honesty={
            "yang_mills_claim": False, "yang_mills_mass_gap_claim": False,
            "continuum_claim": False, "infinite_volume_claim": False,
            "volume_uniform_finite_graph_family_verified": family_passed,
            "finite_graph_neutral_gap_verified": passed and not family,
        },
        meta={
            "analytic_implication": "docs/api/gauge-invariant-vacuum-fourier.md",
            "scope": "exact arithmetic plus written fixed-coupling analytic implication",
            "transcend_backend": "not_used",
        },
    )
    return {
        "status": "PASS" if passed else "INCONCLUSIVE",
        "finite_gate_verified": passed,
        "verification_kind": "EXACT_RATIONAL_WITH_WRITTEN_ANALYTIC_IMPLICATION",
        "witness": witness, "certificate": certificate,
        "digest_verified": verify_certificate_digest(certificate),
        "explicit_graph_neutral_gap_verified": passed and not family,
        "volume_uniform_finite_graph_family_verified": family_passed,
        "volume_uniform_charged_linear_bound_verified": family_passed,
        "volume_uniform_factorization_bound_verified": membership and bool(
            family_witness["arithmetic"]["factorization_bound_verified"]),
        "infinite_volume_claim": False, "uniform_in_a_claim": False,
        "continuum_claim": False, "yang_mills_claim": False,
        "yang_mills_mass_gap_claim": False, "static_confinement_claim": False,
        "string_tension_claim": False, "theorem_prover_verified": False,
        "mathlib_verified": False, "analytic_implication_formally_verified": False,
    }


def invariant_vacuum_fourier_family(
    group: str, kappa: int | Q, *, correction_radius: int | Q,
    weighted_incidence_cap: int | Q = 4, minimum_girth: int = 4,
    max_cycle_length: int = 4, max_cycle_diameter: int = 2,
    decay_base: int | Q = 1, gap_method: str = "curvature",
) -> dict[str, Any]:
    """Seal a quantified finite-family bound, not membership of unseen graphs.

    The radius bounds the correction to the explicit Wilson log-vacuum seed.
    SU(2) uses the sharper spin norm; SU(3) uses the Casimir norm. Rational
    input and strict contraction are required. The gap method is either
    global curvature or conditional Haar comparison plus actual-vacuum
    variance factorization. A failed selected gate is inconclusive.
    """
    _constants(group)
    if gap_method not in ("curvature", "factorization"):
        raise ValueError("gap_method must be 'curvature' or 'factorization'")
    coupling, decay, radius, _ = _parameters(kappa, decay_base, correction_radius, 0)
    cap, length, diameter = _caps(weighted_incidence_cap, max_cycle_length, max_cycle_diameter)
    girth = _girth_bound(minimum_girth)
    if length < 3:
        raise ValueError("simple-graph cycle length cap must be at least three")
    return _report(_family_witness(group, coupling, cap, girth, length, diameter,
                                  decay, radius, gap_method), family=True)


def invariant_vacuum_fourier_bounds(
    n_vertices: int, edges: Sequence[Edge], *, group: str, kappa: int | Q,
    correction_radius: int | Q, plaquettes: Sequence[Sequence[int]] = (),
    magnetic_weights: Sequence[int | Q] | None = None,
    weighted_incidence_cap: int | Q = 4, minimum_girth: int = 4,
    max_cycle_length: int = 4, max_cycle_diameter: int = 2,
    decay_base: int | Q = 1, gap_method: str = "curvature",
) -> dict[str, Any]:
    """Inspect the entire electric graph, including edges absent from plaquettes.

    Parallel edges and self-loops are refused. A shorter electric cycle fails
    membership even when all magnetic plaquettes have four edges. Disconnected
    components and forests are allowed; constants do not depend on their sizes.
    """
    family = invariant_vacuum_fourier_family(
        group, kappa, correction_radius=correction_radius,
        weighted_incidence_cap=weighted_incidence_cap, minimum_girth=minimum_girth,
        max_cycle_length=max_cycle_length, max_cycle_diameter=max_cycle_diameter,
        decay_base=decay_base,
        gap_method=gap_method,
    )["witness"]
    graph = _graph(n_vertices, edges)
    if not graph:
        raise ValueError("at least one electric edge is required")
    girth = _simple_graph_girth(n_vertices, graph)
    loops = _cycles(graph, plaquettes)
    magnetic = tuple(_rational(x, "magnetic weight") for x in (
        [1] * len(loops) if magnetic_weights is None else magnetic_weights))
    if len(magnetic) != len(loops) or any(x < 0 for x in magnetic):
        raise ValueError("one nonnegative magnetic weight per plaquette is required")
    diameters = _cycle_diameters(graph, loops)
    coupling, decay, radius, _ = _parameters(kappa, decay_base, correction_radius, 0)
    n, casimir, _, _, _ = _constants(group)
    g = 4 / coupling**2
    incidence, forcing, hessian, influence, oscillation = ([Q(0)] * len(graph) for _ in range(5))
    for loop, weight, diameter in zip(loops, magnetic, diameters, strict=True):
        length = len(loop)
        for token in loop:
            i = abs(token) - 1
            incidence[i] += weight
            forcing[i] += g * _forcing_multiplier(group, length) * decay**diameter * weight
            hessian[i] += g * weight / (2 * casimir)
            influence[i] += 2 * n * g * weight * (length - 1) / (length * casimir)
            oscillation[i] += 4 * n * g * weight / (length * casimir)
    cap_checks = {
        "entire_electric_graph_girth": girth is None or girth >= minimum_girth,
        "weighted_incidence": max(incidence) <= Q(family["weighted_incidence_cap"]),
        "cycle_length": max(map(len, loops), default=0) <= max_cycle_length,
        "cycle_diameter": max(diameters, default=0) <= max_cycle_diameter,
    }
    witness = {
        "group": group, "n_vertices": n_vertices, "edges": [list(e) for e in graph],
        "plaquettes": [list(p) for p in loops], "magnetic_weights": list(map(str, magnetic)),
        "actual_electric_graph_girth": girth, "forest": girth is None,
        "cycle_ambient_line_graph_diameters": list(diameters),
        "weighted_incidence": list(map(str, incidence)),
        "forcing_by_edge": list(map(str, forcing)),
        "structural_cap_checks": cap_checks, "structural_caps_verified": all(cap_checks.values()),
        "family": family,
        "arithmetic": _arithmetic(group, coupling, girth or minimum_girth, max(forcing),
                                  max(hessian), max(influence), radius, max(oscillation), gap_method),
    }
    return _report(witness, family=False)


def replay_invariant_vacuum_fourier_certificate(certificate: dict[str, Any]) -> bool:
    """Recompute canonical inputs, graph membership, arithmetic, and honesty fields."""
    if not isinstance(certificate, dict):
        return False
    try:
        if not verify_certificate_digest(certificate):
            return False
        payload = certificate["payload"]
        witness = payload["witness"]
        kind = payload["type"]
        if kind not in {"invariant_vacuum_fourier_family_v1", "invariant_vacuum_fourier_graph_v1"}:
            return False
        family = witness if kind.endswith("family_v1") else witness["family"]
        parameters = {
            "group": family["group"], "kappa": Q(family["kappa"]),
            "correction_radius": Q(family["correction_radius"]),
            "weighted_incidence_cap": Q(family["weighted_incidence_cap"]),
            "minimum_girth": family["minimum_girth"],
            "max_cycle_length": family["max_cycle_length"],
            "max_cycle_diameter": family["max_cycle_diameter"],
            "decay_base": Q(family["decay_base"]),
            "gap_method": family["gap_method"],
        }
        if kind.endswith("family_v1"):
            report = invariant_vacuum_fourier_family(**parameters)
        else:
            report = invariant_vacuum_fourier_bounds(
                witness["n_vertices"], [tuple(e) for e in witness["edges"]],
                plaquettes=witness["plaquettes"],
                magnetic_weights=[Q(x) for x in witness["magnetic_weights"]], **parameters,
            )
        return bool(report["finite_gate_verified"] and report["certificate"] == certificate)
    except (KeyError, TypeError, ValueError, ZeroDivisionError, OverflowError, IndexError):
        return False
