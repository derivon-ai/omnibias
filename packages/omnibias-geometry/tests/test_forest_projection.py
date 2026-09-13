# SPDX-License-Identifier: Apache-2.0
"""Exact gauge-action oracles for the relative-forest projection criterion."""

from __future__ import annotations

from copy import deepcopy
from fractions import Fraction as Q
from itertools import combinations, product
from random import Random
from typing import Any

import pytest
from omnibias.core.proof.certificate import seal_certificate
from omnibias.geometry.gauge.transfer.forest_projection import (
    gauge_forest_projection,
)
from omnibias.geometry.gauge.transfer.forest_projection import (
    replay_gauge_forest_projection_certificate as replay,
)

Edge = tuple[int, int]
Quaternion = tuple[Q, Q, Q, Q]
ONE: Quaternion = (Q(1), Q(0), Q(0), Q(0))


def _z2_orbit_size(n: int, edges: list[Edge], observed: list[int], fixed: list[int]) -> int:
    """Enumerate gauge transforms of the identity; no graph/rank algorithm."""
    available = [v for v in range(n) if v not in fixed]
    orbit = set()
    for bits in product((0, 1), repeat=len(available)):
        gauges = dict(zip(available, bits, strict=True))
        orbit.add(
            tuple(gauges.get(edges[e - 1][0], 0) ^ gauges.get(edges[e - 1][1], 0) for e in observed)
        )
    return len(orbit)


def _mul(a: Quaternion, b: Quaternion) -> Quaternion:
    s, x, y, z = a
    t, u, v, w = b
    return (
        s * t - x * u - y * v - z * w,
        s * u + x * t + y * w - z * v,
        s * v - x * w + y * t + z * u,
        s * w + x * v - y * u + z * t,
    )


def _inverse(a: Quaternion) -> Quaternion:
    return (a[0], -a[1], -a[2], -a[3])


def _holonomy(links: list[Quaternion], tokens: list[int]) -> Quaternion:
    result = ONE
    for token in tokens:
        value = links[abs(token) - 1]
        result = _mul(result, value if token > 0 else _inverse(value))
    return result


def test_empty_observed_and_empty_graph_are_trivial() -> None:
    for n in (0, 1, 4):
        result = gauge_forest_projection(n, [], [])
        assert result["status"] == "PASS"
        assert result["empty_observed_set"] is True
        assert result["components"] == []
        assert replay(result["certificate"])
    result = gauge_forest_projection(2, [(0, 0), (0, 1), (0, 1)], [], fixed_vertices=[0, 1])
    assert result["status"] == "PASS"


def test_theta_both_global_forest_blocks_pass() -> None:
    graph = [(0, 1), (1, 2), (3, 4), (4, 5), (0, 3), (1, 4), (2, 5)]
    for observed in ([1, 2, 6], [3, 4, 5, 7]):
        result = gauge_forest_projection(6, graph, observed)
        assert result["status"] == "PASS"
        assert result["actual_vacuum_verified"] is False
        assert result["actual_measure_verified"] is False
        assert result["input_gauge_invariance_verified"] is False
        assert replay(result["certificate"])


def test_exact_finite_z2_oracle_all_graphs_on_four_vertices_and_all_fixed_sets() -> None:
    # Keep K4 as ambient graph and enumerate its 64 observed subgraphs.
    # Unobserved cycles are therefore tested simultaneously.
    edges = list(combinations(range(4), 2))
    for edge_bits in product((0, 1), repeat=6):
        observed = [i + 1 for i, selected in enumerate(edge_bits) if selected]
        for vertex_bits in product((0, 1), repeat=4):
            fixed = [i for i, selected in enumerate(vertex_bits) if selected]
            oracle = _z2_orbit_size(4, edges, observed, fixed) == 2 ** len(observed)
            result = gauge_forest_projection(4, edges, observed, fixed_vertices=fixed)
            assert result["available_gauge_action_transitive"] == oracle
            assert (result["status"] == "PASS") == oracle


def test_multigraph_z2_oracle_includes_selfloops_and_parallel_cycles() -> None:
    edges = [(0, 0), (0, 1), (1, 0), (1, 2), (2, 2)]
    for edge_bits in product((0, 1), repeat=len(edges)):
        observed = [i + 1 for i, selected in enumerate(edge_bits) if selected]
        for vertex_bits in product((0, 1), repeat=3):
            fixed = [i for i, selected in enumerate(vertex_bits) if selected]
            result = gauge_forest_projection(3, edges, observed, fixed_vertices=fixed)
            oracle = _z2_orbit_size(3, edges, observed, fixed) == 2 ** len(observed)
            assert (result["status"] == "PASS") == oracle


def test_fixed_vertices_only_count_inside_each_observed_component() -> None:
    result = gauge_forest_projection(
        7, [(0, 1), (1, 2), (3, 4)], [1, 2, 3], fixed_vertices=[0, 4, 5, 6]
    )
    assert result["status"] == "PASS"
    assert [c["fixed_vertices"] for c in result["components"]] == [[0], [4]]
    assert [c["root"] for c in result["components"]] == [0, 4]


def test_fixed_boundary_path_is_an_obstruction_even_without_cycles() -> None:
    result = gauge_forest_projection(3, [(0, 1), (1, 2)], [1, 2], fixed_vertices=[0, 2])
    assert result["status"] == "INCONCLUSIVE"
    assert result["observed_forest_verified"] is True
    assert result["fixed_terminal_condition_verified"] is False
    assert result["obstructions"][0]["oriented_path_edge_ids"] == [1, 2]
    assert result["conditional_conclusion"] is None
    assert replay(result["certificate"])


@pytest.mark.parametrize(
    "edges,observed",
    [
        ([(0, 0)], [1]),
        ([(0, 1), (0, 1)], [1, 2]),
        ([(0, 1), (1, 2), (2, 0)], [1, 2, 3]),
    ],
)
def test_cycle_obstructions_are_closed_oriented_walks(
    edges: list[Edge], observed: list[int]
) -> None:
    result = gauge_forest_projection(3, edges, observed)
    assert result["status"] == "INCONCLUSIVE"
    obstruction = result["obstructions"][0]
    tokens = obstruction["closed_oriented_edge_ids"]
    oriented = [edges[t - 1] if t > 0 else tuple(reversed(edges[-t - 1])) for t in tokens]
    assert all(oriented[i][1] == oriented[(i + 1) % len(oriented)][0] for i in range(len(oriented)))
    assert len({abs(t) for t in tokens}) == len(tokens)
    assert replay(result["certificate"])


def test_original_orientations_do_not_change_transitivity() -> None:
    original = [(0, 1), (2, 1), (2, 3), (4, 2)]
    expected = gauge_forest_projection(5, original, [1, 2, 3, 4], fixed_vertices=[3])
    for bits in product((0, 1), repeat=len(original)):
        graph = [(v, u) if bit else (u, v) for (u, v), bit in zip(original, bits, strict=True)]
        result = gauge_forest_projection(5, graph, [1, 2, 3, 4], fixed_vertices=[3])
        assert result["status"] == expected["status"] == "PASS"


def test_rooted_elimination_on_noncommuting_rational_quaternions() -> None:
    rng = Random(18941)
    bank: list[Quaternion] = [
        ONE,
        (Q(0), Q(1), Q(0), Q(0)),
        (Q(0), Q(0), Q(1), Q(0)),
        (Q(3, 5), Q(4, 5), Q(0), Q(0)),
        (Q(1, 2), Q(1, 2), Q(1, 2), Q(1, 2)),
    ]
    base = [(0, 1), (1, 2), (1, 3), (3, 4), (5, 6)]
    for bits in product((0, 1), repeat=len(base)):
        graph = [(v, u) if bit else (u, v) for (u, v), bit in zip(base, bits, strict=True)]
        result = gauge_forest_projection(7, graph, [1, 2, 3, 4, 5], fixed_vertices=[4, 5])
        assert result["status"] == "PASS"
        for _ in range(8):
            links = [rng.choice(bank) for _ in graph]
            gauges = [ONE for _ in range(7)]
            for component in result["components"]:
                for step in component["spanning_tree_gauge_order"]:
                    token = step["oriented_edge_id"]
                    value = links[abs(token) - 1]
                    transport = value if token > 0 else _inverse(value)
                    gauges[step["child"]] = _mul(gauges[step["parent"]], transport)
            assert gauges[4] == gauges[5] == ONE
            for (u, v), link in zip(graph, links, strict=True):
                assert _mul(_mul(gauges[u], link), _inverse(gauges[v])) == ONE


def test_path_and_cycle_nonconstant_controls_use_exact_su2_gauge_actions() -> None:
    examples = [
        ([(0, 1), (1, 2)], [0, 2], [1, 2]),
        ([(0, 1), (1, 2), (2, 0)], [], [1, 2, 3]),
    ]
    minus_one: Quaternion = (Q(-1), Q(0), Q(0), Q(0))
    for graph, fixed, tokens in examples:
        for special in (ONE, minus_one):
            links = [special] + [ONE] * (len(graph) - 1)
            gauges: list[Quaternion] = [
                ONE if v in fixed else (Q(1, 2), Q(1, 2), Q(1, 2), Q(1, 2)) for v in range(3)
            ]
            transformed = [
                _mul(_mul(gauges[u], link), _inverse(gauges[v]))
                for (u, v), link in zip(graph, links, strict=True)
            ]
            assert 2 * _holonomy(transformed, tokens)[0] == 2 * special[0]
        result = gauge_forest_projection(
            3, graph, list(range(1, len(graph) + 1)), fixed_vertices=fixed
        )
        assert result["status"] == "INCONCLUSIVE"


def test_exact_z2_conditional_expectation_control() -> None:
    # On a triangle, averaging an invariant cycle parity over the unobserved
    # edge makes its conditional expectation on a two-edge forest zero.
    for observed_bits in product((0, 1), repeat=2):
        values = [(-1) ** (sum(observed_bits) + third) for third in (0, 1)]
        assert sum(values) == 0
    # Fixing both endpoints of a two-edge path makes its parity invariant
    # under the middle gauge, so conditioning on the full path preserves it.
    for links in product((0, 1), repeat=2):
        value = (-1) ** sum(links)
        for middle in (0, 1):
            assert (-1) ** sum(bit ^ middle for bit in links) == value


def test_sorted_ids_are_canonical_but_edge_orientation_and_order_are_data() -> None:
    graph = [(0, 1), (1, 2)]
    a = gauge_forest_projection(3, graph, [2, 1], fixed_vertices=[2])
    b = gauge_forest_projection(3, graph, [1, 2], fixed_vertices=[2])
    assert a["certificate"] == b["certificate"]
    forged = deepcopy(a["certificate"])
    forged["payload"]["inputs"]["observed_edge_ids"] = [2, 1]
    assert not replay(seal_certificate(forged))


@pytest.mark.parametrize(
    "case",
    [
        "conclusion",
        "source",
        "topology",
        "root",
        "cycle",
        "fixed",
        "honesty",
        "meta",
        "claim",
        "parent",
    ],
)
def test_rehashed_tampering_rejected(case: str) -> None:
    cert = gauge_forest_projection(3, [(0, 1), (1, 2), (2, 0)], [1, 2])["certificate"]
    cert = deepcopy(cert)
    p = cert["payload"]
    if case == "conclusion":
        p["conditional_conclusion"] = "arbitrary frozen exteriors are controlled"
    elif case == "source":
        p["actual_vacuum_verified"] = True
    elif case == "topology":
        p["available_gauge_action_transitive"] = False
    elif case == "root":
        p["components"][0]["root"] = 2
    elif case == "cycle":
        p["inputs"]["observed_edge_ids"] = [1, 2, 3]
    elif case == "fixed":
        p["inputs"]["fixed_vertices"] = [0, 2]
    elif case == "honesty":
        cert["honesty"]["input_gauge_invariance_verified"] = True
    elif case == "meta":
        cert["meta"]["transcend_backend"] = "forged"
    elif case == "claim":
        cert["claim"] = "actual continuum gauge theorem"
    else:
        p["continuum_claim"] = True
    assert not replay(seal_certificate(cert))


def test_failed_certificate_forged_as_pass_is_rejected() -> None:
    result = gauge_forest_projection(2, [(0, 1)], [1], fixed_vertices=[0, 1])
    cert = deepcopy(result["certificate"])
    cert["payload"]["status"] = "PASS"
    cert["payload"]["finite_gate_verified"] = True
    assert not replay(seal_certificate(cert))


def test_report_mutation_cannot_change_certificate_or_future_report() -> None:
    result = gauge_forest_projection(2, [(0, 1)], [1])
    expected = deepcopy(result["certificate"])
    result["components"][0]["root"] = 1
    result["inputs"]["edges"][0][0] = 1
    assert result["certificate"] == expected
    assert gauge_forest_projection(2, [(0, 1)], [1])["certificate"] == expected


@pytest.mark.parametrize("value", [True, 2.0, Q(2), "2", None, [], {}])
def test_vertex_count_requires_python_integer(value: Any) -> None:
    with pytest.raises((TypeError, ValueError)):
        gauge_forest_projection(value, [], [])


@pytest.mark.parametrize(
    "edges",
    [
        "01",
        None,
        {},
        [1],
        [(0,)],
        [(0, 1, 2)],
        [(0, True)],
        [(0, Q(1))],
        [(0, 1.0)],
        [(-1, 0)],
        [(0, 2)],
    ],
)
def test_malformed_edges_rejected(edges: Any) -> None:
    with pytest.raises((TypeError, ValueError)):
        gauge_forest_projection(2, edges, [])


@pytest.mark.parametrize(
    "observed", [[0], [-1], [2], [1, 1], [True], [1.0], [Q(1)], "1", {1}, None]
)
def test_malformed_observed_ids_rejected(observed: Any) -> None:
    with pytest.raises((TypeError, ValueError)):
        gauge_forest_projection(2, [(0, 1)], observed)


@pytest.mark.parametrize("fixed", [[-1], [2], [0, 0], [True], [1.0], [Q(1)], "1", {1}, None])
def test_malformed_fixed_vertices_rejected(fixed: Any) -> None:
    with pytest.raises((TypeError, ValueError)):
        gauge_forest_projection(2, [(0, 1)], [1], fixed_vertices=fixed)


def test_negative_count_and_edges_in_empty_graph_rejected() -> None:
    with pytest.raises(ValueError):
        gauge_forest_projection(-1, [], [])
    with pytest.raises(ValueError):
        gauge_forest_projection(0, [(0, 0)], [1])


def test_malformed_replay_top_levels_and_nested_payloads() -> None:
    values: list[Any] = [None, True, [], {}, "", {"payload": []}]
    for value in values:
        assert not replay(value)
    cert = gauge_forest_projection(0, [], [])["certificate"]
    cert["payload"]["inputs"]["edges"] = {"0": [0, 1]}
    assert not replay(seal_certificate(cert))
