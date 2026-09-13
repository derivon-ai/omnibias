# SPDX-License-Identifier: Apache-2.0
"""Independent graph metrics, exact thresholds, quantifiers and replay attacks."""

import random
from copy import deepcopy
from fractions import Fraction as Q
from itertools import combinations, product

import pytest
from omnibias.core.proof.certificate import seal_certificate, verify_certificate_digest
from omnibias.geometry.gauge.transfer.vacuum_fourier import (
    replay_su2_vacuum_fourier_certificate,
    su2_vacuum_fourier_bounds,
    su2_vacuum_fourier_family,
)

SQUARE = [(0, 1), (1, 2), (2, 3), (3, 0)]
LOOP = [[1, 2, 3, 4]]


def _box(side):
    vertices = list(product(range(side), repeat=3))
    index = {v: i for i, v in enumerate(vertices)}
    edges = []
    for v in vertices:
        for axis in range(3):
            w = list(v)
            w[axis] += 1
            if w[axis] < side:
                edges.append((index[v], index[tuple(w)]))
    edge_ids = {edge: i + 1 for i, edge in enumerate(edges)}
    loops = []
    for v in vertices:
        for a, b in combinations(range(3), 2):
            if v[a] + 1 == side or v[b] + 1 == side:
                continue
            va, vb, vab = list(v), list(v), list(v)
            va[a] += 1
            vb[b] += 1
            vab[a] += 1
            vab[b] += 1
            corners = [index[v], index[tuple(va)], index[tuple(vab)], index[tuple(vb)]]
            tokens = []
            for u, w in zip(corners, corners[1:] + corners[:1], strict=True):
                tokens.append(edge_ids[(u, w)] if (u, w) in edge_ids else -edge_ids[(w, u)])
            loops.append(tokens)
    return len(vertices), edges, loops


def test_exact_cubic_family_gate_and_optional_actual_factorization():
    report = su2_vacuum_fourier_family(64, 4)
    assert report["finite_gate_verified"]
    assert report["volume_uniform_finite_graph_family_verified"]
    assert report["volume_uniform_factorization_bound_verified"]
    assert replay_su2_vacuum_fourier_certificate(report["certificate"])
    w = report["witness"]
    a = w["arithmetic"]
    assert w["electric_weights"] == "one on every edge"
    assert "aH=kappa/2" in w["normalization"]
    assert w["energy_units"] == "dimensionless aH"
    assert Q(a["forcing_norm_upper"]) == Q(1, 32)
    assert Q(a["quadratic_correction"]) == Q(3, 256)
    assert Q(a["self_map_slack"]) == Q(1, 256)
    assert Q(a["contraction_upper"]) == Q(1, 2)
    assert Q(a["actual_log_vacuum_hessian_row_upper"]) == Q(1, 32)
    assert Q(a["neutral_gap_lower"]) == 14
    assert Q(a["actual_tv_influence_row_upper"]) == Q(121, 196)
    assert Q(a["actual_factorization_constant_upper"]) == Q(196, 75)
    assert a["exponential_hessian_tail_verified"] is False
    for flag in (
        "infinite_volume_claim",
        "uniform_in_a_claim",
        "continuum_claim",
        "yang_mills_claim",
        "static_confinement_claim",
        "string_tension_claim",
        "theorem_prover_verified",
        "mathlib_verified",
        "analytic_implication_formally_verified",
    ):
        assert report[flag] is False


def test_weighted_family_hessian_and_actual_influence_tails():
    report = su2_vacuum_fourier_family(128, 4, decay_base=2, tail_radius=3)
    a = report["witness"]["arithmetic"]
    assert report["finite_gate_verified"]
    assert a["exponential_hessian_tail_verified"]
    assert Q(a["neutral_gap_lower"]) == 28
    assert Q(a["actual_log_vacuum_hessian_tail_row_upper"]) == Q(1, 256)
    assert Q(a["actual_tv_influence_tail_upper"]) == Q(121, 1568)
    assert replay_su2_vacuum_fourier_certificate(report["certificate"])


def test_factorization_remains_a_separate_gate_at_larger_banach_radius():
    report = su2_vacuum_fourier_family(1000, 4, radius=Q(9, 100))
    assert report["finite_gate_verified"]
    assert report["volume_uniform_neutral_gap_claim"]
    assert not report["volume_uniform_factorization_bound_verified"]
    a = report["witness"]["arithmetic"]
    assert Q(a["contraction_upper"]) < 1
    assert Q(a["actual_tv_influence_row_upper"]) > 1
    assert a["actual_factorization_constant_upper"] is None
    assert replay_su2_vacuum_fourier_certificate(report["certificate"])


def test_inclusive_self_map_boundary_and_strict_contraction_boundary():
    # At kappa=64, g=1/1024 and square forcing=cap/128.
    report = su2_vacuum_fourier_family(64, Q(9, 2))
    assert report["finite_gate_verified"]
    assert Q(report["witness"]["arithmetic"]["self_map_slack"]) == 0
    too_large = su2_vacuum_fourier_family(64, Q(9, 2) + Q(1, 1000000))
    assert not too_large["finite_gate_verified"]
    borderline = su2_vacuum_fourier_family(64, 0, radius=Q(3, 32))
    assert not borderline["finite_gate_verified"]
    assert not borderline["witness"]["arithmetic"]["strict_contraction_verified"]


@pytest.mark.parametrize("kappa", [1, 16, 32])
def test_failed_forcing_gate_does_not_certify_a_gap_or_replay_as_passed(kappa):
    report = su2_vacuum_fourier_family(kappa, 4)
    assert report["status"] == "INCONCLUSIVE"
    assert not report["volume_uniform_neutral_gap_claim"]
    assert not report["volume_uniform_factorization_bound_verified"]
    a = report["witness"]["arithmetic"]
    assert a["neutral_gap_lower"] == "0"
    assert a["actual_log_vacuum_hessian_row_upper"] is None
    assert a["actual_tv_influence_row_upper"] is None
    assert verify_certificate_digest(report["certificate"])
    assert not replay_su2_vacuum_fourier_certificate(report["certificate"])


@pytest.mark.parametrize("side", [2, 3, 4, 5])
def test_growing_cubic_volumes_check_every_local_family_hypothesis(side):
    n, edges, loops = _box(side)
    report = su2_vacuum_fourier_bounds(
        n,
        edges,
        plaquettes=loops,
        kappa=128,
        decay_base=2,
        weighted_incidence_cap=4,
        max_cycle_length=4,
        max_cycle_diameter=2,
    )
    assert report["finite_gate_verified"]
    assert report["volume_uniform_finite_graph_family_verified"]
    assert report["explicit_graph_neutral_gap_verified"]
    w = report["witness"]
    assert all(w["structural_cap_checks"].values())
    assert max(map(Q, w["weighted_incidence"])) <= 4
    assert set(w["cycle_ambient_line_graph_diameters"]) == {2}
    assert Q(w["family"]["arithmetic"]["neutral_gap_lower"]) == 28
    assert replay_su2_vacuum_fourier_certificate(report["certificate"])


def test_ambient_shortcuts_reduce_cycle_diameter():
    ring = [(i, (i + 1) % 6) for i in range(6)]
    pairs = {frozenset(edge) for edge in ring}
    complete = ring + [
        (i, j) for i, j in combinations(range(6), 2) if frozenset((i, j)) not in pairs
    ]
    cycle = [list(range(1, 7))]
    plain = su2_vacuum_fourier_bounds(6, ring, plaquettes=cycle, kappa=512, decay_base=2)
    short = su2_vacuum_fourier_bounds(6, complete, plaquettes=cycle, kappa=512, decay_base=2)
    assert plain["witness"]["cycle_ambient_line_graph_diameters"] == [3]
    assert short["witness"]["cycle_ambient_line_graph_diameters"] == [2]
    assert Q(plain["witness"]["forcing_by_edge"][0]) == 2 * Q(
        short["witness"]["forcing_by_edge"][0]
    )


def test_metric_matches_independent_floyd_warshall_grid_random():
    rng = random.Random(204)
    for n in range(3, 10):
        ring = [(i, (i + 1) % n) for i in range(n)]
        present = {frozenset(e) for e in ring}
        extras = [
            (i, j)
            for i, j in combinations(range(n), 2)
            if frozenset((i, j)) not in present and rng.random() < 0.35
        ]
        edges = ring + extras
        count = len(edges)
        distance = [
            [
                0 if i == j else 1 if set(edges[i]) & set(edges[j]) else count + 1
                for j in range(count)
            ]
            for i in range(count)
        ]
        for k in range(count):
            for i in range(count):
                for j in range(count):
                    distance[i][j] = min(distance[i][j], distance[i][k] + distance[k][j])
        expected = max(distance[i][j] for i in range(n) for j in range(n))
        report = su2_vacuum_fourier_bounds(n, edges, plaquettes=[list(range(1, n + 1))], kappa=4096)
        assert report["witness"]["cycle_ambient_line_graph_diameters"] == [expected]


def test_weighted_two_plaquette_incidence_and_orientation():
    edges = SQUARE + [(1, 4), (4, 5), (5, 2)]
    loops = LOOP + [[5, 6, 7, -2]]
    weights = [Q(3, 2), Q(1, 4)]
    report = su2_vacuum_fourier_bounds(6, edges, plaquettes=loops, magnetic_weights=weights)
    w = report["witness"]
    assert list(map(Q, w["weighted_incidence"])) == [
        Q(3, 2),
        Q(7, 4),
        Q(3, 2),
        Q(3, 2),
        Q(1, 4),
        Q(1, 4),
        Q(1, 4),
    ]
    assert max(map(Q, w["forcing_by_edge"])) == Q(7, 512)
    reversed_loops = [[-token for token in reversed(loop)] for loop in loops]
    inverse = su2_vacuum_fourier_bounds(
        6, edges, plaquettes=reversed_loops, magnetic_weights=weights
    )
    assert inverse["witness"]["arithmetic"] == w["arithmetic"]
    assert inverse["witness"]["cycle_ambient_line_graph_diameters"] == [2, 2]


def test_disconnected_components_and_free_edges_do_not_create_infinite_diameters():
    edges = SQUARE + [(4, 5), (5, 6), (6, 7), (7, 4)] + [(8, 9)]
    report = su2_vacuum_fourier_bounds(
        10, edges, plaquettes=LOOP + [[5, 6, 7, 8]], decay_base=2, kappa=128
    )
    assert report["finite_gate_verified"]
    assert report["witness"]["cycle_ambient_line_graph_diameters"] == [2, 2]
    assert report["witness"]["forcing_by_edge"][-1] == "0"
    assert "componentwise" in report["witness"]["family"]["disconnected_components"]


@pytest.mark.parametrize(
    "kwargs,key",
    [
        ({"weighted_incidence_cap": Q(1, 2)}, "weighted_incidence"),
        ({"max_cycle_length": 3}, "cycle_length"),
        ({"max_cycle_diameter": 1}, "cycle_diameter"),
    ],
)
def test_supplied_structural_caps_are_checked_not_trusted(kwargs, key):
    report = su2_vacuum_fourier_bounds(4, SQUARE, plaquettes=LOOP, **kwargs)
    assert not report["finite_gate_verified"]
    assert report["explicit_graph_neutral_gap_verified"]
    assert not report["volume_uniform_finite_graph_family_verified"]
    assert not report["witness"]["structural_cap_checks"][key]
    assert not replay_su2_vacuum_fourier_certificate(report["certificate"])


def test_coarse_family_gate_can_fail_while_direct_graph_gate_passes():
    report = su2_vacuum_fourier_bounds(4, SQUARE, plaquettes=LOOP, weighted_incidence_cap=100)
    assert report["finite_gate_verified"]
    assert report["explicit_graph_neutral_gap_verified"]
    assert not report["volume_uniform_finite_graph_family_verified"]
    assert replay_su2_vacuum_fourier_certificate(report["certificate"])


@pytest.mark.parametrize("family", [False, True])
@pytest.mark.parametrize(
    "change", ["forcing", "tail", "factorization", "gap", "scope", "claim", "honesty", "meta"]
)
def test_rehashed_fabrications_fail_full_replay(family, change):
    report = (
        su2_vacuum_fourier_family(128, 4, decay_base=2)
        if family
        else su2_vacuum_fourier_bounds(4, SQUARE, plaquettes=LOOP, kappa=128, decay_base=2)
    )
    cert = deepcopy(report["certificate"])
    w = cert["payload"]["witness"]
    if change == "forcing":
        w["arithmetic"]["forcing_norm_upper"] = "0"
    elif change == "tail":
        w["arithmetic"]["actual_log_vacuum_hessian_tail_row_upper"] = "0"
    elif change == "factorization":
        w["arithmetic"]["actual_factorization_constant_upper"] = "1"
    elif change == "gap":
        w["arithmetic"]["neutral_gap_lower"] = "1000000"
    elif change == "scope":
        w["family_class" if family else "normalization"] = "continuum quantum Yang-Mills"
    elif change == "claim":
        cert["claim"] = "positive continuum string tension"
    elif change == "honesty":
        cert["honesty"]["yang_mills_claim"] = True
    else:
        cert["meta"]["scope"] = "continuum"
    cert = seal_certificate(cert)
    assert verify_certificate_digest(cert)
    assert not replay_su2_vacuum_fourier_certificate(cert)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"kappa": True},
        {"kappa": 1.0},
        {"kappa": 0},
        {"weighted_incidence_cap": -1},
        {"weighted_incidence_cap": 1.0},
        {"max_cycle_length": True},
        {"max_cycle_length": 1},
        {"max_cycle_diameter": -1},
        {"max_cycle_diameter": 2.0},
        {"decay_base": Q(1, 2)},
        {"decay_base": 2.0},
        {"radius": 0},
        {"radius": 0.01},
        {"tail_radius": True},
        {"tail_radius": -1},
    ],
)
def test_invalid_or_inexact_family_parameters_refused(kwargs):
    with pytest.raises((ValueError, TypeError)):
        su2_vacuum_fourier_family(**({"kappa": 128, "weighted_incidence_cap": 4} | kwargs))


@pytest.mark.parametrize(
    "edges,loops,weights",
    [
        ([], [], None),
        ([(0, 0)], [], None),
        (SQUARE, [[1, 2, 3, 4, 1]], None),
        (SQUARE, [[1, 2, 3]], None),
        (SQUARE, [[1, 2, 3, True]], None),
        (SQUARE, LOOP, [-1]),
        (SQUARE, LOOP, []),
        (SQUARE, LOOP, [1.0]),
    ],
)
def test_shared_graph_validation_and_magnetic_weights_refused(edges, loops, weights):
    with pytest.raises((TypeError, ValueError)):
        su2_vacuum_fourier_bounds(4, edges, plaquettes=loops, magnetic_weights=weights)


@pytest.mark.parametrize("malformed", [None, [], (), "not a certificate", 42, True])
def test_malformed_replay_refused(malformed):
    assert not replay_su2_vacuum_fourier_certificate(malformed)
