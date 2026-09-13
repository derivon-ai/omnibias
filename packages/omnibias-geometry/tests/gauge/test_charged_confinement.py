# SPDX-License-Identifier: Apache-2.0
"""Charged center parity, disjoint graph cuts and actual-vacuum certificate chains."""

import random
from copy import deepcopy
from fractions import Fraction as Q
from itertools import combinations, product

import numpy as np
import pytest
from omnibias.core.proof.certificate import seal_certificate, verify_certificate_digest
from omnibias.geometry.gauge.transfer.charged_confinement import (
    replay_su2_static_confinement_certificate,
    su2_static_confinement_bounds,
    su2_static_confinement_family,
)
from omnibias.geometry.gauge.transfer.charged_sectors import charged_spin_network_dimension
from omnibias.geometry.gauge.transfer.vacuum_fourier import (
    su2_vacuum_fourier_bounds,
    su2_vacuum_fourier_family,
)

SQUARE = [(0, 1), (1, 2), (2, 3), (3, 0)]
LOOP = [[1, 2, 3, 4]]


def _fourier():
    return su2_vacuum_fourier_bounds(4, SQUARE, plaquettes=LOOP)["certificate"]


def test_square_actual_static_bounds_and_complete_bfs_witness():
    report = su2_static_confinement_bounds(_fourier(), 0, 2)
    assert report["finite_gate_verified"]
    assert report["finite_graph_static_confinement"]
    assert report["finite_graph_family_confinement"]
    assert replay_su2_static_confinement_certificate(report["certificate"])
    w = report["witness"]
    assert w["static_energy_enclosure"] == ["28", "48"]
    assert w["linear_energy_coefficients"] == ["14", "24"]
    assert w["conditional_curvature_lower"] == "7/16"
    assert w["source_bare_rest_energy"] == "0"
    assert w["graph_distance"] == 2 and w["path"] == [1, 2]
    cut = w["cut_witness"]
    assert cut["distance_labels"] == [0, 1, 2, 1]
    assert [c["edges"] for c in cut["cuts"]] == [[1, 4], [2, 3]]
    assert cut["edge_cut_multiplicity"] == [1, 1, 1, 1]
    assert w["path_trial_pointwise_HS_norm_squared"] == "1"
    assert w["path_trial_total_casimir"] == "3/2"


def test_family_is_connected_finite_graph_static_confinement_not_a_tension_limit():
    family = su2_vacuum_fourier_family(128, 4, decay_base=2)["certificate"]
    report = su2_static_confinement_family(family)
    assert report["finite_graph_family_confinement"]
    assert not report["finite_graph_static_confinement"]
    assert report["witness"]["linear_energy_coefficients"] == ["28", "48"]
    assert "connected" in report["witness"]["family_class"]
    assert replay_su2_static_confinement_certificate(report["certificate"])
    for flag in (
        "infinite_volume_static_potential_claim",
        "asymptotic_string_tension_claim",
        "string_tension_limit_claim",
        "infinite_volume_claim",
        "uniform_in_a_claim",
        "continuum_claim",
        "yang_mills_claim",
        "theorem_prover_verified",
        "mathlib_verified",
        "analytic_implication_formally_verified",
    ):
        assert report[flag] is False


def test_fundamental_gauss_oracle_forces_odd_center_parity_on_every_cut():
    fixtures = [
        (4, SQUARE, LOOP, 2, 2),
        (6, SQUARE + [(1, 4), (4, 5), (5, 2)], LOOP + [[5, 6, 7, -2]], 5, 2),
    ]
    accepted = 0
    for n, edges, loops, target, max_spin in fixtures:
        parent = su2_vacuum_fourier_bounds(n, edges, plaquettes=loops)["certificate"]
        w = su2_static_confinement_bounds(parent, 0, target)["witness"]
        cut_sets = [set(c["edges"]) for c in w["cut_witness"]["cuts"]]
        path = set(map(abs, w["path"]))
        for cut in cut_sets:
            assert len(cut & path) % 2 == 1
            assert all(len(cut & set(map(abs, loop))) % 2 == 0 for loop in loops)
        for spins in product(range(max_spin + 1), repeat=len(edges)):
            dimension = charged_spin_network_dimension(n, edges, spins, {0: [1], target: [1]})
            if not dimension:
                continue
            accepted += 1
            assert all(sum(spins[e - 1] for e in cut) % 2 == 1 for cut in cut_sets)
    assert accepted > 10


def test_distances_match_independent_floyd_warshall_on_grid_random_graphs():
    rng = random.Random(924)
    for n in range(3, 11):
        edges = [(i, i + 1) for i in range(n - 1)]
        edges += [(i, j) for i, j in combinations(range(n), 2) if j > i + 1 and rng.random() < 0.25]
        distance = [[0 if i == j else n for j in range(n)] for i in range(n)]
        for u, v in edges:
            distance[u][v] = distance[v][u] = 1
        for k in range(n):
            for i in range(n):
                for j in range(n):
                    distance[i][j] = min(distance[i][j], distance[i][k] + distance[k][j])
        kappa = Q(rng.randrange(1, 201), 3)
        parent = su2_vacuum_fourier_bounds(n, edges, kappa=kappa)["certificate"]
        for source, target in [(0, n - 1), (n - 1, 0), (rng.randrange(n - 1), n - 1)]:
            report = su2_static_confinement_bounds(parent, source, target)
            w = report["witness"]
            assert w["graph_distance"] == distance[source][target]
            assert w["cut_witness"]["distance_labels"] == distance[source]
            cuts = [set(c["edges"]) for c in w["cut_witness"]["cuts"]]
            assert all(not (a & b) for a, b in combinations(cuts, 2))
            # The zero-magnetic all-spin source energy is exactly the electric
            # shortest-path cost, giving an independent exact endpoint oracle.
            exact = 3 * kappa * distance[source][target] / 8
            lo, hi = map(Q, w["static_energy_enclosure"])
            assert lo <= exact == hi


def test_wilson_path_trial_has_unit_color_norm_and_fundamental_electric_cost():
    rng = np.random.default_rng(870)
    pauli = [np.array([[0, 1], [1, 0]]), np.array([[0, -1j], [1j, 0]]), np.diag([1, -1])]
    generators = [1j * sigma / 2 for sigma in pauli]
    edges = [(1, 0), (1, 2), (3, 2), (3, 4)]
    parent = su2_vacuum_fourier_bounds(5, edges)["certificate"]
    w = su2_static_confinement_bounds(parent, 0, 4)["witness"]
    assert w["path"] == [-1, 2, -3, 4]
    for _ in range(12):
        links = []
        for _edge in edges:
            q = rng.normal(size=4)
            q /= np.linalg.norm(q)
            links.append(q[0] * np.eye(2) + 1j * sum(q[i + 1] * pauli[i] for i in range(3)))
        oriented = [links[abs(t) - 1] if t > 0 else links[abs(t) - 1].conj().T for t in w["path"]]
        transport = np.linalg.multi_dot(oriented) / np.sqrt(2)
        assert np.linalg.norm(transport, "fro") ** 2 == pytest.approx(1, abs=2e-14)
        total = 0.0
        for i, token in enumerate(w["path"]):
            for generator in generators:
                derivative = list(oriented)
                derivative[i] = generator @ oriented[i] if token > 0 else -oriented[i] @ generator
                total += np.linalg.norm(np.linalg.multi_dot(derivative) / np.sqrt(2), "fro") ** 2
        assert total == pytest.approx(3, abs=5e-14)
        # Simultaneous cut flips give a minus sign on the actual path matrix.
        for cut in w["cut_witness"]["cuts"]:
            changed = [
                (-m if abs(t) in cut["edges"] else m)
                for t, m in zip(w["path"], oriented, strict=True)
            ]
            np.testing.assert_allclose(
                np.linalg.multi_dot(changed) / np.sqrt(2), -transport, atol=2e-14, rtol=0
            )


def test_no_factorization_gate_is_needed_and_coarse_family_failure_stays_separate():
    parent = su2_vacuum_fourier_bounds(4, SQUARE, plaquettes=LOOP, kappa=1000, radius=Q(9, 100))
    assert not parent["volume_uniform_factorization_bound_verified"]
    report = su2_static_confinement_bounds(parent["certificate"], 0, 2)
    assert report["finite_graph_family_confinement"]
    coarse = su2_vacuum_fourier_bounds(4, SQUARE, plaquettes=LOOP, weighted_incidence_cap=100)
    specific = su2_static_confinement_bounds(coarse["certificate"], 0, 2)
    assert specific["finite_graph_static_confinement"]
    assert not specific["finite_graph_family_confinement"]


@pytest.mark.parametrize("source,target", [(0, 0), (-1, 2), (0, 4), (True, 2), (0, 1.0)])
def test_bad_sources_refused(source, target):
    with pytest.raises((ValueError, TypeError)):
        su2_static_confinement_bounds(_fourier(), source, target)


def test_disconnected_graph_wrong_certificate_kind_and_supplied_rho_refused():
    disconnected = su2_vacuum_fourier_bounds(4, [(0, 1), (2, 3)])["certificate"]
    with pytest.raises(ValueError):
        su2_static_confinement_bounds(disconnected, 0, 1)
    family = su2_vacuum_fourier_family(64, 4)["certificate"]
    with pytest.raises(ValueError):
        su2_static_confinement_bounds(family, 0, 1)
    with pytest.raises(ValueError):
        su2_static_confinement_family(_fourier())
    with pytest.raises(TypeError):
        su2_static_confinement_bounds(_fourier(), 0, 1, rho=Q(1))
    failed = su2_vacuum_fourier_family(1, 4)["certificate"]
    with pytest.raises(ValueError):
        su2_static_confinement_family(failed)


@pytest.mark.parametrize(
    "change",
    [
        "rho",
        "cut",
        "parity",
        "distance",
        "path",
        "source",
        "energy",
        "upstream",
        "claim",
        "honesty",
        "meta",
    ],
)
def test_rehashed_graph_fabrications_fail_full_chained_replay(change):
    cert = deepcopy(su2_static_confinement_bounds(_fourier(), 0, 2)["certificate"])
    w = cert["payload"]["witness"]
    if change == "rho":
        w["conditional_curvature_lower"] = "1"
    elif change == "cut":
        w["cut_witness"]["cuts"][0]["edges"] = []
    elif change == "parity":
        w["cut_witness"]["cuts"][0]["external_fundamental_source_parity"] = 0
    elif change == "distance":
        w["graph_distance"] = 100
    elif change == "path":
        w["path"][0] *= -1
    elif change == "source":
        w["source"] = 1
    elif change == "energy":
        w["static_energy_enclosure"][0] = "1000000"
    elif change == "upstream":
        parent = w["fourier_certificate"]
        parent["payload"]["witness"]["arithmetic"]["curvature_candidate_lower"] = "1"
        w["fourier_certificate"] = seal_certificate(parent)
    elif change == "claim":
        cert["claim"] = "continuum color confinement"
    elif change == "honesty":
        cert["honesty"]["yang_mills_claim"] = True
    else:
        cert["meta"]["scope"] = "continuum"
    cert = seal_certificate(cert)
    assert verify_certificate_digest(cert)
    assert not replay_su2_static_confinement_certificate(cert)


@pytest.mark.parametrize("change", ["slope", "scope", "parent"])
def test_rehashed_family_fabrications_fail_full_replay(change):
    parent = su2_vacuum_fourier_family(64, 4)["certificate"]
    cert = deepcopy(su2_static_confinement_family(parent)["certificate"])
    w = cert["payload"]["witness"]
    if change == "slope":
        w["linear_energy_coefficients"][0] = "1000"
    elif change == "scope":
        w["family_class"] = "all continuum theories"
    else:
        w["fourier_certificate"] = _fourier()
    cert = seal_certificate(cert)
    assert not replay_su2_static_confinement_certificate(cert)


@pytest.mark.parametrize("malformed", [None, [], (), "not a certificate", 42, True])
def test_malformed_input_never_passes(malformed):
    assert not replay_su2_static_confinement_certificate(malformed)
    with pytest.raises(ValueError):
        su2_static_confinement_bounds(malformed, 0, 1)
    with pytest.raises(ValueError):
        su2_static_confinement_family(malformed)
