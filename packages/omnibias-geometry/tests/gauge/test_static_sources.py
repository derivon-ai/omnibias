# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Independent graph/spin checks and adversarial replay for source bounds."""
import random
from copy import deepcopy
from fractions import Fraction as Q
from itertools import product

import pytest
from omnibias.core.proof.certificate import seal_certificate
from omnibias.geometry.gauge.transfer.static_sources import (
    replay_su2_static_source_certificate,
    su2_static_source_bounds,
)

SQUARE = [(0, 1), (1, 2), (2, 3), (3, 0)]


def _w(*args, **kwargs):
    return su2_static_source_bounds(*args, **kwargs)["witness"]


def test_pure_electric_square_exact_all_spin_identity():
    for source, target, distance in ((0, 1, 1), (0, 2, 2), (3, 1, 2)):
        w = _w(4, SQUARE, source, target, kappa=Q(7, 3))
        assert list(map(Q, w["static_energy_enclosure"])) == [Q(7, 8) * distance] * 2
        assert w["pure_electric_identity"] and w["exact_static_energy"]


def test_public_transfer_exports_are_the_same_implementations():
    from omnibias.geometry.gauge import transfer

    assert transfer.su2_static_source_bounds is su2_static_source_bounds
    assert transfer.replay_su2_static_source_certificate is replay_su2_static_source_certificate
    assert transfer.su2_singlet_multiplicity((1, 1, 1, 1)) == 2
    assert transfer.charged_spin_network_dimension(2, [(0, 1)], [1], {0: [1], 1: [1]}) == 1


def test_weighted_shortest_path_uses_exact_cost_not_hops():
    w = _w(4, SQUARE, 0, 1, edge_weights=[10, 1, Q(1, 2), 1])
    assert w["path"] == [-4, -3, -2]
    assert Q(w["weighted_distance"]) == Q(5, 2)
    assert Q(w["static_energy_enclosure"][0]) == Q(15, 16)


def test_parallel_edges_are_not_bridges():
    w = _w(2, [(0, 1), (0, 1)], 0, 1, plaquettes=[(1, -2)])
    assert w["source_separating_bridges"] == []
    assert w["static_energy_enclosure"] == ["0", "3/8"]


def test_attached_interacting_square_does_not_change_bridge_energy():
    edges = [(0, 1), (1, 2), (1, 3), (3, 4), (4, 5), (5, 1)]
    for coupling in (Q(1, 8), Q(1), Q(9)):
        w = _w(6, edges, 0, 2, kappa=coupling, plaquettes=[(3, 4, 5, 6)],
               magnetic_weights=[100])
        assert w["source_separating_bridges"] == [1, 2]
        assert list(map(Q, w["static_energy_enclosure"])) == [3 * coupling / 4] * 2
        assert not w["pure_electric_identity"]
        assert w["exact_static_energy"]


def test_volume_subtraction_is_not_a_positive_lower_bound():
    w = _w(4, SQUARE, 0, 2, plaquettes=[(1, 2, 3, 4)])
    assert w["static_energy_enclosure"] == ["0", "3/4"]
    assert Q(w["finite_volume_haar_lower"]) == -Q(13, 4)
    assert not w["positive_static_energy_lower_bound"]
    assert not w["string_tension_claim"]


def test_strong_coupling_finite_haar_lower():
    w = _w(4, SQUARE, 0, 2, kappa=4, plaquettes=[(1, 2, 3, 4)])
    assert w["static_energy_enclosure"] == ["2", "3"]
    assert w["positive_static_energy_lower_bound"]
    assert not w["string_tension_claim"]


def test_exhaustive_center_parity_lower_bound_and_attaining_path():
    # An independent finite spin enumeration checks the center-parity theorem.
    # Completeness for arbitrary spins is supplied by its written proof.
    edges = SQUARE + [(0, 2)]
    weights = [Q(3), Q(1, 2), Q(2), Q(5, 3), Q(7, 4)]
    w = _w(4, edges, 1, 3, edge_weights=weights)
    exact = Q(w["pure_electric_path_energy"])
    admissible_energies = []
    for spins in product(range(4), repeat=len(edges)):
        parity = [0] * 4
        for (u, v), spin in zip(edges, spins, strict=True):
            parity[u] ^= spin % 2
            parity[v] ^= spin % 2
        if parity == [0, 1, 0, 1]:
            energy = sum((weight * spin * (spin + 2) / 8
                          for spin, weight in zip(spins, weights, strict=True)), Q(0))
            admissible_energies.append(energy)
    assert min(admissible_energies) == exact


def test_grid_and_random_weighted_graphs_against_bellman_ford():
    rng = random.Random(1907)
    for n in range(2, 8):
        for trial in range(12):
            edges = [(i, i + 1) for i in range(n - 1)]
            edges += [(u, v) for u in range(n) for v in range(u + 2, n)
                      if rng.randrange(3) == 0]
            weights = [Q(1 + rng.randrange(19), 1 + rng.randrange(7)) for _ in edges]
            coupling = Q(trial + 1, 7)
            w = _w(n, edges, 0, n - 1, kappa=coupling, edge_weights=weights)
            distances = [Q(0)] + [None] * (n - 1)
            for _ in range(n - 1):
                for (u, v), weight in zip(edges, weights, strict=True):
                    for a, b in ((u, v), (v, u)):
                        if distances[a] is not None:
                            proposal = distances[a] + weight
                            if distances[b] is None or proposal < distances[b]:
                                distances[b] = proposal
            assert list(map(Q, w["shortest_distances"])) == distances
            assert Q(w["static_energy_enclosure"][0]) == 3 * coupling * distances[-1] / 8


@pytest.mark.parametrize("change", ["bound", "bridge", "scope", "normalization", "trial"])
def test_rehashed_fabricated_certificate_is_rejected(change):
    cert = su2_static_source_bounds(4, SQUARE, 0, 2, plaquettes=[(1, 2, 3, 4)])["certificate"]
    assert replay_su2_static_source_certificate(cert)
    bad = deepcopy(cert)
    w = bad["payload"]["witness"]
    if change == "bound":
        w["static_energy_enclosure"][0] = "1/100"
    elif change == "bridge":
        w["source_separating_bridges"] = [1]
    elif change == "scope":
        bad["honesty"]["string_tension_claim"] = True
    elif change == "normalization":
        w["normalization"] = "physical continuum QCD"
    else:
        w["path_two_spins"] = [0] * 4
    assert not replay_su2_static_source_certificate(seal_certificate(bad))


@pytest.mark.parametrize("kwargs", [
    {"kappa": 1.0}, {"kappa": True}, {"kappa": 0},
    {"edge_weights": [1, 1, 1, 0]}, {"edge_weights": [1]},
    {"plaquettes": [(1, 2, 3)]}, {"plaquettes": [(1, 1)]},
    {"plaquettes": [(0, 2)]}, {"plaquettes": [(1, 2, 3, 5)]},
    {"plaquettes": [(1, 2, 3, 4)], "magnetic_weights": [-1]},
])
def test_invalid_model_refused(kwargs):
    with pytest.raises((TypeError, ValueError)):
        su2_static_source_bounds(4, SQUARE, 0, 2, **kwargs)


def test_disconnected_graph_and_self_loop_refused():
    with pytest.raises(ValueError, match="connected"):
        su2_static_source_bounds(3, [(0, 1)], 0, 1)
    with pytest.raises(ValueError, match="self-loops"):
        su2_static_source_bounds(2, [(0, 0), (0, 1)], 0, 1)
