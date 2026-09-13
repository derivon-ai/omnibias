# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Exact Gauss-law dimensions, with an independent SU(2) weight-character check."""

from __future__ import annotations

import copy
import math
import random
from itertools import permutations, product

import pytest
from omnibias.geometry.gauge.transfer.charged_sectors import (
    charged_spin_network_dimension,
    su2_singlet_multiplicity,
)


def _weight_character_singlets(spins: tuple[int, ...]) -> int:
    """Independent oracle: weight-zero minus weight-two multiplicity.

    Each integer-spin irrep has one weight zero; each nontrivial one also
    has one weight two. Subtracting cancels every nontrivial irrep. Odd
    twice-spin tensor products have neither weight.
    """
    weights = {0: 1}
    for spin in spins:
        updated: dict[int, int] = {}
        for weight, count in weights.items():
            for magnetic in range(-spin, spin + 1, 2):
                total = weight + magnetic
                updated[total] = updated.get(total, 0) + count
        weights = updated
    return weights.get(0, 0) - weights.get(2, 0)


def test_empty_trivial_factors_and_higher_valence_multiplicities() -> None:
    assert su2_singlet_multiplicity(()) == 1
    assert su2_singlet_multiplicity((0, 0, 0)) == 1
    assert su2_singlet_multiplicity((1,)) == 0
    assert su2_singlet_multiplicity((2,)) == 0
    assert su2_singlet_multiplicity((1, 1)) == 1
    assert su2_singlet_multiplicity((1, 1, 1)) == 0
    assert su2_singlet_multiplicity((1, 1, 1, 1)) == 2
    assert su2_singlet_multiplicity((1,) * 6) == 5
    assert su2_singlet_multiplicity((2,) * 6) == 15
    assert su2_singlet_multiplicity((0, 1, 0, 1, 0, 1, 1)) == 2


def test_exhaustive_small_fusion_grid_matches_weight_characters() -> None:
    for valence in range(7):
        for labels in product(range(4), repeat=valence):
            assert su2_singlet_multiplicity(labels) == _weight_character_singlets(labels)


def test_random_mixed_spins_and_permutation_invariance() -> None:
    rng = random.Random(1978)
    for _ in range(256):
        labels = tuple(rng.randrange(8) for _ in range(rng.randrange(13)))
        actual = su2_singlet_multiplicity(labels)
        assert actual == _weight_character_singlets(labels)
        assert actual == su2_singlet_multiplicity(tuple(reversed(labels)))
    for labels in permutations((1, 2, 3, 4, 2)):
        assert su2_singlet_multiplicity(labels) == su2_singlet_multiplicity((1, 2, 3, 4, 2))


def test_large_integer_multiplicities_have_no_float_or_fixed_width_rounding() -> None:
    assert su2_singlet_multiplicity((1,) * 100) == math.comb(100, 50) // 51
    spin = 10**100
    assert su2_singlet_multiplicity((spin, spin)) == 1
    assert su2_singlet_multiplicity((spin, spin, spin)) == 1
    assert su2_singlet_multiplicity((spin,) * 4) == spin + 1


def test_fundamental_charge_pair_requires_an_admissible_flux_path() -> None:
    path = ((0, 1), (1, 2), (2, 3))
    charges = {0: (1,), 3: (1,)}
    assert charged_spin_network_dimension(4, path, (1, 1, 1), charges) == 1
    assert charged_spin_network_dimension(4, path, (1, 1, 1)) == 0
    assert charged_spin_network_dimension(4, path, (0, 0, 0), charges) == 0
    assert charged_spin_network_dimension(4, path, (1, 0, 1), charges) == 0
    assert charged_spin_network_dimension(4, path, (1, 1, 1), {0: (1,)}) == 0
    assert charged_spin_network_dimension(4, tuple((b, a) for a, b in path),
                                          (1, 1, 1), charges) == 1


def test_theta_graph_trivalent_admissibility_and_parallel_links() -> None:
    theta = ((0, 1), (0, 1), (0, 1))
    for a, b, c in product(range(6), repeat=3):
        expected = int((a + b + c) % 2 == 0 and abs(a - b) <= c <= a + b)
        assert charged_spin_network_dimension(2, theta, (a, b, c)) == expected
    assert charged_spin_network_dimension(2, ((0, 1),) * 4, (1,) * 4) == 4
    assert charged_spin_network_dimension(2, ((0, 1),) * 6, (1,) * 6) == 25


def test_self_loops_count_both_representation_factors() -> None:
    for spin in range(12):
        assert charged_spin_network_dimension(1, ((0, 0),), (spin,)) == 1
    assert charged_spin_network_dimension(1, ((0, 0),) * 3, (1,) * 3) == 5
    assert charged_spin_network_dimension(1, ((0, 0),) * 2, (2,) * 2) == 3
    assert charged_spin_network_dimension(1, ((0, 0),), (1,), {0: (2,)}) == 1
    assert charged_spin_network_dimension(1, ((0, 0),), (1,), {0: (1,)}) == 0


def test_six_valent_periodic_cube_and_charged_parity_defect() -> None:
    # A 2x2x2 periodic cubic lattice: the 24 oriented positive-axis links
    # include parallel geometric neighbors; every vertex has six ends.
    edges = tuple((vertex, vertex ^ (1 << axis)) for vertex in range(8) for axis in range(3))
    assert len(edges) == 24
    assert charged_spin_network_dimension(8, edges, (1,) * 24) == 5**8
    assert charged_spin_network_dimension(8, edges, (2,) * 24) == 15**8
    charges = {0: (1,), 1: (1,)}
    assert charged_spin_network_dimension(8, edges, (1,) * 24, charges) == 0
    changed = (0,) + (1,) * 23
    assert edges[0] == (0, 1)
    assert charged_spin_network_dimension(8, edges, changed, charges) == 5**8
    minimal = (1,) + (0,) * 23
    assert charged_spin_network_dimension(8, edges, minimal, charges) == 1


def test_empty_graphs_isolated_vertices_and_labelled_external_factors() -> None:
    assert charged_spin_network_dimension(0, (), ()) == 1
    assert charged_spin_network_dimension(10**10, (), ()) == 1
    assert charged_spin_network_dimension(1, (), (), {0: (1, 1)}) == 1
    assert charged_spin_network_dimension(1, (), (), {0: (1, 1, 1, 1)}) == 2
    assert charged_spin_network_dimension(2, (), (), {0: (1,), 1: (1,)}) == 0
    assert charged_spin_network_dimension(1, (), (), {0: (2,)}) == 0


@pytest.mark.parametrize("invalid", (True, False, 1.0, "1", None))
def test_strict_spin_and_vertex_integer_types(invalid: object) -> None:
    with pytest.raises(TypeError):
        su2_singlet_multiplicity((0, invalid))  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        charged_spin_network_dimension(invalid, (), ())  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        charged_spin_network_dimension(2, ((0, invalid),), (1,))  # type: ignore[arg-type]


def test_invalid_graphs_and_external_charges_do_not_silently_return_zero() -> None:
    with pytest.raises(ValueError, match="nonnegative"):
        su2_singlet_multiplicity((1, -1))
    with pytest.raises(TypeError, match="sequence"):
        su2_singlet_multiplicity({1, 2})  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="sequence"):
        su2_singlet_multiplicity(b"\x01\x01")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="one twice-spin"):
        charged_spin_network_dimension(2, ((0, 1),), ())
    with pytest.raises(ValueError, match="exactly two"):
        charged_spin_network_dimension(2, ((0, 1, 0),), (1,))  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="smaller"):
        charged_spin_network_dimension(2, ((0, 2),), (1,))
    with pytest.raises(ValueError, match="smaller"):
        charged_spin_network_dimension(0, ((0, 0),), (0,))
    with pytest.raises(ValueError, match="smaller"):
        charged_spin_network_dimension(2, (), (), {2: ()})
    with pytest.raises(TypeError, match="integer"):
        charged_spin_network_dimension(2, (), (), {False: (1,)})
    with pytest.raises(TypeError, match="integer"):
        charged_spin_network_dimension(2, (), (), {0: (1,), 1: (1.0,)})  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="nonnegative"):
        charged_spin_network_dimension(2, (), (), {0: (1,), 1: (-1,)})
    with pytest.raises(TypeError, match="map"):
        charged_spin_network_dimension(2, (), (), [(0, (1,))])  # type: ignore[arg-type]


def test_input_sequences_and_charge_lists_are_unchanged() -> None:
    edges = [(0, 1), (0, 1)]
    spins = [1, 1]
    charges = {0: [1, 1], 1: [0]}
    original = copy.deepcopy((edges, spins, charges))
    assert charged_spin_network_dimension(2, edges, spins, charges) == 2
    assert (edges, spins, charges) == original
