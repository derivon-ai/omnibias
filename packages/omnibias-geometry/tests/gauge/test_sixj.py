# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Racah 6j textbook containment and two-plaquette sixj magnetics."""

from __future__ import annotations

from fractions import Fraction

import numpy as np
from omnibias.geometry.gauge.transfer.hamiltonian import (
    COUPLING_LOCK,
    certified_hamiltonian_gap,
    su2_two_plaquette_hamiltonian,
)
from omnibias.geometry.gauge.transfer.sixj import (
    TEXTBOOK_SIXJ,
    VANISHING_SIXJ,
    racah_sixj,
)


def test_textbook_sixj_are_contained() -> None:
    for labels, value in TEXTBOOK_SIXJ:
        enclosure = racah_sixj(*labels)
        assert enclosure.contains(float(value)), (labels, enclosure, value)


def test_vanishing_triangles_are_exactly_zero() -> None:
    for labels in VANISHING_SIXJ:
        enclosure = racah_sixj(*labels)
        assert enclosure.lo == 0.0 and enclosure.hi == 0.0


def test_all_half_is_the_illegal_triad() -> None:
    assert VANISHING_SIXJ[0] == (1, 1, 1, 1, 1, 1)
    assert racah_sixj(1, 1, 1, 1, 1, 1).lo == 0.0


def test_default_two_plaquette_uses_sixj_and_certifies() -> None:
    sixj = su2_two_plaquette_hamiltonian(COUPLING_LOCK, j_max=1)
    character = su2_two_plaquette_hamiltonian(
        COUPLING_LOCK, j_max=1, magnetic="character"
    )
    assert sixj.parameters["magnetic"] == "sixj"
    assert character.parameters["magnetic"] == "character"
    sixj_mid = np.array([[0.5 * (c.lo + c.hi) for c in row] for row in sixj.entries])
    char_mid = np.array(
        [[0.5 * (c.lo + c.hi) for c in row] for row in character.entries]
    )
    assert not np.allclose(sixj_mid, char_mid, atol=1e-9)
    np.testing.assert_allclose(sixj_mid, sixj_mid.T, atol=1e-12)
    result = certified_hamiltonian_gap(sixj)
    assert result.certified is True
    assert result.spectral_gap_lower > 0.0


def test_character_magnetic_still_certifies() -> None:
    hamiltonian = su2_two_plaquette_hamiltonian(
        COUPLING_LOCK, j_max=1, magnetic="character"
    )
    result = certified_hamiltonian_gap(hamiltonian)
    assert result.certified is True
