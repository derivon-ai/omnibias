# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
"""NPA words, moment-matrix PSD, isotypic split, linear Hamiltonian bound."""

from __future__ import annotations

from omnibias.sos.npa import (
    moment_matrix_from_functional,
    moment_matrix_is_certified_psd,
    npa_linear_hamiltonian_bound,
    pauli_pair_generators,
    z2_isotypic_blocks,
)


def test_identity_moment_matrix_is_certified_psd() -> None:
    gens = pauli_pair_generators()
    words = gens.words_up_to_length(1)
    moment = moment_matrix_from_functional(gens, words, {(): 1.0})
    assert moment_matrix_is_certified_psd(moment)
    blocks = z2_isotypic_blocks(words)
    assert () in blocks.even
    assert (0,) in blocks.odd
    assert (1,) in blocks.odd


def test_linear_hamiltonian_npa_bound() -> None:
    gens = pauli_pair_generators()
    result = npa_linear_hamiltonian_bound(gens, (-1.0, -1.0), level=1)
    assert result.certified
    assert result.lower_bound == -2.0
    assert result.full_diagonalization_claim is False
