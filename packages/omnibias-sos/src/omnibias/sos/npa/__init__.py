# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
r"""NPA moment hierarchy: noncommutative words, moment matrices, certified PSD.

The Navascués–Pironio–Acín (NPA) hierarchy produces **lower** bounds on a
ground-state energy by relaxing ``<psi|H|psi>`` to a linear functional on
words whose moment matrix is positive semidefinite.  This submodule

* enumerates reduced words (:mod:`omnibias.sos.npa.words`),
* assembles a moment matrix from a linear functional,
* certifies that matrix PSD via interval ``LDL^T``
  (:func:`~omnibias.core.verified.eig_operator.is_positive_definite`),
* splits a word basis into ``Z_2`` isotypic blocks (even / odd length).

It never claims a full diagonalization of an infinite operator, never a
continuum QFT bound, and never ``theorem_prover_verified`` (the PSD check
is the sound-enclosure tier).  Lower bounds only.
"""

from __future__ import annotations

from omnibias.sos.npa.hierarchy import (
    NPALowerBound,
    npa_linear_hamiltonian_bound,
    pauli_pair_generators,
)
from omnibias.sos.npa.isotypic import IsotypicBlocks, z2_isotypic_blocks
from omnibias.sos.npa.moments import MomentMatrix, moment_matrix_from_functional
from omnibias.sos.npa.psd import moment_matrix_is_certified_psd
from omnibias.sos.npa.words import NCGenerators, SignedWord, Word

__all__ = [
    "IsotypicBlocks",
    "MomentMatrix",
    "NCGenerators",
    "NPALowerBound",
    "SignedWord",
    "Word",
    "moment_matrix_from_functional",
    "moment_matrix_is_certified_psd",
    "npa_linear_hamiltonian_bound",
    "pauli_pair_generators",
    "z2_isotypic_blocks",
]
