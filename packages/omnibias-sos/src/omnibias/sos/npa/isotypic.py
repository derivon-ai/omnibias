# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
r"""``Z_2`` isotypic splitting of an NPA word basis (even / odd length).

A length-parity grading is a genuine representation of ``Z_2`` on the free
algebra (even words vs odd words).  Block-diagonalizing the moment matrix
along this grading is an *exact* reduction, not an approximation: words of
different parity never mix under a parity-preserving Hamiltonian.  This is
one isotypic component, not a full character table of a larger group.
"""

from __future__ import annotations

from dataclasses import dataclass

from omnibias.sos.npa.words import Word


@dataclass(frozen=True)
class IsotypicBlocks:
    """Even-length and odd-length words (the two ``Z_2`` isotypic components)."""

    even: tuple[Word, ...]
    odd: tuple[Word, ...]


def z2_isotypic_blocks(words: tuple[Word, ...]) -> IsotypicBlocks:
    """Split a word list by length parity."""
    even = tuple(w for w in words if len(w) % 2 == 0)
    odd = tuple(w for w in words if len(w) % 2 == 1)
    return IsotypicBlocks(even=even, odd=odd)


__all__ = [
    "IsotypicBlocks",
    "z2_isotypic_blocks",
]
