# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
r"""NPA moment matrices from a linear functional on reduced words."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from omnibias.sos.npa.words import NCGenerators, Word


@dataclass(frozen=True)
class MomentMatrix:
    """A real symmetric moment matrix ``M[i,j] = phi(w_i^* w_j)``."""

    words: tuple[Word, ...]
    entries: tuple[tuple[float, ...], ...]

    def __post_init__(self) -> None:
        n = len(self.words)
        if len(self.entries) != n or any(len(row) != n for row in self.entries):
            raise ValueError("entries must be a square table indexed by words")


def moment_matrix_from_functional(
    gens: NCGenerators,
    words: tuple[Word, ...],
    functional: Mapping[Word, float],
) -> MomentMatrix:
    """Assemble ``M[i,j] = sign * phi(reduced(w_i^* w_j))``.

    Missing functional entries default to ``0``.  The identity word must
    typically carry ``phi(()) = 1`` for a state, but that is the caller's
    normalization -- this function does not impose it.
    """
    n = len(words)
    table: list[list[float]] = []
    for i in range(n):
        row: list[float] = []
        adj_i = gens.adjoint(words[i])
        for j in range(n):
            sign, reduced = gens.multiply_signed(adj_i, (1, words[j]))
            row.append(float(sign) * float(functional.get(reduced, 0.0)))
        table.append(row)
    return MomentMatrix(words=words, entries=tuple(tuple(row) for row in table))


__all__ = [
    "MomentMatrix",
    "moment_matrix_from_functional",
]
