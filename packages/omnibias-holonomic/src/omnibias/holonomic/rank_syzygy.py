# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Holonomic consumer of rank collapse: an exact ``Q`` matrix, then a syzygy.

Clear denominators, then accept only with
:func:`~omnibias.core.collapse.rank.rank_collapse`. A float SVD is not a
proof. This does not certify a special-function identity and does not
settle the Jacobian conjecture.
"""

from __future__ import annotations

from collections.abc import Sequence
from fractions import Fraction
from math import lcm

from omnibias.core.collapse.rank import RankReport, rank_collapse
from omnibias.core.proof.lift import as_fraction

HOLONOMIC_SYZYGY = "holonomic_syzygy"


def integerize_matrix(matrix: Sequence[Sequence[object]]) -> list[list[int]]:
    """Clear denominators of a ``Q`` matrix. Floats and bools are refused."""

    rows: list[list[Fraction]] = []
    width = 0
    for row in matrix:
        if not isinstance(row, Sequence) or isinstance(row, str | bytes):
            raise TypeError("matrix rows must be sequences")
        fracs: list[Fraction] = []
        for entry in row:
            if isinstance(entry, bool) or isinstance(entry, float):
                raise TypeError(
                    "holonomic syzygy requires an exact Q matrix; "
                    "a float singular value is not a certificate"
                )
            if not isinstance(entry, int | Fraction):
                raise TypeError(
                    "holonomic syzygy requires an exact Q matrix; "
                    "a float singular value is not a certificate"
                )
            fracs.append(as_fraction(entry))
        if not fracs:
            raise ValueError("matrix rows must be non-empty")
        if width == 0:
            width = len(fracs)
        elif len(fracs) != width:
            raise ValueError("matrix rows must share a width")
        rows.append(fracs)
    if not rows:
        raise ValueError("matrix must be non-empty")
    denom = 1
    for row in rows:
        for entry in row:
            denom = lcm(denom, entry.denominator)
    return [[int(entry * denom) for entry in row] for row in rows]


def certify_holonomic_syzygy(matrix: Sequence[Sequence[object]]) -> RankReport:
    """Integerize ``matrix`` and accept with rank collapse."""

    return rank_collapse(integerize_matrix(matrix))


__all__ = [
    "HOLONOMIC_SYZYGY",
    "RankReport",
    "certify_holonomic_syzygy",
    "integerize_matrix",
]
