# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Algebraic Normal Form (Reed-Muller) over GF(2) via the Mobius transform.

Every Boolean function has a unique multilinear polynomial over GF(2),

.. math::

    f(x) = \bigoplus_{S \subseteq [n]} a_S \prod_{i \in S} x_i,
    \qquad a_S \in \{0, 1\},

the *algebraic normal form* (ANF), equivalently its Reed-Muller expansion. The
coefficients ``a_S`` are obtained from the truth table by the binary Mobius
transform -- an in-place XOR butterfly that is its own inverse over GF(2). The
ANF is indexed by subset bitmask with the same LSB-first convention as
:mod:`omnibias.boolean._core.truth_table` (bit ``j`` set <=> ``x_j`` in the
monomial).
"""

from __future__ import annotations

from omnibias.boolean._core.truth_table import TruthTable, check_truth_table, num_vars

ANF = tuple[int, ...]


def anf_from_truth_table(table: TruthTable) -> ANF:
    """ANF/Reed-Muller coefficients of ``table`` via the GF(2) Mobius transform."""
    check_truth_table(table)
    a = list(table)
    size = len(a)
    step = 1
    while step < size:
        for i in range(0, size, step << 1):
            for j in range(i, i + step):
                a[j + step] ^= a[j]
        step <<= 1
    return tuple(a)


def truth_table_from_anf(anf: ANF) -> TruthTable:
    """Inverse transform: rebuild the truth table from ANF coefficients.

    The binary Mobius transform is an involution, so this is the same butterfly.
    """
    check_truth_table(anf)
    return anf_from_truth_table(anf)


def algebraic_degree(anf: ANF) -> int:
    """Algebraic degree = largest monomial size with a non-zero ANF coefficient.

    The all-zero function (``f == 0``) has degree ``-1`` by convention; a non-zero
    constant has degree ``0``.
    """
    deg = -1
    for mask, coeff in enumerate(anf):
        if coeff:
            deg = max(deg, bin(mask).count("1"))
    return deg


def anf_monomials(anf: ANF) -> list[frozenset[int]]:
    """Variable-index sets of the monomials present in the ANF (``a_S == 1``)."""
    n = num_vars(anf)
    return [
        frozenset(j for j in range(n) if (mask >> j) & 1)
        for mask, coeff in enumerate(anf)
        if coeff
    ]


def anf_to_string(anf: ANF) -> str:
    """Human-readable ANF, e.g. ``"x0*x1 + x2 + 1"`` (``"0"`` for the zero function)."""
    n = num_vars(anf)
    terms: list[str] = []
    for mask, coeff in enumerate(anf):
        if not coeff:
            continue
        if mask == 0:
            terms.append("1")
        else:
            terms.append("*".join(f"x{j}" for j in range(n) if (mask >> j) & 1))
    return " + ".join(terms) if terms else "0"


__all__ = [
    "ANF",
    "algebraic_degree",
    "anf_from_truth_table",
    "anf_monomials",
    "anf_to_string",
    "truth_table_from_anf",
]
