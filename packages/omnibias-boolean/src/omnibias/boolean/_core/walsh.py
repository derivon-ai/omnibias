# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Walsh-Hadamard / Fourier analysis of Boolean functions.

In the ``{+1, -1}`` ("spin") basis every ``f : \{0,1\}^n \to \mathbb{R}`` has a
unique Fourier expansion over the Walsh characters
``chi_S(x) = prod_{i in S} (-1)^{x_i}``,

.. math::

    f(x) = \sum_{S \subseteq [n]} \hat f(S)\, \chi_S(x),
    \qquad \hat f(S) = 2^{-n} \sum_x f(x)\, \chi_S(x).

The coefficients come from the fast Walsh-Hadamard transform (a sum/difference
butterfly) divided by ``2^n``. Subsets are indexed by bitmask (LSB-first, as in
:mod:`omnibias.boolean._core.truth_table`). For a ``{+1,-1}``-valued function the
*influence* of coordinate ``i`` is both the combinatorial flip probability and the
Fourier weight ``sum_{S in i} hat f(S)^2`` -- this module computes (and the tests
cross-check) both.
"""

from __future__ import annotations

from collections.abc import Sequence

from omnibias.boolean._core.truth_table import (
    TruthTable,
    check_truth_table,
    is_power_of_two,
    num_vars,
)


def walsh_hadamard(values: Sequence[float]) -> list[float]:
    """Unnormalized fast Walsh-Hadamard transform of a power-of-two-length array.

    Returns ``W`` with ``W[k] = sum_x (-1)^{<k, x>} values[x]`` where ``<k, x>`` is
    the bitwise dot product (parity of ``k & x``).
    """
    a = [float(v) for v in values]
    size = len(a)
    if not is_power_of_two(size):
        raise ValueError(f"values length must be a power of two, got {size}")
    step = 1
    while step < size:
        for i in range(0, size, step << 1):
            for j in range(i, i + step):
                u = a[j]
                v = a[j + step]
                a[j] = u + v
                a[j + step] = u - v
        step <<= 1
    return a


def fourier_coeffs(table: TruthTable, encoding: str = "pm1") -> tuple[float, ...]:
    """Fourier coefficients ``hat f(S)`` indexed by subset bitmask.

    ``encoding="pm1"`` maps the output to ``{+1, -1}`` (``f = 1 - 2b``, the usual
    convention for which Parseval gives ``sum_S hat f(S)^2 = 1``); ``encoding="01"``
    keeps the ``{0, 1}`` output.
    """
    check_truth_table(table)
    n = num_vars(table)
    if encoding == "pm1":
        vals = [float(1 - 2 * v) for v in table]
    elif encoding == "01":
        vals = [float(v) for v in table]
    else:
        raise ValueError(f"encoding must be 'pm1' or '01', got {encoding!r}")
    wht = walsh_hadamard(vals)
    scale = 1.0 / (1 << n)
    return tuple(c * scale for c in wht)


def walsh_spectrum(table: TruthTable, encoding: str = "pm1") -> dict[frozenset[int], float]:
    """Fourier coefficients as a ``{variable-set: hat f(S)}`` mapping."""
    n = num_vars(table)
    coeffs = fourier_coeffs(table, encoding=encoding)
    return {
        frozenset(j for j in range(n) if (mask >> j) & 1): c
        for mask, c in enumerate(coeffs)
    }


def influences(table: TruthTable) -> list[float]:
    """Combinatorial influence ``Inf_i = Pr_x[f(x) != f(x ^ e_i)]`` per coordinate."""
    check_truth_table(table)
    n = num_vars(table)
    size = 1 << n
    out: list[float] = []
    for i in range(n):
        bit = 1 << i
        diff_pairs = 0
        for x in range(size):
            if (x & bit) == 0 and table[x] != table[x | bit]:
                diff_pairs += 1
        out.append(diff_pairs / (size >> 1))
    return out


def total_influence(table: TruthTable) -> float:
    """Total influence ``I(f) = sum_i Inf_i`` (average sensitivity)."""
    return sum(influences(table))


def fourier_influences(table: TruthTable) -> list[float]:
    """Influence via the Fourier weight ``Inf_i = sum_{S in i} hat f(S)^2``.

    Equal to :func:`influences` for a ``{+1,-1}``-valued function (cross-checked in
    the tests).
    """
    n = num_vars(table)
    coeffs = fourier_coeffs(table, encoding="pm1")
    out = [0.0] * n
    for mask, c in enumerate(coeffs):
        for i in range(n):
            if (mask >> i) & 1:
                out[i] += c * c
    return out


def parseval_defect(table: TruthTable) -> float:
    """``|sum_S hat f(S)^2 - 1|`` for the ``{+1,-1}`` encoding (zero up to rounding)."""
    coeffs = fourier_coeffs(table, encoding="pm1")
    return abs(sum(c * c for c in coeffs) - 1.0)


__all__ = [
    "fourier_coeffs",
    "fourier_influences",
    "influences",
    "parseval_defect",
    "total_influence",
    "walsh_hadamard",
    "walsh_spectrum",
]
