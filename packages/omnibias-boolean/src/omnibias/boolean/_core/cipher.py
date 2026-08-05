# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""S-box cryptanalysis: differential / linear spectra and higher-order derivatives.

An **S-box** is a vector Boolean function ``S : \{0,1\}^n -> \{0,1\}^m`` stored as
a lookup table.  Its security against the two classical attacks is read off two
exact integer spectra built from the same closed-form machinery as the rest of
:mod:`omnibias.boolean._core`:

* the **difference distribution table** ``DDT[a][b] = #{x : S(x) ^ S(x+a) = b}``
  and its maximum off the first row, the **differential uniformity** -- the
  exact bias available to a differential attacker;
* the **linear approximation table** ``LAT[a][b] = sum_x (-1)^{<a,x> + <b,S(x)>}``
  (the Walsh transform of each component ``<b, S(x)>``) and its maximum, the
  **linearity**, hence the **nonlinearity** ``2^{n-1} - L/2``.

**Higher-order differential cryptanalysis** is the exact ``k``-th Boolean
derivative ``D_{a_1}...D_{a_k} S``: a function of algebraic degree ``d`` is
annihilated by every ``(d+1)``-th order derivative, the structural weakness those
attacks exploit.  Each S-box component is a single Boolean function, so the
rigorous interval bias bounds of :mod:`omnibias.boolean._core.verified` apply
verbatim to a differentiable / noisy S-box relaxation via :meth:`SBox.component`.

Conventions match the package: indices and masks are LSB-first.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from omnibias.boolean._core.anf import algebraic_degree, anf_from_truth_table
from omnibias.boolean._core.truth_table import TruthTable, is_power_of_two


def _parity(v: int) -> int:
    """Parity (XOR of bits) of a non-negative integer."""
    return bin(v).count("1") & 1


def _int_walsh_hadamard(values: list[int]) -> list[int]:
    """Exact integer fast Walsh-Hadamard transform (sum/difference butterfly)."""
    a = list(values)
    size = len(a)
    step = 1
    while step < size:
        for i in range(0, size, step << 1):
            for j in range(i, i + step):
                u, v = a[j], a[j + step]
                a[j] = u + v
                a[j + step] = u - v
        step <<= 1
    return a


@dataclass(frozen=True)
class SBox:
    """An ``n x m`` S-box: ``table[x]`` is the ``m``-bit output ``S(x)`` (LSB-first)."""

    table: tuple[int, ...]
    out_bits: int

    def __post_init__(self) -> None:
        size = len(self.table)
        if not is_power_of_two(size):
            raise ValueError(f"S-box table length must be a power of two, got {size}")
        if self.out_bits < 1:
            raise ValueError("out_bits must be >= 1")
        hi = 1 << self.out_bits
        for v in self.table:
            if not 0 <= v < hi:
                raise ValueError(f"S-box output {v} out of range for {self.out_bits} bits")

    @property
    def in_bits(self) -> int:
        return len(self.table).bit_length() - 1

    def is_bijective(self) -> bool:
        """``True`` iff the S-box is a permutation (``n == m`` and table is a bijection)."""
        return self.in_bits == self.out_bits and len(set(self.table)) == len(self.table)

    def component(self, mask: int) -> TruthTable:
        r"""The single Boolean component function ``x -> <mask, S(x)>`` as a truth table."""
        if not 0 <= mask < (1 << self.out_bits):
            raise ValueError(f"mask {mask} out of range for {self.out_bits} output bits")
        return tuple(_parity(mask & y) for y in self.table)


def sbox_from_table(table: Sequence[int], out_bits: int | None = None) -> SBox:
    """Build an :class:`SBox`; ``out_bits`` defaults to the width of the largest output."""
    tbl = tuple(int(v) for v in table)
    if out_bits is None:
        peak = max(tbl) if tbl else 0
        out_bits = max(1, peak.bit_length())
    return SBox(table=tbl, out_bits=out_bits)


# --------------------------------------------------------------------------- #
# Differential spectrum.
# --------------------------------------------------------------------------- #
def difference_distribution_table(sbox: SBox) -> list[list[int]]:
    """The DDT: ``ddt[a][b] = #{x : S(x) ^ S(x ^ a) = b}`` (rows over input differences)."""
    size = 1 << sbox.in_bits
    cols = 1 << sbox.out_bits
    table = sbox.table
    ddt = [[0] * cols for _ in range(size)]
    for a in range(size):
        row = ddt[a]
        for x in range(size):
            row[table[x] ^ table[x ^ a]] += 1
    return ddt


def differential_uniformity(sbox: SBox) -> int:
    """Differential uniformity ``max_{a != 0, b} DDT[a][b]`` (lower = stronger)."""
    ddt = difference_distribution_table(sbox)
    return max(max(ddt[a]) for a in range(1, len(ddt)))


# --------------------------------------------------------------------------- #
# Linear spectrum.
# --------------------------------------------------------------------------- #
def linear_approximation_table(sbox: SBox) -> list[list[int]]:
    r"""The LAT: ``lat[a][b] = sum_x (-1)^{<a,x> + <b,S(x)>}`` (Walsh per component)."""
    size = 1 << sbox.in_bits
    cols = 1 << sbox.out_bits
    table = sbox.table
    lat = [[0] * cols for _ in range(size)]
    for b in range(cols):
        spin = [1 - 2 * _parity(b & table[x]) for x in range(size)]
        walsh = _int_walsh_hadamard(spin)
        for a in range(size):
            lat[a][b] = walsh[a]
    return lat


def linearity(sbox: SBox) -> int:
    """Linearity ``L(S) = max_{a, b != 0} |sum_x (-1)^{<a,x>+<b,S(x)>}|``."""
    size = 1 << sbox.in_bits
    cols = 1 << sbox.out_bits
    table = sbox.table
    best = 0
    for b in range(1, cols):
        spin = [1 - 2 * _parity(b & table[x]) for x in range(size)]
        walsh = _int_walsh_hadamard(spin)
        best = max(best, max(abs(w) for w in walsh))
    return best


def nonlinearity(sbox: SBox) -> int:
    """Nonlinearity ``NL(S) = 2^{n-1} - L(S)/2`` (higher = stronger)."""
    return (1 << (sbox.in_bits - 1)) - linearity(sbox) // 2


def sbox_algebraic_degree(sbox: SBox) -> int:
    """Algebraic degree = the maximum ANF degree over all nonzero output components."""
    deg = -1
    for b in range(1, 1 << sbox.out_bits):
        deg = max(deg, algebraic_degree(anf_from_truth_table(sbox.component(b))))
    return deg


# --------------------------------------------------------------------------- #
# Higher-order differentials (exact n-th Boolean derivative).
# --------------------------------------------------------------------------- #
def directional_derivative(table: TruthTable, direction: int) -> TruthTable:
    r"""Directional Boolean derivative ``D_a f(x) = f(x) ^ f(x ^ a)``."""
    size = len(table)
    if not is_power_of_two(size):
        raise ValueError(f"truth-table length must be a power of two, got {size}")
    if not 0 <= direction < size:
        raise ValueError(f"direction {direction} out of range for length {size}")
    return tuple(table[x] ^ table[x ^ direction] for x in range(size))


def higher_order_derivative(table: TruthTable, directions: Iterable[int]) -> TruthTable:
    r"""``k``-th order derivative ``D_{a_1} ... D_{a_k} f = sum_{v in span} f(x ^ v)``.

    Order-independent; collapses to the all-zero table once the directions span a
    space larger than the algebraic degree allows (the higher-order-differential
    distinguisher).
    """
    result = table
    for a in directions:
        result = directional_derivative(result, a)
    return result


def sbox_directional_derivative(sbox: SBox, direction: int) -> tuple[int, ...]:
    r"""Vector directional derivative ``S(x) ^ S(x ^ a)`` of the whole S-box."""
    size = 1 << sbox.in_bits
    if not 0 <= direction < size:
        raise ValueError(f"direction {direction} out of range for length {size}")
    table = sbox.table
    return tuple(table[x] ^ table[x ^ direction] for x in range(size))


def sbox_higher_order_derivative(
    sbox: SBox, directions: Iterable[int]
) -> tuple[int, ...]:
    r"""Whole-S-box ``k``-th order derivative ``sum_{v in span(directions)} S(x ^ v)`` (XOR)."""
    size = 1 << sbox.in_bits
    result = list(sbox.table)
    for a in directions:
        if not 0 <= a < size:
            raise ValueError(f"direction {a} out of range for length {size}")
        result = [result[x] ^ result[x ^ a] for x in range(size)]
    return tuple(result)


__all__ = [
    "SBox",
    "difference_distribution_table",
    "differential_uniformity",
    "directional_derivative",
    "higher_order_derivative",
    "linear_approximation_table",
    "linearity",
    "nonlinearity",
    "sbox_algebraic_degree",
    "sbox_directional_derivative",
    "sbox_from_table",
    "sbox_higher_order_derivative",
]
