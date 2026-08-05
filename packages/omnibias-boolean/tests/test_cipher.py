# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""S-box cryptanalysis against published AES / PRESENT figures of merit."""

from __future__ import annotations

from omnibias.boolean._core.truth_table import truth_table_from_callable
from omnibias.boolean._core.verified import linear_bias_iv
from omnibias.boolean.cipher import (
    SBox,
    difference_distribution_table,
    differential_uniformity,
    higher_order_derivative,
    linear_approximation_table,
    linearity,
    nonlinearity,
    sbox_algebraic_degree,
    sbox_from_table,
    sbox_higher_order_derivative,
)

# PRESENT 4-bit S-box (ISO/IEC 29192-2): du = 4, NL = 4, deg = 3.
PRESENT = sbox_from_table(
    [0xC, 0x5, 0x6, 0xB, 0x9, 0x0, 0xA, 0xD, 0x3, 0xE, 0xF, 0x8, 0x4, 0x7, 0x1, 0x2],
    out_bits=4,
)


# ---- AES S-box generated from the GF(2^8) inverse + affine map -------------- #
def _gf_mul(a: int, b: int) -> int:
    p = 0
    for _ in range(8):
        if b & 1:
            p ^= a
        hi = a & 0x80
        a = (a << 1) & 0xFF
        if hi:
            a ^= 0x1B  # x^8 = x^4 + x^3 + x + 1
        b >>= 1
    return p


def _gf_inv(a: int) -> int:
    if a == 0:
        return 0
    r, base, e = 1, a, 254  # a^254 = a^{-1} in GF(2^8)*
    while e:
        if e & 1:
            r = _gf_mul(r, base)
        base = _gf_mul(base, base)
        e >>= 1
    return r


def _rotl8(x: int, r: int) -> int:
    return ((x << r) | (x >> (8 - r))) & 0xFF


def _aes_table() -> list[int]:
    out = []
    for a in range(256):
        inv = _gf_inv(a)
        out.append(inv ^ _rotl8(inv, 1) ^ _rotl8(inv, 2) ^ _rotl8(inv, 3) ^ _rotl8(inv, 4) ^ 0x63)
    return out


AES = sbox_from_table(_aes_table(), out_bits=8)


# ---- sanity on the constructions ------------------------------------------- #
def test_aes_table_known_entries() -> None:
    assert AES.table[0] == 0x63
    assert AES.table[1] == 0x7C
    assert AES.table[255] == 0x16
    assert AES.is_bijective()


def test_present_is_bijective() -> None:
    assert PRESENT.is_bijective()


# ---- published differential / linear figures of merit ---------------------- #
def test_present_metrics() -> None:
    assert differential_uniformity(PRESENT) == 4
    assert linearity(PRESENT) == 8
    assert nonlinearity(PRESENT) == 4
    assert sbox_algebraic_degree(PRESENT) == 3


def test_aes_metrics() -> None:
    assert differential_uniformity(AES) == 4
    assert linearity(AES) == 32
    assert nonlinearity(AES) == 112
    assert sbox_algebraic_degree(AES) == 7


# ---- DDT / LAT structural identities ---------------------------------------- #
def test_ddt_row_sums_and_origin() -> None:
    ddt = difference_distribution_table(PRESENT)
    n = PRESENT.in_bits
    assert ddt[0][0] == (1 << n)
    for row in ddt:
        assert sum(row) == (1 << n)
        assert all(v % 2 == 0 for v in row)  # differences come in pairs


def test_lat_parseval_and_origin() -> None:
    lat = linear_approximation_table(PRESENT)
    n = PRESENT.in_bits
    assert lat[0][0] == (1 << n)
    # column b = 0 is the constant component: 2^n at a = 0, else 0.
    assert all(lat[a][0] == (0 if a else (1 << n)) for a in range(1 << n))
    # Parseval per output mask: sum_a W(a,b)^2 = 2^{2n}.
    for b in range(1, 1 << PRESENT.out_bits):
        assert sum(lat[a][b] ** 2 for a in range(1 << n)) == (1 << (2 * n))


# ---- higher-order differential cryptanalysis (exact n-th derivative) -------- #
def test_higher_order_derivative_annihilates_above_degree() -> None:
    # f = x0 x1 + x2 x3 has algebraic degree 2.
    f = truth_table_from_callable(lambda a, b, c, d: (a & b) ^ (c & d), 4)
    # The 2nd-order derivative along {e0, e1} is the constant 1 (d^2(x0 x1)=1).
    d2 = higher_order_derivative(f, [0b0001, 0b0010])
    assert set(d2) == {1}
    # Any 3rd-order derivative (degree + 1) is identically zero.
    d3 = higher_order_derivative(f, [0b0001, 0b0010, 0b0100])
    assert set(d3) == {0}


def test_sbox_full_space_derivative_vanishes_below_dimension() -> None:
    # AES has algebraic degree 7 < 8, so the 8th-order derivative over the whole
    # input space annihilates every output bit.
    basis = [1 << i for i in range(8)]
    d8 = sbox_higher_order_derivative(AES, basis)
    assert set(d8) == {0}


def test_present_degree3_eighth_order_on_subspace() -> None:
    # A 4th-order derivative (degree 3 + 1) of any PRESENT component vanishes.
    comp = PRESENT.component(0b0001)
    d4 = higher_order_derivative(comp, [0b0001, 0b0010, 0b0100, 0b1000])
    assert set(d4) == {0}


# ---- bridge: rigorous interval bias bound of an S-box component ------------- #
def test_component_linear_bias_matches_lat() -> None:
    # The verified (interval) linear bias of a component must contain the exact
    # LAT-derived bias W(a,b) / 2^(n+1) -- "now with rigorous bounds".
    lat = linear_approximation_table(PRESENT)
    n = PRESENT.in_bits
    for b in (0b0001, 0b0011, 0b1010):
        comp = PRESENT.component(b)
        for a in range(1 << n):
            exact = lat[a][b] / (1 << (n + 1))
            assert linear_bias_iv(comp, a).contains(exact)


def test_invalid_sbox_rejected() -> None:
    import pytest

    with pytest.raises(ValueError):
        SBox(table=(0, 1, 2), out_bits=2)  # length not a power of two
    with pytest.raises(ValueError):
        SBox(table=(0, 1, 2, 9), out_bits=2)  # output 9 exceeds 2 bits
