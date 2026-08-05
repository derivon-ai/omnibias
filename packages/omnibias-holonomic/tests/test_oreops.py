# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Ore Euclidean domain: right division, GCRD, LCLM, and symmetric product."""

from __future__ import annotations

from fractions import Fraction

import pytest
from omnibias.holonomic._core.ore import diff_algebra, shift_algebra
from omnibias.holonomic._core.oreops import (
    gcrd,
    lclm,
    ore_divmod,
    symmetric_product,
)


def _fib(n: int) -> int:
    a, b = 0, 1
    for _ in range(n):
        a, b = b, a + b
    return a


def _apply_seq(op, seq, n: int) -> Fraction:
    total = Fraction(0)
    from omnibias.holonomic._core.rational_poly import peval

    for i, c in enumerate(op.coeffs):
        if c:
            total += peval(c, n) * Fraction(seq(n + i))
    return total


def test_ore_divmod_identity() -> None:
    S = shift_algebra()
    a = S.operator([[0, 1], [2], [1]])  # (x) + 2 d + d^2
    b = S.operator([[-2], [1]])  # d - 2
    div = ore_divmod(a, b)
    # left_multiplier * a == quotient * b + remainder, with ord(remainder) < ord(b).
    lhs = S.operator([div.left_multiplier]) * a
    rhs = div.quotient * b + div.remainder
    assert lhs.coeffs == rhs.coeffs
    assert div.remainder.order < b.order


def test_gcrd_common_right_factor() -> None:
    S = shift_algebra()
    a = S.operator([[-2], [1]])  # S - 2 (annihilates 2^n)
    c = S.operator([[-3], [1]])  # S - 3
    product = c * a  # a is a right divisor
    g = gcrd(product, a)
    assert g.order == 1
    # g annihilates 2^n (it is a scalar multiple of S - 2).
    for n in range(6):
        assert _apply_seq(g, lambda k: 2**k, n) == 0


def test_gcrd_coprime_is_trivial() -> None:
    S = shift_algebra()
    a = S.operator([[-2], [1]])  # S - 2
    b = S.operator([[-3], [1]])  # S - 3
    g = gcrd(a, b)
    assert g.order <= 0  # no common right factor


def test_lclm_annihilates_sum() -> None:
    S = shift_algebra()
    fib = S.operator([[-1], [-1], [1]])  # order 2
    geo = S.operator([[-2], [1]])  # 2^n, order 1
    lcm = lclm(fib, geo)
    assert lcm.order <= 3

    def seq(n: int) -> int:
        return _fib(n) + 2**n

    for n in range(8):
        assert _apply_seq(lcm, seq, n) == 0


def test_symmetric_product_shift_hadamard() -> None:
    S = shift_algebra()
    fib = S.operator([[-1], [-1], [1]])
    geo = S.operator([[-2], [1]])
    op = symmetric_product(fib, geo)
    assert 1 <= op.order <= 2

    def seq(n: int) -> int:
        return _fib(n) * 2**n

    for n in range(8):
        assert _apply_seq(op, seq, n) == 0


def test_symmetric_product_differential_exp_squared() -> None:
    D = diff_algebra()
    exp = D.operator([[-1], [1]])  # D - 1
    op = symmetric_product(exp, exp)  # annihilates exp(x)^2 = exp(2x)
    assert op.order == 1

    # exp(2x) Taylor coefficients a_m = 2^m / m!.
    from math import factorial

    coeffs = [Fraction(2**m, factorial(m)) for m in range(12)]
    for order in range(10):
        assert op.apply_series(coeffs, order) == 0


def test_symmetric_product_requires_shared_algebra() -> None:
    S, D = shift_algebra(), diff_algebra()
    with pytest.raises(ValueError, match="share an Ore algebra"):
        symmetric_product(S.operator([[-2], [1]]), D.operator([[-1], [1]]))
