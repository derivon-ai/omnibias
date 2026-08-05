# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Ore (skew-polynomial) algebra: non-commutative product and operator application."""

from __future__ import annotations

import math
from fractions import Fraction

from omnibias.holonomic._core.ore import diff_algebra, shift_algebra


def test_shift_commutation_S_times_n() -> None:
    # S . n = (n + 1) . S in the shift algebra.
    S = shift_algebra()
    Sop = S.operator([[], [1]])  # 1 * S^1
    nop = S.operator([[0, 1]])  # n * S^0
    prod = Sop * nop
    # coefficient of S^1 should be (n + 1); of S^0 zero.
    assert prod.coeffs[1] == (Fraction(1), Fraction(1))
    assert prod.order == 1


def test_shift_addition_and_order() -> None:
    S = shift_algebra()
    a = S.operator([[1], [0, 1]])  # 1 + n S
    b = S.operator([[2], [], [3]])  # 2 + 3 S^2
    s = a + b
    assert s.coeffs[0] == (Fraction(3),)
    assert s.coeffs[1] == (Fraction(0), Fraction(1))
    assert s.coeffs[2] == (Fraction(3),)
    assert s.order == 2


def test_shift_operator_annihilates_fibonacci() -> None:
    # S^2 - S - 1 annihilates Fibonacci.
    S = shift_algebra()
    op = S.operator([[-1], [-1], [1]])
    fib = [0, 1]
    for _ in range(20):
        fib.append(fib[-1] + fib[-2])
    seq = lambda n: Fraction(fib[n])  # noqa: E731
    for n in range(15):
        assert op.apply_sequence(seq, n) == 0


def test_diff_operator_annihilates_exp() -> None:
    # (D - 1) annihilates exp(x): Taylor coefficients 1/m!.
    D = diff_algebra()
    op = D.operator([[-1], [1]])
    coeffs = [Fraction(1, math.factorial(m)) for m in range(14)]
    for order in range(10):
        assert op.apply_series(coeffs, order) == 0


def test_diff_operator_annihilates_sin_cos_combo() -> None:
    # (D^2 + 1) annihilates sin and cos.
    D = diff_algebra()
    op = D.operator([[1], [], [1]])
    # cos: 1, 0, -1/2!, 0, 1/4!, ...
    cos = [Fraction(0)] * 14
    for m in range(0, 14, 2):
        cos[m] = Fraction((-1) ** (m // 2), math.factorial(m))
    for order in range(10):
        assert op.apply_series(cos, order) == 0


def test_product_is_associative() -> None:
    S = shift_algebra()
    a = S.operator([[0, 1], [1]])  # n + S
    b = S.operator([[2], [0, 1]])  # 2 + n S
    c = S.operator([[1], [1]])  # 1 + S
    left = (a * b) * c
    right = a * (b * c)
    assert left.coeffs == right.coeffs


def test_factored_operator_matches_expanded_on_sequence() -> None:
    # (S - 1)(S - 1) applied to a sequence == S^2 - 2S + 1 applied.
    S = shift_algebra()
    sm1 = S.operator([[-1], [1]])
    expanded = sm1 * sm1
    seq = lambda n: Fraction(n * n * n)  # noqa: E731
    for n in range(10):
        direct = seq(n + 2) - 2 * seq(n + 1) + seq(n)
        assert expanded.apply_sequence(seq, n) == direct
