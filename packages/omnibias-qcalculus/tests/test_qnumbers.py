# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Exact q-combinatorics: q -> 1 reductions, polynomial forms, q-Pascal, q-Pochhammer."""

from __future__ import annotations

import math
from fractions import Fraction

import pytest
from omnibias.qcalculus import (
    q_binomial,
    q_binomial_poly,
    q_bracket,
    q_bracket_poly,
    q_factorial,
    q_pochhammer,
)


def _poly_eval(coeffs: tuple[int, ...], q: Fraction) -> Fraction:
    return sum((Fraction(c) * q**i for i, c in enumerate(coeffs)), Fraction(0))


@pytest.mark.parametrize("n", range(8))
def test_q_bracket_reduces_to_n_at_q_one(n: int) -> None:
    assert q_bracket(n, 1) == n
    assert q_bracket(n, Fraction(1)) == n


@pytest.mark.parametrize("q", [Fraction(1, 2), Fraction(1, 3), Fraction(2, 3), Fraction(3)])
def test_q_bracket_matches_geometric_sum(q: Fraction) -> None:
    for n in range(7):
        expected = sum((q**i for i in range(n)), Fraction(0))
        assert q_bracket(n, q) == expected
        # closed form (1 - q^n)/(1 - q) for q != 1
        assert q_bracket(n, q) == (1 - q**n) / (1 - q)


def test_q_bracket_poly_is_ones_and_evaluates() -> None:
    for n in range(7):
        assert q_bracket_poly(n) == tuple(1 for _ in range(n))
        assert _poly_eval(q_bracket_poly(n), Fraction(2, 5)) == q_bracket(n, Fraction(2, 5))


@pytest.mark.parametrize("n", range(7))
def test_q_factorial_reduces_to_factorial(n: int) -> None:
    assert q_factorial(n, 1) == math.factorial(n)


@pytest.mark.parametrize("n", range(6))
def test_q_binomial_reduces_to_binomial(n: int) -> None:
    for k in range(-1, n + 2):
        expected = math.comb(n, k) if 0 <= k <= n else 0
        assert q_binomial(n, k, 1) == expected


def test_q_binomial_poly_symmetry_and_value() -> None:
    # Gaussian polynomials are palindromic and evaluate to the Fraction q-binomial.
    for n in range(7):
        for k in range(n + 1):
            poly = q_binomial_poly(n, k)
            assert poly == poly[::-1]  # palindrome
            assert all(c >= 0 for c in poly)  # non-negative integer coefficients
            q = Fraction(3, 7)
            assert _poly_eval(poly, q) == q_binomial(n, k, q)
            assert sum(poly) == math.comb(n, k)  # q = 1


def test_q_binomial_poly_known_value() -> None:
    # [4 choose 2]_q = 1 + q + 2 q^2 + q^3 + q^4  (a textbook Gaussian polynomial).
    assert q_binomial_poly(4, 2) == (1, 1, 2, 1, 1)


def test_q_pascal_recurrence() -> None:
    # [n, k]_q = [n-1, k-1]_q + q^k [n-1, k]_q, verified at a numeric q.
    q = Fraction(2, 3)
    for n in range(1, 8):
        for k in range(1, n):
            lhs = q_binomial(n, k, q)
            rhs = q_binomial(n - 1, k - 1, q) + q**k * q_binomial(n - 1, k, q)
            assert lhs == rhs


def test_q_pochhammer_values() -> None:
    # (q; q)_n = prod_{k=1}^n (1 - q^k); (2; 1/3)_3 = -7/27.
    q = Fraction(1, 3)
    for n in range(6):
        expected = math.prod(1 - q**k for k in range(1, n + 1))
        assert q_pochhammer(q, q, n) == expected
    assert q_pochhammer(2, Fraction(1, 3), 3) == Fraction(-7, 27)


def test_q_pochhammer_matches_mpmath() -> None:
    mp = pytest.importorskip("mpmath")
    q = Fraction(2, 5)
    for a in (Fraction(1, 2), Fraction(3, 2), Fraction(-1, 3)):
        for n in range(6):
            got = float(q_pochhammer(a, q, n))
            ref = float(mp.qp(mp.mpf(a.numerator) / a.denominator, mp.mpf(2) / 5, n))
            assert got == pytest.approx(ref, rel=1e-12, abs=1e-15)


def test_negative_orders_raise() -> None:
    with pytest.raises(ValueError):
        q_bracket(-1, Fraction(1, 2))
    with pytest.raises(ValueError):
        q_factorial(-1, Fraction(1, 2))
    with pytest.raises(ValueError):
        q_pochhammer(1, Fraction(1, 2), -1)
