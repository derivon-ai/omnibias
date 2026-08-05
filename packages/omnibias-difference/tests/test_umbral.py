# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Umbral / Sheffer identities (exact rational arithmetic)."""

from __future__ import annotations

import random
from fractions import Fraction
from math import factorial

import pytest
from omnibias.difference import (
    appell_sequence,
    bernoulli_number,
    bernoulli_polynomial,
    binomial_transform,
    falling_to_monomial,
    forward_difference,
    inverse_binomial_transform,
    monomial_to_falling,
    newton_forward_coeffs,
    newton_forward_value,
    stirling_first_signed_row,
    stirling_second_row,
)


def _eval(coeffs: tuple[Fraction, ...], x: Fraction) -> Fraction:
    return sum((c * x**j for j, c in enumerate(coeffs)), Fraction(0))


@pytest.mark.parametrize("seed", range(8))
def test_newton_interpolation_reproduces_polynomial(seed: int) -> None:
    rng = random.Random(seed)
    deg = rng.randint(0, 5)
    coeffs = [Fraction(rng.randint(-4, 4)) for _ in range(deg + 1)]

    def poly(x: int) -> Fraction:
        return _eval(tuple(coeffs), Fraction(x))

    samples = [poly(i) for i in range(deg + 1)]
    fwd = newton_forward_coeffs(samples)
    # reproduces the polynomial exactly at nodes and well outside them
    for x in (-3, -1, 0, 1, 2, deg + 5, 10):
        assert newton_forward_value(fwd, Fraction(x)) == poly(x)


def test_forward_difference_of_monomial_is_factorial() -> None:
    # Delta^m of (i^m) is the constant m!.
    for m in range(1, 7):
        values = [Fraction(i**m) for i in range(m + 1)]
        diffs = forward_difference(values, m)
        assert diffs == (Fraction(factorial(m)),)


@pytest.mark.parametrize("seed", range(8))
def test_monomial_falling_are_mutual_inverses(seed: int) -> None:
    rng = random.Random(100 + seed)
    coeffs = [Fraction(rng.randint(-5, 5)) for _ in range(rng.randint(1, 6))]
    fall = monomial_to_falling(coeffs)
    back = falling_to_monomial(fall)
    assert back == tuple(coeffs)


@pytest.mark.parametrize("n", range(7))
def test_monomial_to_falling_is_stirling_second(n: int) -> None:
    # x^n = sum_k S(n, k) (x)_k, so the falling coeffs of x^n are the S(n, .) row.
    monomial = [Fraction(0)] * n + [Fraction(1)]
    assert monomial_to_falling(monomial) == tuple(Fraction(s) for s in stirling_second_row(n))


@pytest.mark.parametrize("n", range(7))
def test_falling_to_monomial_is_signed_stirling_first(n: int) -> None:
    # (x)_n = sum_k s(n, k) x^k, so the monomial coeffs of (x)_n are the s(n, .) row.
    falling = [Fraction(0)] * n + [Fraction(1)]
    assert falling_to_monomial(falling) == tuple(Fraction(s) for s in stirling_first_signed_row(n))


@pytest.mark.parametrize("seed", range(8))
def test_binomial_transform_inverse(seed: int) -> None:
    rng = random.Random(200 + seed)
    seq = [Fraction(rng.randint(-6, 6)) for _ in range(rng.randint(1, 7))]
    assert inverse_binomial_transform(binomial_transform(seq)) == tuple(seq)


def test_appell_of_bernoulli_numbers_are_bernoulli_polynomials() -> None:
    constants = [bernoulli_number(n) for n in range(6)]
    seq = appell_sequence(constants)
    for n in range(6):
        assert tuple(seq[n]) == bernoulli_polynomial(n)


def test_appell_derivative_identity() -> None:
    # Appell property: p_n'(x) = n p_{n-1}(x).
    constants = [Fraction(1), Fraction(2), Fraction(-1), Fraction(3), Fraction(0), Fraction(-5)]
    seq = appell_sequence(constants)
    for n in range(1, len(seq)):
        deriv = tuple(k * seq[n][k] for k in range(1, len(seq[n])))
        assert deriv == tuple(Fraction(n) * c for c in seq[n - 1])


def test_appell_of_delta_is_monomials() -> None:
    seq = appell_sequence([Fraction(1)] + [Fraction(0)] * 4)
    for n in range(5):
        expected = tuple([Fraction(0)] * n + [Fraction(1)])
        assert tuple(seq[n]) == expected
