# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Exact univariate rational-polynomial arithmetic."""

from __future__ import annotations

from fractions import Fraction

import pytest
from omnibias.holonomic._core.rational_poly import (
    degree,
    dispersion_set,
    is_zero,
    padd,
    pderiv,
    pdivmod,
    peval,
    pgcd,
    pmonic,
    pmul,
    pscale,
    pshift,
    psub,
    to_poly,
)


def test_to_poly_trims_trailing_zeros() -> None:
    assert to_poly([1, 2, 0, 0]) == (Fraction(1), Fraction(2))
    assert to_poly([0, 0]) == ()
    assert is_zero(to_poly([]))
    assert degree(to_poly([3, 0, 5])) == 2


def test_add_sub_scale() -> None:
    a = to_poly([1, 2, 3])
    b = to_poly([0, 1, 0, 4])
    assert padd(a, b) == (Fraction(1), Fraction(3), Fraction(3), Fraction(4))
    assert psub(a, a) == ()
    assert pscale(a, 2) == (Fraction(2), Fraction(4), Fraction(6))
    assert pscale(a, 0) == ()


def test_mul_and_eval_agree() -> None:
    a = to_poly([1, 1])  # 1 + x
    b = to_poly([-1, 1])  # x - 1
    prod = pmul(a, b)  # x^2 - 1
    assert prod == (Fraction(-1), Fraction(0), Fraction(1))
    for x in range(-3, 4):
        assert peval(prod, x) == peval(a, x) * peval(b, x)


def test_derivative() -> None:
    p = to_poly([5, 3, 0, 2])  # 5 + 3x + 2x^3
    assert pderiv(p) == (Fraction(3), Fraction(0), Fraction(6))  # 3 + 6x^2
    assert pderiv(to_poly([7])) == ()


def test_shift_composition() -> None:
    p = to_poly([0, 0, 1])  # x^2
    shifted = pshift(p, 1)  # (x + 1)^2 = 1 + 2x + x^2
    assert shifted == (Fraction(1), Fraction(2), Fraction(1))
    for x in range(-3, 4):
        assert peval(shifted, x) == peval(p, x + 1)


def test_divmod_reconstructs() -> None:
    a = to_poly([-1, 0, 0, 1])  # x^3 - 1
    b = to_poly([-1, 1])  # x - 1
    q, r = pdivmod(a, b)
    assert r == ()  # exact
    assert pmul(q, b) == a
    assert q == (Fraction(1), Fraction(1), Fraction(1))  # x^2 + x + 1


def test_divmod_by_zero_raises() -> None:
    with pytest.raises(ZeroDivisionError):
        pdivmod(to_poly([1, 1]), ())


def test_gcd_is_monic() -> None:
    a = pmul(to_poly([-1, 1]), to_poly([2, 1]))  # (x-1)(x+2)
    b = pmul(to_poly([-1, 1]), to_poly([3, 1]))  # (x-1)(x+3)
    g = pgcd(a, b)
    assert g == pmonic(to_poly([-1, 1]))  # ~ (x - 1), monic
    assert g[-1] == 1


def test_dispersion_set() -> None:
    # a(x) = x, b(x) = x - 3 -> gcd(a(x), b(x+j)) non-trivial iff j = 3.
    a = to_poly([0, 1])
    b = to_poly([-3, 1])
    assert dispersion_set(a, b) == [3]
    # disjoint shifts -> empty.
    assert dispersion_set(to_poly([0, 1]), to_poly([1, 0, 0, 1])) == []
