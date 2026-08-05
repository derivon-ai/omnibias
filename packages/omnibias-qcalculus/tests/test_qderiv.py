# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Jackson q-derivative / q-integral: exact polynomial ops, roundtrips, q -> 1 limits."""

from __future__ import annotations

import math
from fractions import Fraction

import pytest
from omnibias.qcalculus import (
    q_antiderivative_poly,
    q_bracket,
    q_derivative,
    q_derivative_poly,
    q_integral,
)


def test_q_derivative_poly_monomials() -> None:
    # D_q x^n = [n]_q x^{n-1}.
    q = Fraction(1, 2)
    for n in range(1, 7):
        coeffs = [Fraction(0)] * n + [Fraction(1)]  # x^n
        d = q_derivative_poly(coeffs, q)
        assert len(d) == n
        assert d[n - 1] == q_bracket(n, q)
        assert all(c == 0 for c in d[: n - 1])


def test_q_derivative_poly_reduces_to_ordinary_derivative() -> None:
    # At q = 1, D_q sum c_i x^i = sum i c_i x^{i-1}.
    coeffs = [Fraction(2), Fraction(-3), Fraction(5), Fraction(7)]
    d = q_derivative_poly(coeffs, 1)
    assert d == (Fraction(-3), Fraction(10), Fraction(21))


def test_antiderivative_is_left_inverse_of_derivative() -> None:
    q = Fraction(1, 3)
    coeffs = [Fraction(1), Fraction(2), Fraction(3), Fraction(4)]
    back = q_derivative_poly(q_antiderivative_poly(coeffs, q), q)
    assert tuple(back) == tuple(Fraction(c) for c in coeffs)


def test_numerical_q_derivative_approaches_ordinary_derivative() -> None:
    # For f(x) = x^3, f'(2) = 12; the Jackson quotient -> 12 as q -> 1.
    f = lambda x: x**3  # noqa: E731
    errs = [abs(q_derivative(f, 2.0, q) - 12.0) for q in (0.5, 0.9, 0.99, 0.999)]
    assert errs == sorted(errs, reverse=True)  # monotone decrease toward the limit
    # D_q x^3 at x=2 is [3]_q * 4 = (1+q+q^2) * 4; the residual is O(1-q).
    assert errs[-1] < 2e-2


def test_numerical_q_derivative_matches_exact_on_polynomial() -> None:
    # D_q x^4 at x = 1.5 equals [4]_q * 1.5^3 exactly (up to float rounding).
    q = 0.5
    got = q_derivative(lambda x: x**4, 1.5, q)
    expected = float(q_bracket(4, Fraction(1, 2))) * 1.5**3
    assert got == pytest.approx(expected, rel=1e-12)


def test_q_integral_of_monomial() -> None:
    # int_0^1 x^n d_q x = 1 / [n+1]_q.
    q = 0.5
    for n in range(5):
        got = q_integral(lambda x, n=n: x**n, 0.0, 1.0, q, terms=400)
        expected = 1.0 / float(q_bracket(n + 1, Fraction(1, 2)))
        assert got == pytest.approx(expected, rel=1e-9)


def test_q_fundamental_theorem() -> None:
    # int_0^a (D_q f) d_q x = f(a) - f(0) for f(x) = x^3 + x.
    q = 0.4
    f = lambda x: x**3 + x  # noqa: E731
    dq_f = lambda x: q_derivative(f, x, q) if x != 0 else 1.0  # noqa: E731
    a = 1.3
    got = q_integral(dq_f, 0.0, a, q, terms=500)
    assert got == pytest.approx(f(a) - f(0.0), rel=1e-6)


def test_errors() -> None:
    with pytest.raises(ValueError):
        q_derivative(lambda x: x, 1.0, 1.0)  # q == 1
    with pytest.raises(ValueError):
        q_derivative(lambda x: x, 0.0, 0.5)  # x == 0
    with pytest.raises(ValueError):
        q_integral(lambda x: x, 0.0, 1.0, 1.5)  # q not in (0, 1)


def test_q_integral_matches_math_reference() -> None:
    # Sanity vs a closed form: int_0^1 x^2 d_q x = 1/[3]_q; also independent of `math`.
    q = 0.25
    got = q_integral(lambda x: x**2, 0.0, 1.0, q, terms=300)
    assert got == pytest.approx(1.0 / (1 + q + q**2), rel=1e-10)
    assert math.isfinite(got)
