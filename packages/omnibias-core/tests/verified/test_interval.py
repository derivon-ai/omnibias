# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Rigor tests for the verified interval arithmetic.

Every test certifies *containment*: the exact rational result of an operation on
sample points must lie inside the computed interval enclosure.
"""

from __future__ import annotations

from fractions import Fraction

import pytest
from omnibias.core.verified.interval import Interval

# rational sample points inside [lo, hi] for containment checks.
_FRACS = [Fraction(-7, 3), Fraction(-1, 8), Fraction(0), Fraction(2, 5), Fraction(11, 4)]


def _iv(p: Fraction, q: Fraction) -> Interval:
    return Interval.hull(Interval.from_rational(p), Interval.from_rational(q))


def _interior(p: Fraction, q: Fraction) -> list[Fraction]:
    lo, hi = (p, q) if p <= q else (q, p)
    return [lo, hi, lo + (hi - lo) * Fraction(1, 3), lo + (hi - lo) * Fraction(7, 11)]


@pytest.mark.parametrize("p", _FRACS)
@pytest.mark.parametrize("q", _FRACS)
def test_from_rational_brackets(p: Fraction, q: Fraction) -> None:
    iv = Interval.from_rational(p)
    assert iv.lo <= p <= iv.hi


def test_add_contains_exact() -> None:
    a, b = _iv(Fraction(-1, 3), Fraction(2, 7)), _iv(Fraction(5, 6), Fraction(9, 4))
    r = a + b
    for x in _interior(Fraction(-1, 3), Fraction(2, 7)):
        for y in _interior(Fraction(5, 6), Fraction(9, 4)):
            assert r.lo <= x + y <= r.hi


def test_sub_contains_exact() -> None:
    a, b = _iv(Fraction(-1, 3), Fraction(2, 7)), _iv(Fraction(5, 6), Fraction(9, 4))
    r = a - b
    for x in _interior(Fraction(-1, 3), Fraction(2, 7)):
        for y in _interior(Fraction(5, 6), Fraction(9, 4)):
            assert r.lo <= x - y <= r.hi


def test_mul_contains_exact_with_sign_changes() -> None:
    a, b = _iv(Fraction(-3, 2), Fraction(5, 4)), _iv(Fraction(-7, 8), Fraction(11, 6))
    r = a * b
    for x in _interior(Fraction(-3, 2), Fraction(5, 4)):
        for y in _interior(Fraction(-7, 8), Fraction(11, 6)):
            assert r.lo <= x * y <= r.hi


def test_pow_int_contains_exact() -> None:
    a = _iv(Fraction(-3, 2), Fraction(5, 4))
    for n in range(0, 7):
        r = a.pow_int(n)
        for x in _interior(Fraction(-3, 2), Fraction(5, 4)):
            assert r.lo <= x**n <= r.hi


def test_reciprocal_and_div_away_from_zero() -> None:
    a, b = _iv(Fraction(1, 3), Fraction(2)), _iv(Fraction(3, 4), Fraction(5, 2))
    r = a / b
    for x in _interior(Fraction(1, 3), Fraction(2)):
        for y in _interior(Fraction(3, 4), Fraction(5, 2)):
            assert r.lo <= Fraction(x, 1) / y <= r.hi


def test_reciprocal_straddling_zero_raises() -> None:
    with pytest.raises(ZeroDivisionError):
        Interval(-1.0, 2.0).reciprocal()


def test_sqrt_contains_exact() -> None:
    a = _iv(Fraction(1, 4), Fraction(9, 4))
    r = a.sqrt()
    # sqrt(1/4)=1/2, sqrt(9/4)=3/2 must be inside.
    assert r.lo <= Fraction(1, 2) <= r.hi
    assert r.lo <= Fraction(3, 2) <= r.hi


def test_sqrt_negative_raises() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        Interval(-1.0, 1.0).sqrt()


def test_empty_interval_raises() -> None:
    with pytest.raises(ValueError, match="empty interval"):
        Interval(2.0, 1.0)


def test_width_is_outward() -> None:
    a = Interval(1.0, 1.0 + 2.0**-50)
    assert a.width >= 2.0**-50
