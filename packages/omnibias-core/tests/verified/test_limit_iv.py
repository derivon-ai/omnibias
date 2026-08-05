# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Verified (interval) L'Hopital limit: enclosures must contain the true limit."""

from __future__ import annotations

from fractions import Fraction

import pytest
from omnibias.core.verified.interval import Interval
from omnibias.core.verified.jet import lhopital_ratio_iv


def _point_jet(values: list[Fraction]) -> list[Interval]:
    return [Interval.from_rational(v) for v in values]


def test_sinc_limit_enclosure_contains_one() -> None:
    # sin(t)/t at t->0: leading order 1, coeffs 1 / 1.
    num = _point_jet([Fraction(0), Fraction(1), Fraction(0), Fraction(-1, 6)])
    den = _point_jet([Fraction(0), Fraction(1), Fraction(0), Fraction(0)])
    enclosure = lhopital_ratio_iv(num, den, order=1)
    assert enclosure.contains(1.0)
    assert enclosure.width < 1e-12


def test_one_minus_cos_over_x2_encloses_half() -> None:
    num = _point_jet([Fraction(0), Fraction(0), Fraction(1, 2), Fraction(0)])
    den = _point_jet([Fraction(0), Fraction(0), Fraction(1), Fraction(0)])
    enclosure = lhopital_ratio_iv(num, den, order=2)
    assert enclosure.contains(0.5)
    assert enclosure.width < 1e-12


def test_enclosure_contains_float_backend_result() -> None:
    # A non-trivial ratio with rounding: 2.5 / -0.5 = -5.0.
    num = _point_jet([Fraction(0), Fraction(5, 2)])
    den = _point_jet([Fraction(0), Fraction(-1, 2)])
    enclosure = lhopital_ratio_iv(num, den, order=1)
    assert enclosure.contains(-5.0)


def test_straddling_denominator_is_not_certified_finite() -> None:
    # If the leading denominator coefficient straddles zero the limit is not
    # certified finite: reciprocal raises ZeroDivisionError.
    num = [Interval.point(0.0), Interval.point(1.0)]
    den = [Interval.point(0.0), Interval(-1.0, 1.0)]
    with pytest.raises(ZeroDivisionError):
        lhopital_ratio_iv(num, den, order=1)


def test_order_out_of_range_raises() -> None:
    num = [Interval.point(0.0), Interval.point(1.0)]
    den = [Interval.point(0.0), Interval.point(1.0)]
    with pytest.raises(ValueError, match="exceeds available jet order"):
        lhopital_ratio_iv(num, den, order=5)
