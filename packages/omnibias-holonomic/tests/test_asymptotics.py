# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Leading asymptotics of P-recursive sequences (Poincare-Perron + certified bridge)."""

from __future__ import annotations

from fractions import Fraction
from math import sqrt

import pytest
from omnibias.holonomic._core.asymptotics import (
    certified_asymptotic,
    empirical_rate,
    precursive_asymptotics,
)
from omnibias.holonomic._core.dfinite import PRecursive
from omnibias.holonomic._core.ore import shift_algebra


def _rec(coeffs: list[list[int]], initial: list[int]) -> PRecursive:
    """Build a P-recursive sequence from integer recurrence coefficients + initials."""
    op = shift_algebra().operator([[Fraction(c) for c in poly] for poly in coeffs])
    return PRecursive(op, tuple(Fraction(v) for v in initial))


def test_geometric_rate_and_zero_exponent() -> None:
    # a_n = 2^n: a_{n+1} - 2 a_n = 0.
    est = precursive_asymptotics(_rec([[-2], [1]], [1]))
    assert est.kind == "geometric"
    assert est.rate == pytest.approx(2.0)
    assert est.signed_rate == pytest.approx(2.0)
    assert est.exponent == pytest.approx(0.0)


def test_alternating_rate_is_signed() -> None:
    # a_n = (-2)^n: a_{n+1} + 2 a_n = 0 -> dominant root -2 (magnitude 2).
    est = precursive_asymptotics(_rec([[2], [1]], [1]))
    assert est.rate == pytest.approx(2.0)
    assert est.signed_rate == pytest.approx(-2.0)
    assert est.model(3) == pytest.approx(-8.0)


def test_fibonacci_golden_ratio() -> None:
    # F_{n+2} - F_{n+1} - F_n = 0 -> dominant root the golden ratio.
    est = precursive_asymptotics(_rec([[-1], [-1], [1]], [0, 1]))
    assert est.rate == pytest.approx((1.0 + sqrt(5.0)) / 2.0)
    assert est.exponent == pytest.approx(0.0)


def test_catalan_exponent_is_minus_three_halves() -> None:
    # (n+2) C_{n+1} - (4n+2) C_n = 0 -> C_n ~ 4^n n^{-3/2} / sqrt(pi).
    rec = _rec([[-2, -4], [2, 1]], [1])
    est = precursive_asymptotics(rec)
    assert est.kind == "geometric"
    assert est.rate == pytest.approx(4.0)
    assert est.exponent == pytest.approx(-1.5)
    # The full asymptotic constant is 1/sqrt(pi); check the shape captures it at large n.
    terms = rec.terms(70)
    ratio = float(terms[60]) / est.model(60)
    assert ratio == pytest.approx(1.0 / sqrt(3.141592653589793), rel=0.05)


def test_tribonacci_order_three_uses_durand_kerner() -> None:
    # T_{n+3} = T_{n+2} + T_{n+1} + T_n -> degree-3 characteristic poly (pure-Python roots).
    est = precursive_asymptotics(_rec([[-1], [-1], [-1], [1]], [0, 0, 1]))
    assert est.rate == pytest.approx(1.8392867552141612)  # real root of t^3 = t^2 + t + 1
    assert est.exponent == pytest.approx(0.0)


def test_factorial_is_super_exponential() -> None:
    # a_n = n!: a_{n+1} - (n+1) a_n = 0 -> top-shift coefficient has sub-maximal degree.
    est = precursive_asymptotics(_rec([[-1, -1], [1]], [1]))
    assert est.kind == "factorial"
    assert est.rate == float("inf")
    assert est.signed_rate is None


def test_char_poly_root_beats_empirical_ratio_for_catalan() -> None:
    # The characteristic root is exact; the finite-n ratio a(n+1)/a(n) still lags.
    rec = _rec([[-2, -4], [2, 1]], [1])
    est = precursive_asymptotics(rec)
    emp = empirical_rate(rec, samples=80)
    assert abs(est.rate - 4.0) < abs(emp - 4.0)


def test_certified_asymptotic_encloses_geometric_coefficient() -> None:
    # a_n = 2^n comes from the simple pole 1/(1 - 2x): scale 1, radius 1/2, exponent 1.
    est = certified_asymptotic(rate=2, exponent_alpha=1, scale=1, n=10)
    assert est.exact_coefficient.lo <= 1024 <= est.exact_coefficient.hi
    assert est.leading == pytest.approx(1024.0)
    assert est.rel_error < 1e-9


def test_certified_asymptotic_catalan_leading_matches() -> None:
    # Catalan singular template -2 (1 - 4x)^{1/2}: rate 4, exponent alpha = -1/2, scale -2.
    est = certified_asymptotic(rate=4, exponent_alpha=Fraction(-1, 2), scale=-2, n=30)
    # Leading transfer term is the classical 4^n / (sqrt(pi) n^{3/2}).
    expected = 4.0**30 / (sqrt(3.141592653589793) * 30.0**1.5)
    assert est.leading == pytest.approx(expected, rel=1e-12)
    assert est.exact_coefficient.lo <= est.exact_coefficient.hi


def test_certified_asymptotic_rejects_zero_rate() -> None:
    with pytest.raises(ValueError, match="non-zero"):
        certified_asymptotic(rate=0, exponent_alpha=1, scale=1, n=5)
