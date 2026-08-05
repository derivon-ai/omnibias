# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Certified Euler-Maclaurin: log_gamma / digamma vs mpmath + the general engine."""

from __future__ import annotations

import math
import random
from fractions import Fraction

import pytest
from omnibias.core.verified.euler_maclaurin import (
    digamma_iv,
    euler_maclaurin_sum,
    log_gamma_iv,
)
from omnibias.core.verified.interval import Interval

_SEEDS = range(8)


def _grid_and_random(lo: float, hi: float, seed: int, *, grid: int = 20, rnd: int = 20) -> list[float]:
    step = (hi - lo) / (grid - 1)
    pts = [lo + i * step for i in range(grid)]
    rng = random.Random(seed)
    pts.extend(rng.uniform(lo, hi) for _ in range(rnd))
    return pts


def test_log_gamma_encloses_mpmath_grid_and_random() -> None:
    mp = pytest.importorskip("mpmath")
    for seed in _SEEDS:
        for x in _grid_and_random(0.4, 6.0, seed):
            enc = log_gamma_iv(Interval.point(x))
            with mp.workdps(40):
                true = float(mp.loggamma(x))
            assert enc.lo <= true <= enc.hi, (x, enc, true)
            assert enc.width < 1e-8


def test_digamma_encloses_mpmath_grid_and_random() -> None:
    mp = pytest.importorskip("mpmath")
    for seed in _SEEDS:
        for x in _grid_and_random(0.4, 6.0, seed):
            enc = digamma_iv(Interval.point(x))
            with mp.workdps(40):
                true = float(mp.digamma(x))
            assert enc.lo <= true <= enc.hi, (x, enc, true)
            assert enc.width < 1e-8


def test_log_gamma_known_values() -> None:
    for x, val in ((1.0, 0.0), (2.0, 0.0), (5.0, math.log(24.0))):
        enc = log_gamma_iv(Interval.point(x))
        assert enc.lo <= val <= enc.hi


def test_log_gamma_box_encloses_range() -> None:
    mp = pytest.importorskip("mpmath")
    box = Interval(2.0, 5.0)
    enc = log_gamma_iv(box)
    for seed in _SEEDS:
        for x in _grid_and_random(2.0, 5.0, seed):
            with mp.workdps(40):
                true = float(mp.loggamma(x))
            assert enc.lo <= true <= enc.hi


def test_log_gamma_rejects_nonpositive() -> None:
    with pytest.raises(ValueError, match="strictly positive"):
        log_gamma_iv(Interval(-1.0, 1.0))


def test_euler_maclaurin_sum_encloses_exact_p_series() -> None:
    # f(x) = x^-2, so f^(k)(x) = (-1)^k (k+1)! x^-(k+2); integral_a^b = 1/a - 1/b.
    def deriv(k: int, x: Interval) -> Interval:
        coeff = Interval.from_rational(Fraction((-1) ** k * math.factorial(k + 1)))
        return coeff * x.pow_int(k + 2).reciprocal()

    a, b = 4, 30
    integral = Interval.from_rational(Fraction(1, a) - Fraction(1, b))
    enc = euler_maclaurin_sum(deriv, integral, a, b, terms=4)
    exact = float(sum(Fraction(1, k * k) for k in range(a, b + 1)))
    assert enc.lo <= exact <= enc.hi
    assert enc.width < 1e-4  # tight: Euler-Maclaurin, not a huge remainder


def test_euler_maclaurin_sum_beats_trapezoid_baseline() -> None:
    # Best-in-class: Euler-Maclaurin (Bernoulli corrections) beats plain trapezoid.
    def deriv(k: int, x: Interval) -> Interval:
        coeff = Interval.from_rational(Fraction((-1) ** k * math.factorial(k + 1)))
        return coeff * x.pow_int(k + 2).reciprocal()

    a, b = 4, 30
    integral = Interval.from_rational(Fraction(1, a) - Fraction(1, b))
    exact = float(sum(Fraction(1, k * k) for k in range(a, b + 1)))

    em = euler_maclaurin_sum(deriv, integral, a, b, terms=4)
    em_err = max(em.lo - exact, exact - em.hi, 0.0)
    trapezoid = float(integral.mid) + 0.5 * (1.0 / a**2 + 1.0 / b**2)
    trap_err = abs(trapezoid - exact)
    assert em_err < trap_err


def test_euler_maclaurin_sum_validates_arguments() -> None:
    def deriv(k: int, x: Interval) -> Interval:
        return x

    with pytest.raises(ValueError, match="a <= b"):
        euler_maclaurin_sum(deriv, Interval.point(0.0), 5, 1, terms=2)
    with pytest.raises(ValueError, match="terms"):
        euler_maclaurin_sum(deriv, Interval.point(0.0), 1, 5, terms=0)
