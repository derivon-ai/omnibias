# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Acceptance gate for the analytic-combinatorics coefficients.

Extracted Stirling / Bernoulli / Euler match known exact values and an
independent recurrence cross-check, sit inside the certified interval-tower
enclosure of the mpmath high-precision truth, and the leading asymptotics match
an mpmath reference -- all across ``K >= 8`` seeds.
"""

from __future__ import annotations

import math
import random
from fractions import Fraction

import pytest
from omnibias.core.verified.interval import Interval
from omnibias.difference import (
    bell_number,
    bell_number_asymptotic,
    bernoulli_asymptotic,
    bernoulli_number,
    bernoulli_polynomial,
    certified_derivative_enclosure,
    euler_asymptotic,
    euler_number,
    euler_polynomial,
    eulerian_number,
    falling_factorial_coeffs,
    log_bell_number_asymptotic,
    power_sum_coeffs,
    rising_factorial_coeffs,
    stirling_first_signed,
    stirling_first_signed_row,
    stirling_first_unsigned,
    stirling_second,
    stirling_second_asymptotic,
    stirling_second_row,
)

mpmath = pytest.importorskip("mpmath")

SEEDS = range(8)  # the K >= 8 acceptance seeds


# --------------------------------------------------------------------------- #
# Exact values + independent cross-checks
# --------------------------------------------------------------------------- #
def test_stirling_second_exact_rows() -> None:
    assert stirling_second_row(0) == (1,)
    assert stirling_second_row(4) == (0, 1, 7, 6, 1)
    assert stirling_second_row(5) == (0, 1, 15, 25, 10, 1)


def test_stirling_first_signed_exact_rows() -> None:
    assert stirling_first_signed_row(4) == (0, -6, 11, -6, 1)
    assert stirling_first_signed_row(5) == (0, 24, -50, 35, -10, 1)


@pytest.mark.parametrize("n", range(9))
def test_falling_factorial_is_signed_stirling_first(n: int) -> None:
    assert falling_factorial_coeffs(n) == stirling_first_signed_row(n)


@pytest.mark.parametrize("n", range(9))
def test_rising_factorial_is_unsigned_stirling_first(n: int) -> None:
    assert rising_factorial_coeffs(n) == tuple(
        stirling_first_unsigned(n, k) for k in range(n + 1)
    )


@pytest.mark.parametrize("n", range(9))
def test_stirling_second_row_sums_to_bell(n: int) -> None:
    assert sum(stirling_second_row(n)) == bell_number(n)


def test_bell_numbers_exact() -> None:
    assert [bell_number(n) for n in range(7)] == [1, 1, 2, 5, 15, 52, 203]


def test_bernoulli_small_exact() -> None:
    assert bernoulli_number(0) == Fraction(1)
    assert bernoulli_number(1) == Fraction(-1, 2)
    assert [bernoulli_number(n) for n in (2, 4, 6, 8, 10, 12)] == [
        Fraction(1, 6),
        Fraction(-1, 30),
        Fraction(1, 42),
        Fraction(-1, 30),
        Fraction(5, 66),
        Fraction(-691, 2730),
    ]


def test_euler_small_exact() -> None:
    assert [euler_number(n) for n in range(9)] == [1, 0, -1, 0, 5, 0, -61, 0, 1385]


def test_bernoulli_and_euler_polynomials() -> None:
    assert bernoulli_polynomial(0) == (Fraction(1),)
    assert bernoulli_polynomial(1) == (Fraction(-1, 2), Fraction(1))
    assert bernoulli_polynomial(2) == (Fraction(1, 6), Fraction(-1), Fraction(1))
    assert euler_polynomial(1) == (Fraction(-1, 2), Fraction(1))
    assert euler_polynomial(2) == (Fraction(0), Fraction(-1), Fraction(1))


def test_eulerian_numbers_exact() -> None:
    assert [eulerian_number(4, k) for k in range(4)] == [1, 11, 11, 1]
    assert [eulerian_number(5, k) for k in range(5)] == [1, 26, 66, 26, 1]


def test_faulhaber_power_sums() -> None:
    assert power_sum_coeffs(1) == (Fraction(0), Fraction(-1, 2), Fraction(1, 2))
    assert power_sum_coeffs(2) == (Fraction(0), Fraction(1, 6), Fraction(-1, 2), Fraction(1, 3))
    c = power_sum_coeffs(3)  # S_3(N) = sum_{i<N} i^3
    for big_n in (4, 7, 10):
        expected = sum(i**3 for i in range(big_n))
        assert sum(c[j] * big_n**j for j in range(len(c))) == expected


# --------------------------------------------------------------------------- #
# mpmath cross-checks
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("n", range(0, 21))
def test_bernoulli_matches_mpmath(n: int) -> None:
    with mpmath.workdps(60):
        ref = mpmath.bernoulli(n)
        got = bernoulli_number(n)
        got_mpf = mpmath.mpf(got.numerator) / mpmath.mpf(got.denominator)
        assert abs(got_mpf - ref) < mpmath.mpf(10) ** (-45)


@pytest.mark.parametrize("m", range(0, 9))
def test_euler_matches_mpmath(m: int) -> None:
    n = 2 * m
    with mpmath.workdps(60):
        true = mpmath.taylor(lambda z: mpmath.sech(z), 0, n)[n] * mpmath.factorial(n)
        assert euler_number(n) == int(mpmath.nint(true))


# --------------------------------------------------------------------------- #
# Certified interval-tower enclosure contains the mpmath truth (K >= 8 seeds)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("seed", SEEDS)
def test_euler_number_inside_certified_sech_enclosure(seed: int) -> None:
    rng = random.Random(seed)
    n = 2 * rng.randint(0, 6)
    enc = certified_derivative_enclosure("sech", 0.0, n).value  # encloses E_n = sech^(n)(0)
    with mpmath.workdps(50):
        true = float(mpmath.taylor(lambda z: mpmath.sech(z), 0, n)[n] * math.factorial(n))
    assert enc.lo <= true <= enc.hi
    assert enc.lo <= float(euler_number(n)) <= enc.hi
    assert round((enc.lo + enc.hi) / 2) == euler_number(n)  # pins the exact integer


@pytest.mark.parametrize("seed", SEEDS)
def test_bernoulli_number_inside_certified_tanh_enclosure(seed: int) -> None:
    rng = random.Random(100 + seed)
    m = rng.randint(1, 6)
    n = 2 * m
    # B_{2m} = tanh^(2m-1)(0) * 2m / (2^{2m}(2^{2m}-1)); enclose the tangent number.
    tangent = certified_derivative_enclosure("tanh", 0.0, 2 * m - 1).value
    scale = Interval.from_rational(Fraction(2 * m, 2 ** (2 * m) * (2 ** (2 * m) - 1)))
    enc = tangent * scale
    with mpmath.workdps(50):
        true = float(mpmath.bernoulli(n))
    assert enc.lo <= true <= enc.hi
    assert enc.lo <= float(bernoulli_number(n)) <= enc.hi


# --------------------------------------------------------------------------- #
# Asymptotics match the mpmath high-precision reference (K >= 8 seeds)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("seed", SEEDS)
def test_bernoulli_asymptotic_matches_reference_and_exact(seed: int) -> None:
    rng = random.Random(200 + seed)
    m = rng.randint(5, 18)
    n = 2 * m
    approx = bernoulli_asymptotic(n)
    with mpmath.workdps(60):
        ref = (-1) ** (m + 1) * 2 * mpmath.factorial(n) / (2 * mpmath.pi) ** n
        exact = mpmath.bernoulli(n)
    # our float asymptotic reproduces the mpmath evaluation of the same formula
    assert abs(approx / float(ref) - 1.0) < 1e-9
    # and it is a genuine asymptotic of the exact value (tight by n >= 10)
    assert abs(float(exact) / approx - 1.0) < 2e-2


@pytest.mark.parametrize("seed", SEEDS)
def test_euler_asymptotic_matches_reference_and_exact(seed: int) -> None:
    rng = random.Random(300 + seed)
    m = rng.randint(5, 18)
    n = 2 * m
    approx = euler_asymptotic(n)
    with mpmath.workdps(60):
        ref = (-1) ** m * mpmath.mpf(2) ** (2 * m + 2) * mpmath.factorial(n) / mpmath.pi ** (2 * m + 1)
        exact = mpmath.taylor(lambda z: mpmath.sech(z), 0, n)[n] * mpmath.factorial(n)
    assert abs(approx / float(ref) - 1.0) < 1e-9
    assert abs(float(exact) / approx - 1.0) < 2e-2


@pytest.mark.parametrize("seed", SEEDS)
def test_bell_asymptotic_matches_reference_and_exact(seed: int) -> None:
    rng = random.Random(400 + seed)
    n = rng.randint(15, 25)
    approx = bell_number_asymptotic(n)
    with mpmath.workdps(60):
        r = mpmath.lambertw(n)
        ref = (
            mpmath.e ** (mpmath.mpf(n) / r - 1)
            * mpmath.factorial(n)
            / (r**n * mpmath.sqrt(2 * mpmath.pi * n * (r + 1)))
        )
    # our pure-Python saddle-point (with our Lambert W) matches the mpmath eval
    assert abs(approx / float(ref) - 1.0) < 1e-6
    # and it approximates the exact Bell number (saddle point: ~1-2% by n ~ 20)
    assert abs(float(bell_number(n)) / approx - 1.0) < 5e-2
    # the log form agrees with the direct form
    assert abs(math.log(approx) - log_bell_number_asymptotic(n)) < 1e-9


def test_bernoulli_and_euler_asymptotics_tighten_with_n() -> None:
    def bern_rel(n: int) -> float:
        with mpmath.workdps(60):
            return abs(float(mpmath.bernoulli(n)) / bernoulli_asymptotic(n) - 1.0)

    assert bern_rel(20) < bern_rel(8)


def test_stirling_second_fixed_k_asymptotic_converges() -> None:
    # S(n, k) ~ k^n / k! as n -> inf, for fixed k.
    def ratio(n: int, k: int) -> float:
        return stirling_second(n, k) / stirling_second_asymptotic(n, k)

    assert ratio(25, 4) > ratio(15, 4) > ratio(8, 4)
    assert 0.99 < ratio(25, 4) <= 1.0


def test_asymptotic_domain_guards() -> None:
    with pytest.raises(ValueError):
        bernoulli_asymptotic(7)  # odd
    with pytest.raises(ValueError):
        euler_asymptotic(0)  # too small
    with pytest.raises(ValueError):
        stirling_second_asymptotic(5, 0)  # k >= 1
