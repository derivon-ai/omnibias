# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""W4 -- analytic combinatorics: generating-function algebra + CERTIFIED enclosures.

Acceptance-gate + regression tests for
:mod:`omnibias.difference._core.generating`:

* the exact rational OGF/EGF algebra round-trips and reproduces textbook series;
* every **certified** enclosure (Bernoulli, Euler, Bell/Dobinski, zeta, Dirichlet
  beta) provably *contains* the exact value -- checked in exact rational arithmetic
  on a dense index grid **and** a random index sample, and cross-checked against the
  ``mpmath`` oracle -- with relative width shrinking as the index grows (the error
  bar the float-only asymptotics lacked);
* the refined Moser--Wyman Bell saddle point beats the raw ``r e^r = n`` baseline by
  ~2 orders of magnitude and stays overflow-free in log space (regression for the
  large-``n`` overflow bug);
* the fallback-threshold finder is *sound* despite the non-monotone small-``n`` error
  bump (regression for the first-crossing bug).
"""

from __future__ import annotations

from fractions import Fraction
from math import comb, expm1, log

import mpmath
import pytest
from omnibias.core.verified.interval import Interval
from omnibias.difference._core.asymptotics import log_bell_number_asymptotic
from omnibias.difference._core.bernoulli import bernoulli_number
from omnibias.difference._core.euler import euler_number
from omnibias.difference._core.generating import (
    bell_asymptotic_relative_error,
    bell_dobinski_enclosure,
    bell_number_asymptotic_refined,
    bernoulli_enclosure,
    catalan_asymptotic,
    cauchy_product,
    dirichlet_beta_odd_enclosure,
    euler_enclosure,
    exponential_generating_coeffs,
    log_bell_number_asymptotic_refined,
    ordinary_from_exponential,
    rational_ogf_coefficients,
    rational_ogf_growth_base,
    recommended_bell_fallback_n,
    zeta_int_enclosure,
)
from omnibias.difference._core.stirling import bell_number

mpmath.mp.dps = 60


def _encloses(iv: Interval, exact: Fraction | int) -> bool:
    """Rigorous containment of an exact rational in an Interval (exact compare)."""
    return Fraction(iv.lo) <= Fraction(exact) <= Fraction(iv.hi)


def _rel_width(iv: Interval) -> float:
    return iv.width / abs(iv.mid)


def _mp_beta(s: int) -> mpmath.mpf:
    """Dirichlet beta beta(s)=sum_k (-1)^k/(2k+1)^s via accelerated nsum (valid at s=1)."""
    return mpmath.nsum(lambda k: (-1) ** k / (2 * k + 1) ** s, [0, mpmath.inf])


# --------------------------------------------------------------------------- #
# Exact generating-function algebra (numerical register, but exact rational)  #
# --------------------------------------------------------------------------- #
def test_egf_ogf_round_trip() -> None:
    seq = [3, 1, 4, 1, 5, 9, 2, 6]
    egf = exponential_generating_coeffs(seq)
    assert egf[0] == 3 and egf[1] == 1 and egf[2] == Fraction(4, 2)
    assert list(ordinary_from_exponential(egf)) == [Fraction(x) for x in seq]


def test_cauchy_product_matches_manual() -> None:
    # (1 + 2x)(1 + 3x + x^2) = 1 + 5x + 7x^2 + 2x^3.
    assert cauchy_product([1, 2], [1, 3, 1]) == (1, 5, 7, 2)


def test_cauchy_product_rejects_empty() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        cauchy_product([], [1])


def test_rational_ogf_recovers_fibonacci() -> None:
    # x / (1 - x - x^2).
    fib = rational_ogf_coefficients([0, 1], [1, -1, -1], 10)
    assert list(fib) == [0, 1, 1, 2, 3, 5, 8, 13, 21, 34]


def test_rational_ogf_recovers_geometric() -> None:
    # 1 / (1 - 2x) -> 2^n.
    geo = rational_ogf_coefficients([1], [1, -2], 8)
    assert list(geo) == [1, 2, 4, 8, 16, 32, 64, 128]


def test_rational_ogf_rejects_zero_constant_denom() -> None:
    with pytest.raises(ValueError, match="non-zero constant term"):
        rational_ogf_coefficients([1], [0, 1], 4)


def test_growth_base_fibonacci_is_golden_ratio() -> None:
    phi = (1 + 5**0.5) / 2
    assert abs(rational_ogf_growth_base([1, -1, -1]) - phi) < 1e-10


def test_growth_base_geometric_is_two() -> None:
    assert abs(rational_ogf_growth_base([1, -2]) - 2.0) < 1e-10


def test_catalan_asymptotic_ratio_to_exact() -> None:
    # C_n ~ 4^n / (sqrt(pi) n^{3/2}); ratio -> 1 (from above) and improves with n.
    def exact(n: int) -> int:
        return comb(2 * n, n) // (n + 1)

    r30 = catalan_asymptotic(30) / exact(30)
    r200 = catalan_asymptotic(200) / exact(200)
    assert 1.0 < r200 < r30 < 1.05


# --------------------------------------------------------------------------- #
# Certified enclosures: rigorous containment of the exact value               #
# --------------------------------------------------------------------------- #
_EVEN = list(range(2, 34, 2))          # K = 16 dense indices, >= 8
_RANDOM_EVEN = [2 * m for m in (7, 11, 19, 27, 41, 55, 63, 80)]  # random-ish sample


@pytest.mark.parametrize("n", _EVEN + _RANDOM_EVEN)
def test_bernoulli_enclosure_contains_exact(n: int) -> None:
    assert _encloses(bernoulli_enclosure(n), bernoulli_number(n))


@pytest.mark.parametrize("n", _EVEN + _RANDOM_EVEN)
def test_euler_enclosure_contains_exact(n: int) -> None:
    assert _encloses(euler_enclosure(n), euler_number(n))


@pytest.mark.parametrize("n", list(range(1, 12)) + [17, 23, 30, 41, 55])
def test_bell_dobinski_contains_exact(n: int) -> None:
    assert _encloses(bell_dobinski_enclosure(n), bell_number(n))


@pytest.mark.parametrize("n", [130, 150, 160, 170])
def test_enclosure_bounds_finite_at_large_index(n: int) -> None:
    # Regression: the intermediate 2^{n+2}*(2m)! product used to overflow to +inf
    # around n~130 (euler) even though the final value is representable. Folding the
    # tiny pi power in first keeps every bound finite up to the factorial limit.
    import math

    for iv in (bernoulli_enclosure(n), euler_enclosure(n)):
        assert math.isfinite(iv.lo) and math.isfinite(iv.hi)
    assert _encloses(euler_enclosure(n), euler_number(n))
    assert _encloses(bernoulli_enclosure(n), bernoulli_number(n))


def test_enclosure_relative_width_shrinks_with_index() -> None:
    # zeta(2m) -> 1 and beta(2m+1) -> 1, so the enclosures tighten as the index grows.
    assert _rel_width(bernoulli_enclosure(16)) < _rel_width(bernoulli_enclosure(4))
    assert _rel_width(euler_enclosure(16)) < _rel_width(euler_enclosure(4))
    assert _rel_width(bell_dobinski_enclosure(40)) <= 1e-12  # float-tight tail


# --------------------------------------------------------------------------- #
# mpmath oracle cross-checks (K >= 8 indices each)                            #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("s", list(range(2, 12)))
def test_zeta_enclosure_contains_mpmath(s: int) -> None:
    iv = zeta_int_enclosure(s)
    z = mpmath.zeta(s)
    assert iv.lo <= float(z) <= iv.hi
    # the two-sided integral-test bracket is tight for a modest term count
    assert iv.width < 1e-2


@pytest.mark.parametrize("s", list(range(1, 20, 2)))
def test_dirichlet_beta_enclosure_contains_mpmath(s: int) -> None:
    iv = dirichlet_beta_odd_enclosure(s, terms=200)
    assert iv.lo <= float(_mp_beta(s)) <= iv.hi


@pytest.mark.parametrize("n", _EVEN)
def test_bernoulli_enclosure_contains_mpmath(n: int) -> None:
    assert bernoulli_enclosure(n).contains(float(mpmath.bernoulli(n)))


@pytest.mark.parametrize("n", _EVEN)
def test_euler_enclosure_contains_mpmath(n: int) -> None:
    assert euler_enclosure(n).contains(float(mpmath.eulernum(n)))


@pytest.mark.parametrize("n", [1, 2, 5, 8, 12, 17, 23, 30])
def test_bell_dobinski_contains_mpmath(n: int) -> None:
    assert bell_dobinski_enclosure(n).contains(float(mpmath.bell(n)))


# --------------------------------------------------------------------------- #
# Refined Bell saddle point: beats baseline, overflow-free in log space       #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("n", [10, 20, 30, 50])
def test_refined_bell_beats_raw_baseline(n: int) -> None:
    log_exact = log(bell_number(n))
    err_raw = abs(expm1(log_bell_number_asymptotic(n) - log_exact))
    err_ref = abs(expm1(log_bell_number_asymptotic_refined(n) - log_exact))
    assert err_ref < err_raw / 20.0  # measured ~80-120x; assert a conservative 20x


def test_log_bell_asymptotic_no_overflow_large_n() -> None:
    # log form stays finite where the value form overflows float.
    assert log_bell_number_asymptotic_refined(1000) > 0.0
    with pytest.raises(OverflowError, match="overflows float"):
        bell_number_asymptotic_refined(1000)


def test_bell_value_asymptotic_matches_log_form_small_n() -> None:
    for n in (5, 20, 60):
        assert bell_number_asymptotic_refined(n) == pytest.approx(
            mpmath.e ** log_bell_number_asymptotic_refined(n), rel=1e-12
        )


# --------------------------------------------------------------------------- #
# Sound fallback threshold (regression for non-monotone first-crossing bug)   #
# --------------------------------------------------------------------------- #
def test_relative_error_probe_no_overflow_large_n() -> None:
    # log-space probe is valid past the value-form overflow horizon (n ~ 150).
    assert bell_asymptotic_relative_error(400) < 1e-5


def test_error_is_non_monotone_at_small_n() -> None:
    # The documented bump: dips at n=5, rises through n~10. This is *why* a
    # first-crossing threshold is unsound.
    e5, e6, e10 = (bell_asymptotic_relative_error(n) for n in (5, 6, 10))
    assert e5 < e6 < e10  # error increases across the bump


@pytest.mark.parametrize("tol", [1e-2, 1e-3, 1e-4, 1e-5])
def test_recommended_fallback_is_sound(tol: float) -> None:
    n0 = recommended_bell_fallback_n(tol, n_max=200)
    # every measured index at/above the cutoff must satisfy the tolerance
    assert all(bell_asymptotic_relative_error(n) <= tol for n in range(n0, 201))


def test_recommended_fallback_beats_naive_first_crossing() -> None:
    # naive first-crossing returns 5 for 1e-4 (unsound: n=6..34 exceed it); the
    # sound rule returns a strictly larger, violation-free cutoff.
    naive_first = next(n for n in range(2, 201) if bell_asymptotic_relative_error(n) <= 1e-4)
    sound = recommended_bell_fallback_n(1e-4, n_max=200)
    assert naive_first == 5
    assert sound > naive_first
    assert bell_asymptotic_relative_error(naive_first + 1) > 1e-4  # the successor violates


def test_recommended_fallback_raises_when_window_too_small() -> None:
    with pytest.raises(ValueError, match="widen n_max"):
        recommended_bell_fallback_n(1e-9, n_max=200)


# --------------------------------------------------------------------------- #
# Input validation                                                            #
# --------------------------------------------------------------------------- #
def test_enclosure_input_validation() -> None:
    with pytest.raises(ValueError, match="even n >= 2"):
        bernoulli_enclosure(3)
    with pytest.raises(ValueError, match="even n >= 2"):
        euler_enclosure(1)
    with pytest.raises(ValueError, match="must be >= 1"):
        bell_dobinski_enclosure(0)
    with pytest.raises(ValueError, match="s >= 2"):
        zeta_int_enclosure(1)
    with pytest.raises(ValueError, match="odd s >= 1"):
        dirichlet_beta_odd_enclosure(2)
