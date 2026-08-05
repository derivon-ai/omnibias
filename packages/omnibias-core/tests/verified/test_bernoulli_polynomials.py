# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Exact Bernoulli polynomials + generalized Bernoulli numbers
and the shared certified ratio-series driver.

These feed the seven-direction expansion (Hurwitz-zeta special values, Dirichlet
``L`` special values, and the basic-hypergeometric / polylog enclosures).
"""

from __future__ import annotations

import importlib
from fractions import Fraction

import pytest
from omnibias.core.verified.coeffs import (
    bernoulli_number_exact,
    bernoulli_polynomial_exact,
    generalized_bernoulli_exact,
)
from omnibias.core.verified.interval import Interval
from omnibias.core.verified.series import certified_ratio_series_sum


def _mpmath():
    try:
        return importlib.import_module("mpmath")
    except ImportError:  # pragma: no cover - environment dependent
        return None


class TestBernoulliPolynomials:
    @pytest.mark.parametrize("n", list(range(0, 10)))
    def test_value_at_zero_is_bernoulli_number(self, n: int) -> None:
        assert bernoulli_polynomial_exact(n, 0) == bernoulli_number_exact(n)

    def test_low_order_closed_forms(self) -> None:
        x = Fraction(3, 7)
        assert bernoulli_polynomial_exact(0, x) == 1
        assert bernoulli_polynomial_exact(1, x) == x - Fraction(1, 2)
        assert bernoulli_polynomial_exact(2, x) == x * x - x + Fraction(1, 6)

    def test_reflection_B_n_of_one(self) -> None:
        # B_n(1) = (-1)^n B_n.
        for n in range(0, 12):
            assert bernoulli_polynomial_exact(n, 1) == (-1) ** n * bernoulli_number_exact(n)

    @pytest.mark.skipif(_mpmath() is None, reason="mpmath not installed")
    def test_matches_mpmath_bernpoly(self) -> None:
        mp = _mpmath()
        for n in range(0, 9):
            for x in [Fraction(1, 2), Fraction(1, 3), Fraction(3, 7), Fraction(-2, 5)]:
                got = float(bernoulli_polynomial_exact(n, x))
                ref = float(mp.bernpoly(n, mp.mpf(x.numerator) / x.denominator))
                assert got == pytest.approx(ref, rel=1e-12, abs=1e-12)


class TestGeneralizedBernoulli:
    def test_trivial_character_recovers_bernoulli(self) -> None:
        # Principal character mod 1: B_{n,chi} = B_n except n=1 (B_1(1) = +1/2).
        assert generalized_bernoulli_exact(0, (1,)) == bernoulli_number_exact(0)
        assert generalized_bernoulli_exact(1, (1,)) == Fraction(1, 2)
        for n in range(2, 6):
            assert generalized_bernoulli_exact(n, (1,)) == bernoulli_number_exact(n)

    def test_chi4_l_values_match_euler_numbers(self) -> None:
        # chi_4 (non-principal mod 4): L(1-n, chi) = -B_{n,chi}/n, and
        # L(-2m, chi_4) = E_{2m}/2 -> B_{2m+1,chi_4} = -(2m+1) E_{2m} / 2.
        chi4 = (0, 1, 0, -1)
        assert generalized_bernoulli_exact(1, chi4) == Fraction(-1, 2)  # L(0)=1/2
        # n=3 -> L(-2)=E_2/2=-1/2 ; n=5 -> L(-4)=E_4/2=5/2.
        assert -generalized_bernoulli_exact(3, chi4) / 3 == Fraction(-1, 2)
        assert -generalized_bernoulli_exact(5, chi4) / 5 == Fraction(5, 2)

    def test_input_guards(self) -> None:
        with pytest.raises(ValueError):
            generalized_bernoulli_exact(1, ())
        with pytest.raises(ValueError):
            generalized_bernoulli_exact(-1, (1,))
        with pytest.raises(ValueError):
            bernoulli_polynomial_exact(-1, 0)


class TestCertifiedRatioSeries:
    def test_geometric_sum_is_enclosed(self) -> None:
        # a_n = (1/3)^n -> sum = 3/2, ratio q = 1/3.
        enc = certified_ratio_series_sum(
            lambda k: Interval.from_rational(Fraction(1, 3)) ** k,
            Fraction(1, 3),
            num_terms=30,
        )
        assert enc.contains(1.5)
        assert enc.width < 1e-12

    def test_alternating_sum_is_enclosed(self) -> None:
        # a_n = (-1/2)^n -> sum = 1/(1+1/2) = 2/3; |ratio| = 1/2.
        enc = certified_ratio_series_sum(
            lambda k: Interval.from_rational(Fraction(-1, 2)) ** k,
            Fraction(1, 2),
            num_terms=40,
        )
        assert enc.contains(2.0 / 3.0)

    def test_num_terms_guard(self) -> None:
        with pytest.raises(ValueError):
            certified_ratio_series_sum(lambda k: Interval.point(0.0), Fraction(1, 3), num_terms=0)
