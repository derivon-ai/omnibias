# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Rigorous weighted sequence spaces: norm, tail bounds, Banach algebra."""

from __future__ import annotations

import random
from fractions import Fraction

import pytest
from omnibias.core.verified.interval import Interval
from omnibias.core.verified.sequence_space import (
    ValidatedSeries,
    convolve,
    ell1_nu_norm,
    fourier_nu_norm,
    geometric_tail_bound,
)


def test_ell1_nu_norm_value() -> None:
    # ||a||_nu = |1| + |−2| nu + |3| nu^2 at nu = 2 -> 1 + 4 + 12 = 17.
    coeffs = [Interval.point(1.0), Interval.point(-2.0), Interval.point(3.0)]
    norm = ell1_nu_norm(coeffs, 2.0)
    assert norm.lo <= 17.0 <= norm.hi


def test_chebyshev_weighting() -> None:
    coeffs = [Interval.point(1.0), Interval.point(1.0)]
    # |c0| + 2|c1| nu at nu=1 -> 1 + 2 = 3.
    norm = ell1_nu_norm(coeffs, 1.0, chebyshev=True)
    assert norm.lo <= 3.0 <= norm.hi


def test_fourier_two_sided_norm() -> None:
    coeffs = {-1: Interval.point(2.0), 0: Interval.point(1.0), 1: Interval.point(2.0)}
    # 1 + 2 nu + 2 nu at nu=3 -> 1 + 12 = 13.
    norm = fourier_nu_norm(coeffs, 3.0)
    assert norm.lo <= 13.0 <= norm.hi


def test_geometric_tail_encloses_exact_series() -> None:
    # sum_{k>2} M q^k nu^k with M=1, q=0.5, nu=1, exact = r^3/(1-r), r=0.5 -> 0.25.
    bound = geometric_tail_bound(1.0, 0.5, 1.0, 2)
    assert bound.lo <= 0.125 / (1 - 0.5) <= bound.hi
    assert bound.hi < 0.26


def test_geometric_tail_requires_convergent_weight() -> None:
    with pytest.raises(ValueError):
        geometric_tail_bound(1.0, 2.0, 1.0, 0)  # nu*ratio = 2 >= 1


def test_convolution_matches_polynomial_product() -> None:
    a = [Interval.point(1.0), Interval.point(2.0)]  # 1 + 2x
    b = [Interval.point(3.0), Interval.point(4.0)]  # 3 + 4x
    c = convolve(a, b)  # 3 + 10x + 8x^2
    vals = [iv.mid for iv in c]
    assert vals == [3.0, 10.0, 8.0]


def test_validated_series_product_is_sound_vs_exact() -> None:
    # Two finite polynomials -> product is exact; truncating folds tail rigorously.
    rng = random.Random(0)
    nu = 1.3
    for _ in range(100):
        a_coeffs = [Fraction(rng.randint(-3, 3)) for _ in range(5)]
        b_coeffs = [Fraction(rng.randint(-3, 3)) for _ in range(5)]
        a = ValidatedSeries.from_coeffs([Interval.from_value(c) for c in a_coeffs], nu)
        b = ValidatedSeries.from_coeffs([Interval.from_value(c) for c in b_coeffs], nu)
        prod = a * b

        # Exact product coefficients (rationals).
        exact = [Fraction(0)] * (len(a_coeffs) + len(b_coeffs) - 1)
        for i, ai in enumerate(a_coeffs):
            for j, bj in enumerate(b_coeffs):
                exact[i + j] += ai * bj

        n = prod.order
        for k in range(n + 1):
            assert prod.coeffs[k].lo <= exact[k] <= prod.coeffs[k].hi
        # The tail must bound the weighted norm of the dropped coefficients.
        dropped_norm = sum(
            abs(exact[k]) * Fraction(nu).__pow__(k) for k in range(n + 1, len(exact))
        )
        assert Fraction(prod.tail.hi) >= dropped_norm


def test_banach_algebra_submultiplicative() -> None:
    rng = random.Random(1)
    nu = 1.5
    for _ in range(100):
        a = ValidatedSeries.from_coeffs(
            [Interval.point(rng.uniform(-2, 2)) for _ in range(4)], nu, tail=abs(rng.uniform(0, 0.3))
        )
        b = ValidatedSeries.from_coeffs(
            [Interval.point(rng.uniform(-2, 2)) for _ in range(4)], nu, tail=abs(rng.uniform(0, 0.3))
        )
        prod = a * b
        # ||a*b|| <= ||a|| ||b||  (submultiplicativity of the weighted norm).
        assert prod.norm().hi <= a.banach_algebra_bound(b).hi * (1 + 1e-12)
