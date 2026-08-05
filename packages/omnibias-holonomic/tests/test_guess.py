# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Guessed-then-verified D-finite and algebraic annihilators from exact series."""

from __future__ import annotations

from fractions import Fraction
from math import comb, factorial

from omnibias.holonomic._core.dfinite import DFinite
from omnibias.holonomic._core.guess import guess_algebraic, guess_dfinite, guess_recurrence
from omnibias.holonomic._core.poly2 import Poly2


def _sqrt1px(m: int) -> list[Fraction]:
    """Taylor coefficients of sqrt(1+x): binomial(1/2, k)."""
    out = []
    for k in range(m):
        c = Fraction(1)
        for t in range(k):
            c *= Fraction(1, 2) - t
        out.append(c / factorial(k))
    return out


def _catalan(m: int) -> list[Fraction]:
    return [Fraction(comb(2 * n, n), n + 1) for n in range(m)]


def _algebraic_residual(poly: Poly2, series: list[Fraction]) -> list[Fraction]:
    m = len(series)
    powers = [[Fraction(0)] * m for _ in range(max(j for _i, j in poly) + 1)]
    powers[0][0] = Fraction(1)
    for j in range(1, len(powers)):
        prev = powers[j - 1]
        cur = powers[j]
        for s in range(m):
            if prev[s] == 0:
                continue
            for t in range(m - s):
                if series[t]:
                    cur[s + t] += prev[s] * series[t]
    residual = [Fraction(0)] * m
    for (i, j), c in poly.items():
        for s in range(m):
            if 0 <= s - i < m and powers[j][s - i]:
                residual[s] += c * powers[j][s - i]
    return residual


def test_guess_dfinite_exp() -> None:
    series = [Fraction(1, factorial(m)) for m in range(14)]
    op = guess_dfinite(series)
    assert op is not None
    assert op.order == 1
    # regenerate exactly: D - 1 annihilates exp.
    d = DFinite(op, tuple(series[: op.order]))
    assert d.taylor(14) == series


def test_guess_dfinite_geometric() -> None:
    series = [Fraction(1)] * 14
    op = guess_dfinite(series)
    assert op is not None
    assert op.order == 1
    assert all(op.apply_series(series, s) == 0 for s in range(12))


def test_guess_dfinite_sqrt() -> None:
    series = _sqrt1px(14)
    op = guess_dfinite(series)
    assert op is not None
    assert op.order == 1
    d = DFinite(op, tuple(series[: op.order]))
    assert d.taylor(14) == series


def test_guess_dfinite_none_for_double_exponential() -> None:
    series = [Fraction(2) ** (2**n) for n in range(8)]
    assert guess_dfinite(series, max_order=3, max_degree=3) is None


def test_guess_algebraic_rational() -> None:
    series = [Fraction(1)] * 14  # 1/(1-x)
    poly = guess_algebraic(series)
    assert poly is not None
    assert max(j for _i, j in poly) == 1  # linear in y -> rational
    assert all(v == 0 for v in _algebraic_residual(poly, series))


def test_guess_algebraic_catalan() -> None:
    series = _catalan(14)
    poly = guess_algebraic(series)
    assert poly is not None
    assert max(j for _i, j in poly) == 2  # x y^2 - y + 1 = 0
    assert (1, 2) in poly
    assert all(v == 0 for v in _algebraic_residual(poly, series))


def test_guess_algebraic_sqrt() -> None:
    series = _sqrt1px(14)
    poly = guess_algebraic(series)
    assert poly is not None
    assert max(j for _i, j in poly) == 2  # y^2 - x - 1 = 0
    assert all(v == 0 for v in _algebraic_residual(poly, series))


def test_guess_algebraic_none_for_exp() -> None:
    series = [Fraction(1, factorial(m)) for m in range(12)]
    assert guess_algebraic(series, max_x_degree=3, max_y_degree=3) is None


def test_guess_recurrence_still_works() -> None:
    # regression: the original guesser is unchanged (factorials).
    op = guess_recurrence([Fraction(factorial(n)) for n in range(10)])
    assert op is not None
    assert op.order == 1
