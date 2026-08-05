# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Petkovsek's Hyper: hypergeometric-term solutions of shift recurrences."""

from __future__ import annotations

from fractions import Fraction
from math import comb, factorial

import pytest
from omnibias.holonomic._core.ore import diff_algebra, shift_algebra
from omnibias.holonomic._core.petkovsek import hyper, term_ratio_annihilates
from omnibias.holonomic._core.ratfunc import RatFunc
from omnibias.holonomic._core.rational_poly import peval, to_poly


def _ratio_at(ratio: RatFunc, n: int) -> Fraction:
    num, den = ratio
    return peval(num, n) / peval(den, n)


def _is_constant(ratio: RatFunc) -> bool:
    num, den = ratio
    return len(den) == 1 and len(num) <= 1


def test_factorial_ratio() -> None:
    # y(n+1) - (n+1) y(n) = 0  ->  y(n) = n!, ratio n+1.
    S = shift_algebra()
    op = S.operator([[-1, -1], [1]])
    sols = hyper(op)
    assert len(sols) == 1
    ratio = sols[0]
    assert term_ratio_annihilates(op, ratio)
    for n in range(6):
        assert _ratio_at(ratio, n) == Fraction(factorial(n + 1), factorial(n))


def test_geometric_ratio() -> None:
    # y(n+1) - 2 y(n) = 0  ->  y(n) = 2^n, constant ratio 2.
    S = shift_algebra()
    op = S.operator([[-2], [1]])
    sols = hyper(op)
    assert len(sols) == 1
    assert _is_constant(sols[0])
    assert _ratio_at(sols[0], 3) == Fraction(2)


def test_catalan_ratio() -> None:
    # (n+2) C(n+1) - (4n+2) C(n) = 0  ->  ratio 2(2n+1)/(n+2).
    S = shift_algebra()
    op = S.operator([[-2, -4], [2, 1]])
    sols = hyper(op)
    assert len(sols) == 1
    ratio = sols[0]
    assert term_ratio_annihilates(op, ratio)

    def catalan(n: int) -> int:
        return comb(2 * n, n) // (n + 1)

    for n in range(6):
        assert _ratio_at(ratio, n) == Fraction(catalan(n + 1), catalan(n))


def test_multiple_geometric_solutions() -> None:
    # (S - 2)(S - 3) = S^2 - 5S + 6: two hypergeometric solutions 2^n and 3^n.
    S = shift_algebra()
    op = S.operator([[6], [-5], [1]])
    sols = hyper(op)
    constants = {_ratio_at(s, 0) for s in sols if _is_constant(s)}
    assert constants == {Fraction(2), Fraction(3)}


def test_fibonacci_has_no_rational_hypergeometric_solution() -> None:
    # y(n+2) - y(n+1) - y(n) = 0: solutions phi^n / psi^n are irrational -> out of scope.
    S = shift_algebra()
    op = S.operator([[-1], [-1], [1]])
    assert hyper(op) == []


def test_term_ratio_annihilates_rejects_wrong_ratio() -> None:
    S = shift_algebra()
    op = S.operator([[-1, -1], [1]])  # factorial: true ratio n+1
    wrong: RatFunc = (to_poly([2, 1]), to_poly([1]))  # n + 2
    assert not term_ratio_annihilates(op, wrong)


def test_zero_leading_or_trailing_raises() -> None:
    S = shift_algebra()
    op = S.operator([[], [-1], [1]])  # trailing coefficient is zero
    with pytest.raises(NotImplementedError):
        hyper(op)


def test_wrong_algebra_raises() -> None:
    D = diff_algebra()
    op = D.operator([[-1], [1]])
    with pytest.raises(ValueError, match="shift algebra"):
        hyper(op)
