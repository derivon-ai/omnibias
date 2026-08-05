# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Gosper's algorithm: indefinite hypergeometric summation with an exact certificate."""

from __future__ import annotations

from fractions import Fraction
from math import factorial

import pytest
from omnibias.holonomic._core.gosper import gosper_sum
from omnibias.holonomic._core.zeilberger import gosper_definite_sum


def _brute(term, a, b) -> Fraction:  # type: ignore[no-untyped-def]
    return sum((Fraction(term(k)) for k in range(a, b)), Fraction(0))


def test_sum_of_k() -> None:
    # t(k) = k, ratio (k + 1)/k. sum_{k=1}^{b-1} k = b(b-1)/2.
    res = gosper_sum((1, 1), (0, 1))
    assert res.summable
    for b in range(2, 12):
        assert gosper_definite_sum((1, 1), (0, 1), Fraction(1), 1, b) == _brute(lambda k: k, 1, b)


def test_sum_of_k_squared() -> None:
    # t(k) = k^2, ratio (k + 1)^2/k^2.
    res = gosper_sum((1, 2, 1), (0, 0, 1))
    assert res.summable
    for b in range(2, 12):
        got = gosper_definite_sum((1, 2, 1), (0, 0, 1), Fraction(1), 1, b)
        assert got == _brute(lambda k: k * k, 1, b)


def test_sum_of_powers_of_two() -> None:
    # t(k) = 2^k, ratio 2. sum_{0}^{b-1} 2^k = 2^b - 1.
    res = gosper_sum((2,), (1,))
    assert res.summable
    for b in range(1, 12):
        assert gosper_definite_sum((2,), (1,), Fraction(1), 0, b) == 2**b - 1


def test_sum_k_times_k_factorial() -> None:
    # t(k) = k * k!, ratio (k + 1)^2/k. sum_{1}^{b-1} k*k! = b! - 1.
    res = gosper_sum((1, 2, 1), (0, 1))
    assert res.summable
    for b in range(2, 8):
        assert gosper_definite_sum((1, 2, 1), (0, 1), Fraction(1), 1, b) == factorial(b) - 1


def test_central_binomial_not_summable() -> None:
    # t(k) = C(2k, k), ratio 2(2k + 1)/(k + 1); partial sums have no closed form.
    res = gosper_sum((2, 4), (1, 1))
    assert not res.summable
    assert gosper_definite_sum((2, 4), (1, 1), Fraction(1), 0, 5) is None


def test_reciprocal_factorial_not_summable() -> None:
    # t(k) = 1/k!, ratio 1/(k + 1); sum -> e, not Gosper-summable.
    res = gosper_sum((1,), (1, 1))
    assert not res.summable


def test_certificate_requires_summable() -> None:
    res = gosper_sum((2, 4), (1, 1))
    with pytest.raises(ValueError, match="not Gosper-summable"):
        res.certificate(3)


def test_empty_range_is_zero() -> None:
    assert gosper_definite_sum((1, 1), (0, 1), Fraction(1), 3, 3) == 0


def test_definite_sum_guards_vanishing_denominator() -> None:
    # den(k) = k vanishes at k = 0 -> a clear error, not a raw ZeroDivisionError.
    with pytest.raises(ValueError, match="denominator vanishes"):
        gosper_definite_sum((1, 1), (0, 1), Fraction(1), 0, 5)
