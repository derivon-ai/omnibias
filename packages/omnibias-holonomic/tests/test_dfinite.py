# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""D-finite / P-recursive objects and their closure operations."""

from __future__ import annotations

import math
from fractions import Fraction

import pytest
from omnibias.holonomic._core.dfinite import (
    DFinite,
    PRecursive,
    dfinite_add,
    dfinite_cauchy,
    dfinite_hadamard,
)
from omnibias.holonomic._core.ore import diff_algebra, shift_algebra


def _fib() -> PRecursive:
    S = shift_algebra()
    return PRecursive(S.operator([[-1], [-1], [1]]), (Fraction(0), Fraction(1)))


def _factorial() -> PRecursive:
    # a_{n+1} - (n + 1) a_n = 0.
    S = shift_algebra()
    return PRecursive(S.operator([[-1, -1], [1]]), (Fraction(1),))


def test_precursive_generates_fibonacci() -> None:
    fib = _fib()
    assert [int(x) for x in fib.terms(11)] == [0, 1, 1, 2, 3, 5, 8, 13, 21, 34, 55]
    assert fib.term(10) == 55


def test_precursive_generates_factorial() -> None:
    fact = _factorial()
    assert [int(x) for x in fact.terms(7)] == [math.factorial(n) for n in range(7)]


def test_precursive_requires_enough_initials() -> None:
    S = shift_algebra()
    with pytest.raises(ValueError, match="initial values"):
        PRecursive(S.operator([[-1], [-1], [1]]), (Fraction(0),))


def test_precursive_rejects_order_zero() -> None:
    S = shift_algebra()
    with pytest.raises(ValueError, match="order >= 1"):
        PRecursive(S.operator([[1]]), (Fraction(1),))


def test_dfinite_exp_taylor() -> None:
    D = diff_algebra()
    exp = DFinite(D.operator([[-1], [1]]), (Fraction(1),))
    coeffs = exp.taylor(10)
    assert coeffs == [Fraction(1, math.factorial(m)) for m in range(10)]


def test_dfinite_cos_taylor() -> None:
    # (D^2 + 1) cos = 0, cos(0) = 1, cos'(0) = 0.
    D = diff_algebra()
    cos = DFinite(D.operator([[1], [], [1]]), (Fraction(1), Fraction(0)))
    coeffs = cos.taylor(10)
    expected = [
        Fraction((-1) ** (m // 2), math.factorial(m)) if m % 2 == 0 else Fraction(0)
        for m in range(10)
    ]
    assert coeffs == expected


def test_closure_add_matches_termwise() -> None:
    fib, fact = _fib(), _factorial()
    s = dfinite_add(fib, fact, terms=30)
    for n in range(10):
        assert s.term(n) == fib.term(n) + fact.term(n)


def test_closure_hadamard_matches_termwise() -> None:
    fib, fact = _fib(), _factorial()
    h = dfinite_hadamard(fib, fact, terms=30)
    for n in range(10):
        assert h.term(n) == fib.term(n) * fact.term(n)


def test_closure_cauchy_matches_convolution() -> None:
    fib, fact = _fib(), _factorial()
    c = dfinite_cauchy(fib, fact, terms=30)
    for n in range(8):
        expected = sum((fib.term(i) * fact.term(n - i) for i in range(n + 1)), Fraction(0))
        assert c.term(n) == expected


def test_closure_returns_verified_annihilator() -> None:
    # The fitted operator must exactly regenerate the combined sequence.
    fib, fact = _fib(), _factorial()
    s = dfinite_add(fib, fact, terms=30)
    regenerated = s.terms(30)
    combined = [fib.term(n) + fact.term(n) for n in range(30)]
    assert regenerated == combined
