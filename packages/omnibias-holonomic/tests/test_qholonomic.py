# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""q-holonomic bridge: q-shift algebra, q-Gosper, and guessed-then-verified q-Zeilberger."""

from __future__ import annotations

from fractions import Fraction

import pytest
from omnibias.holonomic._core.qholonomic import (
    q_apply,
    q_dilate,
    q_gosper,
    q_gosper_definite_sum,
    q_shift_algebra,
    q_zeilberger,
)
from omnibias.holonomic._core.rational_poly import Poly, peval, to_poly
from omnibias.qcalculus import q_binomial, q_bracket


def _term_values(num: Poly, den: Poly, t0: Fraction, a: int, b: int, q: Fraction) -> dict[int, Fraction]:
    t = {a: t0}
    for k in range(a, b):
        x = q**k
        t[k + 1] = t[k] * peval(num, x) / peval(den, x)
    return t


def test_q_dilate() -> None:
    # q_dilate(1 + x + x^2, s) = 1 + s x + s^2 x^2.
    assert q_dilate((Fraction(1), Fraction(1), Fraction(1)), Fraction(2)) == (
        Fraction(1),
        Fraction(2),
        Fraction(4),
    )


def test_q_shift_algebra_sigma_is_dilation() -> None:
    alg = q_shift_algebra(3)
    assert alg.name == "q-shift"
    assert alg.sigma((Fraction(1), Fraction(1))) == (Fraction(1), Fraction(3))  # 1 + 3x
    assert alg.delta((Fraction(5),)) == ()


def test_q_shift_requires_valid_q() -> None:
    with pytest.raises(ValueError, match="q must not be 1"):
        q_shift_algebra(1)
    with pytest.raises(ValueError, match="q must be non-zero"):
        q_shift_algebra(0)


def test_q_gosper_geometric_constant_ratio() -> None:
    # t(k) = 3^k has constant ratio 3; sum_{0}^{b-1} 3^k = (3^b - 1)/2.
    q = Fraction(2)
    res = q_gosper((3,), (1,), q)
    assert res.summable
    total = q_gosper_definite_sum((3,), (1,), Fraction(1), 0, 5, q)
    assert total == sum(Fraction(3) ** k for k in range(5))
    assert total == Fraction(3**5 - 1, 2)


def test_q_gosper_designed_certificate() -> None:
    # Designed so R(x) = x solves the q-Gosper equation: rho(x) = (x+1)/(q x).
    q = Fraction(2)
    num, den = to_poly([1, 1]), to_poly([0, q])
    res = q_gosper(num, den, q)
    assert res.summable
    # certificate R(q^k) = q^k.
    for k in range(5):
        assert res.certificate(k) == q**k
    # definite sum matches the brute rational sum of the term.
    t = _term_values(num, den, Fraction(1), 0, 6, q)
    brute = sum(t[k] for k in range(0, 6))
    assert q_gosper_definite_sum(num, den, Fraction(1), 0, 6, q) == brute


def test_q_gosper_refuses_non_summable() -> None:
    # t(k) = 1 / [k]_q!  ->  ratio 1/[k+1]_q = (1-q)/(1 - q x); not q-Gosper-summable.
    q = Fraction(2)
    num, den = to_poly([1 - q]), to_poly([1, -q])
    res = q_gosper(num, den, q)
    assert not res.summable
    assert q_gosper_definite_sum(num, den, Fraction(1), 0, 5, q) is None


def test_q_gosper_across_q_sweep_and_limit() -> None:
    # The geometric closed form holds exactly for every rational q != 1.
    for q in (Fraction(2), Fraction(3), Fraction(1, 2), Fraction(5, 3)):
        total = q_gosper_definite_sum((3,), (1,), Fraction(1), 0, 4, q)
        assert total == sum(Fraction(3) ** k for k in range(4))


def test_q_zeilberger_q_bracket_sum() -> None:
    # S(n) = sum_{k=0}^n q^k = [n+1]_q obeys an order-1 q-recurrence.
    q = Fraction(2)
    rec = q_zeilberger(lambda n, k: q**k, q, n_max=14)
    assert rec is not None
    assert rec.order == 1
    assert rec.max_residual() == 0
    # cross-check the sampled values against the exact q-bracket.
    for n in range(6):
        assert rec.values[n] == q_bracket(n + 1, q)


def test_q_zeilberger_q_binomial_theorem() -> None:
    # q-binomial theorem: sum_k [n,k]_q q^{C(k,2)} = prod_{j<n} (1 + q^j) = (-1; q)_n.
    q = Fraction(3)

    def summand(n: int, k: int) -> Fraction:
        return q_binomial(n, k, q) * q ** (k * (k - 1) // 2)

    rec = q_zeilberger(summand, q, n_max=12)
    assert rec is not None
    assert rec.order == 1
    assert rec.max_residual() == 0
    for n in range(5):
        expected = Fraction(1)
        for j in range(n):
            expected *= 1 + q**j
        assert rec.values[n] == expected


def test_q_apply_matches_recurrence() -> None:
    q = Fraction(2)
    rec = q_zeilberger(lambda n, k: q**k, q, n_max=12)
    assert rec is not None
    for n in range(rec.checked_upto - rec.order + 1):
        assert q_apply(rec.operator, lambda m: rec.values[m], n, q) == 0
