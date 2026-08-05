# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Sheffer-sequence generation and the umbral operator layer (exact rational identities)."""

from __future__ import annotations

import random
from fractions import Fraction
from math import comb, factorial

import pytest
from omnibias.difference import (
    appell_sequence,
    bernoulli_polynomial,
    falling_factorial_coeffs,
    stirling_second_row,
)
from omnibias.difference.umbral import (
    associated_sequence,
    delta_operator_apply,
    pincherle_derivative,
    series_reciprocal,
    sheffer_sequence,
    shift_polynomial,
    umbral_composition,
    umbral_functional,
)

N = 7


def _eval(coeffs: tuple[Fraction, ...], x: Fraction) -> Fraction:
    return sum((c * x**j for j, c in enumerate(coeffs)), Fraction(0))


def _exp_minus_one(order: int) -> list[Fraction]:
    """The delta series ``e^t - 1`` to the given order (ascending coeffs)."""
    return [Fraction(0)] + [Fraction(1, factorial(k)) for k in range(1, order + 1)]


def _log_one_plus(order: int) -> list[Fraction]:
    """The delta series ``log(1 + t)`` to the given order (ascending coeffs)."""
    return [Fraction(0)] + [Fraction((-1) ** (k + 1), k) for k in range(1, order + 1)]


def _bernoulli_g(order: int) -> list[Fraction]:
    """The Sheffer ``g`` for Bernoulli polynomials: ``(e^t - 1)/t = sum t^k/(k+1)!``."""
    return [Fraction(1, factorial(k + 1)) for k in range(order + 1)]


def test_associated_of_t_are_monomials() -> None:
    seq = associated_sequence((0, 1), N)
    for n in range(N + 1):
        assert tuple(seq[n]) == tuple([Fraction(0)] * n + [Fraction(1)])


@pytest.mark.parametrize("n", range(N + 1))
def test_associated_of_exp_minus_one_is_falling_factorial(n: int) -> None:
    # e^t - 1 is the delta series of the forward difference: its associated
    # (binomial-type) sequence is the falling factorial (x)_n.
    seq = associated_sequence(_exp_minus_one(N), N)
    assert tuple(seq[n]) == tuple(Fraction(c) for c in falling_factorial_coeffs(n))


@pytest.mark.parametrize("n", range(N + 1))
def test_associated_of_log_is_touchard_stirling_second(n: int) -> None:
    # log(1 + t) has compositional inverse e^t - 1, so its associated sequence is the
    # Bell / Touchard polynomial whose coefficients are the Stirling second-kind row.
    seq = associated_sequence(_log_one_plus(N), N)
    assert tuple(seq[n]) == tuple(Fraction(s) for s in stirling_second_row(n))


@pytest.mark.parametrize("n", range(N + 1))
def test_sheffer_bernoulli_pair_is_bernoulli_polynomial(n: int) -> None:
    # Bernoulli polynomials are Sheffer for (g, f) = ((e^t-1)/t, t) (an Appell sequence).
    seq = sheffer_sequence(_bernoulli_g(N), (0, 1), N)
    assert tuple(seq[n]) == bernoulli_polynomial(n)


@pytest.mark.parametrize("seed", range(6))
def test_sheffer_with_f_identity_reduces_to_appell(seed: int) -> None:
    # With f = t, sheffer_sequence(g, t) is the Appell sequence of 1/g; choosing g as the
    # reciprocal of the moment EGF must reproduce appell_sequence exactly.
    rng = random.Random(seed)
    moments = [Fraction(rng.randint(-4, 4), rng.randint(1, 4)) for _ in range(N + 1)]
    moments[0] = Fraction(rng.choice((-3, -2, -1, 1, 2, 3)))  # a_0 = 1/g(0) must be non-zero
    egf = [moments[j] / factorial(j) for j in range(N + 1)]  # A(t) = 1/g = sum a_j t^j/j!
    g = series_reciprocal(egf, N)
    she = sheffer_sequence(g, (0, 1), N)
    app = appell_sequence(moments)
    for n in range(N + 1):
        assert tuple(she[n]) == tuple(app[n])


@pytest.mark.parametrize("seed", range(6))
def test_associated_sequence_binomial_identity(seed: int) -> None:
    # Binomial-type identity: p_n(x + y) = sum_k C(n, k) p_k(x) p_{n-k}(y).
    rng = random.Random(50 + seed)
    f = [Fraction(0), Fraction(1)] + [
        Fraction(rng.randint(-3, 3), rng.randint(1, 3)) for _ in range(N - 1)
    ]
    seq = associated_sequence(f, N)
    x, y = Fraction(2, 3), Fraction(-5, 4)
    for n in range(N + 1):
        lhs = _eval(tuple(seq[n]), x + y)
        rhs = sum(
            (comb(n, k) * _eval(tuple(seq[k]), x) * _eval(tuple(seq[n - k]), y) for k in range(n + 1)),
            Fraction(0),
        )
        assert lhs == rhs


def test_umbral_composition_inverse_gives_monomials() -> None:
    # Falling factorials (assoc of e^t - 1) umbral-composed with the Bell polynomials
    # (assoc of log(1 + t), the compositional inverse) recover the monomials x^n.
    ff = associated_sequence(_exp_minus_one(N), N)
    bell = associated_sequence(_log_one_plus(N), N)
    comp = umbral_composition(ff, bell)
    for n in range(N + 1):
        assert tuple(comp[n]) == tuple([Fraction(0)] * n + [Fraction(1)])


def test_shift_polynomial_exact() -> None:
    # (x^2 + 1) shifted by 3 -> x^2 + 6x + 10.
    assert shift_polynomial((Fraction(1), Fraction(0), Fraction(1)), 3) == (
        Fraction(10),
        Fraction(6),
        Fraction(1),
    )


@pytest.mark.parametrize("seed", range(6))
def test_shift_polynomial_matches_evaluation(seed: int) -> None:
    rng = random.Random(70 + seed)
    coeffs = tuple(Fraction(rng.randint(-5, 5)) for _ in range(rng.randint(1, 6)))
    a = Fraction(rng.randint(-3, 3), rng.randint(1, 3))
    shifted = shift_polynomial(coeffs, a)
    for x in (Fraction(-2), Fraction(0), Fraction(3, 2), Fraction(4)):
        assert _eval(shifted, x) == _eval(coeffs, x + a)


def test_delta_operator_D_is_derivative() -> None:
    # Q = f(D) with f = t is just D.
    p = (Fraction(2), Fraction(-1), Fraction(3), Fraction(1))
    assert delta_operator_apply((0, 1), p) == (Fraction(-1), Fraction(6), Fraction(3))


@pytest.mark.parametrize("seed", range(6))
def test_pincherle_is_commutator_with_multiplication(seed: int) -> None:
    # [f(D), X] = f'(D): (Q X - X Q) p == delta_operator_apply(pincherle_derivative(f), p).
    rng = random.Random(90 + seed)
    f = [Fraction(0)] + [Fraction(rng.randint(-3, 3), rng.randint(1, 3)) for _ in range(N)]
    p = tuple(Fraction(rng.randint(-4, 4)) for _ in range(rng.randint(1, 6)))

    def mul_x(poly: tuple[Fraction, ...]) -> tuple[Fraction, ...]:
        return (Fraction(0),) + poly

    def strip(poly: tuple[Fraction, ...]) -> tuple[Fraction, ...]:
        out = list(poly)
        while len(out) > 1 and out[-1] == 0:
            out.pop()
        return tuple(out)

    qxp = delta_operator_apply(f, mul_x(p))
    xqp = mul_x(delta_operator_apply(f, p))
    width = max(len(qxp), len(xqp))
    left = list(qxp) + [Fraction(0)] * (width - len(qxp))
    right = list(xqp) + [Fraction(0)] * (width - len(xqp))
    commutator = strip(tuple(a - b for a, b in zip(left, right, strict=True)))
    assert commutator == delta_operator_apply(pincherle_derivative(f), p)


def test_umbral_functional_evaluation() -> None:
    # Moments mu_k = a^k realise the evaluation functional L(p) = p(a).
    p = (Fraction(2), Fraction(-1), Fraction(3), Fraction(1))
    a = Fraction(3, 2)
    moments = [a**k for k in range(len(p))]
    assert umbral_functional(moments, p) == _eval(p, a)


def test_umbral_functional_needs_enough_moments() -> None:
    with pytest.raises(ValueError):
        umbral_functional([Fraction(1)], (Fraction(1), Fraction(2)))


def test_sheffer_rejects_non_delta_series() -> None:
    with pytest.raises(ValueError):
        sheffer_sequence((1,), (1, 1), 3)  # f(0) != 0 is not a delta series
    with pytest.raises(ValueError):
        sheffer_sequence((0, 1), (0, 1), 3)  # g(0) == 0 is not invertible
