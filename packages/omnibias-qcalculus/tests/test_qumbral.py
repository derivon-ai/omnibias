# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""q-umbral / q-Sheffer identities: exact q-identities plus q -> 1 collapse to the classical umbral."""

from __future__ import annotations

import random
from fractions import Fraction

import pytest

# Classical umbral (the q -> 1 reference; omnibias-qcalculus depends on omnibias-difference).
from omnibias.difference import (
    appell_sequence,
    bernoulli_number,
    bernoulli_polynomial,
    binomial_transform,
    falling_factorial_coeffs,
    falling_to_monomial,
    inverse_binomial_transform,
    monomial_to_falling,
    newton_forward_coeffs,
    pincherle_derivative,
    sheffer_sequence,
    stirling_first_signed_row,
    stirling_second_row,
    umbral_composition,
)
from omnibias.qcalculus import q_bracket, q_derivative_poly
from omnibias.qcalculus.umbral import (
    q_appell_sequence,
    q_associated_sequence,
    q_binomial_transform,
    q_delta_operator_apply,
    q_falling_factorial_coeffs,
    q_falling_to_monomial,
    q_inverse_binomial_transform,
    q_monomial_to_falling,
    q_newton_forward_coeffs,
    q_newton_forward_value,
    q_pincherle_derivative,
    q_sheffer_classify,
    q_sheffer_sequence,
    q_stirling_first_signed,
    q_stirling_first_signed_row,
    q_stirling_second,
    q_stirling_second_row,
    q_umbral_composition,
)

QS = [Fraction(1, 2), Fraction(2, 3), Fraction(3, 2), Fraction(3)]
ONE = Fraction(1)
N = 6


def _strip(coeffs: tuple[Fraction, ...]) -> tuple[Fraction, ...]:
    out = list(coeffs)
    while len(out) > 1 and out[-1] == 0:
        out.pop()
    return tuple(out)


def _pad(coeffs: tuple[Fraction, ...], width: int) -> tuple[Fraction, ...]:
    return tuple(list(coeffs) + [Fraction(0)] * (width - len(coeffs)))


def _same_poly(a: tuple[Fraction, ...], b: tuple[Fraction, ...]) -> bool:
    a, b = _strip(a), _strip(b)
    width = max(len(a), len(b))
    return _pad(a, width) == _pad(b, width)


def _eval(coeffs: tuple[Fraction, ...], x: Fraction) -> Fraction:
    return sum((c * x**j for j, c in enumerate(coeffs)), Fraction(0))


# --------------------------------------------------------------------------- #
# q-Stirling numbers                                                           #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("q", QS)
def test_q_stirling_second_recurrence(q: Fraction) -> None:
    for n in range(1, N + 1):
        for k in range(1, n + 1):
            expected = q_stirling_second(n - 1, k - 1, q) + q_bracket(k, q) * q_stirling_second(
                n - 1, k, q
            )
            assert q_stirling_second(n, k, q) == expected


@pytest.mark.parametrize("q", QS)
def test_q_stirling_kinds_are_inverse(q: Fraction) -> None:
    # sum_j s_q(n, j) S_q(j, m) = delta_{n, m}.
    for n in range(N + 1):
        for m in range(n + 1):
            total = sum(
                (q_stirling_first_signed(n, j, q) * q_stirling_second(j, m, q) for j in range(n + 1)),
                Fraction(0),
            )
            assert total == (Fraction(1) if n == m else Fraction(0))


@pytest.mark.parametrize("q", QS)
def test_q_falling_factorial_is_signed_first_kind(q: Fraction) -> None:
    for n in range(N + 1):
        assert q_falling_factorial_coeffs(n, q) == q_stirling_first_signed_row(n, q)


@pytest.mark.parametrize("n", range(N + 1))
def test_q_stirling_reduces_to_classical(n: int) -> None:
    assert tuple(int(x) for x in q_stirling_second_row(n, ONE)) == stirling_second_row(n)
    assert tuple(int(x) for x in q_stirling_first_signed_row(n, ONE)) == stirling_first_signed_row(n)
    assert tuple(int(x) for x in q_falling_factorial_coeffs(n, ONE)) == falling_factorial_coeffs(n)


# --------------------------------------------------------------------------- #
# q-transforms                                                                 #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("q", QS)
def test_q_stirling_transforms_are_inverse(q: Fraction) -> None:
    coeffs = [Fraction(1), Fraction(-2), Fraction(3), Fraction(0), Fraction(4)]
    assert q_falling_to_monomial(q_monomial_to_falling(coeffs, q), q) == tuple(coeffs)


@pytest.mark.parametrize("q", QS)
def test_q_binomial_transform_inverse(q: Fraction) -> None:
    rng = random.Random(int(q.numerator) * 7 + int(q.denominator))
    seq = [Fraction(rng.randint(-6, 6)) for _ in range(7)]
    assert q_inverse_binomial_transform(q_binomial_transform(seq, q), q) == tuple(seq)


def test_q_transforms_reduce_to_classical() -> None:
    coeffs = [Fraction(1), Fraction(-2), Fraction(3), Fraction(0), Fraction(4)]
    assert q_monomial_to_falling(coeffs, ONE) == monomial_to_falling(coeffs)
    assert q_falling_to_monomial(coeffs, ONE) == falling_to_monomial(coeffs)
    seq = [Fraction(1), Fraction(3), Fraction(-2), Fraction(5), Fraction(0), Fraction(-1)]
    assert q_binomial_transform(seq, ONE) == binomial_transform(seq)
    assert q_inverse_binomial_transform(seq, ONE) == inverse_binomial_transform(seq)


# --------------------------------------------------------------------------- #
# q-Appell sequences                                                           #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("q", QS)
def test_q_appell_derivative_property(q: Fraction) -> None:
    # D_q p_n = [n]_q p_{n-1}.
    constants = [Fraction(1), Fraction(2), Fraction(-1), Fraction(3), Fraction(0), Fraction(-2), Fraction(1)]
    seq = q_appell_sequence(constants, q)
    for n in range(1, len(seq)):
        lhs = _strip(q_derivative_poly(seq[n], q))
        rhs = _strip(tuple(q_bracket(n, q) * c for c in seq[n - 1]))
        assert _same_poly(lhs, rhs)


def test_q_appell_reduces_to_classical() -> None:
    constants = [Fraction(1), Fraction(2), Fraction(-1), Fraction(3), Fraction(0), Fraction(-2)]
    q_seq = [tuple(p) for p in q_appell_sequence(constants, ONE)]
    classical = [tuple(p) for p in appell_sequence(constants)]
    assert q_seq == classical


@pytest.mark.parametrize("q", QS)
def test_q_bernoulli_polynomials_via_appell(q: Fraction) -> None:
    from omnibias.qcalculus import q_bernoulli

    # The q-Bernoulli polynomials B_n(x; q) = sum_k [n,k]_q B_{n-k}(q) x^k are the q-Appell
    # sequence of the q-Bernoulli numbers: constant term B_n(q), q-derivative [n]_q B_{n-1}.
    constants = [q_bernoulli(j, q) for j in range(N + 1)]
    seq = q_appell_sequence(constants, q)
    for n in range(N + 1):
        assert seq[n][0] == q_bernoulli(n, q)
    for n in range(1, N + 1):
        lhs = _strip(q_derivative_poly(seq[n], q))
        rhs = _strip(tuple(q_bracket(n, q) * c for c in seq[n - 1]))
        assert _same_poly(lhs, rhs)


def test_q_bernoulli_polynomials_reduce_to_classical() -> None:
    from omnibias.qcalculus import q_bernoulli

    constants = [q_bernoulli(j, ONE) for j in range(N + 1)]
    assert constants == [bernoulli_number(j) for j in range(N + 1)]
    seq = q_appell_sequence(constants, ONE)
    for n in range(N + 1):
        assert tuple(seq[n]) == bernoulli_polynomial(n)


# --------------------------------------------------------------------------- #
# q-Newton interpolation                                                       #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("q", QS)
def test_q_newton_reproduces_polynomial(q: Fraction) -> None:
    poly = (Fraction(2), Fraction(-1), Fraction(3), Fraction(1), Fraction(-2))
    samples = [_eval(poly, q_bracket(j, q)) for j in range(len(poly))]
    coeffs = q_newton_forward_coeffs(samples, q)
    for x in (Fraction(-2), Fraction(0), Fraction(1), Fraction(5, 3), Fraction(7)):
        assert q_newton_forward_value(coeffs, x, q) == _eval(poly, x)


def test_q_newton_reduces_to_classical() -> None:
    poly = (Fraction(2), Fraction(-1), Fraction(3), Fraction(1))
    samples = [_eval(poly, Fraction(j)) for j in range(len(poly))]
    assert q_newton_forward_coeffs(samples, ONE) == newton_forward_coeffs(samples)


# --------------------------------------------------------------------------- #
# Full q-Sheffer layer                                                         #
# --------------------------------------------------------------------------- #
_G = [Fraction(1), Fraction(1, 2), Fraction(-1, 3), Fraction(1, 4)]
_F = [Fraction(0), Fraction(1), Fraction(1, 2), Fraction(-1, 5), Fraction(1, 6)]


@pytest.mark.parametrize("q", QS)
def test_q_sheffer_recurrence(q: Fraction) -> None:
    # The exact q-Sheffer recurrence Q s_n = [n]_q s_{n-1} with Q = f(D_q).
    seq = q_sheffer_sequence(_G, _F, N, q)
    for n in range(1, N + 1):
        lhs = q_delta_operator_apply(_F, seq[n], q)
        rhs = tuple(q_bracket(n, q) * c for c in seq[n - 1])
        assert _same_poly(lhs, rhs)


@pytest.mark.parametrize("q", QS)
def test_q_associated_of_t_are_monomials(q: Fraction) -> None:
    seq = q_associated_sequence((0, 1), N, q)
    for n in range(N + 1):
        assert tuple(seq[n]) == tuple([Fraction(0)] * n + [Fraction(1)])


@pytest.mark.parametrize("q", QS)
def test_q_delta_operator_of_t_is_q_derivative(q: Fraction) -> None:
    poly = (Fraction(2), Fraction(-1), Fraction(3), Fraction(1), Fraction(-2))
    assert _same_poly(q_delta_operator_apply((0, 1), poly, q), q_derivative_poly(poly, q))


def test_q_sheffer_reduces_to_classical() -> None:
    q_seq = [tuple(p) for p in q_sheffer_sequence(_G, _F, 5, ONE)]
    classical = [tuple(p) for p in sheffer_sequence(_G, _F, 5)]
    assert q_seq == classical


def test_q_pincherle_reduces_to_classical() -> None:
    assert q_pincherle_derivative(_F, ONE) == pincherle_derivative(_F)


def test_q_umbral_composition_is_q_independent() -> None:
    # Delegates to the classical (q-independent) umbral composition.
    ff = q_associated_sequence([Fraction(0), Fraction(1), Fraction(1, 2), Fraction(1, 6)], 4, Fraction(2, 3))
    bell = q_associated_sequence([Fraction(0), Fraction(1), Fraction(-1, 2), Fraction(1, 3)], 4, Fraction(2, 3))
    assert q_umbral_composition(ff, bell) == umbral_composition(ff, bell)


# --------------------------------------------------------------------------- #
# Classification + errors                                                      #
# --------------------------------------------------------------------------- #
def test_q_sheffer_classify() -> None:
    assert q_sheffer_classify((1,), (0, 1)).kind == "appell"
    assert q_sheffer_classify((1,), (0, 1, 1)).kind == "associated"
    assert q_sheffer_classify((1, 2), (0, 1, 1)).kind == "sheffer"


def test_q_sheffer_rejects_invalid_pairs() -> None:
    with pytest.raises(ValueError):
        q_sheffer_sequence((1,), (1, 1), 3, Fraction(1, 2))  # f not a delta series
    with pytest.raises(ValueError):
        q_sheffer_sequence((0, 1), (0, 1), 3, Fraction(1, 2))  # g not invertible


# --------------------------------------------------------------------------- #
# Headline q -> 1 collapse: every op agrees with the classical difference umbral #
# --------------------------------------------------------------------------- #
def test_q_to_one_collapse_matches_classical_umbral() -> None:
    # q-Stirling rows.
    for n in range(N + 1):
        assert tuple(int(x) for x in q_stirling_second_row(n, ONE)) == stirling_second_row(n)
    # transforms.
    coeffs = [Fraction(1), Fraction(-2), Fraction(3), Fraction(0), Fraction(4)]
    assert q_monomial_to_falling(coeffs, ONE) == monomial_to_falling(coeffs)
    assert q_falling_to_monomial(coeffs, ONE) == falling_to_monomial(coeffs)
    # q-Appell -> Appell.
    constants = [Fraction(1), Fraction(-3), Fraction(2), Fraction(5)]
    assert [tuple(p) for p in q_appell_sequence(constants, ONE)] == [
        tuple(p) for p in appell_sequence(constants)
    ]
    # q-Sheffer -> Sheffer, q-Pincherle -> Pincherle.
    assert [tuple(p) for p in q_sheffer_sequence(_G, _F, 5, ONE)] == [
        tuple(p) for p in sheffer_sequence(_G, _F, 5)
    ]
    assert q_pincherle_derivative(_F, ONE) == pincherle_derivative(_F)
