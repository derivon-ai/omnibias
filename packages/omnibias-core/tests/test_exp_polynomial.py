# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Independent identities and resource refusals for sparse exact functions."""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from dataclasses import FrozenInstanceError, replace
from fractions import Fraction
from itertools import repeat
from typing import cast

import pytest
from omnibias.core.exp_polynomial import (
    ExponentialBudget,
    ExponentialTerm,
    ExpPolynomial,
    Rational,
    TermInput,
)


def test_canonicalization_merges_sorts_and_snapshots_inputs() -> None:
    frequency: list[Rational] = [Fraction(1, 2), 0]
    powers = [1, 2]
    terms: list[TermInput] = [(frequency, powers, 2), ((0, 0), (0, 0), 3),
                             (frequency, powers, Fraction(-1, 2)), ((3, 4), (2, 0), 0)]
    result = ExpPolynomial.from_terms(2, terms)
    frequency[0], powers[0] = 9, 9
    terms.clear()
    assert result.terms == (((Fraction(0), Fraction(0)), (0, 0), Fraction(3)),
                            ((Fraction(1, 2), Fraction(0)), (1, 2), Fraction(3, 2)))
    assert all(isinstance(value, Fraction) for frequency, _, coefficient in result.terms
               for value in (*frequency, coefficient))
    assert hash(result) == hash(ExpPolynomial.from_terms(2, result.terms))
    with pytest.raises(FrozenInstanceError):
        result.dimension = 3  # type: ignore[misc]


def test_direct_constructor_and_factory_are_equivalent() -> None:
    raw = [([1, 0], [2, 3], Fraction(2, 3)), ([1, 0], [2, 3], Fraction(1, 3))]
    direct = ExpPolynomial(2, cast(tuple[ExponentialTerm, ...], raw))
    assert direct == ExpPolynomial.from_terms(2, cast(Iterable[TermInput], raw))
    defaulted = ExpPolynomial(1, (), cast(ExponentialBudget, None))
    assert defaulted.budget == ExponentialBudget()
    assert (defaulted + ExpPolynomial.constant(1, 1)).terms == ExpPolynomial.constant(1, 1).terms


def test_zero_constant_and_generators() -> None:
    zero = ExpPolynomial.from_terms(3, iter(()))
    assert zero.is_zero()
    assert zero == ExpPolynomial.constant(0, 3)
    assert zero.dimension == 3
    assert zero.terms == ()
    assert ExpPolynomial.from_terms(1, (((2,), (1,), c) for c in (2, -2))).is_zero()
    assert ExpPolynomial.exponential((0, 0), Fraction(2, 3)) == ExpPolynomial.constant(Fraction(2, 3), 2)
    assert ExpPolynomial.monomial((0, 0), -2) == ExpPolynomial.constant(-2, 2)


def test_ring_identities_with_rational_frequencies_and_polynomial_factors() -> None:
    p = ExpPolynomial.from_terms(2, [((Fraction(1, 2), -1), (2, 0), Fraction(3, 5)),
                                    ((0, 0), (0, 1), -2)])
    q = ExpPolynomial.from_terms(2, [((0, 2), (1, 1), 3), ((0, 0), (0, 0), 1)])
    r = ExpPolynomial.exponential((Fraction(-1, 2), 1), Fraction(2, 3))
    one, zero = ExpPolynomial.constant(1, 2), ExpPolynomial.constant(0, 2)
    assert p + q == q + p
    assert (p + q) + r == p + (q + r)
    assert p * q == q * p
    assert (p * q) * r == p * (q * r)
    assert p * (q + r) == p * q + p * r
    assert p + zero == p * one == p
    assert (p - p).is_zero()
    assert (p + -p).is_zero()
    assert (p * zero).is_zero()
    assert p.scale(Fraction(2, 3)) + p.scale(Fraction(1, 3)) == p
    assert p.scale(0).is_zero()


def test_frequency_and_power_addition_in_product() -> None:
    p = ExpPolynomial.from_terms(2, [((Fraction(1, 2), -2), (2, 1), Fraction(3, 5))])
    q = ExpPolynomial.from_terms(2, [((Fraction(2, 3), 2), (1, 3), Fraction(-5, 2))])
    assert (p * q).terms == (((Fraction(7, 6), Fraction(0)), (3, 4), Fraction(-3, 2)),)


@pytest.mark.parametrize("exponent", [0, 1, 2, 3, 7])
def test_powers_match_independent_binomial_coefficients(exponent: int) -> None:
    x = ExpPolynomial.monomial((1,))
    exponential = ExpPolynomial.exponential((Fraction(1, 3),))
    expected = ExpPolynomial.from_terms(1, [((Fraction(k, 3),), (exponent - k,), math.comb(exponent, k))
                                           for k in range(exponent + 1)])
    assert (x + exponential) ** exponent == expected
    assert ExpPolynomial.constant(0, 1) ** 0 == ExpPolynomial.constant(1, 1)
    assert (ExpPolynomial.constant(0, 1) ** 4).is_zero()


@pytest.mark.parametrize(("power", "order"), [(0, 0), (0, 6), (2, 1), (2, 2), (2, 6), (5, 4)])
@pytest.mark.parametrize("frequency", [Fraction(0), Fraction(-2, 3), Fraction(3, 2)])
def test_derivatives_match_independent_leibniz_coefficients(power: int, order: int, frequency: Fraction) -> None:
    coefficient = Fraction(3, 7)
    polynomial = ExpPolynomial.from_terms(2, [((frequency, Fraction(1, 3)), (power, 3), coefficient)])
    expected: list[TermInput] = []
    for j in range(min(order, power) + 1):
        falling = math.factorial(power) // math.factorial(power - j)
        value = coefficient * math.comb(order, j) * falling * frequency ** (order - j)
        expected.append(((frequency, Fraction(1, 3)), (power - j, 3), value))
    assert polynomial.differentiate(0, order) == ExpPolynomial.from_terms(2, expected)


def test_mixed_derivatives_commute_and_obey_product_rule() -> None:
    p = ExpPolynomial.from_terms(2, [((2, -1), (3, 2), Fraction(2, 3)), ((0, 0), (0, 4), 5)])
    q = ExpPolynomial.from_terms(2, [((1, 1), (1, 0), -2), ((0, 0), (2, 1), 3)])
    assert p.differentiate(0, 2).differentiate(1, 3) == p.differentiate(1, 3).differentiate(0, 2)
    assert (p * q).differentiate(0) == p.differentiate(0) * q + p * q.differentiate(0)
    assert (p + q).differentiate(1, 3) == p.differentiate(1, 3) + q.differentiate(1, 3)
    assert ExpPolynomial.monomial((4, 2)).differentiate(0, 5).is_zero()
    assert ExpPolynomial.constant(0, 2).differentiate(1, 128).is_zero()
    assert p.differentiate(1, 0) is p


def test_exact_heat_modes_and_spectral_parameter_derivative_mode() -> None:
    # Coordinates (t,x): exp(lambda²*t+lambda*x) and its lambda derivative
    # both satisfy the heat equation, including the polynomial multiplier.
    lam = Fraction(2, 3)
    carrier = ExpPolynomial.exponential((lam * lam, lam))
    multiplier = ExpPolynomial.monomial((1, 0), 2 * lam) + ExpPolynomial.monomial((0, 1))
    solution = carrier + multiplier * carrier
    assert (solution.differentiate(0) - solution.differentiate(1, 2)).is_zero()


def test_high_precision_values_match_analytic_mixed_derivative() -> None:
    mp = pytest.importorskip("mpmath")
    p, q, i, j = 4, 3, 3, 2
    a, b, coefficient = Fraction(2, 3), Fraction(-3, 5), Fraction(7, 11)
    expression = ExpPolynomial.from_terms(2, [((a, b), (p, q), coefficient)])
    derivative = expression.differentiate(0, i).differentiate(1, j)
    with mp.workdps(90):
        x, y = mp.mpf("0.37"), mp.mpf("-0.81")

        def number(value: Fraction) -> object:
            return mp.mpf(value.numerator) / value.denominator

        actual = sum(number(c) * x ** powers[0] * y ** powers[1] *
                     mp.exp(number(frequency[0]) * x + number(frequency[1]) * y)
                     for frequency, powers, c in derivative.terms)
        expected = mp.diff(lambda xx, yy: number(coefficient) * xx**p * yy**q *
                           mp.exp(number(a) * xx + number(b) * yy), (x, y), (i, j))
        assert mp.almosteq(actual, expected)


def test_known_global_sign_is_strict_and_only_sufficient() -> None:
    zero = ExpPolynomial.constant(0, 2)
    one = ExpPolynomial.constant(1, 2)
    square = ExpPolynomial.monomial((2, 0))
    exponential = ExpPolynomial.exponential((1, -2))
    assert zero.known_global_sign() == 0
    assert exponential.known_global_sign() == 1
    assert (-exponential).known_global_sign() == -1
    assert (one + square).known_global_sign() == 1
    assert (-one - square).known_global_sign() == -1
    assert (exponential + square).known_global_sign() == 1
    assert square.known_global_sign() is None  # It vanishes on x=0.
    assert (one - square).known_global_sign() is None
    assert (one + ExpPolynomial.monomial((1, 0))).known_global_sign() is None
    # This polynomial is strictly positive, but the sufficient coefficient
    # rule intentionally cannot prove it because it has a negative coefficient.
    assert (one + (square - one) ** 2).known_global_sign() is None


@pytest.mark.parametrize("value", [True, 0.5, "1", complex(1), None])
def test_inexact_or_malformed_coefficients_and_frequencies_refused(value: object) -> None:
    with pytest.raises(TypeError, match="int or Fraction"):
        ExpPolynomial.constant(cast(Rational, value), 1)
    with pytest.raises(TypeError, match="int or Fraction"):
        ExpPolynomial.exponential((cast(Rational, value),))
    with pytest.raises(TypeError, match="int or Fraction"):
        ExpPolynomial.constant(1, 1).scale(cast(Rational, value))


@pytest.mark.parametrize("dimension", [0, -1, True, 1.0, Fraction(1)])
def test_invalid_dimensions_refused(dimension: object) -> None:
    with pytest.raises((TypeError, ValueError), match="dimension"):
        ExpPolynomial.from_terms(cast(int, dimension), ())


@pytest.mark.parametrize("powers", [(-1,), (True,), (1.0,), (Fraction(1),)])
def test_invalid_powers_refused(powers: object) -> None:
    with pytest.raises((TypeError, ValueError), match="power"):
        ExpPolynomial.monomial(cast(Sequence[int], powers))


@pytest.mark.parametrize("terms", [[((0,), (0,))], [((0,), (0,), 1, 2)],
                                 [((0, 1), (0,), 1)], [((0,), (), 1)],
                                 [(None, (0,), 1)], [((0,), None, 1)], [1]])
def test_malformed_triples_and_shapes_refused(terms: object) -> None:
    with pytest.raises(ValueError):
        ExpPolynomial.from_terms(1, cast(Iterable[TermInput], terms))


@pytest.mark.parametrize("order", [-1, True, 1.0, Fraction(1)])
def test_invalid_derivative_orders_and_exponents_refused(order: object) -> None:
    expression = ExpPolynomial.exponential((1,))
    with pytest.raises((TypeError, ValueError)):
        expression.differentiate(0, cast(int, order))
    with pytest.raises((TypeError, ValueError)):
        expression ** cast(int, order)


@pytest.mark.parametrize("axis", [-1, 1, True, 0.0])
def test_invalid_derivative_axis_refused(axis: object) -> None:
    with pytest.raises((TypeError, ValueError), match="axis"):
        ExpPolynomial.constant(0, 1).differentiate(cast(int, axis), 0)


def test_binary_operations_require_compatible_polynomials() -> None:
    polynomial = ExpPolynomial.constant(1, 1)
    for other in (ExpPolynomial.constant(1, 2),
                  ExpPolynomial.constant(1, 1, budget=ExponentialBudget(max_terms=20))):
        with pytest.raises(ValueError, match="must match"):
            polynomial + other
        with pytest.raises(ValueError, match="must match"):
            polynomial - other
        with pytest.raises(ValueError, match="must match"):
            polynomial * other
    with pytest.raises(TypeError, match="ExpPolynomial"):
        polynomial + cast(ExpPolynomial, 1)
    with pytest.raises(TypeError, match="ExpPolynomial"):
        polynomial - cast(ExpPolynomial, 1)
    with pytest.raises(TypeError, match="ExpPolynomial"):
        polynomial * cast(ExpPolynomial, 1)


@pytest.mark.parametrize("name", ["max_terms", "max_work", "max_coefficient_bits", "max_dimension"])
@pytest.mark.parametrize("value", [0, -1, True, 1.0])
def test_invalid_positive_budget_fields_refused(name: str, value: object) -> None:
    with pytest.raises((TypeError, ValueError), match=name):
        ExponentialBudget(**{name: cast(int, value)})


@pytest.mark.parametrize("name", ["max_total_degree", "max_derivative_order"])
@pytest.mark.parametrize("value", [-1, True, 1.0])
def test_invalid_nonnegative_budget_fields_refused(name: str, value: object) -> None:
    with pytest.raises((TypeError, ValueError), match=name):
        ExponentialBudget(**{name: cast(int, value)})


def test_zero_degree_and_derivative_budgets_are_valid() -> None:
    budget = ExponentialBudget(max_total_degree=0, max_derivative_order=0)
    expression = ExpPolynomial.exponential((1,), budget=budget)
    assert expression.differentiate(0, 0) == expression
    with pytest.raises(ValueError, match="derivative order"):
        expression.differentiate(0)
    with pytest.raises(ValueError, match="total degree"):
        ExpPolynomial.monomial((1,), budget=budget)


def test_budget_is_frozen_and_type_checked() -> None:
    budget = ExponentialBudget()
    with pytest.raises(FrozenInstanceError):
        budget.max_terms = 1  # type: ignore[misc]
    with pytest.raises(TypeError, match="ExponentialBudget"):
        ExpPolynomial.constant(1, 1, budget=cast(ExponentialBudget, 1))


def test_sparse_support_budget_covers_construction_products_and_derivatives() -> None:
    budget = ExponentialBudget(max_terms=2)
    x = ExpPolynomial.monomial((1,), budget=budget)
    one = ExpPolynomial.constant(1, 1, budget=budget)
    with pytest.raises(ValueError, match="terms exceed"):
        ExpPolynomial.from_terms(1, [((0,), (i,), 1) for i in range(3)], budget=budget)
    with pytest.raises(ValueError, match="terms exceed"):
        (x + one) * (x + one)
    expression = ExpPolynomial.from_terms(1, [((1,), (2,), 1)], budget=budget)
    assert len(expression.differentiate(0).terms) == 2
    with pytest.raises(ValueError, match="terms exceed"):
        expression.differentiate(0, 2)
    # Cancellation is exact when bounded intermediate support suffices.
    assert ExpPolynomial.from_terms(1, [((0,), (0,), 1), ((0,), (0,), -1)], budget=budget).is_zero()


def test_degree_and_dimension_limits_preflight_growth() -> None:
    budget = ExponentialBudget(max_total_degree=3, max_dimension=2)
    x_squared = ExpPolynomial.monomial((2,), budget=budget)
    with pytest.raises(ValueError, match="total degree"):
        x_squared * x_squared
    with pytest.raises(ValueError, match="total degree"):
        x_squared ** 2
    with pytest.raises(ValueError, match="total degree"):
        ExpPolynomial.monomial((2, 2), budget=budget)
    with pytest.raises(ValueError, match="dimension"):
        ExpPolynomial.constant(0, 3, budget=budget)
    with pytest.raises(ValueError, match="work"):
        ExpPolynomial.constant(0, 10**9, budget=ExponentialBudget(max_dimension=10**9, max_work=1))


def test_bit_limits_cover_inputs_coefficient_growth_and_frequency_growth() -> None:
    budget = ExponentialBudget(max_coefficient_bits=2)
    terms: list[TermInput] = [((4,), (0,), 1), ((0,), (0,), 4), ((0,), (0,), Fraction(1, 4))]
    for term in terms:
        with pytest.raises(ValueError, match="bit budget"):
            ExpPolynomial.from_terms(1, [term], budget=budget)
    constant = ExpPolynomial.constant(3, 1, budget=budget)
    with pytest.raises(ValueError, match="bit budget"):
        constant + constant
    with pytest.raises(ValueError, match="bit budget"):
        constant.scale(2)
    exponential = ExpPolynomial.exponential((3,), budget=budget)
    with pytest.raises(ValueError, match="bit budget"):
        exponential * exponential
    with pytest.raises(ValueError, match="bit budget"):
        ExpPolynomial.monomial((3,), 3, budget=budget).differentiate(0)


def test_cancelled_intermediate_coefficients_still_respect_bit_limits() -> None:
    budget = ExponentialBudget(max_coefficient_bits=2)
    with pytest.raises(ValueError, match="bit budget"):
        ExpPolynomial.from_terms(1, [((0,), (0,), 3), ((0,), (0,), 3), ((0,), (0,), -3)], budget=budget)


def test_power_work_is_cumulative_across_substeps() -> None:
    budget = ExponentialBudget(max_work=20)
    exponential = ExpPolynomial.exponential((1,), budget=budget)
    square = exponential * exponential
    assert square * square == ExpPolynomial.exponential((4,), budget=budget)
    with pytest.raises(ValueError, match="work"):
        exponential ** 4


def test_derivative_work_is_cumulative_across_substeps() -> None:
    budget = ExponentialBudget(max_work=20)
    exponential = ExpPolynomial.exponential((1,), budget=budget)
    assert exponential.differentiate(0) == exponential
    with pytest.raises(ValueError, match="work"):
        exponential.differentiate(0, 5)


def test_infinite_zero_term_stream_cannot_bypass_work_budget() -> None:
    with pytest.raises(ValueError, match="work"):
        ExpPolynomial.from_terms(1, repeat(((0,), (0,), 0)), budget=ExponentialBudget(max_work=20))


def test_product_work_preflight_and_huge_power_refusal() -> None:
    budget = ExponentialBudget(max_work=100)
    expression = ExpPolynomial.from_terms(1, [((i,), (0,), 1) for i in range(4)], budget=budget)
    with pytest.raises(ValueError, match="work"):
        expression * expression
    with pytest.raises(ValueError, match="work"):
        ExpPolynomial.constant(1, 1, budget=replace(budget, max_work=20)) ** (1 << 1000)
