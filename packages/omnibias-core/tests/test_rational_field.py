# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Exact quotient identities, regular domains and finite-library nullspaces."""

from __future__ import annotations

import math
from dataclasses import FrozenInstanceError, replace
from fractions import Fraction
from typing import Any, cast

import omnibias.core.rational_field as rational_field_module
import pytest
from omnibias.core.exp_polynomial import ExponentialBudget, ExpPolynomial
from omnibias.core.rational_field import RationalField, discover_rational_relations
from omnibias.core.ridge_system import RidgeBudget


def _constant(value: int | Fraction, dimension: int = 1, *, budget: ExponentialBudget | None = None) -> ExpPolynomial:
    return ExpPolynomial.constant(value, dimension, budget=budget)


def test_constructor_and_embedding_are_immutable_exact_metadata() -> None:
    one, x = _constant(1), ExpPolynomial.monomial((1,))
    field = RationalField(one, x)
    assert field.dimension == 1
    assert field.budget == ExponentialBudget()
    assert field.numerator == one
    assert field.denominator_factors == ((x, 1),)
    assert field.nonzero_conditions == (x,)
    assert field.denominator == x
    assert hash(field) == hash(RationalField(one, x))
    assert RationalField.from_polynomial(x) == RationalField(x)
    assert RationalField(x).denominator == one
    assert RationalField(x).nonzero_conditions == ()
    with pytest.raises(FrozenInstanceError):
        field.numerator = x  # type: ignore[misc]


@pytest.mark.parametrize("operation", ["subtract", "differentiate", "scale", "multiply"])
def test_cancelled_zero_retains_original_nonzero_condition(operation: str) -> None:
    one, x = _constant(1), ExpPolynomial.monomial((1,))
    field = RationalField(x, x)
    if operation == "subtract":
        result = field - RationalField(one)
    elif operation == "differentiate":
        result = field.differentiate(0)
    elif operation == "scale":
        result = field.scale(0)
    else:
        result = field * RationalField(_constant(0))
    assert result.is_zero()
    assert result.denominator_factors == ()
    assert result.denominator == one
    assert result.nonzero_conditions == (x,)


def test_zero_power_preserves_every_original_pole() -> None:
    one, x = _constant(1), ExpPolynomial.monomial((1,))
    field = RationalField(one, x) * RationalField(one, x + one)
    for result in (field ** 0, (field - field) ** 0):
        assert result.numerator == one
        assert result.denominator_factors == ()
        assert result.nonzero_conditions == field.nonzero_conditions


def test_zero_numerator_direct_construction_still_validates_and_retains_denominator() -> None:
    zero, x = _constant(0), ExpPolynomial.monomial((1,))
    field = RationalField(zero, x)
    assert field.is_zero()
    assert field.nonzero_conditions == (x,)
    assert field.differentiate(0, 5).nonzero_conditions == (x,)
    for numerator in (zero, _constant(1)):
        with pytest.raises(ZeroDivisionError, match="identically-zero"):
            RationalField(numerator, zero)


def test_factor_merging_uses_structural_maxima_in_addition_and_sums_in_products() -> None:
    one, x = _constant(1), ExpPolynomial.monomial((1,))
    x_copy = ExpPolynomial.from_terms(1, x.terms)
    first, second = RationalField(one, x), RationalField(one, x_copy)
    square = first * second
    assert square.denominator_factors == ((x, 2),)
    assert square.nonzero_conditions == (x,)
    summed = first + square
    assert summed.denominator_factors == ((x, 2),)
    assert summed.numerator == x + one
    assert summed.denominator == x ** 2
    # No algebraic GCD is promised for distinct, proportional factors.
    independent = first * RationalField(one, x.scale(2))
    assert set(independent.denominator_factors) == {(x, 1), (x.scale(2), 1)}


def test_guard_and_factor_order_is_canonical_across_operand_order() -> None:
    one, x = _constant(1), ExpPolynomial.monomial((1,))
    a, b = RationalField(one, x), RationalField(one, x + one)
    assert a * b == b * a
    assert a + b == b + a
    assert (a - a) * b == b * (a - a)


def test_division_and_reciprocal_retain_old_domains_and_add_numerator_guard() -> None:
    one, x = _constant(1), ExpPolynomial.monomial((1,))
    p, q = x + one, x - one
    field = RationalField(p, q)
    reciprocal = field ** -1
    assert set(reciprocal.nonzero_conditions) == {p, q}
    assert reciprocal.denominator_factors == ((p, 1),)
    assert reciprocal.numerator == q
    assert (reciprocal - RationalField(q, p)).is_zero()
    assert (field / field - RationalField(one)).is_zero()
    assert set((field / field).nonzero_conditions) == {p, q}
    for power in (-3, -1, 0, 1, 4):
        expected = RationalField(p ** power, q ** power) if power >= 0 else RationalField(q ** -power, p ** -power)
        assert (field ** power - expected).is_zero()


def test_zero_divisor_refused_even_with_zero_dividend_and_cancelled_guards() -> None:
    x = ExpPolynomial.monomial((1,))
    original = RationalField(_constant(1), x)
    zero = original - original
    for dividend in (zero, original):
        with pytest.raises(ZeroDivisionError, match="identically-zero"):
            dividend / zero
    with pytest.raises(ZeroDivisionError, match="identically-zero"):
        zero ** -1


@pytest.mark.parametrize("order", [0, 1, 2, 3, 4, 6])
def test_raw_reciprocal_derivatives_match_factorial_formula(order: int) -> None:
    x = ExpPolynomial.monomial((1,))
    field = RationalField(_constant(1), x)
    result = field.differentiate(0, order)
    assert result.numerator == _constant((-1) ** order * math.factorial(order))
    assert result.denominator_factors == ((x, order + 1),)
    assert result.nonzero_conditions == (x,)
    if order == 0:
        assert result is field


def test_mixed_derivatives_increment_only_axis_dependent_factors() -> None:
    one = _constant(1, 2)
    x, y = ExpPolynomial.monomial((1, 0)), ExpPolynomial.monomial((0, 1))
    field = RationalField(one, x) * RationalField(one, y)
    dx = field.differentiate(0)
    assert dx.numerator == one.scale(-1)
    assert dict(dx.denominator_factors) == {x: 2, y: 1}
    dxy = dx.differentiate(1)
    assert dxy.numerator == one
    assert dict(dxy.denominator_factors) == {x: 2, y: 2}
    assert dxy == field.differentiate(1).differentiate(0)
    assert set(dxy.nonzero_conditions) == {x, y}


def test_factored_derivative_avoids_expanding_high_degree_denominator() -> None:
    denominator = ExpPolynomial.monomial((80,))
    result = RationalField(_constant(1), denominator).differentiate(0)
    assert result.numerator == ExpPolynomial.monomial((79,), -80)
    assert result.denominator_factors == ((denominator, 2),)
    with pytest.raises(ValueError, match="degree"):
        _ = result.denominator


def test_logistic_differential_polynomial_identity_is_exact() -> None:
    exponential = ExpPolynomial.exponential((1,))
    field = RationalField(exponential, _constant(1) + exponential)
    derivative = field.differentiate(0)
    assert (derivative - field + field ** 2).is_zero()
    discovered = discover_rational_relations([[derivative, field, field ** 2, RationalField(_constant(1))]])
    assert discovered.rank == 3
    assert discovered.relations == ((Fraction(1), Fraction(-1), Fraction(1), Fraction(0)),)
    assert discovered.nonzero_conditions == (field.nonzero_conditions,)


def test_general_quotient_derivatives_against_independent_high_precision_oracle() -> None:
    mp = pytest.importorskip("mpmath")
    numerator = ExpPolynomial.from_terms(2, [((Fraction(1, 3), -1), (2, 0), Fraction(2, 5)),
                                             ((0, 0), (0, 1), 1)])
    denominator = _constant(3, 2) + ExpPolynomial.exponential((1, Fraction(1, 2))) + ExpPolynomial.monomial((0, 2))
    field = RationalField(numerator, denominator)

    def rational(value: Fraction) -> Any:
        return mp.mpf(value.numerator) / value.denominator

    def evaluate_polynomial(poly: ExpPolynomial, x: Any, y: Any) -> Any:
        return mp.fsum(rational(c) * mp.exp(rational(f[0]) * x + rational(f[1]) * y)
                       * x ** powers[0] * y ** powers[1] for f, powers, c in poly.terms)

    def evaluate(result: RationalField, x: Any, y: Any) -> Any:
        return evaluate_polynomial(result.numerator, x, y) / mp.fprod(
            evaluate_polynomial(poly, x, y) ** order for poly, order in result.denominator_factors)

    with mp.workdps(75):
        x, y = mp.mpf("0.4"), mp.mpf("-0.7")

        def analytic(a: Any, b: Any) -> Any:
            return (mp.mpf(2) / 5 * a * a * mp.exp(a / 3 - b) + b) / (3 + mp.exp(a + b / 2) + b * b)

        for dx, dy in ((0, 0), (1, 0), (0, 2), (1, 1), (2, 1)):
            derivative = field.differentiate(0, dx).differentiate(1, dy)
            assert mp.almosteq(evaluate(derivative, x, y), mp.diff(analytic, (x, y), (dx, dy)),
                               rel_eps=mp.mpf("1e-65"), abs_eps=mp.mpf("1e-65"))


def test_multiple_families_remove_accidental_single_family_relations() -> None:
    one = RationalField(_constant(1))
    single = discover_rational_relations([[one, one.scale(2)]])
    assert single.relations == ((Fraction(-2), Fraction(1)),)
    assert single.rank == 1
    assert single.candidate_count == 2
    assert single.constraint_count == 1
    both = discover_rational_relations([[one, one.scale(2)], [one, one.scale(3)]])
    assert both.relations == ()
    assert both.rank == 2
    assert both.constraint_count == 2


def test_discovery_common_denominator_clearing_and_exact_frequency_distinction() -> None:
    x = ExpPolynomial.monomial((1,))
    reciprocal = RationalField(_constant(1), x)
    result = discover_rational_relations([[reciprocal, reciprocal.scale(2), RationalField(_constant(1))]])
    assert result.relations == ((Fraction(-2), Fraction(1), Fraction(0)),)
    assert result.rank == result.constraint_count == 2
    a = RationalField(ExpPolynomial.exponential((Fraction(1, 10**40),)))
    b = RationalField(ExpPolynomial.exponential((Fraction(1, 10**40) + 1,)))
    assert discover_rational_relations([[a, b]]).relations == ()


def test_discovery_preserves_guards_of_cancelled_features_for_every_family() -> None:
    x, one = ExpPolynomial.monomial((1,)), _constant(1)
    a, b = RationalField(one, x), RationalField(one, x + one)
    result = discover_rational_relations([[a - a, RationalField(one)], [b - b, RationalField(one)]])
    assert result.relations == ((Fraction(1), Fraction(0)),)
    assert result.nonzero_conditions == ((x,), (x + one,))
    with pytest.raises(FrozenInstanceError):
        result.rank = 3  # type: ignore[misc]


def test_all_zero_candidates_have_full_exact_nullspace_and_no_constraints() -> None:
    zero = RationalField(_constant(0))
    result = discover_rational_relations([[zero, zero]])
    assert result.rank == result.constraint_count == 0
    assert result.candidate_count == 2
    assert result.relations == ((Fraction(1), Fraction(0)), (Fraction(0), Fraction(1)))
    assert result.nonzero_conditions == ((),)


@pytest.mark.parametrize("bad", [True, 0.5, "1", None])
def test_nonexact_scalar_and_order_types_are_rejected(bad: object) -> None:
    field = RationalField(_constant(1))
    with pytest.raises(TypeError):
        field.scale(cast(int, bad))
    with pytest.raises(TypeError):
        field ** cast(int, bad)
    with pytest.raises(TypeError):
        field.differentiate(cast(int, bad))
    with pytest.raises(TypeError):
        field.differentiate(0, cast(int, bad))


@pytest.mark.parametrize("axis,order", [(-1, 0), (1, 0), (0, -1), (0, 129)])
def test_invalid_derivative_indices_and_order_budget_are_rejected(axis: int, order: int) -> None:
    with pytest.raises(ValueError):
        RationalField(_constant(1)).differentiate(axis, order)


@pytest.mark.parametrize("bad", [1, 1.5, Fraction(1, 2), None])
def test_nonpolynomial_constructor_data_and_nonfield_binary_operands_refused(bad: object) -> None:
    field = RationalField(_constant(1))
    with pytest.raises(TypeError):
        RationalField(cast(ExpPolynomial, bad))
    if bad is not None:
        with pytest.raises(TypeError):
            RationalField(_constant(1), cast(ExpPolynomial, bad))
    with pytest.raises(TypeError):
        field + cast(RationalField, bad)
    with pytest.raises(TypeError):
        field - cast(RationalField, bad)
    with pytest.raises(TypeError):
        field * cast(RationalField, bad)
    with pytest.raises(TypeError):
        field / cast(RationalField, bad)


def test_dimensions_and_polynomial_budgets_must_agree() -> None:
    one = _constant(1)
    different_dimension = _constant(1, 2)
    different_budget = _constant(1, budget=replace(ExponentialBudget(), max_terms=99))
    for polynomial in (different_dimension, different_budget):
        with pytest.raises(ValueError):
            RationalField(one, polynomial)
        with pytest.raises(ValueError):
            RationalField(one) + RationalField(polynomial)
        with pytest.raises(ValueError):
            discover_rational_relations([[RationalField(one)], [RationalField(polynomial)]])


def test_empty_ragged_nonfield_families_and_invalid_discovery_budget_refused() -> None:
    one = RationalField(_constant(1))
    for families in ([], [[]], [[one], []], [[one], [one, one]]):
        with pytest.raises(ValueError):
            discover_rational_relations(families)
    with pytest.raises(TypeError):
        discover_rational_relations([[cast(RationalField, 1)]])
    with pytest.raises(TypeError):
        discover_rational_relations([[one, cast(RationalField, 1)]])
    with pytest.raises(TypeError):
        discover_rational_relations([[one]], budget=cast(RidgeBudget, ExponentialBudget()))


def test_derivative_work_budget_is_cumulative_across_substeps() -> None:
    budget = replace(ExponentialBudget(), max_work=400)
    field = RationalField(_constant(1, budget=budget), ExpPolynomial.monomial((1,), budget=budget))
    # Each individual call fits; the requested two-step operation does not.
    assert field.differentiate(0).differentiate(0).numerator == _constant(2, budget=budget)
    with pytest.raises(ValueError, match="work"):
        field.differentiate(0, 2)


def test_discovery_meters_all_families_before_elimination() -> None:
    one = RationalField(_constant(1))
    limits = replace(RidgeBudget(), max_elimination_work=200)
    assert discover_rational_relations([[one, one.scale(2)]], budget=limits).rank == 1
    with pytest.raises(ValueError, match="work"):
        discover_rational_relations([[one, one.scale(2)]] * 2, budget=limits)


def test_discovery_also_honors_aggregate_polynomial_budget() -> None:
    budget = replace(ExponentialBudget(), max_work=200)
    one = RationalField(_constant(1, budget=budget))
    assert discover_rational_relations([[one, one.scale(2)]]).rank == 1
    with pytest.raises(ValueError, match="work"):
        discover_rational_relations([[one, one.scale(2)]] * 3)


def test_discovery_refuses_storage_and_exact_coefficient_growth() -> None:
    one = RationalField(_constant(1))
    with pytest.raises(ValueError, match="matrix"):
        discover_rational_relations([[one, one]], budget=replace(RidgeBudget(), max_matrix_entries=2))
    with pytest.raises(ValueError, match="coefficient"):
        discover_rational_relations([[one, one.scale(16)]], budget=replace(RidgeBudget(), max_coefficient_bits=4))
    field = RationalField(ExpPolynomial.exponential((Fraction(1, 17),)))
    with pytest.raises(ValueError, match="coefficient"):
        discover_rational_relations([[field]], budget=replace(RidgeBudget(), max_coefficient_bits=4))


def test_discovery_preflights_dimensionful_unit_storage_even_for_zero_features(monkeypatch: pytest.MonkeyPatch) -> None:
    zero = RationalField(_constant(0, 32))

    def forbidden(*args: object, **kwargs: object) -> ExpPolynomial:
        raise AssertionError("a unit polynomial was allocated before storage preflight")

    monkeypatch.setattr(ExpPolynomial, "constant", forbidden)
    with pytest.raises(ValueError, match="matrix"):
        discover_rational_relations([[zero]], budget=replace(RidgeBudget(), max_matrix_entries=2))


def test_discovery_refuses_exhausted_union_sort_work_before_sorting(monkeypatch: pytest.MonkeyPatch) -> None:
    one = RationalField(_constant(1))

    def checked_sorted(values: Any) -> list[Any]:
        if isinstance(values, set):
            raise AssertionError("the coefficient-key union was sorted before its work preflight")
        return sorted(values)

    monkeypatch.setattr(rational_field_module, "sorted", checked_sorted, raising=False)
    with pytest.raises(ValueError, match="work"):
        discover_rational_relations([[one, one.scale(2)]], budget=replace(RidgeBudget(), max_elimination_work=90))
