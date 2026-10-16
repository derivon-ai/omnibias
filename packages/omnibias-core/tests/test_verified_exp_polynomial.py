# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Independent enclosure, denominator-domain, and backend-state regressions."""

from fractions import Fraction
from math import inf

import pytest
from omnibias.core.exp_polynomial import ExponentialBudget, ExpPolynomial
from omnibias.core.rational_field import RationalField
from omnibias.core.verified import transcend
from omnibias.core.verified.exp_polynomial import (
    evaluate_exp_polynomial,
    evaluate_rational_field,
)
from omnibias.core.verified.interval import Interval


def test_polynomial_without_optional_backend_encloses_exact_rational_values(monkeypatch) -> None:
    monkeypatch.setattr(transcend, "_mpmath", lambda: None)
    polynomial = ExpPolynomial.from_terms(2, [((0, 0), (2, 1), Fraction(1, 3)),
                                             ((0, 0), (0, 0), Fraction(1, 7))])
    box = (Interval(-0.5, 1.0), Interval(-0.25, 0.5))
    enclosure = evaluate_exp_polynomial(polynomial, box)
    for x in (Fraction(-1, 2), Fraction(0), Fraction(1)):
        for y in (Fraction(-1, 4), Fraction(1, 4), Fraction(1, 2)):
            exact = x*x*y/3 + Fraction(1, 7)
            assert enclosure.lo <= exact <= enclosure.hi


def test_nonzero_exponential_requires_rigorous_backend(monkeypatch) -> None:
    monkeypatch.setattr(transcend, "_mpmath", lambda: None)
    with pytest.raises(RuntimeError, match="rigorous transcendental backend"):
        evaluate_exp_polynomial(ExpPolynomial.exponential((1,)), (Interval(0.0, 1.0),))


def test_rational_frequencies_and_polynomial_factors_high_precision() -> None:
    mp = pytest.importorskip("mpmath")
    polynomial = ExpPolynomial.from_terms(2, [
        ((Fraction(1, 3), Fraction(-2, 7)), (2, 1), Fraction(2, 5)),
        ((Fraction(-3, 4), 0), (0, 0), Fraction(-1, 11)),
    ])
    box = (Interval(-0.5, 0.75), Interval(-0.25, 0.5))
    interval = evaluate_exp_polynomial(polynomial, box)
    with mp.workdps(80):
        for i in range(9):
            x = -mp.mpf("0.5") + mp.mpf("1.25") * i / 8
            for j in range(9):
                y = -mp.mpf("0.25") + mp.mpf("0.75") * j / 8
                exact = mp.mpf(2)/5*x*x*y*mp.exp(x/3-2*y/7) - mp.exp(-3*x/4)/11
                assert mp.mpf(interval.lo) <= exact <= mp.mpf(interval.hi)


@pytest.mark.parametrize("operation", ["difference", "zero_scale", "power_zero", "ratio"])
def test_cancellation_does_not_erase_original_domain(operation) -> None:
    x = ExpPolynomial.monomial((1,))
    one = ExpPolynomial.constant(1, 1)
    original = RationalField(one, x)
    if operation == "difference":
        result = original - original
    elif operation == "zero_scale":
        result = original.scale(0)
    elif operation == "power_zero":
        result = original**0
    else:
        result = original / original
    with pytest.raises(ZeroDivisionError, match="nonzero condition"):
        evaluate_rational_field(result, (Interval(-1.0, 1.0),))
    enclosed = evaluate_rational_field(result, (Interval(1.0, 2.0),))
    assert enclosed.contains(0.0 if operation in ("difference", "zero_scale") else 1.0)


def test_unknown_sign_is_inconclusive_even_when_denominator_is_positive() -> None:
    # x²-x+1 is everywhere positive, but the sufficient coefficient sign rule
    # and the natural interval form on [0,2] cannot establish that fact.
    x = ExpPolynomial.monomial((1,))
    one = ExpPolynomial.constant(1, 1)
    denominator = x*x - x + one
    assert denominator.known_global_sign() is None
    result = RationalField(one, denominator)
    with pytest.raises(ZeroDivisionError, match="nonzero condition"):
        evaluate_rational_field(result, (Interval(0.0, 2.0),))
    assert evaluate_rational_field(result, (Interval.point(0.5),)).contains(float(Fraction(4, 3)))


def test_zero_identity_with_positive_guard_needs_no_transcendental_backend(monkeypatch) -> None:
    monkeypatch.setattr(transcend, "_mpmath", lambda: None)
    tau = ExpPolynomial.constant(1, 1) + ExpPolynomial.exponential((1,))
    original = RationalField(tau.differentiate(0), tau)
    zero = original - original
    assert zero.nonzero_conditions
    assert evaluate_rational_field(zero, (Interval(-1e300, 1e300),)) == Interval.point(0.0)


@pytest.mark.parametrize("prior", [False, True])
@pytest.mark.parametrize("fails", [False, True])
def test_backend_state_and_prior_provenance_survive_evaluation(monkeypatch, prior, fails) -> None:
    pytest.importorskip("mpmath")
    monkeypatch.setattr(transcend, "_LIBM_FALLBACK_USED", True)
    monkeypatch.setattr(transcend, "_STRICT_BACKEND", prior)
    polynomial = ExpPolynomial.exponential((1,))
    if fails:
        with pytest.raises(ArithmeticError, match="nonfinite"):
            evaluate_exp_polynomial(polynomial, (Interval.point(1000.0),))
    else:
        evaluate_exp_polynomial(polynomial, (Interval.point(0.0),))
    assert transcend.strict_backend() is prior
    assert transcend.libm_fallback_used()


def test_polynomial_overflow_is_explicit_refusal() -> None:
    polynomial = ExpPolynomial.monomial((2,))
    with pytest.raises(ArithmeticError, match="nonfinite"):
        evaluate_exp_polynomial(polynomial, (Interval(1e307, 1e308),))


@pytest.mark.parametrize("endpoint", [Fraction(1, 10), True, 2**53 + 1, inf])
def test_invalid_endpoint_semantics_are_refused(endpoint) -> None:
    polynomial = ExpPolynomial.constant(0, 1)
    with pytest.raises((TypeError, ValueError), match="endpoint"):
        evaluate_exp_polynomial(polynomial, (Interval(endpoint, endpoint),))


def test_invalid_dimensions_and_input_types_are_refused() -> None:
    polynomial = ExpPolynomial.constant(1, 1)
    with pytest.raises(ValueError, match="dimension"):
        evaluate_exp_polynomial(polynomial, ())
    with pytest.raises(TypeError, match="Interval objects"):
        evaluate_exp_polynomial(polynomial, ((0.0, 1.0),))
    with pytest.raises(TypeError, match="ExpPolynomial"):
        evaluate_exp_polynomial(1, (Interval(0.0, 1.0),))
    with pytest.raises(TypeError, match="RationalField"):
        evaluate_rational_field(polynomial, (Interval(0.0, 1.0),))


def test_dimension_refusal_precedes_sequence_materialization() -> None:
    class Oversized:
        def __len__(self):
            return 10**12

        def __getitem__(self, index):
            raise AssertionError("invalid box should not be traversed")

    with pytest.raises(ValueError, match="dimension"):
        evaluate_exp_polynomial(ExpPolynomial.constant(0, 1), Oversized())


def test_interval_work_budget_is_enforced_across_power_arithmetic() -> None:
    # Exact metadata is compact; evaluating its very high degree still has to
    # account for every interval multiplication and input-coordinate visit.
    budget = ExponentialBudget(max_work=10, max_total_degree=1_000_000)
    polynomial = ExpPolynomial.monomial((1_000_000,), budget=budget)
    with pytest.raises(ValueError, match="work budget"):
        evaluate_exp_polynomial(polynomial, (Interval.point(1.0),))
