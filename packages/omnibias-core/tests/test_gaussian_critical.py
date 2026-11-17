# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Independent critical-branch composition and bounded exact arithmetic checks."""

from dataclasses import FrozenInstanceError
from fractions import Fraction as F
from importlib import import_module
from math import factorial

import pytest
from omnibias.core.gaussian_critical import (
    GaussianCriticalSeries,
    _build_critical_series,
    gaussian_critical_series,
)
from omnibias.core.gaussian_jet import _Arithmetic
from omnibias.core.ridge_system import RidgeBudget
from omnibias.core.verified import transcend


def _multiply(left, right, order):
    result = [F(0)]*(order+1)
    for i, a in enumerate(left):
        for j, b in enumerate(right):
            if i+j <= order:
                result[i+j] += a*b
    return result


def _original_equation_residual(coefficients, order):
    # Independently form cosh(chi) and sinh(chi) from factorial powers in t.
    # No z-series, desingularized H, or production convolution is reused.
    cosh, sinh = [F(0)]*(order+1), [F(0)]*(order+1)
    power = [F(1)]+[F(0)]*order
    for n in range(order+1):
        destination = cosh if n % 2 == 0 else sinh
        for k, value in enumerate(power):
            destination[k] += value/F(factorial(n))
        power = _multiply(power, coefficients, order)
    product = _multiply(coefficients, cosh, order)
    return tuple(3*(a-b)-(cosh[k-3] if k >= 3 else 0)
                 for k, (a, b) in enumerate(zip(product, sinh, strict=True)))


def _mp(mp, value):
    value = F(value)
    return mp.mpf(value.numerator)/value.denominator


def test_known_universal_coefficients_through_nine():
    result = gaussian_critical_series(9)
    assert result.coefficients == (0, 1, 0, F(2, 15), 0, F(3, 175), 0, F(2, 1575), 0, F(-16, 202125))
    assert result.z_coefficients == (1, F(2, 15), F(3, 175), F(2, 1575), F(-16, 202125))
    assert result.order == 9
    assert all(isinstance(value, F) for value in result.coefficients+result.z_coefficients)


@pytest.mark.parametrize("order", [0, 1, 2, 3, 4, 8, 9, 10])
def test_zero_and_even_orders_have_exact_lengths_and_preserve_prefixes(order):
    result = gaussian_critical_series(order)
    reference = gaussian_critical_series(11)
    assert result.order == order and len(result.coefficients) == order+1
    assert result.coefficients == reference.coefficients[:order+1]
    assert result.z_coefficients == reference.z_coefficients[:(order+1)//2]
    assert len(result.z_coefficients) == (order+1)//2
    assert all(result.coefficients[k] == 0 for k in range(0, order+1, 2))
    if order == 0:
        assert result.coefficients == (0,) and result.z_coefficients == ()


@pytest.mark.parametrize("order", [1, 3, 5, 9, 15, 20])
def test_original_sinh_cosh_equation_vanishes_to_the_implied_order(order):
    result = gaussian_critical_series(order)
    highest_odd = 2*((order-1)//2)+1
    residual = _original_equation_residual(result.coefficients, highest_odd+2)
    assert all(value == 0 for value in residual)
    # The finite polynomial is not an exact solution of the original equation.
    later = _original_equation_residual(result.coefficients, highest_odd+4)
    assert any(later)


def test_independent_high_precision_critical_inverse_root_and_error_order():
    mp = pytest.importorskip("mpmath")
    result = gaussian_critical_series(9)
    with mp.workdps(90):
        errors = []
        for t in (mp.mpf("0.1"), mp.mpf("0.05")):
            root = mp.findroot(lambda y, t=t: y-mp.tanh(y)-t**3/3, (t, t*(1+t*t)))
            polynomial = sum(_mp(mp, coefficient)*t**n for n, coefficient in enumerate(result.coefficients))
            errors.append(abs(root-polynomial))
            assert abs(root-t) > mp.mpf("0.1")*t**3
            assert abs(root-mp.tanh(root)-t**3/3) < mp.mpf("1e-85")
        assert 1900 < errors[0]/errors[1] < 2200  # First omitted odd term is t**11.
        assert errors[0] < mp.mpf("1e-12")
        negative = mp.mpf("-0.15")
        root = mp.findroot(lambda y: y-mp.tanh(y)-negative**3/3, (negative, negative*(1+negative**2)))
        polynomial = sum(_mp(mp, coefficient)*negative**n for n, coefficient in enumerate(result.coefficients))
        assert root < 0 and abs(root-polynomial) < mp.mpf("1e-10")


def test_even_order_has_the_same_recurrence_work_and_explicit_trailing_zero(monkeypatch):
    observed = []

    def record(self, count):
        observed.append(count)
        assert count <= self.max_elimination_work

    monkeypatch.setattr(RidgeBudget, "_work", record)
    odd = gaussian_critical_series(31)
    odd_work = observed[-1]-observed[0]
    observed.clear()
    even = gaussian_critical_series(32)
    even_work = observed[-1]-observed[0]
    assert even.coefficients == odd.coefficients+(F(0),)
    assert even.z_coefficients == odd.z_coefficients
    assert odd_work == even_work
    assert odd_work <= 3*15**2+10*15


def test_streaming_recurrence_operation_growth_is_quadratic(monkeypatch):
    observed = []

    def record(self, count):
        observed.append(count)
        assert count <= self.max_elimination_work

    monkeypatch.setattr(RidgeBudget, "_work", record)
    for highest in (4, 8, 16, 32):
        observed.clear()
        result = gaussian_critical_series(2*highest+1)
        assert len(result.z_coefficients) == highest+1
        actual = observed[-1]-observed[0]
        assert actual <= 3*highest**2+10*highest
    assert gaussian_critical_series(11).coefficients[-1] == F(-362, 9384375)


def test_factory_only_frozen_record_and_exact_evaluation():
    limits = RidgeBudget(max_order=9)
    result = gaussian_critical_series(9, budget=limits)
    assert result.budget is limits
    with pytest.raises(TypeError):
        GaussianCriticalSeries()
    with pytest.raises(FrozenInstanceError):
        result.coefficients = ()
    for t in (F(0), F(1, 3), F(-2, 7)):
        expected = sum(coefficient*t**n for n, coefficient in enumerate(result.coefficients))
        assert result.evaluate(t) == expected
        assert result.evaluate(-t) == -expected
        assert t*sum(value*(t*t)**k for k, value in enumerate(result.z_coefficients)) == expected
    assert gaussian_critical_series(0).evaluate(F(19, 2)) == 0


@pytest.mark.parametrize("order,error", [(-1, ValueError), (True, TypeError), (False, TypeError),
    (1.0, TypeError), (F(1), TypeError), (None, TypeError), ("3", TypeError), (complex(1), TypeError)])
def test_order_contracts(order, error):
    with pytest.raises(error):
        gaussian_critical_series(order)


@pytest.mark.parametrize("budget", [0, {}, True, "budget"])
def test_budget_type_contract(budget):
    with pytest.raises(TypeError):
        gaussian_critical_series(1, budget=budget)


@pytest.mark.parametrize("parameter", [True, 0.0, complex(0), None])
def test_evaluation_requires_exact_rational_data_even_for_zero_order(parameter):
    with pytest.raises(TypeError):
        gaussian_critical_series(0).evaluate(parameter)


def test_no_factorization_singular_jacobian_solve_or_transcendental(monkeypatch):
    def fail(*args, **kwargs):
        pytest.fail("universal critical coefficients attempted a solve or transcendental")

    for name, functions in (("omnibias.core._rational_linear", ("factor_positive_definite", "solve_affine")),
                            ("omnibias.core.verified.inverse_gaussian_heat", ("inverse_gaussian_heat", "certify_gaussian_inverse"))):
        module = import_module(name)
        for function in functions:
            monkeypatch.setattr(module, function, fail)
    monkeypatch.setattr(transcend, "exp_iv", fail)
    monkeypatch.setattr(transcend, "_mpmath", lambda: None)
    monkeypatch.setattr(transcend, "_STRICT_BACKEND", True)
    monkeypatch.setattr(transcend, "_LIBM_FALLBACK_USED", True)
    result = gaussian_critical_series(9)
    assert result.coefficients[9] == F(-16, 202125)
    assert transcend.strict_backend() and transcend.libm_fallback_used()


@pytest.mark.parametrize("limits", [RidgeBudget(max_order=8), RidgeBudget(max_matrix_entries=1), RidgeBudget(max_elimination_work=1)])
def test_resource_preflight_precedes_coefficient_recurrence(limits, monkeypatch):
    monkeypatch.setattr(_Arithmetic, "mul", lambda *_: pytest.fail("preflight missed recurrence"))
    with pytest.raises(ValueError):
        gaussian_critical_series(9, budget=limits)


def test_recurrence_and_evaluation_intermediates_obey_bit_budgets():
    small = gaussian_critical_series(3, budget=RidgeBudget(max_coefficient_bits=4))
    assert small.coefficients[-1] == F(2, 15)
    with pytest.raises(ValueError, match="bit"):
        gaussian_critical_series(5, budget=RidgeBudget(max_coefficient_bits=4))
    with pytest.raises(ValueError, match="bit"):
        small.evaluate(2**10)
    with pytest.raises(ValueError, match="bit"):
        small.evaluate(F(3))


def test_shared_builder_keeps_a_precharged_meter_and_checks_budget_identity():
    limits = RidgeBudget(max_elimination_work=2000)
    arithmetic = _Arithmetic(limits)
    arithmetic._work(1999)
    with pytest.raises(ValueError, match="work"):
        _build_critical_series(9, limits, arithmetic)
    assert arithmetic.used > limits.max_elimination_work
    with pytest.raises(ValueError, match="agree"):
        _build_critical_series(1, RidgeBudget(max_elimination_work=3000), _Arithmetic(limits))


def test_all_recurrence_arithmetic_uses_one_cumulative_counter(monkeypatch):
    observed = []

    def record(self, count):
        observed.append(count)
        assert count <= self.max_elimination_work

    monkeypatch.setattr(RidgeBudget, "_work", record)
    result = gaussian_critical_series(15)
    assert result.order == 15 and len(observed) > 100
    assert all(a < b for a, b in zip(observed, observed[1:], strict=False))
    assert observed[-1] > observed[0]
