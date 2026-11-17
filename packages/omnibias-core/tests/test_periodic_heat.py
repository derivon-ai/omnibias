# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Independent Fourier/Taylor coefficients and exact prescribed-data checks."""

from __future__ import annotations

import math
from dataclasses import FrozenInstanceError, replace
from fractions import Fraction
from typing import Any, cast

import pytest
from omnibias.core.exp_polynomial import ExponentialBudget
from omnibias.core.periodic_heat import PeriodicHeatPlan


def _table(plan: PeriodicHeatPlan) -> dict[tuple[int, ...], tuple[Fraction, Fraction]]:
    return {frequency: (real, imaginary) for frequency, real, imaginary in plan.heat_coefficients}


def test_plan_snapshots_sources_and_is_immutable() -> None:
    cosine = {(1, 0): Fraction(2, 3)}
    sine = {(0, 1): Fraction(-1, 5)}
    drift = [Fraction(2), Fraction(-1)]
    plan = PeriodicHeatPlan.from_log_potential(2, cosine, sine, degree=3, viscosity=Fraction(3, 2), drift=drift)
    before = (plan.potential_coefficients, plan.heat_coefficients, plan.drift)
    cosine.clear()
    sine[(0, 1)] = 90
    drift[0] = Fraction(100)
    assert (plan.potential_coefficients, plan.heat_coefficients, plan.drift) == before
    assert plan.dimension == 2
    assert plan.degree == 3
    assert plan.viscosity == Fraction(3, 2)
    assert plan.norm_bound == Fraction(13, 15)
    assert plan.max_frequency == 1
    assert all(isinstance(value, Fraction) for _, re, im in plan.heat_coefficients for value in (re, im))
    assert hash(plan)
    with pytest.raises(FrozenInstanceError):
        plan.degree = 4  # type: ignore[misc]
    with pytest.raises(TypeError, match="from_log_potential"):
        PeriodicHeatPlan()


def test_trigonometric_signs_frequency_reversal_and_constant_gauge_are_exact() -> None:
    # cos(-x)=cos(x), sin(-x)=-sin(x); only the consolidated nonconstant remains.
    plan = PeriodicHeatPlan.from_log_potential(1, {(0,): 9, (1,): 2, (-1,): 3},
                                              {(0,): 17, (1,): 7, (-1,): 2}, degree=1)
    assert plan.potential_coefficients == (((-1,), Fraction(5, 2), Fraction(5, 2)),
                                           ((1,), Fraction(5, 2), Fraction(-5, 2)))
    assert plan.norm_bound == 10
    assert plan.heat_coefficients == (((-1,), Fraction(5, 2), Fraction(5, 2)),
                                      ((0,), Fraction(1), Fraction(0)),
                                      ((1,), Fraction(5, 2), Fraction(-5, 2)))
    canonical = PeriodicHeatPlan.from_log_potential(1, {(1,): 5}, {(1,): 5}, degree=1)
    assert plan == canonical


@pytest.mark.parametrize("degree", [0, 1, 2, 3, 7, 16])
@pytest.mark.parametrize("amplitude", [Fraction(1), Fraction(-2, 3), Fraction(7, 5)])
def test_cosine_taylor_coefficients_match_independent_binomial_formula(degree: int, amplitude: Fraction) -> None:
    plan = PeriodicHeatPlan.from_log_potential(1, {(3,): amplitude}, degree=degree)
    expected = {}
    for frequency in range(-degree, degree + 1):
        value = sum((amplitude ** order * Fraction(math.comb(order, (order + frequency) // 2),
                                                   2**order * math.factorial(order))
                     for order in range(abs(frequency), degree + 1) if (order + frequency) % 2 == 0), Fraction(0))
        if value:
            expected[(3 * frequency,)] = (value, Fraction(0))
    assert _table(plan) == expected
    assert plan.norm_bound == abs(amplitude)
    assert plan.max_frequency == 3
    if degree:
        assert (3 * degree,) in _table(plan)  # No harmonic truncation to the input support.


def test_sine_square_has_negative_second_harmonic_and_exact_imaginary_signs() -> None:
    plan = PeriodicHeatPlan.from_log_potential(1, {}, {(1,): 2}, degree=2)
    assert _table(plan) == {(-2,): (Fraction(-1, 2), Fraction(0)),
                            (-1,): (Fraction(0), Fraction(1)),
                            (0,): (Fraction(2), Fraction(0)),
                            (1,): (Fraction(0), Fraction(-1)),
                            (2,): (Fraction(-1, 2), Fraction(0))}


def test_multidimensional_coefficients_match_total_degree_product_formula() -> None:
    a, b, degree = Fraction(2, 3), Fraction(-3, 5), 4
    plan = PeriodicHeatPlan.from_log_potential(2, {(1, 0): a, (0, 1): b}, degree=degree)
    expected = {}
    for k in range(-degree, degree + 1):
        for ell in range(-degree, degree + 1):
            value = Fraction(0)
            for p in range(abs(k), degree + 1):
                for q in range(abs(ell), degree + 1 - p):
                    if (p + k) % 2 == (q + ell) % 2 == 0:
                        value += (a**p * b**q * math.comb(p, (p + k) // 2) * math.comb(q, (q + ell) // 2)
                                  / (2**(p + q) * math.factorial(p) * math.factorial(q)))
            if value:
                expected[(k, ell)] = (value, Fraction(0))
    assert _table(plan) == expected
    assert (3, 1) in expected
    assert (4, 1) not in expected


def test_canonical_conjugate_symmetry_after_mixed_complex_convolutions() -> None:
    plan = PeriodicHeatPlan.from_log_potential(2, {(1, 2): Fraction(1, 3), (-2, 0): Fraction(1, 5)},
                                              {(2, -1): Fraction(-2, 7)}, degree=5)
    for coefficients in (plan.potential_coefficients, plan.heat_coefficients):
        table = {frequency: (real, imaginary) for frequency, real, imaginary in coefficients}
        assert tuple(table) == tuple(sorted(table))
        assert all(real or imaginary for real, imaginary in table.values())
        for frequency, (real, imaginary) in table.items():
            assert table[tuple(-entry for entry in frequency)] == (real, -imaginary)
    assert plan.max_frequency == 3


def test_compiled_polynomial_matches_independent_high_precision_point_evaluation() -> None:
    mp = pytest.importorskip("mpmath")
    plan = PeriodicHeatPlan.from_log_potential(2, {(1, 2): Fraction(2, 5), (2, -1): Fraction(-3, 7)},
                                              {(1, 0): Fraction(1, 3)}, degree=6)

    def rational(value: Fraction) -> Any:
        return mp.mpf(value.numerator) / value.denominator

    with mp.workdps(80):
        x, y = mp.mpf("0.23"), mp.mpf("-0.41")
        potential = mp.mpf(2) / 5 * mp.cos(x + 2 * y) - mp.mpf(3) / 7 * mp.cos(2 * x - y) + mp.sin(x) / 3
        expected = mp.fsum(potential ** order / mp.factorial(order) for order in range(plan.degree + 1))
        actual = mp.fsum((rational(real) + mp.j * rational(imaginary)) * mp.exp(mp.j * (frequency[0] * x + frequency[1] * y))
                        for frequency, real, imaginary in plan.heat_coefficients)
        assert abs(actual.imag) < mp.mpf("1e-75")
        assert abs(actual.real - expected) < mp.mpf("1e-75")


def test_empty_cancelled_and_constant_inputs_are_the_unit_heat_plan() -> None:
    for cosine, sine in (({}, None), ({(0,): Fraction(9, 2)}, {(0,): 8}),
                         ({(1,): 4, (-1,): -4}, {(2,): 3, (-2,): 3})):
        plan = PeriodicHeatPlan.from_log_potential(1, cosine, sine, degree=16, drift=(2,))
        assert plan.potential_coefficients == ()
        assert plan.heat_coefficients == (((0,), Fraction(1), Fraction(0)),)
        assert plan.norm_bound == plan.max_frequency == 0
        assert plan.drift == (Fraction(2),)
    assert PeriodicHeatPlan.from_velocity({}, mean=3).heat_coefficients == (((0,), Fraction(1), Fraction(0)),)


def test_mapping_insertion_order_does_not_change_canonical_plan() -> None:
    cosine = {(1,): Fraction(2, 3), (-1,): Fraction(1, 3), (3,): Fraction(-1, 5)}
    sine = {(1,): Fraction(1, 9), (-2,): Fraction(2, 7)}
    direct = PeriodicHeatPlan.from_log_potential(1, cosine, sine, degree=4)
    reverse = PeriodicHeatPlan.from_log_potential(1, dict(reversed(tuple(cosine.items()))),
                                                 dict(reversed(tuple(sine.items()))), degree=4)
    assert direct == reverse


def test_from_velocity_recovers_prescribed_nonconstant_fourier_coefficients_exactly() -> None:
    cosine = {1: Fraction(2, 3), 3: Fraction(-4, 7)}
    sine = {1: Fraction(5, 11), 2: Fraction(-3, 5)}
    viscosity, mean = Fraction(3, 2), Fraction(-1, 4)
    plan = PeriodicHeatPlan.from_velocity(cosine, sine, mean=mean, viscosity=viscosity, degree=3)
    assert plan.drift == (mean,)
    actual = {}
    for (n,), real, imaginary in plan.potential_coefficients:
        # Fourier coefficient of -2*nu*d_x g: multiply by -2*nu*i*n.
        actual[n] = (2 * viscosity * n * imaginary, -2 * viscosity * n * real)
    for n in (1, 2, 3):
        assert actual[n] == (cosine.get(n, Fraction(0)) / 2, -sine.get(n, Fraction(0)) / 2)
        assert actual[-n] == (cosine.get(n, Fraction(0)) / 2, sine.get(n, Fraction(0)) / 2)
    expected = PeriodicHeatPlan.from_log_potential(1,
        {(n,): value / (2 * viscosity * n) for n, value in sine.items()},
        {(n,): -value / (2 * viscosity * n) for n, value in cosine.items()},
        viscosity=viscosity, degree=3, drift=(mean,))
    assert plan == expected


@pytest.mark.parametrize("bad", [True, 1.0, Fraction(1, 2), "1"])
def test_invalid_integer_metadata_types(bad: Any) -> None:
    with pytest.raises(TypeError):
        PeriodicHeatPlan.from_log_potential(bad, {}, degree=0)
    with pytest.raises(TypeError):
        PeriodicHeatPlan.from_log_potential(1, {}, degree=bad)
    with pytest.raises(TypeError):
        PeriodicHeatPlan.from_log_potential(1, {(bad,): 1}, degree=0)
    with pytest.raises(TypeError):
        PeriodicHeatPlan.from_velocity({bad: 1}, degree=0)


@pytest.mark.parametrize("bad", [True, 1.0, float("nan"), float("inf"), complex(1), "1"])
def test_only_exact_rational_coefficients_viscosity_and_drift_are_accepted(bad: Any) -> None:
    with pytest.raises(TypeError):
        PeriodicHeatPlan.from_log_potential(1, {(1,): bad}, degree=0)
    with pytest.raises(TypeError):
        PeriodicHeatPlan.from_log_potential(1, {}, viscosity=bad, degree=0)
    with pytest.raises(TypeError):
        PeriodicHeatPlan.from_log_potential(1, {}, drift=(bad,), degree=0)
    with pytest.raises(TypeError):
        PeriodicHeatPlan.from_velocity({1: bad}, degree=0)
    with pytest.raises(TypeError):
        PeriodicHeatPlan.from_velocity({}, mean=bad, degree=0)


def test_invalid_domains_shapes_and_container_types() -> None:
    for dimension in (0, -1):
        with pytest.raises(ValueError):
            PeriodicHeatPlan.from_log_potential(dimension, {})
    with pytest.raises(ValueError):
        PeriodicHeatPlan.from_log_potential(1, {}, degree=-1)
    for viscosity in (0, -1, Fraction(-1, 3)):
        with pytest.raises(ValueError, match="positive"):
            PeriodicHeatPlan.from_log_potential(1, {}, viscosity=viscosity)
    for frequency in ((), (1, 2), 1):
        with pytest.raises(ValueError, match="frequency|frequencies"):
            PeriodicHeatPlan.from_log_potential(1, {cast(Any, frequency): 1})
    with pytest.raises(ValueError, match="drift"):
        PeriodicHeatPlan.from_log_potential(2, {}, drift=(1,))
    for mapping in (None, [], ((1, 2),)):
        with pytest.raises(TypeError, match="mapping"):
            PeriodicHeatPlan.from_log_potential(1, cast(Any, mapping))
        with pytest.raises(TypeError, match="mapping"):
            PeriodicHeatPlan.from_velocity(cast(Any, mapping))
    for frequency in (0, -1):
        with pytest.raises(ValueError):
            PeriodicHeatPlan.from_velocity({frequency: 1})
    with pytest.raises(TypeError, match="budget"):
        PeriodicHeatPlan.from_log_potential(1, {}, budget=cast(Any, 1))


def test_term_degree_dimension_and_exact_bit_budgets_refuse_without_truncation() -> None:
    with pytest.raises(ValueError, match="support"):
        PeriodicHeatPlan.from_log_potential(1, {(1,): 1}, degree=2, budget=replace(ExponentialBudget(), max_terms=4))
    with pytest.raises(ValueError, match="max_total_degree"):
        PeriodicHeatPlan.from_log_potential(1, {}, degree=3, budget=replace(ExponentialBudget(), max_total_degree=2))
    with pytest.raises(ValueError, match="dimension"):
        PeriodicHeatPlan.from_log_potential(2, {}, budget=replace(ExponentialBudget(), max_dimension=1))
    with pytest.raises(ValueError, match="bit budget"):
        PeriodicHeatPlan.from_log_potential(1, {(1,): 32}, degree=0, budget=replace(ExponentialBudget(), max_coefficient_bits=5))
    # The frequency input fits five bits; the second convolution creates 32.
    with pytest.raises(ValueError, match="bit budget"):
        PeriodicHeatPlan.from_log_potential(1, {(16,): 2}, degree=2, budget=replace(ExponentialBudget(), max_coefficient_bits=5))
    # Rational-coefficient products are checked before later cancellation.
    with pytest.raises(ValueError, match="bit budget"):
        PeriodicHeatPlan.from_log_potential(1, {(1,): 30}, degree=2, budget=replace(ExponentialBudget(), max_coefficient_bits=5))


def test_large_harmonic_is_not_confused_with_taylor_degree_budget() -> None:
    frequency = 10**100
    plan = PeriodicHeatPlan.from_log_potential(1, {(frequency,): 1}, degree=2,
                                              budget=replace(ExponentialBudget(), max_total_degree=2))
    assert plan.max_frequency == frequency
    assert (2 * frequency,) in _table(plan)
    assert len(plan.heat_coefficients) == 5


def test_compilation_work_budget_accumulates_across_all_taylor_orders() -> None:
    budget = replace(ExponentialBudget(), max_work=300)
    assert PeriodicHeatPlan.from_log_potential(1, {(1,): 1}, degree=2, budget=budget).degree == 2
    with pytest.raises(ValueError, match="work"):
        PeriodicHeatPlan.from_log_potential(1, {(1,): 1}, degree=3, budget=budget)


def test_velocity_integration_and_compilation_share_one_work_budget() -> None:
    budget = replace(ExponentialBudget(), max_work=260)
    plan = PeriodicHeatPlan.from_log_potential(1, {}, {(1,): Fraction(-1, 2)}, degree=2, budget=budget)
    assert plan.degree == 2
    with pytest.raises(ValueError, match="work"):
        PeriodicHeatPlan.from_velocity({1: 1}, degree=2, budget=budget)


def test_zero_velocity_coefficients_do_not_consume_nonzero_support() -> None:
    plan = PeriodicHeatPlan.from_velocity({n: 0 for n in range(1, 20)}, degree=0,
                                          budget=replace(ExponentialBudget(), max_terms=1))
    assert plan.potential_coefficients == ()
    assert plan.heat_coefficients == (((0,), Fraction(1), Fraction(0)),)


def test_derivative_budget_does_not_limit_taylor_compilation() -> None:
    plan = PeriodicHeatPlan.from_log_potential(1, {(1,): 1}, degree=5,
                                              budget=replace(ExponentialBudget(), max_derivative_order=0))
    assert plan.degree == 5
