# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Independent exact values for compressed scalar differential functionals."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from math import comb, factorial
from typing import Any

import pytest
from omnibias.core.jet_algebra import JetBudget
from omnibias.core.operator_polynomial import (
    coordinate_polynomial,
    operator_product,
    operator_product_plan,
    operator_readout,
    operator_sum,
    operator_unary,
    operator_unary_plan,
)


def _tower(polynomial: dict[int, int], value: Any, order: int) -> list[Any]:
    return [sum((coefficient * comb(power, k) * value ** (power - k)
                 for power, coefficient in polynomial.items() if power >= k), value * 0)
            for k in range(order + 1)]


@pytest.mark.parametrize("order", [0, 1, 2, 8, 24, 32])
def test_exp_bell_transform_against_inclusion_exclusion(order: int) -> None:
    # e^(-t) D_x^n exp(t exp(x)) at x=0 has Stirling-number coefficients.
    # Independent inclusion-exclusion avoids the production Bell recurrence.
    expected = [Fraction(int(order == 0))] + [
        Fraction(sum((-1) ** (k - j) * comb(k, j) * j ** order for j in range(k + 1)), factorial(k))
        for k in range(1, order + 1)]
    coefficients = coordinate_polynomial(order, Fraction(1))
    tower = [Fraction(1, factorial(k)) for k in range(order + 1)]
    plan = operator_unary_plan(order)
    transformed = operator_unary(coefficients, tower, plan=plan)
    assert transformed == expected
    assert operator_readout(transformed, tower) == sum(expected)


def test_unary_composition_and_tower_normalization() -> None:
    point, order = Fraction(2, 3), 5
    original = coordinate_polynomial(order, Fraction(1))
    first_tower = _tower({2: 1, 1: 1}, point, order)
    first = operator_unary(original, first_tower)
    second = operator_unary(first, _tower({3: 1}, first_tower[0], order))
    direct = operator_unary(original, _tower({6: 1, 5: 3, 4: 3, 3: 1}, point, order))
    assert second == direct
    assert operator_readout(original, first_tower) == 0  # fifth derivative of a quadratic
    assert operator_readout(coordinate_polynomial(2), first_tower) == 2


@pytest.mark.parametrize("point", [Fraction(0), Fraction(2, 3), Fraction(-1, 2)])
def test_disjoint_product_against_separable_monomials(point: Fraction) -> None:
    # For H(s)=s^p and s=xy, D_x^a D_y^b H(xy) is the product
    # of two ordinary monomial derivatives, including at x=0 or y=0.
    for a, b in ((0, 0), (0, 3), (2, 3), (5, 2)):
        left, right = coordinate_polynomial(a, Fraction(1)), coordinate_polynomial(b, Fraction(1))
        for other in (Fraction(0), Fraction(3, 2)):
            polynomial = operator_product(left, right, point, other,
                                          plan=operator_product_plan(a, b))
            for power in (0, 2, 6):
                actual = operator_readout(polynomial, _tower({power: 1}, point * other, a + b))
                expected = (Fraction(0) if power < max(a, b) else
                            factorial(power) // factorial(power - a)
                            * (factorial(power) // factorial(power - b))
                            * point ** (power - a) * other ** (power - b))
                assert actual == expected


def test_product_associativity_and_inactive_branch() -> None:
    a, b, c = [coordinate_polynomial(n, Fraction(1)) for n in (2, 3, 1)]
    x, y, z = Fraction(0), Fraction(2), Fraction(-3)
    left = operator_product(operator_product(a, b, x, y), c, x * y, z)
    right = operator_product(a, operator_product(b, c, y, z), x, y * z)
    assert left == right
    assert operator_sum([Fraction(1)], a) == a
    assert operator_unary([Fraction(1)], [Fraction(7)]) == [1]
    assert operator_product([Fraction(1)], coordinate_polynomial(3), Fraction(2), Fraction(5)) == [0, 0, 0, 8]


@dataclass(frozen=True)
class _Dual:
    value: Fraction
    gradient: Fraction = Fraction(0)

    @staticmethod
    def _coerce(other: Any) -> _Dual:
        return other if isinstance(other, _Dual) else _Dual(Fraction(other))

    def __add__(self, other: Any) -> _Dual:
        other = self._coerce(other)
        return _Dual(self.value + other.value, self.gradient + other.gradient)

    __radd__ = __add__

    def __mul__(self, other: Any) -> _Dual:
        other = self._coerce(other)
        return _Dual(self.value * other.value, self.gradient * other.value + self.value * other.gradient)

    __rmul__ = __mul__

    def __pow__(self, exponent: int) -> _Dual:
        if exponent == 0:
            return _Dual(Fraction(1))
        return _Dual(self.value ** exponent, exponent * self.value ** (exponent - 1) * self.gradient)


@pytest.mark.parametrize("alpha", [(0, 0), (2, 3), (4, 2), (6, 8)])
def test_repeated_mixed_derivatives_and_shared_parameter_gradient(alpha: tuple[int, int]) -> None:
    # f=((theta*x+x²)*(theta*y+y³))^4 + ((theta*x+x²)*(theta*y+y³))².
    # The oracle expands binomial powers directly, independently of the
    # compressed operator propagation; dual theta checks differentiability.
    a, b = alpha
    x, y, theta = Fraction(2, 3), Fraction(-1, 2), Fraction(3, 2)
    parameter = _Dual(theta, Fraction(1))
    u = parameter * x + x * x
    v = parameter * y + y ** 3
    u_tower = [u, parameter + 2 * x, _Dual(Fraction(1))] + [_Dual(Fraction(0))] * a
    v_tower = [v, parameter + 3 * y * y, _Dual(3 * y), _Dual(Fraction(1))] + [_Dual(Fraction(0))] * b
    left = operator_unary(coordinate_polynomial(a, _Dual(Fraction(1))), u_tower)
    right = operator_unary(coordinate_polynomial(b, _Dual(Fraction(1))), v_tower)
    product = operator_product(left, right, u, v)
    actual = operator_readout(product, _tower({4: 1, 2: 1}, u * v, a + b))
    value = gradient = Fraction(0)
    for power in (2, 4):
        for j in range(power + 1):
            for k in range(power + 1):
                x_power, y_power, parameter_power = power + j, power + 2 * k, 2 * power - j - k
                if x_power < a or y_power < b:
                    continue
                scale = (comb(power, j) * comb(power, k)
                         * (factorial(x_power) // factorial(x_power - a))
                         * (factorial(y_power) // factorial(y_power - b))
                         * x ** (x_power - a) * y ** (y_power - b))
                value += scale * theta ** parameter_power
                if parameter_power:
                    gradient += scale * parameter_power * theta ** (parameter_power - 1)
    assert actual == _Dual(value, gradient)


def test_preallocation_resource_refusals_and_schedule_checks() -> None:
    budget = JetBudget(max_coefficients=100, max_multiplication_terms=30)
    with pytest.raises(ValueError, match="multiplication"):
        operator_product_plan(5, 5, budget=budget)
    with pytest.raises(ValueError, match="multiplication"):
        operator_unary_plan(5, budget=budget)
    with pytest.raises(ValueError, match="coefficients"):
        coordinate_polynomial(10**9)
    with pytest.raises(ValueError, match="constant"):
        operator_sum([], [1])
    with pytest.raises(ValueError, match="tower"):
        operator_unary([0, 0, 1], [1, 2])
    with pytest.raises(ValueError, match="tower"):
        operator_readout([0, 0, 1], [1, 2])
    with pytest.raises(ValueError, match="schedule"):
        operator_product([0, 1], [0, 1], 2, 3, plan=())
    with pytest.raises(ValueError, match="schedule"):
        operator_unary([0, 1], [2, 3], plan=())


@pytest.mark.parametrize("order", [24, 32])
def test_high_order_product_schedules_and_zero_value(order: int) -> None:
    n, m = order // 2, order - order // 2
    schedule = operator_product_plan(n, m)
    assert len(schedule) == sum(min(i, j) + 1 for i in range(n + 1) for j in range(m + 1))
    result = operator_product(coordinate_polynomial(n, Fraction(1)),
                              coordinate_polynomial(m, Fraction(1)), Fraction(0), Fraction(0), plan=schedule)
    expected = [Fraction(0)] * (order + 1)
    expected[n] = Fraction(factorial(n))
    assert result == expected


@pytest.mark.parametrize("order", [-1, True, 1.5])
def test_invalid_coordinate_order(order: Any) -> None:
    with pytest.raises(ValueError):
        coordinate_polynomial(order)
