# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Exact finite algebra checks behind coalescing-bias/Taylor equivalence.

These identities exercise rational moments and nilpotent arithmetic directly;
they do not estimate derivatives from nearby floating-point samples.
"""

from fractions import Fraction
from math import comb, factorial, prod

import pytest
from omnibias.core.jet_algebra import JetSpec
from omnibias.core.jet_series import multiply


@pytest.mark.parametrize("order", range(1, 9))
def test_binomial_bias_moments_select_exactly_the_requested_derivative(order: int) -> None:
    nodes = tuple(Fraction(j) for j in range(order + 1))
    weights = tuple(Fraction((-1)**(order-j) * comb(order, j)) for j in range(order + 1))
    for degree in range(order + 1):
        moment = sum((weight * node**degree for weight, node in zip(weights, nodes, strict=True)), Fraction())
        assert moment == (factorial(order) if degree == order else 0)
    # Merely coalescing an unweighted collection does not cancel its constant.
    assert sum(Fraction(1) for _ in nodes) == order + 1


@pytest.mark.parametrize("order", range(1, 9))
def test_irregular_divided_difference_moments_and_centered_leading_error(order: int) -> None:
    nodes = tuple(Fraction(j*j+j, 3) for j in range(order + 1))
    weights = tuple(Fraction(factorial(order), 1) /
                    prod(node - other for k, other in enumerate(nodes) if k != j)
                    for j, node in enumerate(nodes))
    for degree in range(order + 1):
        assert sum(weight * node**degree for weight, node in zip(weights, nodes, strict=True)) == (
            factorial(order) if degree == order else 0)

    centered = tuple(Fraction(j) - Fraction(order, 2) for j in range(order + 1))
    binomial = tuple(Fraction((-1)**(order-j)*comb(order, j)) for j in range(order + 1))
    odd_error = sum(w * x**(order+1) for w, x in zip(binomial, centered, strict=True))
    leading_error = sum(w * x**(order+2) for w, x in zip(binomial, centered, strict=True))
    assert odd_error == 0
    assert leading_error / factorial(order+2) == Fraction(order, 24)


def _dual_product(a: dict[int, Fraction], b: dict[int, Fraction]) -> dict[int, Fraction]:
    result: dict[int, Fraction] = {}
    for left, value in a.items():
        for right, other in b.items():
            if not left & right:
                mask = left | right
                result[mask] = result.get(mask, Fraction()) + value * other
    return {mask: value for mask, value in result.items() if value}


def _symmetric_embedding(coefficients: list[Fraction], order: int) -> dict[int, Fraction]:
    return {mask: coefficients[mask.bit_count()] * factorial(mask.bit_count())
            for mask in range(1 << order) if coefficients[mask.bit_count()]}


@pytest.mark.parametrize("order", range(1, 9))
def test_nested_dual_symmetry_is_the_directional_taylor_algebra(order: int) -> None:
    # A bit mask represents the square-free monomial product of selected eps_i.
    generator = {1 << i: Fraction(1) for i in range(order)}
    power = {0: Fraction(1)}
    for degree in range(order + 2):
        expected = {mask: Fraction(factorial(degree)) for mask in range(1 << order)
                    if mask.bit_count() == degree}
        assert power == expected
        power = _dual_product(power, generator)

    spec = JetSpec.directional(order)
    a = [Fraction((-1)**k, k+1) for k in range(order+1)]
    b = [Fraction(k+2, k+3) for k in range(order+1)]
    assert _dual_product(_symmetric_embedding(a, order), _symmetric_embedding(b, order)) == (
        _symmetric_embedding(multiply(a, b, spec), order))


@pytest.mark.parametrize("order", range(1, 9))
def test_coalescing_quotient_has_nilpotent_special_fiber(order: int) -> None:
    # Expand product(t - delta*c_j) symbolically in (t_degree, delta_degree).
    polynomial = {(0, 0): Fraction(1)}
    for node in (Fraction(j, 3) for j in range(order+1)):
        updated: dict[tuple[int, int], Fraction] = {}
        for (t_degree, delta_degree), coefficient in polynomial.items():
            high = (t_degree+1, delta_degree)
            low = (t_degree, delta_degree+1)
            updated[high] = updated.get(high, Fraction()) + coefficient
            updated[low] = updated.get(low, Fraction()) - coefficient * node
        polynomial = {powers: value for powers, value in updated.items() if value}
    assert {(t, d): c for (t, d), c in polynomial.items() if d == 0} == {(order+1, 0): Fraction(1)}
    assert all(t + d == order + 1 for t, d in polynomial)

    spec = JetSpec.directional(order)
    epsilon = [Fraction(0), Fraction(1)] + [Fraction(0)] * (order-1)
    power = [Fraction(1)] + [Fraction(0)] * order
    for _ in range(order):
        power = multiply(power, epsilon, spec)
    assert power[-1] == 1
    assert multiply(power, epsilon, spec) == [Fraction(0)] * (order+1)
