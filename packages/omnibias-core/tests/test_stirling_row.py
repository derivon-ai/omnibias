# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Independent falling-factorial and Bell checks for derivative tail weights."""

from fractions import Fraction
from math import comb, factorial

import pytest
from omnibias.core.bell import MAX_STIRLING_ORDER, bell_number, bell_partial, stirling_second_row


@pytest.mark.parametrize("n", [0, 1, 2, 5, 12, 32, 128])
def test_stirling_row_converts_powers_to_falling_factorials(n):
    coefficients = stirling_second_row(n)
    assert len(coefficients) == n + 1
    for j in (0, 1, 2, 7, Fraction(3, 2), Fraction(-2, 3)):
        falling = Fraction(1)
        value = coefficients[0] * falling
        for k in range(1, n + 1):
            falling *= j - k + 1
            value += coefficients[k] * falling
        assert value == j**n
    assert sum(coefficients) == bell_number(n)


def test_stirling_coefficients_match_partition_and_inclusion_exclusion_counts():
    for n in range(10):
        row = stirling_second_row(n)
        assert row == tuple(sum(bell_partial(n, k).values()) for k in range(n + 1))
    for n, k in ((25, 8), (64, 12), (128, 20)):
        count = sum((-1)**(k-j) * comb(k, j) * j**n for j in range(k + 1)) // factorial(k)
        assert stirling_second_row(n)[k] == count


@pytest.mark.parametrize("n", [-1, MAX_STIRLING_ORDER + 1])
def test_stirling_order_limits(n):
    with pytest.raises(ValueError, match="Stirling order"):
        stirling_second_row(n)


@pytest.mark.parametrize("n", [True, 2.0, Fraction(2), None])
def test_stirling_order_requires_exact_integer(n):
    with pytest.raises(TypeError, match="integer"):
        stirling_second_row(n)
