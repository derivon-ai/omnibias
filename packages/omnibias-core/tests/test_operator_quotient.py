# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Exact operator readouts and pullbacks in weighted quotient bases."""

from dataclasses import replace
from fractions import Fraction
from math import comb, factorial
from types import SimpleNamespace

import pytest
from omnibias.core.apolar import apolar_algebra
from omnibias.core.jet_algebra import JetBudget, JetSpec
from omnibias.core.operator_quotient import (
    quotient_activation,
    quotient_activation_derivative,
    quotient_compose,
    quotient_estimate,
    quotient_multiply,
    quotient_multiply_transpose,
    quotient_readout,
    quotient_reciprocal,
    quotient_seed,
)


def test_weighted_operator_on_ridge_polynomial() -> None:
    algebra = apolar_algebra({(2, 0): Fraction(3, 2), (1, 1): -2, (0, 2): Fraction(5, 3)})
    assert algebra.size == 4
    x, y = quotient_seed(Fraction(1, 4), 0, algebra), quotient_seed(Fraction(-1, 5), 1, algebra)
    z = [a + 2 * b for a, b in zip(x, y, strict=True)]
    tower = [comb(6, k) * z[0] ** (6 - k) for k in range(algebra.order + 1)]
    lifted = quotient_compose(z, tower, algebra)
    # P(1,2) = 25/6 and d²z⁶/dz² = 30z⁴.
    assert quotient_readout(lifted, algebra) == 125 * z[0] ** 4


def test_nonlinear_exp_path_biharmonic_readout_is_exact() -> None:
    algebra = apolar_algebra({(4, 0): 1, (2, 2): 2, (0, 4): 1})
    x, y = quotient_seed(Fraction(0), 0, algebra), quotient_seed(Fraction(0), 1, algebra)
    x2 = quotient_multiply(x, x, algebra)
    xy = quotient_multiply(x, y, algebra)
    y2 = quotient_multiply(y, y, algebra)
    u = [a + 3 * b + 2 * c for a, b, c in zip(x2, xy, y2, strict=True)]
    ops = SimpleNamespace(zeros_like=lambda _: Fraction(0), ones_like=lambda _: Fraction(1),
                          exp=lambda _: Fraction(1))  # exp is evaluated only at the zero primal.
    lifted = quotient_activation(u, algebra, "exp", ops,
                                 directional_spec=JetSpec.directional(algebra.order))
    reference = quotient_compose(u, [Fraction(1, factorial(k)) for k in range(5)], algebra)
    assert lifted == reference
    # Quartic part is (x²+3xy+2y²)²/2. Its Δ² is 12+52+48.
    assert quotient_readout(lifted, algebra) == 112
    assert quotient_activation_derivative(u, algebra, "exp", ops) == lifted


@pytest.mark.parametrize("symbol", [
    {(2, 0): 2, (0, 2): -3}, {(2, 1): Fraction(1, 3)},
    {(4, 0): 1, (2, 2): 2, (0, 4): 1},
])
def test_associativity_reciprocal_and_weighted_transpose(symbol: dict) -> None:
    algebra = apolar_algebra(symbol)
    a = [Fraction(3)] + [Fraction(i - 3, 7) for i in range(1, algebra.size)]
    b = [Fraction(i + 2, 5) for i in range(algebra.size)]
    c = [Fraction(4 - i, 3) for i in range(algebra.size)]
    multiply = lambda left, right: quotient_multiply(left, right, algebra)  # noqa: E731
    assert multiply(multiply(a, b), c) == multiply(a, multiply(b, c))
    assert multiply(a, quotient_reciprocal(a, algebra)) == list(algebra.unit)
    pullback = quotient_multiply_transpose(c, b, algebra)
    assert sum(u * v for u, v in zip(c, multiply(a, b), strict=True)) == sum(
        u * v for u, v in zip(pullback, a, strict=True))
    assert quotient_multiply(a, b, algebra, product=lambda u, v: 2 * u * v) == [
        2 * v for v in multiply(a, b)]


def test_zero_order_quotient_and_invalid_inputs() -> None:
    algebra = apolar_algebra({(0,): 3})
    assert quotient_seed(Fraction(2), 0, algebra) == [2]
    assert quotient_compose([Fraction(2)], [Fraction(5)], algebra) == [5]
    assert quotient_readout([Fraction(5)], algebra) == 15
    with pytest.raises(ValueError, match="coordinates"):
        quotient_multiply([], [Fraction(1)], algebra)
    with pytest.raises(ValueError, match="coordinate"):
        quotient_seed(Fraction(2), 1, algebra)
    with pytest.raises(ValueError, match="tower"):
        quotient_compose([Fraction(2)], [], algebra)
    with pytest.raises(ValueError, match="operation"):
        quotient_estimate(algebra, "unknown")


def test_budget_refuses_before_tensor_callback_and_checks_layout() -> None:
    algebra = apolar_algebra({(4, 0): 1, (2, 2): 2, (0, 4): 1})

    def forbidden(*args: object) -> None:
        raise AssertionError("backend operations must not run before resource refusal")

    with pytest.raises(ValueError, match="multiplication"):
        quotient_multiply([1] * algebra.size, [1] * algebra.size, algebra, product=forbidden,
                          budget=JetBudget(max_multiplication_terms=1))
    with pytest.raises(ValueError, match="coefficients"):
        quotient_compose([1] * algebra.size, [1] * 5, algebra, budget=JetBudget(max_coefficients=3))
    ops = SimpleNamespace(zeros_like=forbidden, ones_like=forbidden, exp=forbidden)
    with pytest.raises(ValueError, match="directional"):
        quotient_activation([1] * algebra.size, algebra, "exp", ops, directional_spec=JetSpec.directional(2))


@pytest.mark.parametrize("weight", [Fraction(1 << 2000), Fraction(1, 1 << 2000)])
def test_extreme_exact_metadata_requires_explicit_numerical_strategy(weight: Fraction) -> None:
    algebra = apolar_algebra({(1,): 1})
    extreme = replace(algebra, readout=(Fraction(0), weight))
    assert quotient_readout([Fraction(0), Fraction(1)], extreme) == weight
    with pytest.raises(ValueError, match="binary64"):
        quotient_readout([0.0, 1.0], extreme)
