# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Independent high-precision and rational oracles for polynomial jet rules."""

from fractions import Fraction
from math import factorial
from types import SimpleNamespace

import mpmath as mp
import pytest
from omnibias.core.jet_algebra import JetSpec
from omnibias.core.jet_series import (
    activation,
    activation_derivative,
    differential_system,
    multiply,
    reciprocal,
)
from omnibias.core.jet_systems import DifferentialSystemSpec, get_system


def _ops() -> SimpleNamespace:
    return SimpleNamespace(**{name: getattr(mp, name) for name in
        ("exp", "log", "sqrt", "tanh", "sin", "cos", "erf", "erfc")},
        sigmoid=lambda x: 1 / (1 + mp.exp(-x)),
        logaddexp=lambda x, y: mp.log(mp.exp(x) + mp.exp(y)),
        zeros_like=lambda x: x * 0, ones_like=lambda x: x * 0 + 1)


FUNCTIONS = {
    "tanh": mp.tanh, "sigmoid": lambda x: 1 / (1 + mp.exp(-x)),
    "exp": mp.exp, "sin": mp.sin, "cos": mp.cos, "erf": mp.erf, "erfc": mp.erfc,
    "log": mp.log, "sqrt": mp.sqrt, "reciprocal": lambda x: 1 / x,
    "gaussian": lambda x: mp.exp(-x*x/2),
    "gaussian_normalized": lambda x: mp.exp(-x*x/2) / mp.sqrt(2*mp.pi),
    "gaussian_rbf": lambda x: mp.exp(-x*x),
    "softplus": lambda x: mp.log(1 + mp.exp(x)),
    "silu": lambda x: x / (1 + mp.exp(-x)),
    "gelu": lambda x: x * (1 + mp.erf(x/mp.sqrt(2))) / 2,
    "gelu_approx": lambda x: x * (1 + mp.tanh(mp.sqrt(2/mp.pi)*(x+mp.mpf("0.044715")*x**3))) / 2,
}


@pytest.mark.parametrize("name", FUNCTIONS)
def test_nonlinear_directional_path_against_independent_derivatives(name: str) -> None:
    with mp.workdps(60):
        spec = JetSpec.directional(7)
        a = [mp.mpf("0.7"), mp.mpf("0.2"), mp.mpf("-0.1"), mp.mpf("0.03")] + [mp.mpf(0)]*4
        def path(t):
            return sum(value * t**k for k, value in enumerate(a))
        expected = mp.taylor(lambda t: FUNCTIONS[name](path(t)), 0, 7)
        actual = activation(a, spec, name, _ops())
        # Float constants at the tensor boundary (sqrt(pi), GELU coefficient)
        # limit several families to ordinary float64 agreement.
        for got, want in zip(actual, expected, strict=True):
            assert abs(got - want) <= mp.mpf("3e-15") * (1 + abs(want))
        derivative = activation_derivative(a, spec, name, _ops())
        expected_d = mp.taylor(lambda t: mp.diff(FUNCTIONS[name], path(t)), 0, 7)
        for got, want in zip(derivative, expected_d, strict=True):
            assert abs(got - want) <= mp.mpf("5e-15") * (1 + abs(want))


def test_exact_fraction_mixed_product_and_reciprocal() -> None:
    spec = JetSpec.mixed([(3, 2), (1, 4)])
    a = [Fraction(2 if i == 0 else (-1)**i, i+1) for i in range(spec.size)]
    inv = reciprocal(a, spec)
    assert multiply(a, inv, spec) == [Fraction(1)] + [Fraction(0)] * (spec.size-1)


def test_mixed_nonlinear_exp_independent_multivariate_derivatives() -> None:
    with mp.workdps(50):
        spec = JetSpec.mixed([(3, 1), (1, 3)])
        a = [mp.mpf(0)] * spec.size
        for alpha, value in { (0, 0): "0.2", (1, 0): "0.3", (0, 1): "-0.1", (1, 1): "0.4"}.items():
            a[spec.index(alpha)] = mp.mpf(value)
        actual = activation(a, spec, "exp", _ops())
        def function(x, y):
            return mp.exp(mp.mpf("0.2") + mp.mpf("0.3")*x - mp.mpf("0.1")*y + mp.mpf("0.4")*x*y)
        for alpha, got in zip(spec.indices, actual, strict=True):
            want = mp.diff(function, (0, 0), alpha) / (factorial(alpha[0])*factorial(alpha[1]))
            assert abs(got - want) < mp.mpf("1e-45")


def test_rational_system_and_metadata_roundtrip() -> None:
    # y'=2u, y(0)=0 gives y=u^2 exactly, including nonlinear path terms.
    system = DifferentialSystemSpec("square", ("square",),
        (((Fraction(2), (1, 0)),),), ((Fraction(1), (0, 1)),))
    assert DifferentialSystemSpec.from_config(system.to_config()) == system
    spec = JetSpec.directional(5)
    a = [Fraction(2), Fraction(3), Fraction(1), Fraction(0), Fraction(2), Fraction(0)]
    assert differential_system(a, spec, system, [a[0]**2]) == multiply(a, a, spec)
    activation_spec = SimpleNamespace(name="tanh", riccati_polynomial=(1., 0., -1.))
    assert DifferentialSystemSpec.from_activation(activation_spec) == get_system("tanh")


def test_order_zero_and_refusals() -> None:
    spec = JetSpec.directional(0)
    assert activation([mp.mpf(0)], spec, "exp", _ops()) == [1]
    with pytest.raises(ValueError, match="coefficients"):
        multiply([], [1], spec)
    with pytest.raises(NotImplementedError, match="registered"):
        activation([1], spec, "unknown", _ops())


@pytest.mark.parametrize("point", [-8, -10])
@pytest.mark.parametrize("order", [16, 32])
def test_exact_gelu_negative_tail_has_relative_accuracy(point: int, order: int) -> None:
    with mp.workdps(100):
        spec = JetSpec.directional(order)
        a = [mp.mpf(point), mp.mpf(1)] + [mp.mpf(0)] * (order - 1)
        got = activation(a, spec, "gelu", _ops())
        expected = mp.taylor(lambda x: x * mp.erfc(-x / mp.sqrt(2)) / 2, point, order)
        assert got[0] != 0
        for actual, wanted in zip(got, expected, strict=True):
            assert abs(actual - wanted) <= abs(wanted) * mp.mpf("2e-12")


def test_fixed_system_arithmetic_is_quadratic_in_order() -> None:
    class Counted:
        products = 0

        def __init__(self, value: float):
            self.value = value

        def __add__(self, other):
            return Counted(self.value + (other.value if isinstance(other, Counted) else other))

        def __mul__(self, other):
            Counted.products += 1
            return Counted(self.value * (other.value if isinstance(other, Counted) else other))

        def __truediv__(self, other):
            return Counted(self.value / other)

    counts = []
    for order in (8, 16, 32):
        Counted.products = 0
        spec = JetSpec.directional(order)
        differential_system([Counted(0.1)]*(order+1), spec, get_system("tanh"), [Counted(0.2)])
        counts.append(Counted.products)
    assert counts[1] < 4 * counts[0]
    assert counts[2] < 4 * counts[1]
