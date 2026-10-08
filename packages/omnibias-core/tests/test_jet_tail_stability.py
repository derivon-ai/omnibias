# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Binary64 tail coefficients against independent 100-digit derivatives."""

import math
from types import SimpleNamespace

import mpmath as mp
import pytest
from omnibias.core.jet_algebra import JetSpec
from omnibias.core.jet_series import activation, activation_derivative, differential_system
from omnibias.core.jet_systems import get_system


def _float_ops():
    return SimpleNamespace(tanh=math.tanh, exp=math.exp, log=math.log,
                           sigmoid=lambda x: 1 / (1 + math.exp(-x)),
                           zeros_like=lambda x: 0., ones_like=lambda x: 1.)


@pytest.mark.parametrize("name,point", [("tanh", -20), ("tanh", 20),
                                       ("sigmoid", -40), ("sigmoid", 40),
                                       ("softplus", -40), ("softplus", 40)])
@pytest.mark.parametrize("nonlinear", [False, True])
def test_saturated_coefficients_retain_relative_accuracy(name, point, nonlinear):
    with mp.workdps(100):
        def sigmoid(z):
            return 1 / (1 + mp.exp(-z))

        function = {"tanh": mp.tanh, "sigmoid": sigmoid,
                    "softplus": lambda z: mp.log(1 + mp.exp(z))}[name]
        derivative = {"tanh": lambda z: 4 * sigmoid(2*z) * sigmoid(-2*z),
                      "sigmoid": lambda z: sigmoid(z) * sigmoid(-z),
                      "softplus": sigmoid}[name]
        first, second = (0.3, 0.02) if nonlinear else (1., 0.)
        def path(t):
            return point + mp.mpf(first)*t + mp.mpf(second)*t*t

        expected = mp.taylor(lambda t: function(path(t)), 0, 32)
        expected_derivative = mp.taylor(lambda t: derivative(path(t)), 0, 32)
        spec = JetSpec.directional(32)
        incoming = [float(point), first, second] + [0.] * 30
        actual = activation(incoming, spec, name, _float_ops())
        actual_derivative = activation_derivative(incoming, spec, name, _float_ops())
        for order in (1, 2, 16, 32):
            assert actual[order] != 0
            assert abs(mp.mpf(actual[order]) - expected[order]) <= abs(expected[order]) * mp.mpf("5e-13")
        for order in (0, 1, 2, 16, 32):
            assert abs(mp.mpf(actual_derivative[order]) - expected_derivative[order]) <= abs(expected_derivative[order]) * mp.mpf("5e-13")


def test_initial_rhs_override_validation_and_minimal_ops_compatibility():
    spec = JetSpec.directional(2)
    system = get_system("tanh")
    with pytest.raises(ValueError, match="initial RHS"):
        differential_system([0., 1., 0.], spec, system, [0.], initial_rhs=[])
    # A caller supplying only the primitive value function remains supported.
    ops = SimpleNamespace(tanh=math.tanh, zeros_like=lambda x: 0., ones_like=lambda x: 1.)
    value = activation([0.3, 1., 0.], spec, "tanh", ops)
    assert value == differential_system([0.3, 1., 0.], spec, system, [math.tanh(0.3)], ops)
    # Only degree-zero RHS is overridden; subsequent RHS uses the ODE.
    overridden = differential_system([0., 1., 0.], spec, system, [0.], initial_rhs=[2.])
    assert overridden == [0., 2., 0.]


@pytest.mark.parametrize("point", [-6., -10.])
def test_approximate_gelu_tail_value_coefficients_and_pullback(point):
    spec = JetSpec.directional(32)
    incoming = [point, 0.3, 0.02] + [0.] * 30
    got = activation(incoming, spec, "gelu_approx", _float_ops())
    got_slope = activation_derivative(incoming, spec, "gelu_approx", _float_ops())
    with mp.workdps(120):
        c = mp.mpf(0.044715)
        # Use the declared real constant rather than a fitted polynomial.
        factor = mp.mpf(math.sqrt(2 / math.pi))
        def gelu(z):
            return z / (1 + mp.exp(-2 * factor * (z + c*z**3)))

        def path(t):
            return mp.mpf(point) + mp.mpf(0.3)*t + mp.mpf(0.02)*t*t

        wanted = mp.taylor(lambda t: gelu(path(t)), 0, 32)
        slope = mp.taylor(lambda t: mp.diff(gelu, path(t)), 0, 32)
        for k in (0, 1, 2, 8, 16, 32):
            assert got[k] != 0 and got_slope[k] != 0
            assert abs(mp.mpf(got[k]) - wanted[k]) <= abs(wanted[k])*mp.mpf("2e-11")
            assert abs(mp.mpf(got_slope[k]) - slope[k]) <= abs(slope[k])*mp.mpf("2e-11")
