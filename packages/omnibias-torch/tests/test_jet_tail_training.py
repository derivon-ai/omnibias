# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Parameter training and analytic pullbacks in saturated activation tails."""

from math import factorial

import mpmath as mp
import pytest
import torch
from omnibias.torch.experimental import JetSpec, compile_jet


@pytest.mark.parametrize("name,point", [("tanh", -20.), ("tanh", 20.),
                                       ("sigmoid", -40.), ("sigmoid", 40.)])
@pytest.mark.parametrize("backward", ["native", "analytic"])
def test_saturated_nonlinear_path_parameter_gradient_and_double_backward(name, point, backward):
    class Shift(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.bias = torch.nn.Parameter(torch.tensor(0., dtype=torch.float64))

        def forward(self, x):
            return getattr(torch, name)(x + self.bias)

    model = Shift()
    spec = JetSpec.directional(32)
    value = torch.tensor([point], dtype=torch.float64)
    program = compile_jet(model, (value,), spec=spec, backward=backward)
    rows = torch.zeros((33, 1), dtype=torch.float64)
    rows[0], rows[1], rows[2] = value, 0.3, 0.02
    raw = program(rows)[32].sum() * float(factorial(32))
    gradient = torch.autograd.grad(raw, model.bias, create_graph=True)[0]
    second = torch.autograd.grad(gradient, model.bias)[0]
    with mp.workdps(100):
        def sigmoid(z):
            return 1 / (1 + mp.exp(-z))

        function = mp.tanh if name == "tanh" else sigmoid
        slope = (lambda z: 4 * sigmoid(2*z) * sigmoid(-2*z)) if name == "tanh" else (lambda z: sigmoid(z) * sigmoid(-z))
        curvature = (lambda z: -2 * mp.tanh(z) * slope(z)) if name == "tanh" else (lambda z: (1 - 2*sigmoid(z)) * slope(z))
        def path(t):
            return mp.mpf(point) + mp.mpf(0.3)*t + mp.mpf(0.02)*t*t

        for actual, fn in zip((raw, gradient, second), (function, slope, curvature), strict=True):
            expected = mp.taylor(lambda t, fn=fn: fn(path(t)), 0, 32)[32] * factorial(32)
            assert actual.item() != 0
            assert abs(mp.mpf(actual.item()) - expected) <= abs(expected) * mp.mpf("3e-12")
