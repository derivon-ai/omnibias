# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""High-order residual gradients against independent closed-form expressions."""

import math

import pytest
import torch
from omnibias.torch.experimental import (
    JetSpec,
    compile_jet,
    directional_seed,
    extract_derivative,
)


@pytest.mark.parametrize("order", [16, 32])
@pytest.mark.parametrize("backward", ["native", "analytic"])
def test_high_order_parameter_and_direction_gradients(order, backward):
    class Ridge(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor(1.1, dtype=torch.float64))
            self.bias = torch.nn.Parameter(torch.tensor(0.1, dtype=torch.float64))

        def forward(self, x):
            return torch.exp(self.weight * x + self.bias)

    model = Ridge()
    x = torch.tensor(0.2, dtype=torch.float64)
    direction = torch.tensor(0.9, dtype=torch.float64, requires_grad=True)
    spec = JetSpec.directional(order)
    program = compile_jet(model, (x,), spec=spec, backward=backward)
    derivative = extract_derivative(program(directional_seed(x, direction, spec)), spec, order)
    actual = torch.autograd.grad(derivative, (model.weight, model.bias, direction))
    expected_value = (1.1 * 0.9)**order * math.exp(1.1 * 0.2 + 0.1)
    expected = (expected_value * (order / 1.1 + 0.2), expected_value,
                expected_value * order / 0.9)
    assert derivative.item() == pytest.approx(expected_value, rel=2e-13)
    for result, reference in zip(actual, expected, strict=True):
        assert result.item() == pytest.approx(reference, rel=2e-13)
