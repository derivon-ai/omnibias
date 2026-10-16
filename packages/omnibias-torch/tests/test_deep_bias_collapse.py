# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""The user's deep chain: whole-graph collapse versus derivative replacement."""

from math import comb

import mpmath as mp
import pytest
import torch
from omnibias.torch.experimental import (
    JetSpec,
    coefficients_to_derivatives,
    compile_jet,
    directional_seed,
)


class Chain(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.biases = torch.nn.Parameter(torch.tensor([0.1, -0.2, 0.05], dtype=torch.float64))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sin(torch.tanh(torch.sigmoid(x + self.biases[0]) + self.biases[1])
                         + self.biases[2])


def test_deep_graph_tower_matches_high_precision_through_order_eight() -> None:
    model = Chain()
    points = torch.tensor([[-0.7], [0.0], [0.8]], dtype=torch.float64)
    spec = JetSpec.directional(8)
    program = compile_jet(model, (points,), spec=spec)
    actual = coefficients_to_derivatives(
        program(directional_seed(points, torch.ones_like(points), spec)), spec
    )
    with mp.workdps(70):
        # Convert the actual stored binary64 operands, not decimal substitutes.
        b = [mp.mpf(float(value)) for value in model.biases.detach()]

        def oracle(x):
            return mp.sin(mp.tanh(1 / (1 + mp.exp(-(x + b[0]))) + b[1]) + b[2])

        expected = torch.tensor([
            [[float(mp.diff(oracle, mp.mpf(float(point)), order))] for point in points[:, 0]]
            for order in range(9)
        ], dtype=points.dtype)
    torch.testing.assert_close(actual, expected, atol=2e-12, rtol=2e-10)


def test_compiled_deep_derivative_keeps_original_parameter_gradients() -> None:
    model = Chain()
    points = torch.tensor([[-0.2], [0.4]], dtype=torch.float64, requires_grad=True)
    spec = JetSpec.directional(4)
    program = compile_jet(model, (points.detach().clone(),), spec=spec)
    actual = coefficients_to_derivatives(
        program(directional_seed(points, torch.ones_like(points), spec)), spec
    )[4]
    expected = model(points)
    for _ in range(4):
        expected = torch.autograd.grad(expected.sum(), points, create_graph=True)[0]
    actual_gradient, = torch.autograd.grad(actual.square().sum(), model.biases)
    expected_gradient, = torch.autograd.grad(expected.square().sum(), model.biases)
    torch.testing.assert_close(actual_gradient, expected_gradient, atol=2e-12, rtol=2e-10)
    assert any(parameter is model.biases for parameter in program.parameters())


@pytest.mark.parametrize("order", [1, 2, 4])
def test_whole_graph_centered_bias_collapse_converges_to_composite_derivative(order) -> None:
    with mp.workdps(80):
        def f(x):
            return mp.sin(mp.tanh(1 / (1 + mp.exp(-x))))

        point = mp.mpf("0.3")
        exact = mp.diff(f, point, order)
        errors = []
        for h in (mp.mpf("0.04"), mp.mpf("0.02"), mp.mpf("0.01")):
            estimate = sum((-1)**(order-j) * comb(order, j)
                           * f(point + (j - mp.mpf(order)/2) * h)
                           for j in range(order + 1)) / h**order
            errors.append(abs(estimate - exact))
        assert all(mp.mpf("3.8") < left/right < mp.mpf("4.2")
                   for left, right in zip(errors[:-1], errors[1:], strict=True))


def test_local_collapsed_derivative_network_is_not_the_composite_derivative() -> None:
    with mp.workdps(70):
        x = mp.mpf("0.3")
        def sigmoid(value):
            return 1 / (1 + mp.exp(-value))
        s = sigmoid(x)
        true_derivative = mp.cos(mp.tanh(s)) * (1 - mp.tanh(s)**2) * s * (1-s)
        derivative_network = mp.cos(1 - mp.tanh(s * (1-s))**2)
        assert abs(true_derivative - derivative_network) > mp.mpf("0.3")
        h = mp.mpf("0.0001")
        value = x
        for activation in (sigmoid, mp.tanh, mp.sin):
            value = (activation(value + h/2) - activation(value - h/2)) / h
        assert abs(value - derivative_network) < mp.mpf("1e-8")
        # Without normalization the local difference network tends to zero.
        value = x
        for activation in (sigmoid, mp.tanh, mp.sin):
            value = activation(value + h/2) - activation(value - h/2)
        assert abs(value) <= h
