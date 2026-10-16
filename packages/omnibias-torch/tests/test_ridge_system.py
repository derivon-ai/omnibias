# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Analytic and independent spatial-autograd checks of constrained ridge fields."""

from fractions import Fraction

import pytest
import torch
from omnibias.core.operator_spec import OperatorSpec
from omnibias.core.ridge_system import RidgeSystem, RidgeSyzygy
from omnibias.torch.ridge_system import ridge_solution


def _wave_pack(sign: int = 1) -> RidgeSyzygy:
    system = RidgeSystem.from_operators(
        [
            [OperatorSpec.polynomial({(1, 0): -1}), OperatorSpec.mixed((0, 0)), None],
            [OperatorSpec.polynomial({(0, 1): -1}), None, OperatorSpec.mixed((0, 0))],
            [None, OperatorSpec.mixed((1, 0)), OperatorSpec.polynomial({(0, 1): -4})],
        ],
        (2 * sign, 1),
    )
    assert not system.kernel_basis(max_degree=0)
    pack, = system.kernel_basis(max_degree=1)
    assert pack.coefficients == ((1, 0, 0), (0, 2 * sign, 1))
    return pack


@pytest.mark.parametrize("activation", ["tanh", "sigmoid", "softplus", "gaussian", "exp",
                                         "sin", "cos", "sinh", "cosh"])
def test_wave_values_and_spatial_equations(activation: str) -> None:
    x = torch.tensor([[0.1, -0.4], [-0.2, 0.7]], dtype=torch.float64, requires_grad=True)
    actual = ridge_solution(x, _wave_pack(), 0.13, activation=activation)
    z = 2 * x[:, 0] + x[:, 1] + 0.13
    if activation == "gaussian":
        base = torch.exp(-z.square() / 2)
    elif activation == "softplus":
        base = torch.nn.functional.softplus(z)
    else:
        base = getattr(torch, activation)(z)
    derivative = torch.autograd.grad(base.sum(), z, create_graph=True)[0]
    expected = torch.stack((base, 2 * derivative, derivative), dim=-1)
    torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)
    jac = torch.stack([
        torch.autograd.grad(actual[:, c].sum(), x, retain_graph=True)[0] for c in range(3)
    ], dim=1)
    residual = torch.stack((actual[:, 1] - jac[:, 0, 0], actual[:, 2] - jac[:, 0, 1],
                            jac[:, 1, 0] - 4 * jac[:, 2, 1]), dim=-1)
    torch.testing.assert_close(residual, torch.zeros_like(residual), rtol=0, atol=1e-12)


@pytest.mark.parametrize("activation", ["tanh", "sin", "exp"])
def test_trainable_bias_and_scalar_readout(activation: str) -> None:
    x = torch.tensor([[0.1, -0.4], [-0.2, 0.7]], dtype=torch.float64)
    bias = torch.tensor([0.13, -0.17], dtype=x.dtype, requires_grad=True)
    weights = torch.tensor([0.7, -0.4], dtype=x.dtype, requires_grad=True)
    actual = sum(weights[i] * ridge_solution(x, _wave_pack(sign), bias[i], activation=activation)
                 for i, sign in enumerate((1, -1)))
    expected = torch.zeros_like(actual)
    for i, sign in enumerate((1, -1)):
        z = 2 * sign * x[:, 0] + x[:, 1] + bias[i]
        base = getattr(torch, activation)(z)
        derivative = torch.autograd.grad(base.sum(), z, create_graph=True)[0]
        expected = expected + weights[i] * torch.stack((base, 2 * sign * derivative, derivative), dim=-1)
    actual_grad = torch.autograd.grad(actual.square().sum(), (bias, weights))
    expected_grad = torch.autograd.grad(expected.square().sum(), (bias, weights))
    for value, reference in zip(actual_grad, expected_grad, strict=True):
        assert torch.isfinite(value).all()
        assert torch.count_nonzero(value) == value.numel()
        torch.testing.assert_close(value, reference, rtol=1e-11, atol=1e-12)


def test_third_derivative_uses_raw_coefficients() -> None:
    system = RidgeSystem.from_operators(
        [[OperatorSpec.mixed((3,)), OperatorSpec.mixed((0,))]], (Fraction(3, 2),)
    )
    pack, = system.kernel_basis(max_degree=3)
    x = torch.tensor([[-0.2], [0.1], [0.4]], dtype=torch.float64, requires_grad=True)
    result = ridge_solution(x, pack, 0.1, activation="sin")
    z = 1.5 * x[:, 0] + 0.1
    torch.testing.assert_close(result, torch.stack((-8 / 27 * torch.sin(z), -torch.cos(z)), dim=-1))
    derivative = result[:, 0]
    for _ in range(3):
        derivative = torch.autograd.grad(derivative.sum(), x, create_graph=True)[0][:, 0]
    torch.testing.assert_close(derivative + result[:, 1], torch.zeros_like(derivative), rtol=0, atol=1e-12)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_shapes_dtype_and_scalar_batch(dtype: torch.dtype) -> None:
    pack = _wave_pack()
    x = torch.zeros((2, 3, 2), dtype=dtype)
    result = ridge_solution(x, pack)
    assert result.shape == (2, 3, 3) and result.dtype == dtype and result.device == x.device
    assert ridge_solution(x[0, 0], pack).shape == (3,)
    assert ridge_solution(x[:0], pack).shape == (0, 3, 3)


def test_invalid_inputs_and_nonsmooth_profiles() -> None:
    pack = _wave_pack()
    for x in (torch.zeros(()), torch.zeros(3)):
        with pytest.raises(ValueError, match="shape"):
            ridge_solution(x, pack)
    with pytest.raises(TypeError, match="real floating"):
        ridge_solution(torch.zeros(2, dtype=torch.int64), pack)
    with pytest.raises(ValueError, match="scalar"):
        ridge_solution(torch.zeros(2, 2), pack, torch.zeros(2))
    for activation in ("relu", "elu", "abs", "softsign"):
        with pytest.raises(ValueError, match="globally smooth"):
            ridge_solution(torch.zeros(2), pack, activation=activation)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_device_and_parameter_cast_preserve_gradients() -> None:
    x = torch.ones(4, 2, dtype=torch.float32, device="cuda")
    bias = torch.tensor(0.2, dtype=torch.float64, requires_grad=True)
    value = ridge_solution(x, _wave_pack(), bias)
    assert value.device == x.device and value.dtype == x.dtype
    value.square().sum().backward()
    assert bias.grad is not None and torch.isfinite(bias.grad)
