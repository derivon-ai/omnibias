# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Analytic PDE and parameter-gradient oracles for annihilator profiles."""

import pytest
import torch
from omnibias.core.operator_spec import OperatorSpec
from omnibias.core.profile_system import ProfilePack, ProfileSystem
from omnibias.torch.profile_system import profile_solution


@pytest.mark.parametrize("equation", ["heat", "laplace"])
def test_compiled_sine_profile_solves_pde(equation: str) -> None:
    operator = (OperatorSpec.polynomial({(1, 0): 1, (0, 2): -1})
                if equation == "heat" else OperatorSpec.laplacian())
    carrier = (-1, 0) if equation == "heat" else (1, 0)
    system = ProfileSystem.from_operators([[operator]], (0, 1), annihilator=(1, 0, 1), carrier=carrier)
    result = system.solve()
    assert result.consistent
    pack = result.homogeneous_basis[0]
    x = torch.tensor([[0.2, -0.4], [0.3, 0.7]], dtype=torch.float64, requires_grad=True)
    value = profile_solution(x, pack, [0.0, 1.0])[:, 0]
    expected = torch.exp(carrier[0] * x[:, 0]) * torch.sin(x[:, 1])
    torch.testing.assert_close(value, expected, rtol=2e-12, atol=2e-12)
    gradient = torch.autograd.grad(value.sum(), x, create_graph=True)[0]
    second_y = torch.autograd.grad(gradient[:, 1].sum(), x, create_graph=True)[0][:, 1]
    if equation == "heat":
        residual = gradient[:, 0] - second_y
    else:
        second_x = torch.autograd.grad(gradient[:, 0].sum(), x, create_graph=True)[0][:, 0]
        residual = second_x + second_y
    torch.testing.assert_close(residual, torch.zeros_like(residual), rtol=0, atol=2e-12)


def test_repeated_root_and_live_initial_derivatives() -> None:
    pack = ProfilePack((1,), (0,), (1, -2, 1), ((1, 0), (0, 1)))
    x = torch.tensor([[-0.4], [0.0], [0.6]], dtype=torch.float64)
    initial = torch.tensor([0.2, 0.7], dtype=x.dtype, requires_grad=True)
    bias = torch.tensor(0.13, dtype=x.dtype, requires_grad=True)
    weight = torch.tensor(0.8, dtype=x.dtype, requires_grad=True)
    actual = weight * profile_solution(x, pack, initial, bias)
    z = x[:, 0] + bias
    polynomial = initial[0] + (initial[1] - initial[0]) * z
    expected = weight * torch.stack((polynomial * torch.exp(z),
                                     (polynomial + initial[1] - initial[0]) * torch.exp(z)), dim=-1)
    torch.testing.assert_close(actual, expected, rtol=2e-12, atol=2e-12)
    actual_grad = torch.autograd.grad(actual.square().sum(), (initial, bias, weight))
    expected_grad = torch.autograd.grad(expected.square().sum(), (initial, bias, weight))
    for value, reference in zip(actual_grad, expected_grad, strict=True):
        assert torch.isfinite(value).all() and torch.count_nonzero(value) == value.numel()
        torch.testing.assert_close(value, reference, rtol=3e-12, atol=3e-12)
    canonical = profile_solution(x, pack, [0.0, 1.0])
    torch.testing.assert_close(canonical[:, 0], x[:, 0] * torch.exp(x[:, 0]), rtol=2e-12, atol=2e-12)


def test_repeated_oscillatory_root_forced_poisson() -> None:
    pack = ProfilePack((0, 1), (1, 0), (1, 0, 2, 0, 1), ((1,), (0,), (0,), (0,)))
    x = torch.tensor([[0.2, 0.0], [-0.2, 0.4], [0.4, -0.3]], dtype=torch.float64, requires_grad=True)
    value = profile_solution(x, pack, [0.0, 0.0, 0.0, 1.0])[:, 0]
    y = x[:, 1]
    expected = torch.exp(x[:, 0]) * (torch.sin(y) - y * torch.cos(y)) / 2
    torch.testing.assert_close(value, expected, rtol=2e-12, atol=2e-12)
    gradient = torch.autograd.grad(value.sum(), x, create_graph=True)[0]
    laplacian = sum(torch.autograd.grad(gradient[:, i].sum(), x, retain_graph=True)[0][:, i]
                    for i in range(2))
    torch.testing.assert_close(laplacian, torch.exp(x[:, 0]) * torch.sin(y), rtol=2e-12, atol=2e-12)
    torch.testing.assert_close(value[0], value.new_tensor(0), rtol=0, atol=1e-14)
    torch.testing.assert_close(gradient[0, 1], value.new_tensor(0), rtol=0, atol=1e-14)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_exponential_profile_shapes_dtype_and_cast_gradients(dtype: torch.dtype) -> None:
    pack = ProfilePack((1,), (0,), (-1, 1), ((1,),))
    x = torch.full((2, 3, 1), 0.2, dtype=dtype)
    initial = torch.tensor([2.0], dtype=torch.float64, requires_grad=True)
    value = profile_solution(x, pack, initial)
    assert value.shape == (2, 3, 1) and value.dtype == dtype and value.device == x.device
    torch.testing.assert_close(value, 2 * torch.exp(x))
    assert profile_solution(x[0, 0], pack, initial).shape == (1,)
    assert profile_solution(x[:0], pack, initial).shape == (0, 3, 1)
    value.sum().backward()
    assert initial.grad is not None and torch.isfinite(initial.grad).all()


def test_invalid_inputs() -> None:
    pack = ProfilePack((1,), (0,), (1, 0, 1), ((1,), (0,)))
    for x in (torch.zeros(()), torch.zeros(2)):
        with pytest.raises(ValueError, match="shape"):
            profile_solution(x, pack, [0.0, 1.0])
    for dtype in (torch.int64, torch.float16, torch.complex64):
        with pytest.raises(TypeError, match="float32 or float64"):
            profile_solution(torch.zeros(1, dtype=dtype), pack, [0.0, 1.0])
    for initial in ([1.0], [[0.0, 1.0]]):
        with pytest.raises(ValueError, match="initial_jet"):
            profile_solution(torch.zeros(1), pack, initial)
    with pytest.raises(ValueError, match="scalar"):
        profile_solution(torch.zeros(1), pack, [0.0, 1.0], torch.zeros(1))
