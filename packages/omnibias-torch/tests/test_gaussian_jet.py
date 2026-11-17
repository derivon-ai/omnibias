# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Exact Taylor references, implicit oracles, trainable jets and failure paths."""

from __future__ import annotations

import math
from collections.abc import Iterator
from fractions import Fraction as F
from typing import Any

import pytest
import torch
from omnibias.core.gaussian_jet import gaussian_inverse_series
from omnibias.torch.gaussian_heat import gaussian_heat_inverse
from omnibias.torch.gaussian_jet import gaussian_heat_inverse_jet
from torch import Tensor
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils._pytree import tree_flatten


@pytest.fixture(scope="module", autouse=True)
def _one_thread() -> Iterator[None]:
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def _product(left: list[F], right: list[F]) -> list[F]:
    return [sum((left[k] * right[n - k] for k in range(n + 1)), F(0)) for n in range(len(left))]


def _scalar_reference(target: list[F]) -> list[F]:
    # Independent compositional reversion of (x-tanh(x/4))/2, using the
    # analytic tanh coefficients, rather than mixture/quotient recurrences.
    forward = {1: F(3, 8), 3: F(1, 384), 5: -F(1, 15360), 7: F(17, 10321920)}
    output = [F(0)] * len(target)
    for n in range(1, len(target)):
        power = [F(1)] + [F(0)] * (len(target) - 1)
        known = F(0)
        for k in range(1, n + 1):
            power = _product(power, output)
            known += forward.get(k, F(0)) * power[n]
        output[n] = (target[n] - known) / forward[1]
    return output


@pytest.mark.parametrize("mode", ["dense", "low_rank", "auto"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_order_eight_matches_independent_exact_scalar_series_reversion(mode: str, dtype: torch.dtype) -> None:
    target = [F(0), F(1), -F(1, 3), F(1, 5), F(0), F(1, 7), -F(1, 11), F(0), F(1, 13)]
    coefficients = torch.tensor([[float(value)] for value in target], dtype=dtype)
    result = gaussian_heat_inverse_jet(coefficients, [[-1.0], [1.0]], [0.0, 0.0], time=2.0, adjoint=mode)
    expected = coefficients.new_tensor([[float(value)] for value in _scalar_reference(target)])
    tolerance = 1e-6 if dtype == torch.float32 else 5e-14
    torch.testing.assert_close(result.coefficients, expected, rtol=tolerance, atol=tolerance)
    assert result.base_result.converged
    assert result.coefficients.dtype == dtype and result.coefficients.shape == coefficients.shape


@pytest.mark.parametrize("mode", ["dense", "low_rank", "auto"])
def test_coupled_order_eight_matches_exact_rational_core_oracle(mode: str) -> None:
    centers = ((0, 0), (1, 0), (0, 1))
    target = ((-F(1, 3), -F(1, 3)), (F(1), F(2))) + ((F(0), F(0)),) * 7
    expected = gaussian_inverse_series(centers, (F(1, 3),) * 3, (0, 0), target)
    data = torch.tensor([[float(value) for value in row] for row in target], dtype=torch.float64)
    result = gaussian_heat_inverse_jet(data, centers, (0, 0.25, 0.25), adjoint=mode)
    torch.testing.assert_close(result.coefficients, data.new_tensor(expected.coefficients), rtol=1e-12, atol=5e-14)
    torch.testing.assert_close(result.anchor_target, data[0], rtol=0, atol=1e-15)


def test_first_four_coefficients_match_independent_repeated_implicit_backward() -> None:
    target = torch.tensor([0.23], dtype=torch.float64, requires_grad=True)
    centers, biases = [[-0.7], [0.2], [0.5]], [0.2, -0.1, 0.3]
    data = torch.cat((target[None], torch.ones_like(target)[None], torch.zeros((3, 1), dtype=target.dtype)))
    result = gaussian_heat_inverse_jet(data, centers, biases, time=1.3)
    value = gaussian_heat_inverse(target, centers, biases, time=1.3).value
    for order in range(1, 5):
        value = torch.autograd.grad(value.sum(), target, create_graph=True)[0]
        torch.testing.assert_close(result.coefficients[order], value / math.factorial(order), rtol=2e-12, atol=3e-14)


@pytest.mark.parametrize("mode", ["dense", "low_rank", "auto"])
def test_all_parameter_and_target_coefficient_gradcheck_and_gradgradcheck(mode: str) -> None:
    parameters = tuple(torch.tensor(value, dtype=torch.float64, requires_grad=True) for value in (
        [[0.2], [0.7], [-0.3], [0.1]], [[-0.3], [0.4]], [0.2, -0.1], 1.2, 0.8))

    def function(*values: Tensor) -> Tensor:
        return gaussian_heat_inverse_jet(*values, max_iterations=32, adjoint=mode).coefficients

    assert torch.autograd.gradcheck(function, parameters, atol=2e-5, rtol=2e-5)
    assert torch.autograd.gradgradcheck(function, parameters, atol=3e-5, rtol=3e-5)


@pytest.mark.parametrize("mode,size", [("dense", 2), ("low_rank", 3), ("auto", 2)])
def test_exactly_one_factorization_is_reused_for_all_positive_orders(
    monkeypatch: pytest.MonkeyPatch, mode: str, size: int,
) -> None:
    original = torch.linalg.cholesky_ex
    shapes: list[torch.Size] = []

    def recorded(matrix: Tensor, *args: Any, **kwargs: Any) -> Any:
        shapes.append(matrix.shape)
        return original(matrix, *args, **kwargs)

    monkeypatch.setattr(torch.linalg, "cholesky_ex", recorded)
    data = torch.full((9, 4, 2), 0.03, dtype=torch.float64)
    result = gaussian_heat_inverse_jet(data, [[0.0, 0.0], [0.2, 0.0], [0.0, 0.3]], [0.0, 0.1, -0.2], adjoint=mode)
    assert torch.isfinite(result.coefficients).all()
    assert shapes == [torch.Size((4, size, size))]


@pytest.mark.parametrize("mode", ["auto", "low_rank"])
def test_high_dimension_never_materializes_dimension_square(mode: str) -> None:
    dimension = 2000
    data = torch.full((5, dimension), 0.01, dtype=torch.float64, requires_grad=True)
    centers = torch.zeros((3, dimension), dtype=data.dtype, requires_grad=True)
    with torch.no_grad():
        centers[1, 0], centers[2, 1] = 0.2, -0.3
    biases = torch.tensor([0.2, -0.1, 0.1], dtype=data.dtype, requires_grad=True)

    class NoSquare(TorchDispatchMode):
        def __torch_dispatch__(self, function: Any, types: Any, args: Any = (), kwargs: Any = None) -> Any:
            result = function(*args, **({} if kwargs is None else kwargs))
            values, _ = tree_flatten(result)
            assert all(not isinstance(value, Tensor) or value.ndim < 2 or value.shape[-2:] != (dimension, dimension)
                       for value in values)
            return result

    with NoSquare():
        result = gaussian_heat_inverse_jet(data, centers, biases, max_iterations=16, adjoint=mode)
        gradients = torch.autograd.grad(result.coefficients.square().sum(), (data, centers, biases), create_graph=True)
    assert result.base_result.converged
    assert all(torch.isfinite(value).all() for value in gradients)


def test_reported_anchor_is_the_actual_base_image_for_a_loose_solve() -> None:
    data = torch.tensor([[0.3], [1.0], [0.2]], dtype=torch.float64)
    result = gaussian_heat_inverse_jet(data, [[-1.0], [1.0]], [0.0, 0.0], max_iterations=2, tolerance=0.4)
    x0 = result.coefficients[0]
    expected = x0 - torch.tanh(x0 / 2)
    torch.testing.assert_close(result.anchor_target, expected, rtol=1e-14, atol=1e-15)
    torch.testing.assert_close(result.base_target_residual, expected - data[0], rtol=1e-14, atol=1e-15)
    assert result.base_target_residual.abs().item() > 1e-3
    hessian = 1 - (1 - torch.tanh(x0 / 2).square()) / 2
    torch.testing.assert_close(result.coefficients[1], 1 / hessian, rtol=1e-14, atol=1e-14)


def test_order_zero_matches_inverse_without_factorization(monkeypatch: pytest.MonkeyPatch) -> None:
    def refused(*args: Any, **kwargs: Any) -> None:
        pytest.fail("order zero must not factor a matrix")

    monkeypatch.setattr(torch.linalg, "cholesky_ex", refused)
    data = torch.tensor([[[0.2], [-0.3]]], dtype=torch.float64, requires_grad=True)
    result = gaussian_heat_inverse_jet(data, [[-0.3], [0.4]], [0.2, -0.1])
    expected = gaussian_heat_inverse(data[0], [[-0.3], [0.4]], [0.2, -0.1])
    torch.testing.assert_close(result.coefficients[0], expected.value, rtol=0, atol=0)
    assert torch.isfinite(torch.autograd.grad(result.coefficients.sum(), data)[0]).all()


def test_one_center_affine_jet_and_empty_or_multiaxis_batches() -> None:
    data = torch.arange(24, dtype=torch.float64).reshape(4, 2, 3, 1) / 10
    result = gaussian_heat_inverse_jet(data, [[0.7]], [2.0], time=1.3)
    expected = data * 1.3
    expected[0] += 0.7
    torch.testing.assert_close(result.coefficients, expected, rtol=1e-14, atol=1e-15)
    assert result.anchor_target.shape == result.base_target_residual.shape == (2, 3, 1)
    empty = gaussian_heat_inverse_jet(data[:, :0], [[0.7]], [2.0], time=1.3)
    assert empty.coefficients.shape == (4, 0, 3, 1)


def test_dense_and_low_rank_values_and_parameter_gradients_agree() -> None:
    parameters = tuple(torch.tensor(value, dtype=torch.float64, requires_grad=True) for value in (
        [[0.1, -0.2], [0.7, 0.3], [-0.1, 0.2], [0.05, -0.1]],
        [[-0.2, 0.1], [0.4, 0.3], [0.0, -0.1]], [0.1, -0.2, 0.3], 0.8, 1.2))
    dense = gaussian_heat_inverse_jet(*parameters, adjoint="dense")
    low_rank = gaussian_heat_inverse_jet(*parameters, adjoint="low_rank")
    torch.testing.assert_close(dense.coefficients, low_rank.coefficients, rtol=1e-13, atol=1e-14)
    dense_gradient = torch.autograd.grad(dense.coefficients.square().sum(), parameters)
    low_rank_gradient = torch.autograd.grad(low_rank.coefficients.square().sum(), parameters)
    for left, right in zip(dense_gradient, low_rank_gradient, strict=True):
        torch.testing.assert_close(left, right, rtol=2e-12, atol=1e-14)


def test_refused_base_has_nan_coefficients_and_explicit_diagnostics() -> None:
    data = torch.tensor([[0.2], [1.0], [0.0]], dtype=torch.float64)
    for kwargs in ({"viscosity": 0.0}, {"time": 0.1}, {"max_iterations": 0, "tolerance": 1e-15}):
        result = gaussian_heat_inverse_jet(data, [[-1.0], [1.0]], [0.0, 0.0], **kwargs)
        assert not result.base_result.converged
        assert torch.isnan(result.coefficients).all()
        assert torch.isnan(result.anchor_target).all() and torch.isnan(result.base_target_residual).all()


def test_nonfinite_positive_coefficient_fails_only_from_that_order() -> None:
    data = torch.tensor([[0.2], [0.7], [float("nan")], [0.1]], dtype=torch.float64)
    result = gaussian_heat_inverse_jet(data, [[-0.3], [0.4]], [0.2, -0.1])
    assert result.base_result.converged
    assert torch.isfinite(result.coefficients[:2]).all() and torch.isnan(result.coefficients[2:]).all()
    assert torch.isfinite(result.anchor_target).all()


def test_factorization_failure_is_explicit_and_does_not_change_base_diagnostics(monkeypatch: pytest.MonkeyPatch) -> None:
    def refused(matrix: Tensor) -> tuple[Tensor, Tensor]:
        return torch.ones_like(matrix), torch.ones(matrix.shape[:-2], dtype=torch.int32, device=matrix.device)

    monkeypatch.setattr(torch.linalg, "cholesky_ex", refused)
    data = torch.tensor([[0.2], [1.0], [0.1]], dtype=torch.float64)
    result = gaussian_heat_inverse_jet(data, [[-0.3], [0.4]], [0.2, -0.1])
    assert result.base_result.converged and torch.isfinite(result.coefficients[0]).all()
    assert torch.isnan(result.coefficients[1:]).all() and torch.isfinite(result.anchor_target).all()


@pytest.mark.parametrize("data", [torch.ones(2), torch.ones((0, 2)), torch.ones((2, 0))])
def test_invalid_coefficient_shape_is_rejected(data: Tensor) -> None:
    with pytest.raises(ValueError, match="shape"):
        gaussian_heat_inverse_jet(data, [[0.0]], [0.0])


@pytest.mark.parametrize("data", [torch.ones((2, 1), dtype=torch.int64), torch.ones((2, 1), dtype=torch.float16), [[0.0]]])
def test_invalid_coefficient_type_is_rejected(data: Any) -> None:
    with pytest.raises(TypeError, match="float32 or float64"):
        gaussian_heat_inverse_jet(data, [[0.0]], [0.0])


def test_casting_preserves_parameter_source_gradients_and_scalar_precision() -> None:
    data = torch.tensor([[0.2], [1.0]], dtype=torch.float32, requires_grad=True)
    centers = torch.tensor([[-0.3], [0.4]], dtype=torch.float64, requires_grad=True)
    result = gaussian_heat_inverse_jet(data, centers, [0.2, -0.1])
    gradient = torch.autograd.grad(result.coefficients.sum(), centers)[0]
    assert result.coefficients.dtype == torch.float32 and gradient.dtype == torch.float64
    precise = 1.00000000000001
    result64 = gaussian_heat_inverse_jet(torch.ones((2, 1), dtype=torch.float64), [[0.0]], [0.0], time=precise)
    assert result64.coefficients[1].item() == precise


def test_underflowed_posterior_has_finite_parameter_gradients_and_double_backward() -> None:
    data = torch.tensor([[0.2], [0.7], [-0.1]], dtype=torch.float64, requires_grad=True)
    biases = torch.tensor([0.0, -1000.0], dtype=torch.float64, requires_grad=True)
    result = gaussian_heat_inverse_jet(data, [[-0.2], [0.3]], biases, adjoint="low_rank")
    gradients = torch.autograd.grad(result.coefficients.square().sum(), (data, biases), create_graph=True)
    second = torch.autograd.grad(sum(value.sum() for value in gradients), (data, biases))
    assert all(torch.isfinite(value).all() for value in (*gradients, *second))


def test_huge_common_bias_is_removed_before_logits_are_combined() -> None:
    data = torch.tensor([[0.2], [0.7], [-0.1], [0.2]], dtype=torch.float64)
    reference = gaussian_heat_inverse_jet(data, [[-0.2], [0.3]], [0.0, 0.0])
    shifted = gaussian_heat_inverse_jet(data, [[-0.2], [0.3]], [1e200, 1e200])
    torch.testing.assert_close(shifted.coefficients, reference.coefficients, rtol=0, atol=0)
    torch.testing.assert_close(shifted.anchor_target, reference.anchor_target, rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_cuda_preserves_device_and_gradients() -> None:
    data = torch.tensor([[0.2], [0.7], [-0.1]], dtype=torch.float64, device="cuda", requires_grad=True)
    result = gaussian_heat_inverse_jet(data, [[-0.2], [0.3]], [0.1, -0.2])
    assert result.coefficients.device == data.device
    assert torch.isfinite(torch.autograd.grad(result.coefficients.sum(), data)[0]).all()
