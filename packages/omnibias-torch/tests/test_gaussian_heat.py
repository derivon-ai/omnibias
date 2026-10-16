# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Independent inverses, implicit derivatives and explicit numerical failures."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import pytest
import torch
from omnibias.torch.gaussian_heat import gaussian_heat_inverse
from torch import Tensor
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils._pytree import tree_flatten


@pytest.fixture(scope="module", autouse=True)
def _one_thread() -> Iterator[None]:
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def _parameters() -> tuple[Tensor, ...]:
    return tuple(torch.tensor(value, dtype=torch.float64, requires_grad=True) for value in (
        [[0.2, -0.1], [0.4, 0.3]], [[-0.3, 0.2], [0.2, -0.1], [0.1, 0.25]], [0.1, -0.2, 0.3], 1.2, 0.8))


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_symmetric_scalar_inverse_matches_independent_high_precision_root(dtype: torch.dtype) -> None:
    mp = pytest.importorskip("mpmath")
    target = torch.tensor([[0.25], [-0.3]], dtype=dtype)
    result = gaussian_heat_inverse(target, [[-1.0], [1.0]], [0.0, 0.0], time=2.0)
    assert result.value.shape == target.shape
    assert result.converged.shape == result.residual_bound.shape == (2,)
    assert result.contraction_bound.shape == ()
    assert result.value.dtype == result.residual_bound.dtype == result.contraction_bound.dtype == dtype
    assert result.converged.dtype == torch.bool
    assert result.converged.all()
    assert result.contraction_bound == 0.25
    with mp.workdps(70):
        expected = [float(mp.findroot(lambda x, goal=float(value): (x - mp.tanh(x / 4)) / 2 - goal, 2 * float(value)))
                    for value in target[:, 0]]
    torch.testing.assert_close(result.value[:, 0], target.new_tensor(expected), rtol=2e-6 if dtype == torch.float32 else 2e-13,
                               atol=2e-6 if dtype == torch.float32 else 2e-13)
    assert not result.residual_bound.requires_grad and not result.contraction_bound.requires_grad


@pytest.mark.parametrize("mode", ["dense", "low_rank", "auto"])
def test_all_parameter_gradcheck_and_double_backward(mode: str) -> None:
    parameters = _parameters()

    def function(*values: Tensor) -> Tensor:
        return gaussian_heat_inverse(*values, max_iterations=32, tolerance=1e-13, adjoint=mode).value

    assert torch.autograd.gradcheck(function, parameters, atol=2e-5, rtol=2e-5)
    assert torch.autograd.gradgradcheck(function, parameters, atol=3e-5, rtol=3e-5)


def test_gradients_match_independent_gaussian_root_finite_differences() -> None:
    mp = pytest.importorskip("mpmath")
    data = [0.23, -0.4, 0.2, 0.55, 0.1, -0.2, 0.3, 1.1, 0.9]
    target = torch.tensor([data[0]], dtype=torch.float64, requires_grad=True)
    centers = torch.tensor(data[1:4], dtype=torch.float64, requires_grad=True)
    biases = torch.tensor(data[4:7], dtype=torch.float64, requires_grad=True)
    nu, time = (torch.tensor(value, dtype=torch.float64, requires_grad=True) for value in data[7:])
    result = gaussian_heat_inverse(target, centers[:, None], biases, nu, time, tolerance=1e-13)
    actual = torch.autograd.grad(result.value.sum(), (target, centers, biases, nu, time))
    flat_actual = torch.cat([value.reshape(-1) for value in actual])

    def root(values: list[Any]) -> Any:
        goal, locations, offsets, viscosity, smoothing_time = values[0], values[1:4], values[4:7], values[7], values[8]

        def residual(x: Any) -> Any:
            weights = [mp.exp(bias - (x - center) ** 2 / (4 * viscosity * smoothing_time))
                       for center, bias in zip(locations, offsets, strict=True)]
            mean = mp.fsum(weight * center for weight, center in zip(weights, locations, strict=True)) / mp.fsum(weights)
            return x - smoothing_time * goal - mean

        return mp.findroot(residual, (0, 1))

    with mp.workdps(70):
        values = [mp.mpf(value) for value in data]
        step = mp.mpf("1e-6")
        expected = []
        for index in range(len(values)):
            left, right = values.copy(), values.copy()
            left[index] -= step
            right[index] += step
            expected.append(float((root(right) - root(left)) / (2 * step)))
    torch.testing.assert_close(flat_actual, target.new_tensor(expected), rtol=3e-10, atol=3e-11)


def test_bias_gauge_and_simultaneous_center_translation_gradient_identities() -> None:
    parameters = _parameters()
    result = gaussian_heat_inverse(*parameters, tolerance=1e-13)
    cotangent = result.value.new_tensor([[0.7, -0.2], [-0.3, 0.5]])
    gradients = torch.autograd.grad((result.value * cotangent).sum(), parameters)
    torch.testing.assert_close(gradients[2].sum(), result.value.new_tensor(0.0), rtol=0, atol=3e-15)
    torch.testing.assert_close(gradients[1].sum(0), cotangent.sum(0), rtol=3e-14, atol=3e-14)
    assert all(torch.isfinite(value).all() for value in gradients)


def test_dense_lowrank_and_auto_match_values_and_all_parameter_gradients() -> None:
    parameters = _parameters()
    results, gradients = [], []
    for mode in ("dense", "low_rank", "auto"):
        result = gaussian_heat_inverse(*parameters, tolerance=1e-13, adjoint=mode)
        results.append(result.value)
        gradients.append(torch.autograd.grad(result.value.square().sum(), parameters))
    for value, gradient in zip(results[1:], gradients[1:], strict=True):
        torch.testing.assert_close(value, results[0], rtol=0, atol=0)
        for actual, expected in zip(gradient, gradients[0], strict=True):
            torch.testing.assert_close(actual, expected, rtol=2e-13, atol=2e-13)


def test_backward_storage_is_independent_of_forward_iteration_count() -> None:
    parameters = _parameters()
    counts = []
    for iterations in (16, 128):
        stored: list[int] = []

        def pack(value: Tensor, storage: list[int] = stored) -> Tensor:
            storage.append(value.numel())
            return value

        with torch.autograd.graph.saved_tensors_hooks(pack, lambda value: value):
            result = gaussian_heat_inverse(*parameters, max_iterations=iterations)
        assert result.converged.all()
        counts.append(stored)
    assert counts[0] == counts[1]


@pytest.mark.parametrize("mode", ["low_rank", "auto"])
def test_high_dimension_small_mode_adjoint_never_materializes_dimension_square(mode: str) -> None:
    dimension = 5000
    target = torch.full((dimension,), 0.01, dtype=torch.float64, requires_grad=True)
    centers = torch.zeros((3, dimension), dtype=target.dtype, requires_grad=True)
    with torch.no_grad():
        centers[1, 0], centers[2, 1] = 0.2, -0.3
    biases = torch.tensor([0.2, -0.1, 0.1], dtype=target.dtype, requires_grad=True)

    class NoSquare(TorchDispatchMode):
        def __torch_dispatch__(self, function: Any, types: Any, args: Any = (), kwargs: Any = None) -> Any:
            result = function(*args, **({} if kwargs is None else kwargs))
            tensors, _ = tree_flatten(result)
            assert all(not isinstance(value, Tensor) or value.ndim < 2 or value.shape[-2:] != (dimension, dimension)
                       for value in tensors)
            return result

    with NoSquare():
        result = gaussian_heat_inverse(target, centers, biases, max_iterations=16, adjoint=mode)
        gradients = torch.autograd.grad(result.value.sum(), (target, centers, biases), create_graph=True)
    assert result.converged
    assert all(torch.isfinite(value).all() for value in gradients)


def test_batch_shapes_empty_batch_and_scalar_shared_parameter_gradients() -> None:
    target = torch.full((2, 3, 1), 0.2, dtype=torch.float64, requires_grad=True)
    nu = torch.tensor(1.2, dtype=target.dtype, requires_grad=True)
    time = torch.tensor(0.8, dtype=target.dtype, requires_grad=True)
    result = gaussian_heat_inverse(target, [[-0.3], [0.3]], [0.1, -0.2], nu, time)
    assert result.value.shape == (2, 3, 1)
    assert result.converged.shape == result.residual_bound.shape == (2, 3)
    derivatives = torch.autograd.grad(result.value.sum(), (target, nu, time))
    assert derivatives[0].shape == target.shape
    assert derivatives[1].shape == derivatives[2].shape == ()
    empty = gaussian_heat_inverse(target[:0], [[-0.3], [0.3]], [0.1, -0.2], nu, time)
    assert empty.value.shape == (0, 3, 1)
    assert empty.converged.shape == empty.residual_bound.shape == (0, 3)


def test_one_center_and_duplicate_centers_are_trainable() -> None:
    target = torch.tensor([[0.2, -0.3]], dtype=torch.float64, requires_grad=True)
    center = torch.tensor([[1.0, 2.0]], dtype=target.dtype, requires_grad=True)
    result = gaussian_heat_inverse(target, center, [100.0], time=2.0)
    torch.testing.assert_close(result.value, center + 2 * target)
    assert result.converged.all() and result.contraction_bound == 0
    target_gradient, center_gradient = torch.autograd.grad(result.value.sum(), (target, center))
    torch.testing.assert_close(target_gradient, torch.full_like(target, 2.0))
    torch.testing.assert_close(center_gradient, torch.ones_like(center))
    duplicate = gaussian_heat_inverse(target, center.detach().repeat(3, 1), [0.1, 0.2, -0.5], time=2.0)
    torch.testing.assert_close(duplicate.value, result.value)


def test_dtype_cast_preserves_source_parameter_gradients_and_python_scalar_precision() -> None:
    target = torch.tensor([0.3], dtype=torch.float32, requires_grad=True)
    centers = torch.tensor([[-0.2], [0.3]], dtype=torch.float64, requires_grad=True)
    result = gaussian_heat_inverse(target, centers, [0.1, -0.2])
    gradient = torch.autograd.grad(result.value.sum(), centers)[0]
    assert result.value.dtype == torch.float32 and gradient.dtype == torch.float64
    precise = 1.00000000000001
    value = gaussian_heat_inverse(torch.tensor([1.0], dtype=torch.float64), [[0.0]], [0.0], time=precise).value
    assert value.item() == precise


@pytest.mark.parametrize("field,value", [("viscosity", 0.0), ("viscosity", -1.0), ("time", 0.0),
                                         ("time", float("inf")), ("biases", [float("nan"), 0.0]),
                                         ("centers", [[float("inf")], [0.0]])])
def test_invalid_shared_numerics_return_explicit_failure(field: str, value: Any) -> None:
    arguments = dict(target=torch.tensor([[0.2], [0.3]], dtype=torch.float64), centers=[[-0.2], [0.2]],
                     biases=[0.0, 0.0], viscosity=1.0, time=1.0)
    arguments[field] = value
    result = gaussian_heat_inverse(**arguments)
    assert not result.converged.any()
    assert torch.isnan(result.value).all() and torch.isinf(result.residual_bound).all()
    assert torch.isinf(result.contraction_bound)


def test_noncontractive_and_exhausted_iterations_do_not_return_a_plausible_inverse() -> None:
    target = torch.tensor([0.3], dtype=torch.float64)
    outside = gaussian_heat_inverse(target, [[-2.0], [2.0]], [0.0, 0.0])
    assert not outside.converged and outside.contraction_bound == 2
    assert torch.isnan(outside.value).all() and torch.isinf(outside.residual_bound)
    exhausted = gaussian_heat_inverse(target, [[-1.0], [1.0]], [0.0, 0.0], max_iterations=0, tolerance=1e-14)
    assert not exhausted.converged and torch.isnan(exhausted.value).all()
    assert torch.isfinite(exhausted.residual_bound) and exhausted.residual_bound > 1e-14


def test_scaled_residual_norm_does_not_underflow_before_comparison() -> None:
    result = gaussian_heat_inverse(torch.zeros(1, dtype=torch.float32), [[0.0], [1e-30]], [0.0, 0.0],
                                   max_iterations=0, tolerance=1e-35)
    assert not result.converged and torch.isnan(result.value).all()
    assert result.residual_bound > 1e-31


def test_overflowing_contraction_denominator_does_not_masquerade_as_zero_bound() -> None:
    result = gaussian_heat_inverse(torch.zeros(1, dtype=torch.float64), [[0.0], [1.0]], [0.0, 0.0], time=3e307)
    assert not result.converged and torch.isnan(result.value).all()
    assert torch.isinf(result.contraction_bound) and torch.isinf(result.residual_bound)


def test_invalid_target_batch_can_be_masked_without_poisoning_valid_parameter_gradient() -> None:
    target = torch.tensor([[0.2], [float("nan")]], dtype=torch.float64, requires_grad=True)
    centers = torch.tensor([[-0.3], [0.3]], dtype=target.dtype, requires_grad=True)
    result = gaussian_heat_inverse(target, centers, [0.0, 0.0])
    assert result.converged.tolist() == [True, False]
    assert torch.isnan(result.value[1]).all() and torch.isinf(result.residual_bound[1])
    gradients = torch.autograd.grad(result.value[0].sum(), (target, centers))
    assert all(torch.isfinite(value).all() for value in gradients)
    assert gradients[0][1] == 0


def test_used_failed_value_has_undefined_gradient() -> None:
    target = torch.tensor([0.3], dtype=torch.float64, requires_grad=True)
    result = gaussian_heat_inverse(target, [[-2.0], [2.0]], [0.0, 0.0])
    assert torch.isnan(torch.autograd.grad(result.value.sum(), target)[0]).all()


def test_underflowed_softmax_keeps_higher_gradients_finite() -> None:
    target = torch.tensor([0.2, -0.1, 0.3], dtype=torch.float64, requires_grad=True)
    centers = torch.tensor([[0.0, 0.0, 0.0], [0.2, 0.1, -0.2]], dtype=target.dtype, requires_grad=True)
    biases = torch.tensor([0.0, -1000.0], dtype=target.dtype, requires_grad=True)
    result = gaussian_heat_inverse(target, centers, biases, adjoint="low_rank")
    first = torch.autograd.grad(result.value.square().sum(), (target, centers, biases), create_graph=True)
    second = torch.autograd.grad(sum(value.square().sum() for value in first), (target, centers, biases))
    assert result.converged and all(torch.isfinite(value).all() for value in (*first, *second))


@pytest.mark.parametrize("options", [{"max_iterations": -1}, {"max_iterations": True}, {"adjoint": "bad"},
                                     {"tolerance": 0}, {"tolerance": float("inf")}, {"tolerance": True}])
def test_invalid_static_options_raise(options: dict[str, Any]) -> None:
    with pytest.raises((TypeError, ValueError)):
        gaussian_heat_inverse(torch.zeros(1), [[0.0]], [0.0], **options)


def test_shape_dtype_and_nonreal_type_errors() -> None:
    with pytest.raises(TypeError):
        gaussian_heat_inverse(torch.tensor([1]), [[0.0]], [0.0])
    with pytest.raises(ValueError):
        gaussian_heat_inverse(torch.tensor(0.0), [[0.0]], [0.0])
    for centers, biases in (([], []), ([[0.0, 1.0]], [0.0]), ([[0.0]], [])):
        with pytest.raises(ValueError):
            gaussian_heat_inverse(torch.zeros(1), centers, biases)
    with pytest.raises(ValueError):
        gaussian_heat_inverse(torch.zeros(1), [[0.0]], [0.0], viscosity=torch.ones(1))
    for value in (True, 1j):
        with pytest.raises(TypeError):
            gaussian_heat_inverse(torch.zeros(1), [[0.0]], [0.0], viscosity=value)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_cuda_device_and_all_parameter_gradients() -> None:
    parameters = tuple(value.detach().cuda().requires_grad_() for value in _parameters())
    result = gaussian_heat_inverse(*parameters)
    gradients = torch.autograd.grad(result.value.sum(), parameters)
    assert result.converged.all() and result.value.is_cuda
    assert all(value.is_cuda and torch.isfinite(value).all() for value in gradients)
