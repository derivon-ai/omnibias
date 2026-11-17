# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Explicit predictor formulas, truncation checks, gradients and refusal paths."""

from typing import Any

import numpy as np
import pytest
import torch
from omnibias.torch.gaussian_predictor import gaussian_heat_inverse_predictor
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils._pytree import tree_flatten


@pytest.fixture(scope="module", autouse=True)
def _one_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def _parameters():
    return tuple(torch.tensor(value, dtype=torch.float64, requires_grad=True) for value in (
        [[0.2, -0.15], [-0.1, 0.3]], [[0.1, -0.2], [0.8, 0.3], [-0.2, 0.6]], [0.2, -0.1, 0.3], 0.8, 1.3))


@pytest.mark.parametrize("order", [0, 1])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_values_and_bounds_match_independent_physical_center_formula(order, dtype):
    args = tuple(value.detach().to(dtype) for value in _parameters())
    target, centers, biases, viscosity, time = (value.double().numpy() for value in args)
    logits = target @ centers.T / (2 * viscosity) + biases
    weights = np.exp(logits - logits.max(axis=-1, keepdims=True))
    probabilities = weights / weights.sum(axis=-1, keepdims=True)
    mean = probabilities @ centers
    deviations = centers - mean[:, None, :]
    third = np.sum(probabilities[:, :, None] * deviations * np.sum(deviations**2, axis=-1)[:, :, None], axis=1)
    expected = time * target + mean - (third / (4 * viscosity * time) if order else 0)
    diameter = max(np.linalg.norm(left - right) for left in centers for right in centers)
    q = diameter**2 / (8 * viscosity * time)
    bound = diameter * (q**2 if order else q / 2)
    result = gaussian_heat_inverse_predictor(*args, order=order)
    tolerance = 3e-6 if dtype == torch.float32 else 3e-14
    np.testing.assert_allclose(result.value.numpy(), expected, rtol=tolerance, atol=tolerance)
    np.testing.assert_allclose(result.approximation_error_bound.numpy(), bound, rtol=tolerance)
    np.testing.assert_allclose(result.contraction_bound.numpy(), q, rtol=tolerance)
    assert result.valid.tolist() == [True, True]
    assert result.value.dtype == result.approximation_error_bound.dtype == result.contraction_bound.dtype == dtype


@pytest.mark.parametrize("order", [0, 1])
def test_all_parameter_gradcheck_and_gradgradcheck(order):
    args = _parameters()
    def function(*values):
        return gaussian_heat_inverse_predictor(*values, order=order).value
    assert torch.autograd.gradcheck(function, args, rtol=2e-5, atol=2e-5)
    assert torch.autograd.gradgradcheck(function, args, rtol=3e-5, atol=3e-5)
    result = gaussian_heat_inverse_predictor(*args, order=order)
    assert not result.approximation_error_bound.requires_grad and not result.contraction_bound.requires_grad


@pytest.mark.parametrize("order", [0, 1])
def test_error_is_below_analytical_truncation_formula_for_independent_roots(order):
    mp = pytest.importorskip("mpmath")
    centers, biases = [-0.5, -0.25, 0.75], [0.125, -0.25, 0.375]
    for time in (0.5, 1.0, 2.0, 8.0):
        target = torch.tensor([[-0.3], [0.0], [0.25]], dtype=torch.float64)
        result = gaussian_heat_inverse_predictor(target, [[value] for value in centers], biases, time=time, order=order)
        assert result.valid.all()
        with mp.workdps(70):
            for query, actual, bound in zip(target[:, 0], result.value[:, 0], result.approximation_error_bound, strict=True):
                def equation(x, goal=float(query), age=time):
                    weights = [mp.exp(mp.mpf(bias) - (x - center)**2 / (4 * mp.mpf(age)))
                               for center, bias in zip(centers, biases, strict=True)]
                    return x - age * mp.mpf(goal) - sum(weight * center for weight, center in zip(weights, centers, strict=True)) / sum(weights)
                root = mp.findroot(equation, time * float(query))
                # The observed floating error is checked with ample margin;
                # this does not turn a rounded diagnostic into a certificate.
                assert abs(mp.mpf(float(actual)) - root) < mp.mpf(float(bound))


def test_one_center_duplicate_centers_shapes_and_empty_batches():
    target = torch.full((2, 3, 2), 0.25, dtype=torch.float64, requires_grad=True)
    center = torch.tensor([[2.0, -1.0]], dtype=target.dtype, requires_grad=True)
    for order in (0, 1):
        result = gaussian_heat_inverse_predictor(target, center, [3.0], time=1.5, order=order)
        torch.testing.assert_close(result.value, 1.5 * target + center)
        assert result.valid.shape == result.approximation_error_bound.shape == (2, 3)
        assert result.contraction_bound == 0 and (result.approximation_error_bound == 0).all()
        gradients = torch.autograd.grad(result.value.sum(), (target, center), retain_graph=True)
        assert all(torch.isfinite(value).all() for value in gradients)
        empty = gaussian_heat_inverse_predictor(target[:0], center, [3.0], time=1.5, order=order)
        assert empty.value.shape == (0, 3, 2) and empty.valid.shape == (0, 3)
        duplicates = gaussian_heat_inverse_predictor(target, center.repeat(3, 1), [0.1, -0.2, 0.3], time=1.5, order=order)
        torch.testing.assert_close(duplicates.value, result.value)


def test_gauges_and_gradient_identities():
    args = _parameters()
    result = gaussian_heat_inverse_predictor(*args)
    gradient = torch.autograd.grad(result.value.sum(), args)
    torch.testing.assert_close(gradient[2].sum(), args[0].new_tensor(0.0), rtol=0, atol=1e-14)
    torch.testing.assert_close(gradient[1].sum(0), args[0].new_tensor([2.0, 2.0]), rtol=1e-13, atol=1e-13)
    query, centers = args[0].detach(), args[1].detach()
    ordinary = gaussian_heat_inverse_predictor(query, centers, [0.0, 0.0, 0.0])
    shifted = gaussian_heat_inverse_predictor(query, centers + query.new_tensor([8.0, -4.0]), [1e100] * 3)
    torch.testing.assert_close(shifted.value, ordinary.value + query.new_tensor([8.0, -4.0]), rtol=1e-13, atol=1e-13)


def test_high_dimension_allocates_no_square_matrix_or_factorization(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("a one-pass predictor must not factor or solve a matrix")
    monkeypatch.setattr(torch.linalg, "solve", forbidden)
    monkeypatch.setattr(torch.linalg, "solve_ex", forbidden)
    monkeypatch.setattr(torch.linalg, "cholesky_ex", forbidden)
    dimension = 2000
    target = torch.full((2, dimension), 0.01, dtype=torch.float64, requires_grad=True)
    centers = torch.zeros((3, dimension), dtype=target.dtype)
    centers[1, 0], centers[2, 1] = 0.2, -0.3

    class NoSquare(TorchDispatchMode):
        def __torch_dispatch__(self, function: Any, types: Any, args: Any = (), kwargs: Any = None) -> Any:
            result = function(*args, **({} if kwargs is None else kwargs))
            values, _ = tree_flatten(result)
            assert all(not isinstance(value, torch.Tensor) or value.ndim < 2 or value.shape[-2:] != (dimension, dimension)
                       for value in values)
            return result

    with NoSquare():
        result = gaussian_heat_inverse_predictor(target, centers, [0.1, -0.2, 0.3])
        gradient = torch.autograd.grad(result.value.sum(), target)[0]
    assert result.valid.all() and torch.isfinite(gradient).all()


@pytest.mark.parametrize("field,value", [("viscosity", 0.0), ("time", -1.0), ("time", float("inf")),
                                         ("centers", [[float("nan")], [0.0]]), ("biases", [float("inf"), 0.0])])
def test_invalid_shared_parameters_produce_nan_diagnostics(field, value):
    args = dict(target=torch.zeros((2, 1), dtype=torch.float64), centers=[[-0.2], [0.3]], biases=[0.1, 0.2])
    args[field] = value
    result = gaussian_heat_inverse_predictor(**args)
    assert not result.valid.any() and torch.isnan(result.value).all()
    assert torch.isnan(result.approximation_error_bound).all() and torch.isnan(result.contraction_bound)


def test_invalid_target_and_noncontractive_or_overflowing_geometry():
    result = gaussian_heat_inverse_predictor(torch.tensor([[0.2], [float("nan")]], dtype=torch.float64), [[-0.2], [0.3]], [0.1, 0.2])
    assert result.valid.tolist() == [True, False] and torch.isnan(result.value[1]).all()
    assert torch.isnan(result.approximation_error_bound[1]) and torch.isfinite(result.contraction_bound)
    for centers, time in (([[-2.0], [2.0]], 1.0), ([[0.0], [1.0]], 3e307)):
        refused = gaussian_heat_inverse_predictor(torch.zeros(1, dtype=torch.float64), centers, [0.0, 0.0], time=time)
        assert not refused.valid and torch.isnan(refused.value).all() and torch.isnan(refused.approximation_error_bound)


@pytest.mark.parametrize("order", [-1, 2, True, 0.5, "1"])
def test_invalid_order_is_rejected(order):
    with pytest.raises((TypeError, ValueError), match="order"):
        gaussian_heat_inverse_predictor(torch.zeros(1), [[0.0]], [0.0], order=order)


def test_shape_dtype_scalar_conversion_and_underflowed_softmax():
    for target in (torch.ones(2, dtype=torch.int64), torch.ones(2, dtype=torch.float16)):
        with pytest.raises(TypeError):
            gaussian_heat_inverse_predictor(target, [[0.0, 0.0]], [0.0])
    with pytest.raises(ValueError):
        gaussian_heat_inverse_predictor(torch.zeros(2), [[0.0]], [0.0])
    with pytest.raises(TypeError):
        gaussian_heat_inverse_predictor(torch.zeros(1), [[0.0]], [0.0], viscosity=True)
    target = torch.tensor([0.25], dtype=torch.float64, requires_grad=True)
    biases = torch.tensor([0.0, -1000.0], dtype=torch.float64, requires_grad=True)
    result = gaussian_heat_inverse_predictor(target, [[-0.2], [0.3]], biases)
    gradients = torch.autograd.grad(result.value.square().sum(), (target, biases), create_graph=True)
    second = torch.autograd.grad(sum(value.sum() for value in gradients), (target, biases))
    assert all(torch.isfinite(value).all() for value in (*gradients, *second))
    precise = 1.00000000000001
    result = gaussian_heat_inverse_predictor(torch.ones(1, dtype=torch.float64), [[0.0]], [0.0], time=precise)
    assert result.value.item() == precise


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_cuda_preserves_device_and_gradients():
    target = torch.tensor([0.2], dtype=torch.float64, device="cuda", requires_grad=True)
    result = gaussian_heat_inverse_predictor(target, [[-0.2], [0.3]], [0.1, -0.2])
    assert result.value.device == result.valid.device == target.device
    assert torch.isfinite(torch.autograd.grad(result.value.sum(), target)[0]).all()
