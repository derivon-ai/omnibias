# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Aging coefficients, analytical tails, trainability and refusal behavior."""

from fractions import Fraction
from typing import Any

import numpy as np
import pytest
import torch
from omnibias.torch.gaussian_aging import gaussian_heat_inverse_aging
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
        [0.2, -0.15], [[0.1, -0.2], [0.8, 0.3], [-0.2, 0.6]], [0.2, -0.1, 0.3], 0.8, 1.3))


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_coefficients_match_independent_rational_picard_fixture_through_order_eight(dtype):
    # Obtained by whole-series Picard substitution, with exponential powers
    # and geometric reciprocal. This does not use the production recurrences.
    expected = [Fraction(0), Fraction(-5, 216), Fraction(-245, 124416),
                Fraction(-4655, 35831808), Fraction(-257555, 41278242816),
                Fraction(-237013, 3962711310336), Fraction(76827149, 2282521714753536),
                Fraction(24734280845, 4601563776943128576),
                Fraction(2483951492605, 4543715546604414959616)]
    result = gaussian_heat_inverse_aging(torch.zeros(1, dtype=dtype), [[0.0], [1.0], [3.0]], [0.0] * 3, time=8)
    tolerance = 2e-4 if dtype == torch.float32 else 2e-12
    np.testing.assert_allclose(result.correction_coefficients[:, 0], [float(v) for v in expected], rtol=tolerance, atol=0)
    torch.testing.assert_close(result.value, result.value.new_tensor([float(Fraction(4, 3) + sum(expected))]))
    assert bool(result.valid) and result.correction_coefficients.dtype == dtype
    q = Fraction(9, 64)
    assert result.approximation_error_bound.item() == pytest.approx(float(Fraction(3, 2) * (3 * q)**9 / (1 - 3 * q)), rel=tolerance)


@pytest.mark.parametrize("order", [0, 1])
def test_low_orders_reproduce_established_predictors_and_tighter_bounds(order):
    args = _parameters()
    result = gaussian_heat_inverse_aging(*args, order=order)
    previous = gaussian_heat_inverse_predictor(*args, order=order)
    torch.testing.assert_close(result.value, previous.value)
    torch.testing.assert_close(result.approximation_error_bound, previous.approximation_error_bound)
    assert result.correction_coefficients.shape == (order + 1, 2)
    assert torch.equal(result.correction_coefficients[0], torch.zeros(2, dtype=torch.float64))


def test_first_two_corrections_match_independent_moment_formula():
    args = tuple(value.detach() for value in _parameters())
    target, centers, biases, nu, age = (value.numpy() for value in args)
    weights = np.exp(biases + centers @ target / (2 * nu))
    p = weights / weights.sum()
    centered = centers - p @ centers
    cost = np.sum(centered**2, axis=-1) / (4 * nu * age)
    first = -np.sum((p * cost)[:, None] * centered, axis=0)
    covariance = centered.T @ (p[:, None] * centered)
    second = covariance @ first / (2 * nu * age) + np.sum((p * cost**2 / 2)[:, None] * centered, axis=0) + (p @ cost) * first
    result = gaussian_heat_inverse_aging(*args, order=2)
    np.testing.assert_allclose(result.correction_coefficients[1:], [first, second], rtol=3e-13, atol=1e-16)


@pytest.mark.parametrize("order", [0, 1, 4])
def test_all_parameter_and_coefficient_gradcheck_and_gradgradcheck(order):
    args = _parameters()
    def function(*values):
        result = gaussian_heat_inverse_aging(*values, order=order)
        return torch.cat((result.value, result.correction_coefficients.flatten()))
    assert torch.autograd.gradcheck(function, args, rtol=3e-5, atol=2e-6)
    assert torch.autograd.gradgradcheck(function, args, rtol=4e-5, atol=3e-6)
    result = gaussian_heat_inverse_aging(*args, order=order)
    assert not result.approximation_error_bound.requires_grad and not result.contraction_bound.requires_grad


@pytest.mark.parametrize("order", [2, 4, 8])
def test_true_inverse_and_scaled_series_respect_analytical_tail(order):
    mp = pytest.importorskip("mpmath")
    target = torch.tensor([0.125], dtype=torch.float64)
    centers, biases, age = [-0.5, 0.25, 0.75], [0.125, -0.25, 0.375], 2.0
    result = gaussian_heat_inverse_aging(target, [[a] for a in centers], biases, time=age, order=order)
    with mp.workdps(70):
        p = [mp.exp(mp.mpf(b) + a * mp.mpf(0.125) / 2) for a, b in zip(centers, biases, strict=True)]
        mean = sum(a * w for a, w in zip(centers, p, strict=True)) / sum(p)
        for scale in (mp.mpf(1), mp.mpf('0.5')):
            elapsed = age / scale
            def residual(x, elapsed=elapsed):
                weights = [mp.exp(mp.mpf(b) - (x - a)**2 / (4 * elapsed)) for a, b in zip(centers, biases, strict=True)]
                return x - elapsed * mp.mpf(0.125) - sum(a * w for a, w in zip(centers, weights, strict=True)) / sum(weights)
            root = mp.findroot(residual, elapsed * mp.mpf(0.125))
            estimate = elapsed * mp.mpf(0.125) + mean + sum(mp.mpf(float(c)) * scale**n for n, c in enumerate(result.correction_coefficients[:, 0]))
            q = mp.mpf(1.25)**2 / (8 * elapsed)
            tail = mp.mpf(1.25) / 2 * (3 * q)**(order + 1) / (1 - 3 * q)
            assert abs(estimate - root) < tail


def test_gauges_shapes_affine_and_underflowed_posterior():
    target = torch.full((2, 3, 2), 0.25, dtype=torch.float64, requires_grad=True)
    center = torch.tensor([[2.0, -1.0]], dtype=target.dtype, requires_grad=True)
    result = gaussian_heat_inverse_aging(target, center.repeat(3, 1), [0.1, -0.2, 0.3], time=1.5)
    torch.testing.assert_close(result.value, 1.5 * target + center)
    assert result.correction_coefficients.shape == (9, 2, 3, 2) and not result.correction_coefficients.any()
    assert result.valid.all() and result.contraction_bound == 0 and not result.approximation_error_bound.any()
    gradients = torch.autograd.grad(result.value.sum(), (target, center))
    assert all(torch.isfinite(value).all() for value in gradients)
    empty = gaussian_heat_inverse_aging(target[:0], center, [0.0])
    assert empty.value.shape == (0, 3, 2) and empty.correction_coefficients.shape == (9, 0, 3, 2)
    query, centers, _, nu, age = _parameters()
    base = gaussian_heat_inverse_aging(query, centers, [0.0] * 3, nu, age)
    shifted = gaussian_heat_inverse_aging(query, centers + centers.new_tensor([8.0, -4.0]), [1e100] * 3, nu, age)
    torch.testing.assert_close(shifted.value, base.value + centers.new_tensor([8.0, -4.0]), rtol=2e-13, atol=2e-13)
    torch.testing.assert_close(shifted.correction_coefficients, base.correction_coefficients, rtol=2e-11, atol=1e-14)
    biases = torch.tensor([0.0, -1000.0, -2000.0], dtype=query.dtype, requires_grad=True)
    saturated = gaussian_heat_inverse_aging(query, centers, biases, nu, age)
    first = torch.autograd.grad(saturated.value.square().sum(), (query, biases), create_graph=True)
    second = torch.autograd.grad(sum(v.sum() for v in first), (query, biases))
    assert all(torch.isfinite(value).all() for value in (*first, *second))


def test_high_dimension_allocates_no_square_matrix_or_factorization(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("aging recurrence must not factor or solve a matrix")
    for name in ("solve", "solve_ex", "cholesky_ex"):
        monkeypatch.setattr(torch.linalg, name, forbidden)
    dimension = 1000
    target = torch.full((2, dimension), 0.01, dtype=torch.float64, requires_grad=True)
    centers = torch.zeros((3, dimension), dtype=target.dtype)
    centers[1, 0], centers[2, 1] = 0.2, -0.3
    class NoSquare(TorchDispatchMode):
        def __torch_dispatch__(self, function: Any, types: Any, args: Any = (), kwargs: Any = None) -> Any:
            result = function(*args, **({} if kwargs is None else kwargs))
            values, _ = tree_flatten(result)
            assert all(not isinstance(v, torch.Tensor) or v.ndim < 2 or v.shape[-2:] != (dimension, dimension) for v in values)
            return result
    with NoSquare():
        result = gaussian_heat_inverse_aging(target, centers, [0.1, -0.2, 0.3])
        gradient = torch.autograd.grad(result.value.sum(), target)[0]
    assert result.valid.all() and torch.isfinite(gradient).all()


@pytest.mark.parametrize("order", [0, 1, 8])
def test_strict_convergence_cutoff_and_invalid_batch_mask(order):
    result = gaussian_heat_inverse_aging(torch.tensor([[0.2], [float("nan")]], dtype=torch.float64), [[-0.2], [0.3]], [0.1, 0.2], order=order)
    assert result.valid.tolist() == [True, False]
    assert torch.isnan(result.correction_coefficients[:, 1]).all() and torch.isnan(result.value[1]).all()
    for age in (0.375, 0.3):  # q exactly 1/3 and q between 1/3 and 1.
        refused = gaussian_heat_inverse_aging(torch.zeros(1, dtype=torch.float64), [[0.0], [1.0]], [0.0, 0.0], time=age, order=order)
        assert not refused.valid and torch.isnan(refused.value).all() and torch.isnan(refused.correction_coefficients).all()
        assert torch.isnan(refused.approximation_error_bound) and torch.isfinite(refused.contraction_bound)


@pytest.mark.parametrize("field,value", [("viscosity", 0.0), ("time", -1.0), ("time", float("inf")),
                                         ("centers", [[float("nan")], [0.0]]), ("biases", [float("inf"), 0.0])])
def test_invalid_shared_parameters_produce_nan_diagnostics(field, value):
    args = dict(target=torch.zeros((2, 1), dtype=torch.float64), centers=[[-0.2], [0.3]], biases=[0.1, 0.2])
    args[field] = value
    result = gaussian_heat_inverse_aging(**args)
    assert not result.valid.any() and torch.isnan(result.value).all() and torch.isnan(result.correction_coefficients).all()
    assert torch.isnan(result.approximation_error_bound).all() and torch.isnan(result.contraction_bound)


@pytest.mark.parametrize("order", [-1, True, 0.5, "1"])
def test_invalid_order_rejected(order):
    with pytest.raises((TypeError, ValueError), match="order"):
        gaussian_heat_inverse_aging(torch.zeros(1), [[0.0]], [0.0], order=order)


def test_malformed_inputs_and_numerical_overflow_refused():
    for target in (torch.ones(1, dtype=torch.int64), torch.ones(1, dtype=torch.float16)):
        with pytest.raises(TypeError):
            gaussian_heat_inverse_aging(target, [[0.0]], [0.0])
    with pytest.raises(ValueError):
        gaussian_heat_inverse_aging(torch.zeros(2), [[0.0]], [0.0])
    with pytest.raises(TypeError):
        gaussian_heat_inverse_aging(torch.zeros(1), [[0.0]], [0.0], viscosity=True)
    for target, centers, age in ((1e308, [[0.0], [2.0]], 10.0), (0.0, [[0.0], [1.0]], 3e307)):
        result = gaussian_heat_inverse_aging(torch.tensor([target], dtype=torch.float64), centers, [0.0, 0.0], time=age)
        assert not result.valid and torch.isnan(result.value).all() and torch.isnan(result.approximation_error_bound)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_cuda_preserves_device_and_gradients():
    target = torch.tensor([0.2], dtype=torch.float64, device="cuda", requires_grad=True)
    result = gaussian_heat_inverse_aging(target, [[-0.2], [0.3]], [0.1, -0.2])
    assert result.value.device == result.correction_coefficients.device == target.device
    assert torch.isfinite(torch.autograd.grad(result.value.sum(), target)[0]).all()
