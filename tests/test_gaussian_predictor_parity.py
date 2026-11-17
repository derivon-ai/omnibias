# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""The paired explicit predictors share values and ordinary parameter derivatives."""

from fractions import Fraction as F

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import torch
from omnibias.core.verified.gaussian_heat import GaussianHeatMap
from omnibias.core.verified.gaussian_predictor import certify_gaussian_inverse_predictor
from omnibias.core.verified.interval import Interval
from omnibias.jax.gaussian_predictor import gaussian_heat_inverse_predictor as jax_predict
from omnibias.torch.gaussian_heat import gaussian_heat_inverse
from omnibias.torch.gaussian_predictor import gaussian_heat_inverse_predictor as torch_predict

jax.config.update("jax_enable_x64", True)


@pytest.mark.parametrize("order", [0, 1])
@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_live_values_all_parameter_gradients_and_bound_parity(order, dtype):
    values = (np.array([[0.25, -0.125], [-0.375, 0.5]]), np.array([[0., 0.], [1., 0.], [0., 1.]]),
              np.array([0.125, -0.25, 0.375]), np.array(1.5), np.array(2.0))
    args_t = tuple(torch.tensor(value, dtype=getattr(torch, dtype), requires_grad=True) for value in values)
    args_j = tuple(jnp.asarray(value, dtype=getattr(jnp, dtype)) for value in values)
    result_t = torch_predict(*args_t, order=order)
    result_j = jax.jit(lambda *args: jax_predict(*args, order=order))(*args_j)
    tolerance = 5e-6 if dtype == "float32" else 3e-12
    np.testing.assert_allclose(result_t.value.detach().numpy(), result_j.value, rtol=tolerance, atol=tolerance)
    np.testing.assert_allclose(result_t.approximation_error_bound.numpy(), result_j.approximation_error_bound, rtol=tolerance)
    np.testing.assert_allclose(result_t.contraction_bound.numpy(), result_j.contraction_bound, rtol=tolerance)
    np.testing.assert_array_equal(result_t.valid.numpy(), result_j.valid)
    gradients_t = torch.autograd.grad(result_t.value.square().sum(), args_t)
    gradients_j = jax.jit(jax.grad(lambda *args: jnp.square(jax_predict(*args, order=order).value).sum(),
                                 argnums=(0, 1, 2, 3, 4)))(*args_j)
    for actual, expected in zip(gradients_t, gradients_j, strict=True):
        np.testing.assert_allclose(actual.detach().numpy(), expected, rtol=tolerance, atol=tolerance)


@pytest.mark.parametrize("order", [0, 1])
def test_mixed_viscosity_time_hessian_parity(order):
    target = np.array([0.25, -0.125])
    centers = np.array([[0., 0.], [1., 0.], [0., 1.]])
    biases = np.array([0.125, -0.25, 0.375])
    def torch_loss(parameters):
        return torch_predict(torch.tensor(target), centers, biases, parameters[0], parameters[1], order=order).value.square().sum()
    def jax_loss(parameters):
        return jnp.square(jax_predict(jnp.asarray(target), centers, biases, parameters[0], parameters[1], order=order).value).sum()
    parameters = np.array([1.5, 2.0])
    actual = torch.autograd.functional.hessian(torch_loss, torch.tensor(parameters, requires_grad=True))
    expected = jax.jit(jax.hessian(jax_loss))(jnp.asarray(parameters))
    np.testing.assert_allclose(actual.detach().numpy(), expected, rtol=3e-11, atol=3e-12)


def test_invalid_batch_and_geometry_flags_match():
    for time in (2.0, 0.1, 3e307):
        query = np.array([[0.25], [np.nan]])
        result_t = torch_predict(torch.tensor(query), [[-1.0], [1.0]], [0.0, 0.0], time=time)
        result_j = jax.jit(lambda target, age=time: jax_predict(target, [[-1.0], [1.0]], [0.0, 0.0], time=age))(jnp.asarray(query))
        np.testing.assert_array_equal(result_t.valid.numpy(), result_j.valid)
        np.testing.assert_allclose(result_t.value.numpy(), result_j.value, equal_nan=True, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(result_t.approximation_error_bound.numpy(), result_j.approximation_error_bound, equal_nan=True)


@pytest.mark.parametrize("order", [0, 1])
def test_core_box_certificate_encloses_numerical_predictors_and_true_inverse_samples(order):
    pytest.importorskip("mpmath")
    model = GaussianHeatMap(((0, 0), (1, 0), (0, 1)), (F(1, 8), -F(1, 4), F(3, 8)), viscosity=F(3, 2), time=2)
    box = (Interval(-0.25, 0.375), Interval(-0.5, 0.25))
    certified = certify_gaussian_inverse_predictor(model, box, order=order)
    assert certified is not None
    query = np.array([[x, y] for x in (-0.25, 0.0625, 0.375) for y in (-0.5, -0.125, 0.25)])
    centers, biases = np.asarray(model.centers, dtype=np.float64), np.asarray(model.biases, dtype=np.float64)
    result_t = torch_predict(torch.tensor(query), centers, biases, 1.5, 2.0, order=order)
    result_j = jax.jit(lambda target: jax_predict(target, centers, biases, 1.5, 2.0, order=order))(jnp.asarray(query))
    roots = gaussian_heat_inverse(torch.tensor(query), centers, biases, 1.5, 2.0, tolerance=1e-13)
    assert bool(roots.converged.all())
    for axis in range(2):
        predicted = certified.predictor_enclosure[axis]
        for values in (result_t.value.numpy(), np.asarray(result_j.value)):
            assert np.all(predicted.lo <= values[:, axis]) and np.all(values[:, axis] <= predicted.hi)
        enclosed = certified.enclosure[axis]
        assert np.all(enclosed.lo <= roots.value.numpy()[:, axis]) and np.all(roots.value.numpy()[:, axis] <= enclosed.hi)
    np.testing.assert_allclose(result_t.approximation_error_bound.numpy(), float(certified.approximation_error_bound), rtol=2e-14)
    errors = np.linalg.norm(result_t.value.numpy() - roots.value.numpy(), axis=-1)
    assert np.all(errors < result_t.approximation_error_bound.numpy())


def test_order_one_bound_is_not_always_smaller_and_overflowing_logits_are_refused():
    target = torch.tensor([0.25], dtype=torch.float64)
    zero = torch_predict(target, [[-1.0], [1.0]], [0.0, 0.0], time=0.75, order=0)
    first = torch_predict(target, [[-1.0], [1.0]], [0.0, 0.0], time=0.75, order=1)
    assert zero.valid and first.valid and zero.contraction_bound > 0.5
    assert first.approximation_error_bound > zero.approximation_error_bound
    query = np.array([1e308])
    for order in (0, 1):
        result_t = torch_predict(torch.tensor(query), [[0.0], [2.0]], [0.0, 0.0], time=2.0, order=order)
        result_j = jax_predict(jnp.asarray(query), [[0.0], [2.0]], [0.0, 0.0], time=2.0, order=order)
        assert not bool(result_t.valid) and not bool(result_j.valid)
        assert torch.isnan(result_t.value).all() and bool(jnp.isnan(result_j.value).all())
