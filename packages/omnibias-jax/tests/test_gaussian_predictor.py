# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Predictor tracing, explicit gradients, approximation bounds and refusals."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from omnibias.jax.gaussian_predictor import gaussian_heat_inverse_predictor

jax.config.update("jax_enable_x64", True)


def _parameters():
    return tuple(jnp.asarray(value, dtype=jnp.float64) for value in (
        [[0.2, -0.15], [-0.1, 0.3]], [[0.1, -0.2], [0.8, 0.3], [-0.2, 0.6]], [0.2, -0.1, 0.3], 0.8, 1.3))


@pytest.mark.parametrize("order", [0, 1])
@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
def test_jit_vmap_batch_shapes_and_independent_explicit_formula(order, dtype):
    args = tuple(value.astype(dtype) for value in _parameters())
    target, centers, biases, viscosity, time = (np.asarray(value, dtype=np.float64) for value in args)
    logits = target @ centers.T / (2 * viscosity) + biases
    weights = np.exp(logits - logits.max(axis=-1, keepdims=True))
    probabilities = weights / weights.sum(axis=-1, keepdims=True)
    mean = probabilities @ centers
    centered = centers - mean[:, None, :]
    third = np.sum(probabilities[:, :, None] * centered * np.sum(centered**2, axis=-1)[:, :, None], axis=1)
    expected = time * target + mean - (third / (4 * viscosity * time) if order else 0)
    function = jax.jit(lambda *values: gaussian_heat_inverse_predictor(*values, order=order))
    result = function(*args)
    mapped = jax.vmap(lambda query: function(query, *args[1:]))(args[0])
    tolerance = 3e-6 if dtype == jnp.float32 else 3e-14
    np.testing.assert_allclose(result.value, expected, rtol=tolerance, atol=tolerance)
    np.testing.assert_allclose(mapped.value, result.value, rtol=tolerance, atol=tolerance)
    assert result.value.dtype == result.approximation_error_bound.dtype == dtype
    assert result.valid.shape == result.approximation_error_bound.shape == (2,) and result.contraction_bound.shape == ()
    assert bool(result.valid.all())
    empty = function(args[0][:0], *args[1:])
    assert empty.value.shape == (0, 2) and empty.valid.shape == (0,)


@pytest.mark.parametrize("order", [0, 1])
def test_all_parameter_gradients_match_finite_differences_and_higher_derivatives(order):
    args = _parameters()

    def loss(*values):
        return jnp.square(gaussian_heat_inverse_predictor(*values, order=order).value).sum()

    gradients = jax.jit(jax.grad(loss, argnums=(0, 1, 2, 3, 4)))(*args)
    step = 2e-5
    for parameter, gradient in enumerate(gradients):
        expected = np.empty(args[parameter].shape)
        for index in np.ndindex(expected.shape):
            plus, minus = [np.array(value) for value in args], [np.array(value) for value in args]
            plus[parameter][index] += step
            minus[parameter][index] -= step
            expected[index] = (float(loss(*plus)) - float(loss(*minus))) / (2 * step)
        np.testing.assert_allclose(gradient, expected, rtol=3e-7, atol=3e-9)
    hessian = jax.jit(jax.hessian(lambda query: loss(query, *args[1:])))(args[0])
    assert np.isfinite(hessian).all()
    flat = np.asarray(hessian).reshape(4, 4)
    np.testing.assert_allclose(flat, flat.T, rtol=3e-12, atol=3e-12)
    diagnostic_gradient = jax.grad(lambda locations: gaussian_heat_inverse_predictor(
        args[0], locations, *args[2:], order=order).approximation_error_bound.sum())(args[1])
    np.testing.assert_array_equal(diagnostic_gradient, np.zeros_like(args[1]))


@pytest.mark.parametrize("order", [0, 1])
def test_high_precision_scalar_inverse_error_below_truncation_formula(order):
    mp = pytest.importorskip("mpmath")
    target = jnp.asarray([[0.25], [-0.3]], dtype=jnp.float64)
    centers, biases = [-0.5, -0.25, 0.75], [0.125, -0.25, 0.375]
    for time in (0.5, 2.0, 8.0):
        result = gaussian_heat_inverse_predictor(target, [[value] for value in centers], biases, time=time, order=order)
        with mp.workdps(70):
            for query, value, bound in zip(np.asarray(target[:, 0]), np.asarray(result.value[:, 0]),
                                          np.asarray(result.approximation_error_bound), strict=True):
                def residual(x, goal=float(query), age=time):
                    weights = [mp.exp(mp.mpf(bias) - (x - center)**2 / (4 * mp.mpf(age)))
                               for center, bias in zip(centers, biases, strict=True)]
                    return x - age * mp.mpf(goal) - sum(weight * center for weight, center in zip(weights, centers, strict=True)) / sum(weights)
                root = mp.findroot(residual, time * query)
                assert abs(mp.mpf(float(value)) - root) < mp.mpf(float(bound))


def test_one_center_and_underflowed_posterior_have_finite_higher_gradients():
    for order in (0, 1):
        target = jnp.asarray([0.25, -0.3], dtype=jnp.float64)
        result = gaussian_heat_inverse_predictor(target, [[2.0, -1.0]], [3.0], time=1.5, order=order)
        np.testing.assert_allclose(result.value, 1.5 * target + jnp.asarray([2.0, -1.0]))
        assert result.contraction_bound == 0 and result.approximation_error_bound == 0 and bool(result.valid)
    def objective(query, biases):
        return gaussian_heat_inverse_predictor(jnp.reshape(query, (1,)), [[-0.2], [0.3]], biases).value.sum()
    second = jax.jit(jax.hessian(objective, argnums=(0, 1)))(jnp.asarray(0.2), jnp.asarray([0.0, -1000.0]))
    assert all(np.isfinite(value).all() for value in jax.tree_util.tree_leaves(second))


def test_bias_and_translation_gauges():
    target, centers, _, nu, time = _parameters()
    expected = gaussian_heat_inverse_predictor(target, centers, jnp.zeros(3), nu, time)
    shifted = gaussian_heat_inverse_predictor(target, centers + jnp.array([8.0, -4.0]), jnp.full(3, 1e100), nu, time)
    np.testing.assert_allclose(shifted.value, expected.value + jnp.array([8.0, -4.0]), rtol=1e-13, atol=1e-13)


def test_high_dimension_has_no_matrix_factorization_or_dimension_squared_storage():
    dimension = 1000
    target = jnp.full((2, dimension), 0.01, dtype=jnp.float64)
    centers = jnp.zeros((3, dimension)).at[1, 0].set(0.2).at[2, 1].set(-0.3)
    biases = jnp.asarray([0.1, -0.2, 0.3])
    graph = jax.make_jaxpr(gaussian_heat_inverse_predictor)(target, centers, biases)

    def inspect(jaxpr):
        if hasattr(jaxpr, "jaxpr"):
            inspect(jaxpr.jaxpr)
        elif hasattr(jaxpr, "eqns"):
            for equation in jaxpr.eqns:
                assert equation.primitive.name not in ("lu", "cholesky", "triangular_solve", "custom_linear_solve")
                for variable in equation.outvars:
                    shape = getattr(variable.aval, "shape", ())
                    assert len(shape) < 2 or shape[-2:] != (dimension, dimension)
                for value in equation.params.values():
                    if isinstance(value, (tuple, list)):
                        for item in value:
                            inspect(item)
                    else:
                        inspect(value)
    inspect(graph)
    result = jax.jit(gaussian_heat_inverse_predictor)(target, centers, biases)
    assert bool(result.valid.all())


@pytest.mark.parametrize("field,value", [("viscosity", 0.0), ("time", -1.0), ("time", float("inf")),
                                         ("centers", [[float("nan")], [0.0]]), ("biases", [float("inf"), 0.0])])
def test_invalid_shared_parameters_produce_nan_diagnostics(field, value):
    args = dict(target=jnp.zeros((2, 1), dtype=jnp.float64), centers=[[-0.2], [0.3]], biases=[0.1, 0.2])
    args[field] = value
    result = jax.jit(lambda: gaussian_heat_inverse_predictor(**args))()
    assert not bool(result.valid.any()) and bool(jnp.isnan(result.value).all())
    assert bool(jnp.isnan(result.approximation_error_bound).all()) and bool(jnp.isnan(result.contraction_bound))


def test_invalid_target_noncontractive_and_overflowing_geometry():
    result = gaussian_heat_inverse_predictor(jnp.asarray([[0.2], [jnp.nan]]), [[-0.2], [0.3]], [0.1, 0.2])
    np.testing.assert_array_equal(result.valid, [True, False])
    assert bool(jnp.isnan(result.value[1]).all()) and bool(jnp.isnan(result.approximation_error_bound[1]))
    for centers, time in (([[-2.0], [2.0]], 1.0), ([[0.0], [1.0]], 3e307)):
        result = jax.jit(lambda query, locations=centers, age=time: gaussian_heat_inverse_predictor(
            query, locations, [0.0, 0.0], time=age))(jnp.zeros(1))
        assert not bool(result.valid) and bool(jnp.isnan(result.value).all())


@pytest.mark.parametrize("order", [-1, 2, True, 0.5, "1"])
def test_invalid_order_is_rejected(order):
    with pytest.raises((TypeError, ValueError), match="order"):
        gaussian_heat_inverse_predictor(jnp.zeros(1), [[0.0]], [0.0], order=order)


def test_shape_dtype_and_complex_parameters_rejected():
    for target in (jnp.ones(2, dtype=jnp.int64), jnp.ones(2, dtype=jnp.float16)):
        with pytest.raises(TypeError):
            gaussian_heat_inverse_predictor(target, [[0.0, 0.0]], [0.0])
    with pytest.raises(ValueError):
        gaussian_heat_inverse_predictor(jnp.zeros(2), [[0.0]], [0.0])
    with pytest.raises(TypeError):
        gaussian_heat_inverse_predictor(jnp.zeros(1), [[0.0]], [0.0], viscosity=True)
