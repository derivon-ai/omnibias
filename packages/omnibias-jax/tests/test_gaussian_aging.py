# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Aging series tracing, independent coefficients, gradients and validity."""

from fractions import Fraction

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from omnibias.jax.gaussian_aging import GaussianHeatAgingResult, gaussian_heat_inverse_aging
from omnibias.jax.gaussian_predictor import gaussian_heat_inverse_predictor

jax.config.update("jax_enable_x64", True)


def _parameters():
    return tuple(jnp.asarray(value, dtype=jnp.float64) for value in (
        [0.2, -0.15], [[0.1, -0.2], [0.8, 0.3], [-0.2, 0.6]], [0.2, -0.1, 0.3], 0.8, 1.3))


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
def test_jit_coefficients_match_independent_rational_picard_fixture(dtype):
    # Whole-series Picard substitution using polynomial powers and geometric
    # reciprocal supplies these exact coefficients independently of our loops.
    expected = [Fraction(0), Fraction(-5, 216), Fraction(-245, 124416),
                Fraction(-4655, 35831808), Fraction(-257555, 41278242816),
                Fraction(-237013, 3962711310336), Fraction(76827149, 2282521714753536),
                Fraction(24734280845, 4601563776943128576),
                Fraction(2483951492605, 4543715546604414959616)]
    result = jax.jit(gaussian_heat_inverse_aging)(jnp.zeros(1, dtype=dtype), jnp.asarray([[0.0], [1.0], [3.0]]), jnp.zeros(3), time=8)
    tolerance = 2e-4 if dtype == jnp.float32 else 2e-12
    np.testing.assert_allclose(result.correction_coefficients[:, 0], [float(v) for v in expected], rtol=tolerance, atol=0)
    np.testing.assert_allclose(result.value, [float(Fraction(4, 3) + sum(expected))], rtol=tolerance)
    assert bool(result.valid) and result.correction_coefficients.dtype == dtype
    q = Fraction(9, 64)
    assert float(result.approximation_error_bound) == pytest.approx(float(Fraction(3, 2) * (3 * q)**9 / (1 - 3 * q)), rel=tolerance)


@pytest.mark.parametrize("order", [0, 1])
def test_low_orders_match_predictors_and_bounds(order):
    args = _parameters()
    result = gaussian_heat_inverse_aging(*args, order=order)
    previous = gaussian_heat_inverse_predictor(*args, order=order)
    np.testing.assert_allclose(result.value, previous.value, rtol=3e-14, atol=3e-14)
    np.testing.assert_allclose(result.approximation_error_bound, previous.approximation_error_bound, rtol=3e-14)
    np.testing.assert_array_equal(result.correction_coefficients[0], jnp.zeros(2))


def test_multidimensional_batches_vmap_empty_and_affine():
    target, *shared = _parameters()
    batch = jnp.stack((target, target + 0.2, target - 0.15)).reshape(1, 3, 2)
    function = jax.jit(gaussian_heat_inverse_aging)
    result = function(batch, *shared)
    mapped = jax.vmap(lambda query: function(query, *shared), out_axes=GaussianHeatAgingResult(0, 1, 0, None, 0))(batch[0])
    np.testing.assert_allclose(mapped.value, result.value[0], rtol=3e-13, atol=3e-14)
    np.testing.assert_allclose(mapped.correction_coefficients, result.correction_coefficients[:, 0], rtol=3e-12, atol=3e-14)
    empty = function(batch[:0], *shared)
    assert empty.value.shape == (0, 3, 2) and empty.correction_coefficients.shape == (9, 0, 3, 2)
    affine = function(batch, [[2.0, -1.0]] * 3, [0.1, -0.2, 0.3], time=1.5)
    np.testing.assert_allclose(affine.value, 1.5 * batch + jnp.array([2.0, -1.0]))
    np.testing.assert_array_equal(affine.correction_coefficients, jnp.zeros((9, 1, 3, 2)))
    assert bool(affine.valid.all()) and affine.contraction_bound == 0 and not bool(affine.approximation_error_bound.any())


@pytest.mark.parametrize("order", [0, 1, 4])
def test_all_parameter_gradients_and_higher_derivatives(order):
    args = _parameters()
    def loss(*values):
        result = gaussian_heat_inverse_aging(*values, order=order)
        weights = jnp.arange(order + 1, dtype=values[0].dtype)[:, None]
        return jnp.square(result.value).sum() + (weights * result.correction_coefficients).sum()
    gradients = jax.jit(jax.grad(loss, argnums=(0, 1, 2, 3, 4)))(*args)
    step = 2e-5
    for parameter, gradient in enumerate(gradients):
        expected = np.empty(args[parameter].shape)
        for index in np.ndindex(expected.shape):
            plus, minus = [np.array(v) for v in args], [np.array(v) for v in args]
            plus[parameter][index] += step
            minus[parameter][index] -= step
            expected[index] = (float(loss(*plus)) - float(loss(*minus))) / (2 * step)
        np.testing.assert_allclose(gradient, expected, rtol=3e-7, atol=3e-9)
    hessian = jax.jit(jax.hessian(lambda query: loss(query, *args[1:])))(args[0])
    np.testing.assert_allclose(hessian, hessian.T, rtol=3e-12, atol=3e-12)
    mixed = jax.jit(jax.jacfwd(jax.grad(lambda nu, age: loss(*args[:3], nu, age), argnums=0), argnums=1))(*args[3:])
    assert np.isfinite(mixed).all()
    diagnostic = jax.grad(lambda centers: gaussian_heat_inverse_aging(args[0], centers, *args[2:], order=order).approximation_error_bound.sum())(args[1])
    np.testing.assert_array_equal(diagnostic, jnp.zeros_like(args[1]))


def test_gauges_and_underflowed_posterior_have_finite_higher_derivatives():
    target, centers, _, nu, time = _parameters()
    base = gaussian_heat_inverse_aging(target, centers, jnp.zeros(3), nu, time)
    shifted = gaussian_heat_inverse_aging(target, centers + jnp.array([8.0, -4.0]), jnp.full(3, 1e100), nu, time)
    np.testing.assert_allclose(shifted.value, base.value + jnp.array([8.0, -4.0]), rtol=2e-13, atol=2e-13)
    np.testing.assert_allclose(shifted.correction_coefficients, base.correction_coefficients, rtol=2e-11, atol=1e-14)
    def saturated(query, bias):
        return gaussian_heat_inverse_aging(query, centers, bias, nu, time).value.sum()
    second = jax.jit(jax.hessian(saturated, argnums=(0, 1)))(target, jnp.array([0.0, -1000.0, -2000.0]))
    assert all(np.isfinite(v).all() for v in jax.tree_util.tree_leaves(second))


def test_high_dimension_has_no_factorization_or_square_storage():
    dimension = 1000
    target = jnp.full((2, dimension), 0.01, dtype=jnp.float64)
    centers = jnp.zeros((3, dimension)).at[1, 0].set(0.2).at[2, 1].set(-0.3)
    biases = jnp.asarray([0.1, -0.2, 0.3])
    graph = jax.make_jaxpr(gaussian_heat_inverse_aging)(target, centers, biases)
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
    assert bool(jax.jit(gaussian_heat_inverse_aging)(target, centers, biases).valid.all())


@pytest.mark.parametrize("order", [0, 1, 8])
def test_invalid_batch_and_strict_convergence_cutoff(order):
    result = gaussian_heat_inverse_aging(jnp.array([[0.2], [jnp.nan]]), [[-0.2], [0.3]], [0.1, 0.2], order=order)
    np.testing.assert_array_equal(result.valid, [True, False])
    assert bool(jnp.isnan(result.correction_coefficients[:, 1]).all()) and bool(jnp.isnan(result.value[1]).all())
    for time in (0.375, 0.3):
        refused = gaussian_heat_inverse_aging(jnp.zeros(1), [[0.0], [1.0]], [0.0, 0.0], time=time, order=order)
        assert not bool(refused.valid) and bool(jnp.isnan(refused.value).all()) and bool(jnp.isnan(refused.correction_coefficients).all())
        assert bool(jnp.isnan(refused.approximation_error_bound)) and bool(jnp.isfinite(refused.contraction_bound))


@pytest.mark.parametrize("field,value", [("viscosity", 0.0), ("time", -1.0), ("time", float("inf")),
                                         ("centers", [[float("nan")], [0.0]]), ("biases", [float("inf"), 0.0])])
def test_invalid_shared_parameters_nan_diagnostics(field, value):
    args = dict(target=jnp.zeros((2, 1)), centers=[[-0.2], [0.3]], biases=[0.1, 0.2])
    args[field] = value
    result = gaussian_heat_inverse_aging(**args)
    assert not bool(result.valid.any()) and bool(jnp.isnan(result.value).all()) and bool(jnp.isnan(result.correction_coefficients).all())
    assert bool(jnp.isnan(result.approximation_error_bound).all()) and bool(jnp.isnan(result.contraction_bound))


@pytest.mark.parametrize("order", [-1, True, 0.5, "1"])
def test_invalid_order_rejected(order):
    with pytest.raises((TypeError, ValueError), match="order"):
        gaussian_heat_inverse_aging(jnp.zeros(1), [[0.0]], [0.0], order=order)


def test_malformed_inputs_and_overflow_refused():
    for target in (jnp.ones(1, dtype=jnp.int64), jnp.ones(1, dtype=jnp.float16)):
        with pytest.raises(TypeError):
            gaussian_heat_inverse_aging(target, [[0.0]], [0.0])
    with pytest.raises(ValueError):
        gaussian_heat_inverse_aging(jnp.zeros(2), [[0.0]], [0.0])
    with pytest.raises(TypeError):
        gaussian_heat_inverse_aging(jnp.zeros(1), [[0.0]], [0.0], viscosity=True)
    for target, centers, age in ((1e308, [[0.0], [2.0]], 10.0), (0.0, [[0.0], [1.0]], 3e307)):
        result = gaussian_heat_inverse_aging(jnp.array([target]), centers, [0.0, 0.0], time=age)
        assert not bool(result.valid) and bool(jnp.isnan(result.value).all()) and bool(jnp.isnan(result.approximation_error_bound))
