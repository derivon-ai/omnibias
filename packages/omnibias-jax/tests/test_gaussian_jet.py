# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Forward inverse jets: independent Lagrange inversion, exact replay and AD."""

from __future__ import annotations

from fractions import Fraction

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from omnibias.core.gaussian_jet import gaussian_inverse_series
from omnibias.jax.gaussian_jet import gaussian_heat_inverse_jet

jax.config.update("jax_enable_x64", True)


def _data(order=3):
    path = jnp.array([[0.2, -0.15], [0.3, 0.1], [-0.1, 0.04], [0.01, -0.02]])[:order+1]
    return (path, jnp.array([[0.1, -0.2], [0.8, 0.3], [-0.2, 0.6]]),
            jnp.array([0.2, -0.1, 0.3]), jnp.array(0.8), jnp.array(1.3))


@pytest.mark.parametrize("adjoint", ["auto", "dense", "low_rank"])
def test_eighth_order_coefficients_match_independent_lagrange_inversion(adjoint):
    mp = pytest.importorskip("mpmath")
    target = jnp.zeros((9, 1)).at[1, 0].set(1.0)
    result = jax.jit(lambda c: gaussian_heat_inverse_jet(c, [[-0.5], [0.5]], [0.0, 0.0],
                                                       adjoint=adjoint))(target)
    assert bool(result.base_result.converged)
    # v=f(x)=x−a*tanh(c*x), a=1/2,c=1/4. Lagrange inversion gives
    # [v^n] f^-1(v) = [x^(n−1)] (x/f(x))^n / n. This is independent
    # of the implementation's exponential and quotient recurrences.
    with mp.workdps(100):
        a, c = mp.mpf("0.5"), mp.mpf("0.25")
        expected = [0.0]
        for order in range(1, 9):
            def ratio_power(x, order=order):
                return (1/(1-a*c) if x == 0 else 1/(1-a*mp.tanh(c*x)/x))**order
            expected.append(float(mp.diff(ratio_power, 0, order-1)/(order*mp.factorial(order-1))))
    np.testing.assert_allclose(result.coefficients[:, 0], expected, rtol=3e-12, atol=3e-16)
    np.testing.assert_allclose(result.anchor_target, 0.0, atol=1e-15)


@pytest.mark.parametrize("adjoint", ["dense", "low_rank"])
def test_coupled_eighth_order_jet_matches_exact_rational_reference(adjoint):
    centers = ((0, 0), (1, 0), (0, 1))
    path = ((Fraction(-1, 3), Fraction(-1, 3)), (1, 2)) + ((0, 0),) * 7
    exact = gaussian_inverse_series(centers, (Fraction(1, 3),) * 3, (0, 0), path)
    target = jnp.array([[float(value) for value in row] for row in path])
    result = jax.jit(lambda values: gaussian_heat_inverse_jet(values, centers, [0.0, 0.25, 0.25],
                                                            adjoint=adjoint))(target)
    expected = np.array([[float(value) for value in row] for row in exact.coefficients])
    np.testing.assert_allclose(result.coefficients, expected, rtol=3e-11, atol=2e-14)
    np.testing.assert_allclose(result.coefficients[0], [0.0, 0.0], atol=5e-15)


def test_nonlinear_target_path_uses_ordinary_coefficients_not_raw_derivatives():
    # One center gives the exact affine inverse x=a+s*v, at every order.
    path = jnp.array([[0.2, -0.1], [0.5, 0.2], [0.03, -0.04], [-0.07, 0.09], [0.01, 0.02]])
    result = gaussian_heat_inverse_jet(path, [[2.0, -3.0]], [4.0], time=2.5)
    expected = 2.5*path
    expected = expected.at[0].add(jnp.array([2.0, -3.0]))
    np.testing.assert_allclose(result.coefficients, expected, rtol=2e-14, atol=2e-14)


@pytest.mark.parametrize("adjoint", ["dense", "low_rank"])
def test_factorization_occurs_once_for_all_forward_orders(monkeypatch, adjoint):
    original = jnp.linalg.cholesky
    calls = []

    def counting(matrix, *args, **kwargs):
        calls.append(matrix.shape)
        return original(matrix, *args, **kwargs)

    monkeypatch.setattr(jnp.linalg, "cholesky", counting)
    path, centers, biases, nu, time = _data()
    result = gaussian_heat_inverse_jet(path, centers, biases, nu, time, adjoint=adjoint)
    assert len(calls) == 1 and np.isfinite(result.coefficients).all()
    calls.clear()
    gaussian_heat_inverse_jet(path[:1], centers, biases, nu, time, adjoint=adjoint)
    assert calls == []


def _equations(value):
    if hasattr(value, "jaxpr"):
        yield from _equations(value.jaxpr)
        return
    if hasattr(value, "eqns"):
        for equation in value.eqns:
            yield equation
            for nested in equation.params.values():
                yield from _equations(nested)
    if isinstance(value, (tuple, list)):
        for nested in value:
            yield from _equations(nested)
    if isinstance(value, dict):
        for nested in value.values():
            yield from _equations(nested)


@pytest.mark.parametrize("adjoint", ["auto", "low_rank"])
def test_low_rank_jaxpr_has_one_small_factor_and_no_dimension_squared_array(adjoint):
    dimension, count = 129, 3
    centers = jnp.arange(dimension*count, dtype=jnp.float64).reshape(count, dimension) * 1e-5
    path = jnp.zeros((5, dimension)).at[1].set(0.001)
    def function(coefficients):
        return gaussian_heat_inverse_jet(coefficients, centers, jnp.zeros(count), adjoint=adjoint)
    equations = list(_equations(jax.make_jaxpr(function)(path)))
    assert sum(equation.primitive.name == "cholesky" for equation in equations) == 1
    shapes = [tuple(variable.aval.shape) for equation in equations for variable in equation.outvars
              if hasattr(variable.aval, "shape")]
    assert not any(shape[-2:] == (dimension, dimension) for shape in shapes)
    assert (count, count) in shapes
    assert np.isfinite(jax.jit(function)(path).coefficients).all()


def test_all_target_coefficients_and_parameters_keep_jitted_gradients():
    args = _data()
    weights = jnp.array([[0.2, -0.1], [0.1, 0.3], [-0.2, 0.4], [0.4, 0.1]])

    def objective(*parameters):
        return jnp.sum(weights * gaussian_heat_inverse_jet(*parameters).coefficients)

    derivative = jax.jit(jax.grad(objective, argnums=(0, 1, 2, 3, 4)))(*args)
    step = 2e-5
    for argument, actual in enumerate(derivative):
        expected = np.empty(args[argument].shape)
        for index in np.ndindex(expected.shape):
            plus, minus = [np.array(value) for value in args], [np.array(value) for value in args]
            plus[argument][index] += step
            minus[argument][index] -= step
            expected[index] = (float(objective(*plus)) - float(objective(*minus)))/(2*step)
        assert np.isfinite(actual).all()
        np.testing.assert_allclose(actual, expected, rtol=2e-7, atol=2e-9)
    assert np.count_nonzero(derivative[0]) == derivative[0].size
    np.testing.assert_allclose(jnp.sum(derivative[2]), 0.0, atol=3e-14)


def test_jitted_second_parameter_derivative_through_factor_reuse():
    path, centers, biases, nu, time = _data(order=2)

    def objective(parameters):
        return jnp.sum(gaussian_heat_inverse_jet(path, centers, biases, parameters[0], parameters[1],
                                               adjoint="low_rank").coefficients**2)

    hessian = jax.jit(jax.hessian(objective))(jnp.array([nu, time]))
    assert np.isfinite(hessian).all()
    np.testing.assert_allclose(hessian, hessian.T, rtol=3e-11, atol=3e-11)
    step = 1e-4
    center = jnp.array([nu, time])
    reference = (objective(center+jnp.array([step, step])) - objective(center+jnp.array([step, -step]))
                 - objective(center+jnp.array([-step, step])) + objective(center-jnp.array([step, step])))/(4*step**2)
    np.testing.assert_allclose(hessian[0, 1], reference, rtol=3e-6, atol=1e-7)


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
def test_jit_vmap_batch_shapes_and_order_zero(dtype):
    path, centers, biases, nu, time = [value.astype(dtype) for value in _data()]
    batched = jnp.stack((path, path.at[0].multiply(0.5)), axis=1)
    result = jax.jit(gaussian_heat_inverse_jet)(batched, centers, biases, nu, time)
    mapped = jax.jit(jax.vmap(lambda p: gaussian_heat_inverse_jet(p, centers, biases, nu, time),
                            in_axes=1))(batched)
    assert result.coefficients.shape == batched.shape and result.coefficients.dtype == dtype
    np.testing.assert_allclose(result.coefficients, jnp.swapaxes(mapped.coefficients, 0, 1), rtol=3e-6, atol=3e-6)
    assert result.anchor_target.shape == (2, 2) and result.base_target_residual.shape == (2, 2)
    zero = gaussian_heat_inverse_jet(path[:1], centers, biases, nu, time)
    assert zero.coefficients.shape == (1, 2)
    empty = gaussian_heat_inverse_jet(jnp.empty((3, 0, 2), dtype=dtype), centers, biases, nu, time)
    assert empty.coefficients.shape == (3, 0, 2)


def test_loose_base_solve_reports_actual_anchor_and_expands_about_it():
    path = jnp.array([[0.2], [1.0], [0.0]])
    result = gaussian_heat_inverse_jet(path, [[-0.5], [0.5]], [0.0, 0.0], max_iterations=0, tolerance=1.0)
    assert bool(result.base_result.converged)
    x0 = float(result.coefficients[0, 0])
    anchor = x0 - 0.5*np.tanh(x0/4)
    hessian = 1 - 0.125/(np.cosh(x0/4)**2)
    np.testing.assert_allclose(result.anchor_target, [anchor], rtol=3e-14, atol=3e-14)
    np.testing.assert_allclose(result.base_target_residual, [anchor-0.2], rtol=3e-14, atol=3e-14)
    assert abs(float(result.base_target_residual[0])) > 0.1
    np.testing.assert_allclose(result.coefficients[1, 0], 1/hessian, rtol=3e-14)


def test_refused_base_and_bad_higher_coefficient_propagate_nan():
    path, centers, biases, nu, time = _data()
    refused = gaussian_heat_inverse_jet(path, centers, biases, nu, time, max_iterations=0)
    assert not bool(refused.base_result.converged)
    assert np.isnan(refused.coefficients).all() and np.isnan(refused.anchor_target).all()
    bad = gaussian_heat_inverse_jet(path.at[2, 0].set(jnp.inf), centers, biases, nu, time)
    assert bool(bad.base_result.converged)
    assert np.isfinite(bad.coefficients[:2]).all() and np.isnan(bad.coefficients[2:]).all()


def test_invalid_order_axis_shapes_and_dtypes():
    for values in (jnp.zeros(2), jnp.zeros((0, 2)), jnp.zeros((2, 0))):
        with pytest.raises(ValueError):
            gaussian_heat_inverse_jet(values, [[0.0, 0.0]], [0.0])
    with pytest.raises(TypeError, match="float32 or float64"):
        gaussian_heat_inverse_jet(jnp.zeros((2, 2), dtype=jnp.int32), [[0.0, 0.0]], [0.0])


def test_huge_common_bias_is_removed_before_adding_logit_differences():
    path = jnp.array([[0.3], [1.0], [0.1], [0.0]])
    ordinary = gaussian_heat_inverse_jet(path, [[-0.5], [0.5]], [0.0, 0.0])
    shifted = gaussian_heat_inverse_jet(path, [[-0.5], [0.5]], [1e200, 1e200])
    np.testing.assert_allclose(shifted.coefficients, ordinary.coefficients, rtol=3e-14, atol=3e-14)
    np.testing.assert_allclose(shifted.anchor_target, ordinary.anchor_target, rtol=3e-14, atol=3e-14)
