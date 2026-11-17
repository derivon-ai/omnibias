# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Independent calculus checks for the tanh fast path and its parameter AD.

The mpmath oracle differentiates tanh itself, without using package polynomial
coefficients. The separate legacy comparison checks the unchanged primal and
the unoptimized high-order path; it does not certify high-order accuracy.
"""

from functools import lru_cache
from inspect import signature

import numpy as np
import pytest

mp = pytest.importorskip("mpmath")
jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402
from jax.experimental.jet import jet  # noqa: E402
from omnibias.core.polynomials import tanh_polynomial_coeffs  # noqa: E402
from omnibias.jax._fastpath import tanh_nth_derivative  # noqa: E402
from omnibias.jax.laplacian import neural_field_polylaplacian  # noqa: E402

_JET_HAS_FACTORIAL_SCALING = "factorial_scaled" in signature(jet).parameters


@lru_cache(maxsize=512)
def _derivative(z: float, order: int) -> float:
    with mp.workdps(90):
        return float(mp.diff(mp.tanh, mp.mpf(z), order))


def _legacy(z, order):
    t = jnp.tanh(z)
    if order == 0:
        return t
    coefficients = tanh_polynomial_coeffs(order)
    result = jnp.full_like(t, coefficients[-1])
    for coefficient in reversed(coefficients[:-1]):
        result = result * t + coefficient
    return result


def _assert_oracle(actual, expected, dtype, order, *, gradient=False):
    if dtype == np.float64:
        atol, rtol = (1e-8, 2e-9) if gradient else (1e-9, 2e-10)
    else:
        # A fixed, degree-dependent cancellation allowance. This float32
        # regression tolerance is not an accuracy certificate. In particular,
        # a relative-only test is inappropriate at derivative zeros.
        coefficients = tanh_polynomial_coeffs(order)
        atol = 4 * np.finfo(np.float32).eps * sum(abs(c) for c in coefficients)
        rtol = 4e-5
    np.testing.assert_allclose(actual, expected, atol=atol, rtol=rtol)


@pytest.mark.parametrize("order", range(1, 9))
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_values_jvp_vjp_and_vmap_against_independent_tanh(order, dtype):
    # atanh(1/sqrt(3)) is a root of the third derivative. Both sides,
    # saturation, zero, and the known z=3 conditioning probe are included.
    points = np.asarray(
        [-20, -3, -1, -0.6584789484624084, -0.25, 0, 0.125,
         0.6584789484624084, 1, 3, 20],
        dtype=dtype,
    )
    directions = np.asarray([1, -0.5, 2, -1, 0, 0.75, 1.5, -2, 0.25, 1, -1], dtype=dtype)
    z, tangent = jnp.asarray(points), jnp.asarray(directions)
    def evaluate(value):
        return tanh_nth_derivative(value, order)

    expected = np.asarray([_derivative(float(value), order) for value in points])
    expected_gradient = np.asarray([_derivative(float(value), order + 1) for value in points])

    actual, jvp = jax.jit(lambda value, direction: jax.jvp(evaluate, (value,), (direction,)))(
        z, tangent
    )
    _, pullback = jax.vjp(evaluate, z)
    vjp, = pullback(tangent)
    mapped = jax.jit(jax.vmap(jax.grad(lambda value: evaluate(value))))(z)
    _assert_oracle(actual, expected, dtype, order)
    _assert_oracle(jvp, expected_gradient * directions, dtype, order + 1, gradient=True)
    _assert_oracle(vjp, expected_gradient * directions, dtype, order + 1, gradient=True)
    _assert_oracle(mapped, expected_gradient, dtype, order + 1, gradient=True)


@pytest.mark.parametrize("order", [0, 1, 2, 3, 4, 5, 6, 7, 8, 16, 32])
def test_primal_unchanged_and_unoptimized_high_orders_retained(order):
    points = jnp.asarray([-20.0, -3, -1, -0.25, -0.0, 0.0, 0.125, 1, 3, 20])
    def evaluate(value):
        return tanh_nth_derivative(value, order)

    def legacy(value):
        return _legacy(value, order)

    np.testing.assert_array_equal(evaluate(points), legacy(points))
    np.testing.assert_array_equal(jax.jit(evaluate)(points), jax.jit(legacy)(points))
    if order in (0, 1, 2, 3, 16, 32):
        actual = jax.jit(jax.grad(lambda value: evaluate(value).sum()))(points)
        expected = jax.jit(jax.grad(lambda value: legacy(value).sum()))(points)
        np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("dtype", [jnp.float16, jnp.bfloat16])
@pytest.mark.parametrize("order", [4, 7])
def test_low_precision_gradient_keeps_legacy_path(dtype, order):
    # P_7 has binary16-representable coefficients, but P'_7 contains 80640,
    # which would overflow when converted to a binary16 coefficient constant.
    points = jnp.asarray([-0.25, 0.0, 0.25], dtype=dtype)
    def evaluate(value):
        return tanh_nth_derivative(value, order).sum()

    def legacy(value):
        return _legacy(value, order).sum()

    actual = jax.jit(jax.grad(evaluate))(points)
    expected = jax.jit(jax.grad(legacy))(points)
    np.testing.assert_array_equal(np.asarray(actual, dtype=np.float32), np.asarray(expected, dtype=np.float32))
    assert np.isfinite(np.asarray(actual, dtype=np.float32)[1])


@pytest.mark.parametrize("order", [4, 8])
def test_second_derivatives_remain_differentiable(order):
    points = jnp.asarray([-1.0, -0.2, 0.3, 1.0])
    def loss(value):
        return tanh_nth_derivative(value, order).sum()

    hessian = jax.jit(jax.jacfwd(jax.grad(loss)))(points)
    expected = np.diag([_derivative(float(value), order + 2) for value in points])
    np.testing.assert_allclose(hessian, expected, atol=1e-7, rtol=2e-8)


@pytest.mark.parametrize("order", [4, 8])
@pytest.mark.parametrize("map_elements", [False, True])
def test_taylor_transform_of_fastpath_against_independent_oracle(order, map_elements):
    points = jnp.asarray([-0.5, 0.125, 0.8])

    def transformed(value):
        def evaluate(argument):
            return tanh_nth_derivative(argument, order)

        function = jax.vmap(evaluate) if map_elements else evaluate
        series = ((jnp.ones_like(value), jnp.zeros_like(value)),)
        if _JET_HAS_FACTORIAL_SCALING:
            return jet(function, (value,), series, factorial_scaled=False)
        # Older supported JAX versions return physical derivatives only.
        primal, derivatives = jet(function, (value,), series)
        return primal, [derivatives[0], derivatives[1] / 2]

    primal, coefficients = jax.jit(transformed)(points)
    expected = np.asarray([_derivative(float(value), order) for value in points])
    first = np.asarray([_derivative(float(value), order + 1) for value in points])
    second = np.asarray([_derivative(float(value), order + 2) / 2 for value in points])
    np.testing.assert_allclose(primal, expected, atol=1e-9, rtol=2e-10)
    np.testing.assert_allclose(coefficients[0], first, atol=1e-8, rtol=2e-9)
    np.testing.assert_allclose(coefficients[1], second, atol=1e-7, rtol=2e-8)


@pytest.mark.parametrize("order", [4, 8])
def test_taylor_and_gradient_transform_both_composition_orders(order):
    points = jnp.asarray([-0.5, 0.125, 0.8])

    def evaluate(value):
        return tanh_nth_derivative(value, order)

    def first_taylor(function, value):
        series = ((jnp.ones_like(value),),)
        if _JET_HAS_FACTORIAL_SCALING:
            return jet(function, (value,), series, factorial_scaled=False)[1][0]
        return jet(function, (value,), series)[1][0]

    def gradient_after_taylor(value):
        return jax.grad(lambda point: first_taylor(evaluate, point))(value)

    def taylor_after_gradient(value):
        return first_taylor(jax.grad(evaluate), value)

    expected = np.asarray([_derivative(float(value), order + 2) for value in points])
    for function in (gradient_after_taylor, taylor_after_gradient):
        actual = jax.jit(jax.vmap(function))(points)
        np.testing.assert_allclose(actual, expected, atol=1e-7, rtol=2e-8)


@pytest.mark.parametrize("order", [4, 6, 8])
def test_float32_primal_is_unchanged(order):
    points = jnp.asarray([-20, -3, -0.5, 0, 0.125, 0.8, 3, 20], dtype=jnp.float32)

    def evaluate(value):
        return tanh_nth_derivative(value, order)

    def legacy(value):
        return _legacy(value, order)

    np.testing.assert_array_equal(evaluate(points), legacy(points))
    np.testing.assert_array_equal(jax.jit(evaluate)(points), jax.jit(legacy)(points))


@pytest.mark.parametrize("order", [4, 8])
def test_nonlinear_input_series(order):
    points = jnp.asarray([-0.5, 0.125, 0.8])

    def transformed(value):
        def evaluate(argument):
            return tanh_nth_derivative(argument, order)

        first = jnp.full_like(value, 0.7)
        second = jnp.full_like(value, 0.3)
        if _JET_HAS_FACTORIAL_SCALING:
            return jet(evaluate, (value,), ((first, second),), factorial_scaled=False)[1]
        series = jet(evaluate, (value,), ((first, 2 * second),))[1]
        return [series[0], series[1] / 2]

    actual = jax.jit(transformed)(points)
    dn1 = np.asarray([_derivative(float(v), order + 1) for v in points])
    dn2 = np.asarray([_derivative(float(v), order + 2) for v in points])
    np.testing.assert_allclose(actual[0], 0.7 * dn1, atol=1e-8, rtol=2e-9)
    np.testing.assert_allclose(actual[1], 0.3 * dn1 + 0.5 * 0.7**2 * dn2, atol=1e-7, rtol=2e-8)


def _field_oracle(parameters, inputs, cotangent, order):
    weights, biases, readout = [np.asarray(value) for value in parameters]
    with mp.workdps(90):
        w = [[mp.mpf(float(value)) for value in row] for row in weights]
        b = [mp.mpf(float(value)) for value in biases]
        c = [mp.mpf(float(value)) for value in readout]
        value = mp.mpf(0)
        gw = [[mp.mpf(0) for _ in row] for row in w]
        gb = [mp.mpf(0) for _ in b]
        gc = [mp.mpf(0) for _ in c]
        for input_row, cot in zip(inputs, cotangent, strict=True):
            x = [mp.mpf(float(element)) for element in input_row]
            cot = mp.mpf(float(cot))
            for h in range(len(w)):
                z = sum(wh * xh for wh, xh in zip(w[h], x, strict=True)) + b[h]
                r = sum(element**2 for element in w[h])
                dn = mp.diff(mp.tanh, z, 2 * order)
                dn1 = mp.diff(mp.tanh, z, 2 * order + 1)
                value += cot * c[h] * r**order * dn
                gc[h] += cot * r**order * dn
                gb[h] += cot * c[h] * r**order * dn1
                for d in range(len(x)):
                    gw[h][d] += cot * c[h] * (
                        r**order * dn1 * x[d] +
                        2 * order * r**(order - 1) * w[h][d] * dn
                    )
    return float(value), tuple(np.asarray(g, dtype=np.float64) for g in (gw, gb, gc))


@pytest.mark.parametrize("order", [1, 2, 3, 4])
def test_complete_parameter_vjp_and_changed_weights(order):
    inputs = np.asarray([[-0.75, 0.25], [0.2, -0.4], [0.5, 0.8]])
    cotangent = np.asarray([1.0, -0.75, 0.5])
    parameters = (
        np.asarray([[0.7, -0.2], [0.0, 0.5], [0.0, 0.0]]),
        np.asarray([0.1, -0.2, 0.3]),
        np.asarray([0.8, -0.7, 0.1]),
    )

    def objective(params, points, cot):
        return jnp.vdot(cot, neural_field_polylaplacian(points, *params, "tanh", order))

    compiled = jax.jit(jax.value_and_grad(objective))
    previous = None
    for change in (0.0, 0.03):
        updated = (parameters[0] + change, parameters[1] - change, parameters[2])
        expected_value, expected_gradient = _field_oracle(updated, inputs, cotangent, order)
        value, gradient = compiled(tuple(map(jnp.asarray, updated)), jnp.asarray(inputs), jnp.asarray(cotangent))
        np.testing.assert_allclose(value, expected_value, atol=1e-9, rtol=2e-10)
        for actual, expected in zip(gradient, expected_gradient, strict=True):
            np.testing.assert_allclose(actual, expected, atol=1e-8, rtol=2e-9)
        if previous is not None:
            assert float(value) != previous
        previous = float(value)
