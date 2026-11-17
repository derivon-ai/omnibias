# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Separate surrogate AD accuracy from certificates for true inverse derivatives."""

from fractions import Fraction as F
from math import factorial

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import torch
from omnibias.core.gaussian_jet import GaussianInverseSeries
from omnibias.core.verified.gaussian_chebyshev import approximate_gaussian_inverse
from omnibias.core.verified.gaussian_chebyshev_derivative import certify_gaussian_inverse_derivative
from omnibias.core.verified.gaussian_heat import GaussianHeatMap
from omnibias.core.verified.gaussian_tube import certify_gaussian_inverse_tube
from omnibias.jax.gaussian_chebyshev import evaluate_gaussian_inverse_approximation as jax_evaluate
from omnibias.torch.gaussian_chebyshev import (
    evaluate_gaussian_inverse_approximation as torch_evaluate,
)

jax.config.update("jax_enable_x64", True)


def _approximate(model, base, direction, *, degree=24, half_width=F(1, 4)):
    tube = certify_gaussian_inverse_tube(model)
    assert tube is not None
    # A simple exact ellipse avoids inflating rational power denominators from
    # the automatic square-root-based radius; it lies strictly inside these tubes.
    result = approximate_gaussian_inverse(tube, base, direction, half_width=half_width, degree=degree, ellipse_parameter=4)
    assert result is not None
    return result


@pytest.fixture(scope="module")
def coupled():
    pytest.importorskip("mpmath")
    model = GaussianHeatMap(((1, -2), (2, -2), (1, -1)), (0, F(1, 4), F(1, 4)))
    return _approximate(model, (-F(1, 3), -F(1, 3)), (F(2, 5), -F(1, 4)))


@pytest.fixture(scope="module")
def unequal():
    pytest.importorskip("mpmath")
    model = GaussianHeatMap(((-F(1, 2),), (-F(1, 4),), (F(3, 4),)),
                            (F(1, 5), -F(1, 4), F(2, 7)), viscosity=F(3, 2), time=F(5, 4))
    return _approximate(model, (F(2, 5),), (F(2, 3),))


def _power_basis(approximation):
    """Independent exact expansion T_(n+1)=2*z*T_n-T_(n-1)."""
    degree = approximation.degree
    basis = [[F(1)]]
    if degree:
        basis.append([F(0), F(1)])
    for n in range(1, degree):
        row = [F(0)] * (n + 2)
        for k, value in enumerate(basis[-1]):
            row[k + 1] += 2 * value
        for k, value in enumerate(basis[-2]):
            row[k] -= value
        basis.append(row)
    dimension = len(approximation.base_target)
    result = [[F(0)] * dimension for _ in range(max(degree, 1) + 1)]
    for coefficients, polynomial in zip(approximation.coefficients, basis, strict=True):
        for power, value in enumerate(polynomial):
            for axis, coefficient in enumerate(coefficients):
                result[power][axis] += F.from_float(coefficient) * value / approximation.half_width**power
    for axis in range(dimension):
        result[0][axis] += (approximation.tube.model.time * approximation.base_target[axis]
                           + approximation.tube.reference_center[axis])
        result[1][axis] += approximation.tube.model.time * approximation.direction[axis]
    return tuple(tuple(row) for row in result)


def _nominal_derivative(approximation, parameter, order):
    powers = _power_basis(approximation)
    return tuple(sum((powers[n][axis] * F(factorial(n), factorial(n - order)) * parameter**(n - order)
                      for n in range(order, len(powers))), F(0))
                 for axis in range(len(approximation.base_target)))


def _torch_derivative(approximation, parameters, order, dtype):
    query = torch.tensor(parameters, dtype=dtype, requires_grad=True)
    values = torch_evaluate(query, approximation).value
    coordinates = []
    for axis in range(values.shape[-1]):
        derivative = values[:, axis]
        for _ in range(order):
            derivative = (torch.autograd.grad(derivative.sum(), query, create_graph=True, retain_graph=True)[0]
                          if derivative.requires_grad else torch.zeros_like(query))
        coordinates.append(derivative.detach())
    return torch.stack(coordinates, dim=-1).numpy()


def _jax_derivative(approximation, parameters, order, dtype):
    def function(query):
        return jax_evaluate(query, approximation).value
    for _ in range(order):
        function = jax.jacfwd(function)
    return np.asarray(jax.jit(jax.vmap(function))(jnp.asarray(parameters, dtype=dtype)))


@pytest.mark.parametrize("order", [0, 1, 2, 3, 4])
def test_exact_derivative_artifact_agrees_with_independent_power_basis(coupled, order):
    derivative = certify_gaussian_inverse_derivative(coupled, order)
    assert derivative is not None and derivative.order == order
    assert all(isinstance(value, F) for row in derivative.coefficients for value in row)
    for parameter in (-F(1, 4), -F(1, 9), F(0), F(1, 7), F(1, 4)):
        assert derivative.evaluate(parameter) == _nominal_derivative(coupled, parameter, order)
    if order == 0:
        assert derivative.uniform_error_bounds == coupled.uniform_error_bounds


@pytest.mark.parametrize("order", [1, 2, 3, 4])
@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_tensor_ad_matches_exact_stored_polynomial_derivatives(coupled, order, dtype):
    # This is a floating implementation check, deliberately separate from the
    # theorem-backed bounds for derivatives of the true inverse below.
    parameters = np.asarray([-0.25, -0.125, 0.0, 0.125, 0.25], dtype=dtype)
    expected = np.asarray([_nominal_derivative(coupled, F(float(value)), order) for value in parameters], dtype=np.float64)
    actual_torch = _torch_derivative(coupled, parameters, order, getattr(torch, dtype))
    actual_jax = _jax_derivative(coupled, parameters, order, getattr(jnp, dtype))
    rtol, atol = (2e-4, 2e-6) if dtype == "float32" else (3e-10, 3e-12)
    np.testing.assert_allclose(actual_torch, expected, rtol=rtol, atol=atol)
    np.testing.assert_allclose(actual_jax, expected, rtol=rtol, atol=atol)
    np.testing.assert_allclose(actual_torch, actual_jax, rtol=rtol, atol=atol)


@pytest.mark.parametrize("order", [1, 2, 3, 4, 5])
def test_true_coupled_derivative_encloses_exact_implicit_series_at_base(coupled, order):
    series = GaussianInverseSeries.from_equal_logits(coupled.tube.model, (1, -2),
              (coupled.direction, *((F(0), F(0)),) * (order - 1)))
    assert series.target_coefficients[0] == coupled.base_target
    exact = tuple(factorial(order) * value for value in series.coefficients[order])
    assert any(exact), "the true derivative oracle must not be a symmetry-forced zero"
    derivative = certify_gaussian_inverse_derivative(coupled, order)
    assert derivative is not None
    nominal = derivative.evaluate(0)
    for value, center, error, bound in zip(exact, nominal, derivative.uniform_error_bounds,
                                           derivative.enclose(0), strict=True):
        assert F(bound.lo) <= value <= F(bound.hi)
        assert abs(value - center) <= error
    if order > 1:
        assert exact != series.coefficients[order], "the public derivative convention is raw, not Taylor-normalized"


def _mp(value, mp):
    value = F(value)
    return mp.mpf(value.numerator) / value.denominator


def _mp_inverse_function(approximation, mp):
    model = approximation.tube.model
    centers = [_mp(row[0], mp) for row in model.centers]
    biases = [_mp(value, mp) for value in model.biases]
    viscosity, time = _mp(model.viscosity, mp), _mp(model.time, mp)
    base, direction = _mp(approximation.base_target[0], mp), _mp(approximation.direction[0], mp)

    def inverse(parameter):
        target = base + parameter * direction

        def residual(point):
            weights = [mp.exp(bias - (point - center)**2 / (4 * viscosity * time))
                       for center, bias in zip(centers, biases, strict=True)]
            mean = mp.fsum(weight * center for weight, center in zip(weights, centers, strict=True)) / mp.fsum(weights)
            return point - time * target - mean

        return mp.findroot(residual, time * target)

    return inverse


@pytest.mark.parametrize("order", [1, 2, 3, 4])
def test_unequal_bias_true_derivative_bounds_cover_high_precision_inverse(unequal, order):
    mp = pytest.importorskip("mpmath")
    derivative = certify_gaussian_inverse_derivative(unequal, order)
    assert derivative is not None
    with mp.workdps(80):
        inverse = _mp_inverse_function(unequal, mp)
        for parameter in (-F(1, 4), -F(1, 9), F(0), F(1, 7), F(1, 4)):
            exact = mp.diff(inverse, _mp(parameter, mp), order)
            bound = derivative.enclose(parameter)[0]
            assert mp.mpf(bound.lo) <= exact <= mp.mpf(bound.hi)
            nominal = _mp(derivative.evaluate(parameter)[0], mp)
            assert abs(nominal - exact) <= _mp(derivative.uniform_error_bounds[0], mp)


def test_derivative_above_polynomial_degree_retains_nonzero_true_error(unequal):
    mp = pytest.importorskip("mpmath")
    low_degree = _approximate(unequal.tube.model, unequal.base_target, unequal.direction, degree=2)
    derivative = certify_gaussian_inverse_derivative(low_degree, 3)
    assert derivative is not None and derivative.degree == 0
    assert derivative.coefficients == ((F(0),),) and derivative.evaluate(0) == (F(0),)
    assert derivative.uniform_error_bounds[0] > 0
    with mp.workdps(80):
        actual = mp.diff(_mp_inverse_function(low_degree, mp), mp.mpf(0), 3)
        assert abs(actual) > mp.mpf("1e-6")
        bound = derivative.enclose(0)[0]
        assert mp.mpf(bound.lo) <= actual <= mp.mpf(bound.hi)
    np.testing.assert_allclose(_torch_derivative(low_degree, [0.0], 3, torch.float64), [[0.0]], atol=1e-15)
    np.testing.assert_allclose(_jax_derivative(low_degree, [0.0], 3, jnp.float64), [[0.0]], atol=1e-15)


@pytest.mark.parametrize("order", [1, 2, 4])
def test_affine_derivatives_have_zero_error_and_backend_agreement(order):
    model = GaussianHeatMap(((2, -1),), (F(3, 7),), time=F(3, 2))
    approximation = _approximate(model, (F(1, 3), -F(1, 4)), (2, -3), degree=1)
    derivative = certify_gaussian_inverse_derivative(approximation, order)
    assert derivative is not None
    expected = (F(3), -F(9, 2)) if order == 1 else (F(0), F(0))
    assert derivative.uniform_error_bounds == (F(0), F(0))
    assert derivative.evaluate(F(1, 8)) == expected
    for value, bound in zip(expected, derivative.enclose(F(1, 8)), strict=True):
        assert F(bound.lo) <= value <= F(bound.hi)
    numerical = np.asarray([expected], dtype=np.float64)
    np.testing.assert_allclose(_torch_derivative(approximation, [0.125], order, torch.float64), numerical, atol=1e-15)
    np.testing.assert_allclose(_jax_derivative(approximation, [0.125], order, jnp.float64), numerical, atol=1e-15)


def test_constant_target_path_has_certified_zero_derivatives_despite_value_uncertainty(unequal):
    approximation = _approximate(unequal.tube.model, unequal.base_target, (F(0),), degree=8)
    assert approximation.degree == 0 and approximation.uniform_error_bounds[0] > 0
    derivative = certify_gaussian_inverse_derivative(approximation, 2)
    assert derivative is not None
    assert derivative.evaluate(0) == (F(0),) and derivative.uniform_error_bounds == (F(0),)
    np.testing.assert_allclose(_torch_derivative(approximation, [0.0, 0.125], 2, torch.float64), [[0.0], [0.0]], atol=1e-15)
    np.testing.assert_allclose(_jax_derivative(approximation, [0.0, 0.125], 2, jnp.float64), [[0.0], [0.0]], atol=1e-15)
