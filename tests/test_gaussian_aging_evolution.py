# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Physical-age derivative certificates versus independent roots and explicit AD.

Autodiff below differentiates the finite aging polynomial. Only adding the
certified tail makes an enclosure for derivatives of the true physical inverse.
"""

from fractions import Fraction as F
from math import prod

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import torch
from omnibias.core.verified.gaussian_aging import certify_gaussian_inverse_aging
from omnibias.core.verified.gaussian_aging_derivative import certify_gaussian_aging_derivative
from omnibias.core.verified.gaussian_evolution import certify_gaussian_inverse_evolution
from omnibias.core.verified.gaussian_heat import GaussianHeatMap
from omnibias.core.verified.interval import Interval
from omnibias.jax.gaussian_aging import gaussian_heat_inverse_aging as jax_aging
from omnibias.torch.gaussian_aging import gaussian_heat_inverse_aging as torch_aging
from test_gaussian_inverse_aging import _formal_picard

jax.config.update("jax_enable_x64", True)


@pytest.fixture(scope="module", autouse=True)
def _single_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def _mp(mp, value):
    exact = F(value)
    return mp.mpf(exact.numerator)/exact.denominator


def _inverse_at_age(mp, model, target, age):
    """Fresh direct Gaussian root at current mp precision, also for mp.diff.

    Metadata is converted inside the call and root tolerance follows mp.eps:
    finite differences must not differentiate a fixed-precision rounded root.
    """
    centers = [tuple(_mp(mp, value) for value in row) for row in model.centers]
    biases = tuple(_mp(mp, value) for value in model.biases)
    wanted = tuple(_mp(mp, value) for value in target)
    nu = _mp(mp, model.viscosity)

    def residual(*point):
        logits = [b-sum((x-a)**2 for x, a in zip(point, row, strict=True))/(4*nu*age)
                  for b, row in zip(biases, centers, strict=True)]
        weights = [mp.exp(value-max(logits)) for value in logits]
        mean = tuple(sum(p*row[axis] for p, row in zip(weights, centers, strict=True))/sum(weights)
                     for axis in range(model.dimension))
        return tuple((x-mu)/age-v for x, mu, v in zip(point, mean, wanted, strict=True))

    return tuple(mp.findroot(residual, tuple(age*v for v in wanted), tol=mp.eps*32, maxsteps=80))


def _true_derivative(mp, model, target, order):
    age = _mp(mp, model.time)
    return tuple(mp.diff(lambda s, axis=axis: _inverse_at_age(mp, model, target, s)[axis], age, order)
                 for axis in range(model.dimension))


@pytest.mark.parametrize("derivative_order", [0, 1, 2, 3, 5])
def test_exact_polynomial_weights_are_rising_factorials_even_above_truncation_degree(derivative_order):
    centers, probabilities, time, degree = ((F(0),), (F(1),), (F(3),)), (F(1, 3),)*3, F(8), 2
    coefficients, mean = _formal_picard(centers, probabilities, F(1), time, degree)
    aging = certify_gaussian_inverse_aging(GaussianHeatMap(centers, (0, 0, 0), time=time),
                                          (Interval.point(0.0),), order=degree)
    assert aging is not None
    derivative = certify_gaussian_aging_derivative(aging, order=derivative_order)
    assert derivative is not None
    exact = sum((-1)**derivative_order*prod(range(n, n+derivative_order))*row[0]/time**derivative_order
                for n, row in enumerate(coefficients[1:], 1))
    if derivative_order == 0:
        exact += mean[0]
        assert derivative.approximation_error_bound == aging.approximation_error_bound
    interval = derivative.polynomial_enclosure[0]
    assert F(interval.lo) <= exact <= F(interval.hi)
    if derivative_order > degree:
        assert exact != 0  # An inverse-age polynomial is not a polynomial in s.


FAMILIES = (
    (((0,), (1,), (3,)), (F(1, 8), F(-1, 4), F(3, 8)), F(1), F(8), (F(1, 4),)),
    (((0, 0), (1, 0), (0, 2), (2, 1)), (F(1, 8), F(-1, 4), F(3, 8), F(-1, 8)),
     F(3, 2), F(4), (F(1, 4), F(-1, 8))),
)


@pytest.mark.parametrize("family", FAMILIES)
@pytest.mark.parametrize("order", [1, 2, 3])
def test_true_physical_age_derivatives_fit_tail_enclosures_and_existing_evolution(family, order):
    mp = pytest.importorskip("mpmath")
    centers, biases, viscosity, time, target = family
    model = GaussianHeatMap(centers, biases, viscosity=viscosity, time=time)
    box = tuple(Interval.point(float(value)) for value in target)
    aging = certify_gaussian_inverse_aging(model, box, order=5)
    assert aging is not None
    derivative = certify_gaussian_aging_derivative(aging, order=order)
    assert derivative is not None
    with mp.workdps(90):
        true = _true_derivative(mp, model, target, order)
        for value, interval in zip(true, derivative.enclosure, strict=True):
            assert _mp(mp, interval.lo) <= value <= _mp(mp, interval.hi)
        if order == 1:
            evolution = certify_gaussian_inverse_evolution(model, box, absolute_tolerance=1e-13)
            assert evolution is not None
            for value, first, second in zip(true, derivative.enclosure, evolution.time_derivative, strict=True):
                assert _mp(mp, second.lo) <= value <= _mp(mp, second.hi)
                assert max(first.lo, second.lo) <= min(first.hi, second.hi)


def test_independent_root_derivative_is_stable_when_precision_increases():
    mp = pytest.importorskip("mpmath")
    centers, biases, viscosity, time, target = FAMILIES[0]
    model = GaussianHeatMap(centers, biases, viscosity=viscosity, time=time)
    values = []
    for digits in (60, 90):
        with mp.workdps(digits):
            values.append(_true_derivative(mp, model, target, 3)[0])
    with mp.workdps(90):
        assert abs(values[0]-values[1]) < mp.mpf("1e-50")
        assert abs(values[1]) > mp.mpf("1e-5")


def test_finite_polynomial_derivative_does_not_pretend_to_equal_true_inverse_derivative():
    mp = pytest.importorskip("mpmath")
    centers, biases, viscosity, time, target = FAMILIES[0]
    model = GaussianHeatMap(centers, biases, viscosity=viscosity, time=time)
    aging = certify_gaussian_inverse_aging(model, (Interval.point(float(target[0])),), order=1)
    assert aging is not None
    derivative = certify_gaussian_aging_derivative(aging, order=1)
    assert derivative is not None
    with mp.workdps(90):
        true = _true_derivative(mp, model, target, 1)[0]
        polynomial = derivative.polynomial_enclosure[0]
        assert true < _mp(mp, polynomial.lo) or true > _mp(mp, polynomial.hi)
        assert _mp(mp, derivative.enclosure[0].lo) <= true <= _mp(mp, derivative.enclosure[0].hi)


@pytest.mark.parametrize("family", FAMILIES)
def test_tensor_ad_matches_only_the_explicit_polynomial_derivative_enclosures(family):
    centers, biases, viscosity, time, target = family
    model = GaussianHeatMap(centers, biases, viscosity=viscosity, time=time)
    degree = 3
    aging = certify_gaussian_inverse_aging(model, tuple(Interval.point(float(value)) for value in target), order=degree)
    assert aging is not None
    data = tuple(np.asarray(value, dtype=np.float64) for value in (target, centers, biases, viscosity))
    torch_data = tuple(torch.tensor(value, dtype=torch.float64) for value in data)
    jax_data = tuple(jnp.asarray(value) for value in data)

    def function_t(age):
        return torch_aging(*torch_data, age, order=degree).value

    def function_j(age):
        return jax_aging(*jax_data, age, order=degree).value

    for order in range(1, 4):
        previous_t = function_t

        def function_t(age, previous=previous_t):
            return torch.autograd.functional.jacobian(previous, age, create_graph=True)

        function_j = jax.jacfwd(function_j)
        value_t = function_t(torch.tensor(float(time), dtype=torch.float64, requires_grad=True)).detach().numpy()
        value_j = np.asarray(jax.jit(function_j)(jnp.asarray(float(time))))
        np.testing.assert_allclose(value_t, value_j, rtol=3e-10, atol=3e-13)
        derivative = certify_gaussian_aging_derivative(aging, order=order)
        assert derivative is not None
        for values in (value_t, value_j):
            for value, interval in zip(values, derivative.polynomial_enclosure, strict=True):
                assert interval.lo <= value <= interval.hi


@pytest.mark.parametrize("order", [0, 1, 2, 4])
def test_identical_centers_have_exact_affine_age_derivatives(order):
    model = GaussianHeatMap(((2, -3), (2, -3)), (0, F(1, 4)), time=F(3, 2))
    target = (F(1, 4), F(-1, 8))
    aging = certify_gaussian_inverse_aging(model, tuple(Interval.point(float(value)) for value in target), order=3)
    assert aging is not None
    derivative = certify_gaussian_aging_derivative(aging, order=order)
    assert derivative is not None and derivative.approximation_error_bound == 0
    expected = tuple(model.time*v+a for v, a in zip(target, model.centers[0], strict=True)) if order == 0 else (
        target if order == 1 else (F(0), F(0)))
    for exact, interval in zip(expected, derivative.enclosure, strict=True):
        assert F(interval.lo) <= exact <= F(interval.hi)
