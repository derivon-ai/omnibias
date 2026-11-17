# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Aging jets against independent exact formal substitution and true roots.

The test oracle uses polynomial products, finite exponential powers and a
geometric reciprocal inside formal Picard iteration. It does not use the
implementation's exponential/quotient coefficient recurrences.
"""

from fractions import Fraction as F
from math import factorial

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import torch
from omnibias.core.verified.gaussian_aging import certify_gaussian_inverse_aging
from omnibias.core.verified.gaussian_heat import GaussianHeatMap
from omnibias.core.verified.interval import Interval
from omnibias.jax.gaussian_aging import gaussian_heat_inverse_aging as jax_aging
from omnibias.torch.gaussian_aging import gaussian_heat_inverse_aging as torch_aging

jax.config.update("jax_enable_x64", True)


@pytest.fixture(scope="module", autouse=True)
def _single_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def _multiply(left, right):
    size = len(left)
    return [sum(left[j]*right[n-j] for j in range(n+1)) for n in range(size)]


def _exponential(poly):
    assert poly[0] == 0
    size = len(poly)
    zero = poly[0]*0
    power = [zero+1]+[zero]*(size-1)
    total = list(power)
    for n in range(1, size):
        power = _multiply(power, poly)
        total = [left+right/factorial(n) for left, right in zip(total, power, strict=True)]
    return total


def _reciprocal(poly):
    size = len(poly)
    zero = poly[0]*0
    reduced = [zero]+[-value/poly[0] for value in poly[1:]]
    power = [zero+1]+[zero]*(size-1)
    total = list(power)
    for _ in range(1, size):
        power = _multiply(power, reduced)
        total = [left+right for left, right in zip(total, power, strict=True)]
    return [value/poly[0] for value in total]


def _formal_picard(centers, posterior, viscosity, time, order):
    """Independent finite-ring fixed point; works over Fractions or mpmath.

    A substitution gains one correct coefficient because the exponent has a
    factor t. After N substitutions, all coefficients through N are exact.
    """
    dimension = len(centers[0])
    zero = viscosity*0
    mean = tuple(sum(p*a[axis] for p, a in zip(posterior, centers, strict=True)) for axis in range(dimension))
    centered = [tuple(a-mu for a, mu in zip(row, mean, strict=True)) for row in centers]
    h = [[zero]*(order+1) for _ in range(dimension)]
    for _ in range(order):
        exponentials = []
        for row in centered:
            poly = [zero]*(order+1)
            for n in range(1, order+1):
                poly[n] = sum(a*h[axis][n-1] for axis, a in enumerate(row))/(2*viscosity*time)
            poly[1] -= sum(a*a for a in row)/(4*viscosity*time)
            exponentials.append(_exponential(poly))
        denominator = [sum(p*e[n] for p, e in zip(posterior, exponentials, strict=True)) for n in range(order+1)]
        reciprocal = _reciprocal(denominator)
        for axis in range(dimension):
            numerator = [sum(p*row[axis]*e[n] for p, row, e in zip(posterior, centered, exponentials, strict=True))
                         for n in range(order+1)]
            h[axis] = _multiply(numerator, reciprocal)
            h[axis][0] = zero  # E[A]=0 structurally, including a rounded mp oracle.
    return tuple(tuple(h[axis][n] for axis in range(dimension)) for n in range(order+1)), mean


def _mp(mp, value):
    exact = F(value)
    return mp.mpf(exact.numerator)/exact.denominator


def _mp_explicit(mp, target, centers, biases, viscosity, time, order):
    logits = [b+sum(a*v for a, v in zip(row, target, strict=True))/(2*viscosity)
              for b, row in zip(biases, centers, strict=True)]
    weights = [mp.exp(value-max(logits)) for value in logits]
    posterior = [value/sum(weights) for value in weights]
    coefficients, mean = _formal_picard(centers, posterior, viscosity, time, order)
    value = tuple(time*v+mu+sum(row[axis] for row in coefficients)
                  for axis, (v, mu) in enumerate(zip(target, mean, strict=True)))
    return value, coefficients


def _physical_root(mp, model, target):
    centers = [tuple(_mp(mp, value) for value in row) for row in model.centers]
    biases = [_mp(mp, value) for value in model.biases]
    nu, time = _mp(mp, model.viscosity), _mp(mp, model.time)
    wanted = tuple(_mp(mp, value) for value in target)

    def residual(*point):
        logits = [b-sum((x-a)**2 for x, a in zip(point, row, strict=True))/(4*nu*time)
                  for b, row in zip(biases, centers, strict=True)]
        weights = [mp.exp(value-max(logits)) for value in logits]
        mean = tuple(sum(p*row[axis] for p, row in zip(weights, centers, strict=True))/sum(weights)
                     for axis in range(model.dimension))
        return tuple((x-mu)/time-v for x, mu, v in zip(point, mean, wanted, strict=True))

    root = tuple(mp.findroot(residual, tuple(time*v for v in wanted), tol=mp.mpf("1e-80"), maxsteps=80))
    assert mp.norm(residual(*root)) < mp.mpf("1e-75")
    return root


FAMILIES = (
    (((F(0),), (F(1),), (F(3),)), F(1), F(8)),
    (((F(0), F(0)), (F(1), F(0)), (F(0), F(2)), (F(2), F(1))), F(3, 2), F(4)),
)


@pytest.mark.parametrize("family", FAMILIES)
@pytest.mark.parametrize("order", [0, 1, 4, 8])
def test_core_and_both_backends_match_exact_formal_picard_coefficients(family, order):
    centers, viscosity, time = family
    dimension, count = len(centers[0]), len(centers)
    probabilities = (F(1, count),)*count
    expected, mean = _formal_picard(centers, probabilities, viscosity, time, order)
    assert all(isinstance(value, F) for row in expected for value in row)
    model = GaussianHeatMap(centers, (0,)*count, viscosity=viscosity, time=time)
    certificate = certify_gaussian_inverse_aging(model, (Interval.point(0.0),)*dimension, order=order)
    assert certificate is not None
    for actual_row, exact_row in zip(certificate.correction_coefficients, expected, strict=True):
        for actual, exact in zip(actual_row, exact_row, strict=True):
            assert F(actual.lo) <= exact <= F(actual.hi)
    args = (np.zeros(dimension), np.asarray(centers, dtype=np.float64), np.zeros(count), float(viscosity), float(time))
    actual_t = torch_aging(*(torch.tensor(value, dtype=torch.float64) for value in args), order=order)
    actual_j = jax.jit(lambda *values: jax_aging(*values, order=order))(*(jnp.asarray(value) for value in args))
    assert bool(actual_t.valid) and bool(actual_j.valid)
    reference = np.asarray(expected, dtype=np.float64)
    for actual in (actual_t.correction_coefficients.detach().numpy(), np.asarray(actual_j.correction_coefficients)):
        np.testing.assert_allclose(actual, reference, rtol=3e-11, atol=2e-14)
    predicted = np.asarray([mu+sum(row[axis] for row in expected) for axis, mu in enumerate(mean)], dtype=np.float64)
    np.testing.assert_allclose(actual_t.value.detach().numpy(), predicted, rtol=3e-13, atol=3e-14)
    np.testing.assert_allclose(actual_j.value, predicted, rtol=3e-13, atol=3e-14)
    if dimension == 1 and order:
        assert expected[1] == (F(-5, 216),)
        assert all(row[0] != 0 for row in expected[1:])


@pytest.mark.parametrize("family", FAMILIES)
def test_scaled_aging_coefficients_follow_time_inverse_powers(family):
    centers, viscosity, time = family
    dimension, count, order = len(centers[0]), len(centers), 6
    query = np.zeros(dimension)
    values = []
    for age in (time, 2*time):
        args = (query, np.asarray(centers, dtype=np.float64), np.zeros(count), float(viscosity), float(age))
        result = torch_aging(*(torch.tensor(value, dtype=torch.float64) for value in args), order=order)
        assert bool(result.valid)
        values.append(result.correction_coefficients.detach().numpy())
    for n in range(order+1):
        np.testing.assert_allclose(values[1][n], values[0][n]/2**n, rtol=2e-12, atol=2e-15)


@pytest.mark.parametrize("family", FAMILIES)
@pytest.mark.parametrize("order", [0, 1, 3, 6])
def test_independent_physical_roots_and_formal_predictors_obey_certified_tail(family, order):
    mp = pytest.importorskip("mpmath")
    centers, viscosity, time = family
    dimension, count = len(centers[0]), len(centers)
    biases = tuple(F((-1)**j*(j+1), 8) for j in range(count))
    model = GaussianHeatMap(centers, biases, viscosity=viscosity, time=time)
    targets = ((F(1, 4),)*dimension, tuple(F(-3, 8) if axis == 0 else F(1, 8) for axis in range(dimension)))
    with mp.workdps(90):
        for target in targets:
            certificate = certify_gaussian_inverse_aging(model, tuple(Interval.point(float(v)) for v in target), order=order)
            assert certificate is not None
            predicted, coefficients = _mp_explicit(mp, tuple(_mp(mp, v) for v in target),
                [tuple(_mp(mp, a) for a in row) for row in centers], tuple(_mp(mp, b) for b in biases),
                _mp(mp, viscosity), _mp(mp, time), order)
            root = _physical_root(mp, model, target)
            error = mp.norm(tuple(actual-expected for actual, expected in zip(predicted, root, strict=True)))
            assert error <= _mp(mp, certificate.approximation_error_bound)
            for axis in range(dimension):
                for value, interval in ((predicted[axis], certificate.predictor_enclosure[axis]),
                                        (root[axis], certificate.enclosure[axis])):
                    assert _mp(mp, interval.lo) <= value <= _mp(mp, interval.hi)
            for exact_row, interval_row in zip(coefficients, certificate.correction_coefficients, strict=True):
                for exact, interval in zip(exact_row, interval_row, strict=True):
                    assert _mp(mp, interval.lo) <= exact <= _mp(mp, interval.hi)


def test_batched_jit_vmap_values_coefficients_and_diagnostics_match():
    data = (np.array([[0.25, -0.125], [-0.375, 0.5]]), np.array([[0., 0.], [1., 0.], [0., 1.]]),
            np.array([0.125, -0.25, 0.375]), np.array(1.5), np.array(3.0))
    torch_result = torch_aging(*(torch.tensor(value, dtype=torch.float64) for value in data), order=5)
    arrays = tuple(jnp.asarray(value) for value in data)
    jax_result = jax.jit(lambda *args: jax_aging(*args, order=5))(*arrays)
    mapped = jax.jit(jax.vmap(lambda target: jax_aging(target, *arrays[1:], order=5)))(arrays[0])
    np.testing.assert_allclose(torch_result.value.numpy(), jax_result.value, rtol=2e-12, atol=2e-13)
    np.testing.assert_allclose(torch_result.correction_coefficients.numpy(), jax_result.correction_coefficients,
                               rtol=3e-11, atol=2e-14)
    np.testing.assert_allclose(jax_result.correction_coefficients, jnp.swapaxes(mapped.correction_coefficients, 0, 1),
                               rtol=2e-12, atol=2e-14)
    np.testing.assert_allclose(torch_result.approximation_error_bound.numpy(), jax_result.approximation_error_bound, rtol=2e-13)
    np.testing.assert_array_equal(torch_result.valid.numpy(), jax_result.valid)


def test_all_parameter_gradients_match_finite_differences_of_the_explicit_polynomial():
    mp = pytest.importorskip("mpmath")
    data = (np.array([0.25, -0.125]), np.array([[0., 0.], [1., 0.], [0., 1.]]),
            np.array([0.125, -0.25, 0.375]), np.array(1.5), np.array(3.0))
    order = 3
    args_t = tuple(torch.tensor(value, dtype=torch.float64, requires_grad=True) for value in data)
    args_j = tuple(jnp.asarray(value) for value in data)
    result_t = torch_aging(*args_t, order=order)
    loss_t = result_t.value.square().sum()+result_t.correction_coefficients.square().sum()/10
    gradients_t = torch.autograd.grad(loss_t, args_t)

    def loss_j(*args):
        result = jax_aging(*args, order=order)
        return jnp.sum(result.value**2)+jnp.sum(result.correction_coefficients**2)/10

    gradients_j = jax.jit(jax.grad(loss_j, argnums=(0, 1, 2, 3, 4)))(*args_j)
    flattened = np.concatenate([value.reshape(-1) for value in data])
    with mp.workdps(80):
        nominal = [_mp(mp, value) for value in flattened]

        def explicit_loss(values):
            target, centers = values[:2], [values[2+2*j:4+2*j] for j in range(3)]
            prediction, coefficients = _mp_explicit(mp, target, centers, values[8:11], values[11], values[12], order)
            return sum(value*value for value in prediction)+sum(value*value for row in coefficients for value in row)/10

        expected = []
        epsilon = mp.mpf("1e-12")
        for index in range(len(nominal)):
            plus, minus = list(nominal), list(nominal)
            plus[index] += epsilon
            minus[index] -= epsilon
            expected.append(float((explicit_loss(plus)-explicit_loss(minus))/(2*epsilon)))
    for gradients in (gradients_t, gradients_j):
        actual = np.concatenate([np.asarray(value.detach().numpy() if isinstance(value, torch.Tensor) else value).reshape(-1)
                                 for value in gradients])
        np.testing.assert_allclose(actual, expected, rtol=2e-9, atol=2e-11)


@pytest.mark.parametrize("order", [0, 1, 5])
def test_conservative_domain_refusal_does_not_claim_nonexistence(order):
    model = GaussianHeatMap(((-1,), (1,)), (0, 0), time=F(3, 2))
    assert model.contraction_bound == F(1, 3)
    assert certify_gaussian_inverse_aging(model, (Interval.point(0.0),), order=order) is None
    target = np.array([0.0])
    result_t = torch_aging(torch.tensor(target), [[-1.], [1.]], [0., 0.], time=1.5, order=order)
    result_j = jax_aging(jnp.asarray(target), [[-1.], [1.]], [0., 0.], time=1.5, order=order)
    assert not bool(result_t.valid) and not bool(result_j.valid)
    # Nonetheless the real inverse at v=0 exists: x=0, since U is odd.
    assert np.isnan(result_t.value.detach().numpy()).all() and np.isnan(np.asarray(result_j.value)).all()
