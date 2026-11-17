# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Box certificates versus implicit AD, forward jets and independent moments."""

from fractions import Fraction as F
from itertools import product

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import torch
from omnibias.core.verified.gaussian_evolution import certify_gaussian_inverse_evolution
from omnibias.core.verified.gaussian_heat import GaussianHeatMap
from omnibias.core.verified.interval import Interval
from omnibias.jax.gaussian_heat import gaussian_heat_inverse as jax_inverse
from omnibias.jax.gaussian_jet import gaussian_heat_inverse_jet as jax_jet
from omnibias.torch.gaussian_heat import gaussian_heat_inverse as torch_inverse
from omnibias.torch.gaussian_jet import gaussian_heat_inverse_jet as torch_jet

jax.config.update("jax_enable_x64", True)


@pytest.fixture(scope="module", autouse=True)
def _single_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def _model(age=F(3, 2)):
    # Every numerical-backend operand is an exactly representable binary64
    # rational, so the core and tensor probes address the same fixed family.
    return GaussianHeatMap(((0, 0), (1, 0), (0, 2), (1, 1)),
                           (0, F(3, 8), -F(1, 4), F(1, 8)), viscosity=F(5, 4), time=age)


def _box():
    return (Interval(0.125, 0.375), Interval(-0.5, -0.25))


def _points(box):
    points = list(product(*[(axis.lo, (axis.lo + axis.hi) / 2, axis.hi) for axis in box]))
    for left, right in ((0.25, 0.75), (0.75, 0.25), (0.125, 0.875), (0.875, 0.125)):
        points.append((box[0].lo + left * (box[0].hi - box[0].lo),
                       box[1].lo + right * (box[1].hi - box[1].lo)))
    return np.asarray(points, dtype=np.float64)


def _enclosed(values, intervals):
    values = np.asarray(values)
    lower = np.asarray([[bound.lo for bound in row] for row in intervals] if values.ndim == 2
                       else [bound.lo for bound in intervals])
    upper = np.asarray([[bound.hi for bound in row] for row in intervals] if values.ndim == 2
                       else [bound.hi for bound in intervals])
    assert np.isfinite(values).all()
    assert np.all(lower <= values), (values, lower, upper)
    assert np.all(values <= upper), (values, lower, upper)


def _moments(point, centers, biases, viscosity, age, target):
    """Physical Gaussian formulas independent of both inverse implementations."""
    squared_distance = np.sum((point - centers)**2, axis=-1)
    logits = biases - squared_distance / (4 * viscosity * age)
    weights = np.exp(logits - logits.max())
    probability = weights / weights.sum()
    mean = probability @ centers
    centered = centers - mean
    covariance = (centered.T * probability) @ centered
    third_trace = (probability * np.sum(centered**2, axis=-1)) @ centered
    hessian = np.eye(point.size) - covariance / (2 * viscosity * age)
    diffusion = np.linalg.solve(hessian, third_trace / (4 * viscosity * age**2))
    mean_time = (probability * squared_distance / (4 * viscosity * age**2)) @ centered
    time_derivative = np.linalg.solve(hessian, target + mean_time)
    return age * np.linalg.inv(hessian), time_derivative, diffusion, hessian / age


@pytest.mark.parametrize("age", [F(3, 2), F(3)])
def test_full_target_box_contains_both_backend_derivatives_and_forward_jet_diffusion(age):
    pytest.importorskip("mpmath")
    model, box = _model(age), _box()
    certificate = certify_gaussian_inverse_evolution(model, box)
    assert certificate is not None and certificate.target_box == box
    centers = np.asarray(model.centers, dtype=np.float64)
    biases = np.asarray(model.biases, dtype=np.float64)
    viscosity, time = float(model.viscosity), float(model.time)

    def jax_function(target, smoothing_time):
        return jax_inverse(target, centers, biases, viscosity, smoothing_time, tolerance=1e-13).value

    derivative_jax = jax.jit(jax.jacfwd(jax_function, argnums=(0, 1)))
    value_jax = jax.jit(jax_function)
    jet_jax = jax.jit(lambda path: jax_jet(path, centers, biases, viscosity, time, tolerance=1e-13).coefficients)
    diffusion_sizes, coupling_sizes = [], []
    for target in _points(box):
        wanted = torch.tensor(target, dtype=torch.float64, requires_grad=True)
        smoothing_time = torch.tensor(time, dtype=torch.float64, requires_grad=True)
        result = torch_inverse(wanted, centers, biases, viscosity, smoothing_time, tolerance=1e-13)
        assert bool(result.converged)
        derivatives = [torch.autograd.grad(component, (wanted, smoothing_time), retain_graph=True)
                       for component in result.value]
        target_jacobian = torch.stack([entry[0] for entry in derivatives]).detach().numpy()
        time_derivative = torch.stack([entry[1] for entry in derivatives]).detach().numpy()
        point = result.value.detach().numpy()
        jacobian_jax, time_jax = derivative_jax(jnp.asarray(target), jnp.asarray(time))
        np.testing.assert_allclose(value_jax(jnp.asarray(target), jnp.asarray(time)), point, rtol=2e-12, atol=2e-12)
        np.testing.assert_allclose(jacobian_jax, target_jacobian, rtol=3e-12, atol=3e-12)
        np.testing.assert_allclose(time_jax, time_derivative, rtol=3e-12, atol=3e-12)
        _enclosed(point, certificate.spatial_enclosure)
        _enclosed(target_jacobian, certificate.target_jacobian)
        _enclosed(jacobian_jax, certificate.target_jacobian)
        _enclosed(time_derivative, certificate.time_derivative)
        _enclosed(time_jax, certificate.time_derivative)

        moment_jacobian, moment_time, moment_diffusion, velocity_jacobian = _moments(
            point, centers, biases, viscosity, time, target)
        np.testing.assert_allclose(target_jacobian, moment_jacobian, rtol=3e-12, atol=3e-12)
        np.testing.assert_allclose(time_derivative, moment_time, rtol=3e-12, atol=3e-12)
        np.testing.assert_allclose(time_derivative, target + moment_diffusion, rtol=3e-12, atol=3e-12)

        # J^-1 columns are frozen path directions. Differentiating these
        # directions would change the nondivergence-form hodograph operator.
        paths = np.stack((np.broadcast_to(target, (2, 2)), velocity_jacobian.T, np.zeros((2, 2))))
        jets = torch_jet(torch.tensor(paths), centers, biases, viscosity, time, tolerance=1e-13)
        assert bool(jets.base_result.converged.all())
        torch_diffusion = 2 * viscosity * jets.coefficients[2].detach().numpy().sum(axis=0)
        jax_diffusion = 2 * viscosity * np.asarray(jet_jax(jnp.asarray(paths)))[2].sum(axis=0)
        for diffusion in (moment_diffusion, torch_diffusion, jax_diffusion):
            np.testing.assert_allclose(diffusion, time_derivative - target, rtol=3e-10, atol=3e-12)
            _enclosed(diffusion, certificate.diffusion)
            _enclosed(target + diffusion, certificate.time_derivative)
            assert np.linalg.norm(diffusion) <= float(certificate.diffusion_norm_bound)
        diffusion_sizes.append(np.linalg.norm(moment_diffusion))
        coupling_sizes.append(abs(target_jacobian[0, 1]))
    assert max(diffusion_sizes) > 1e-4 and max(coupling_sizes) > 1e-4


def _mp(value, mp):
    value = F(value)
    return mp.mpf(value.numerator) / value.denominator


def test_point_box_encloses_independent_high_precision_physical_moments():
    mp = pytest.importorskip("mpmath")
    model = _model()
    target = (0.25, -0.375)
    certificate = certify_gaussian_inverse_evolution(model, tuple(Interval.point(value) for value in target))
    assert certificate is not None
    with mp.workdps(80):
        centers = [mp.matrix([_mp(value, mp) for value in row]) for row in model.centers]
        biases = [_mp(value, mp) for value in model.biases]
        viscosity, age = _mp(model.viscosity, mp), _mp(model.time, mp)
        wanted = mp.matrix([_mp(value, mp) for value in target])

        def posterior(point):
            weights = [mp.exp(bias - sum((point[k] - center[k])**2 for k in range(2)) / (4 * viscosity * age))
                       for center, bias in zip(centers, biases, strict=True)]
            return [weight / sum(weights) for weight in weights]

        def residual(x, y):
            point = mp.matrix([x, y])
            probability = posterior(point)
            mean = sum((weight * center for weight, center in zip(probability, centers, strict=True)), mp.zeros(2, 1))
            value = point - age * wanted - mean
            return value[0], value[1]

        point = mp.findroot(residual, (1, 0))
        probability = posterior(point)
        mean = sum((weight * center for weight, center in zip(probability, centers, strict=True)), mp.zeros(2, 1))
        centered = [center - mean for center in centers]
        covariance = sum((weight * row * row.T for weight, row in zip(probability, centered, strict=True)), mp.zeros(2))
        third_trace = sum((weight * sum(value**2 for value in row) * row
                           for weight, row in zip(probability, centered, strict=True)), mp.zeros(2, 1))
        hessian = mp.eye(2) - covariance / (2 * viscosity * age)
        jacobian = age * hessian**-1
        diffusion = mp.lu_solve(hessian, third_trace / (4 * viscosity * age**2))
        time_derivative = wanted + diffusion
        for values, bounds in ((point, certificate.spatial_enclosure), (diffusion, certificate.diffusion),
                              (time_derivative, certificate.time_derivative)):
            for value, bound in zip(values, bounds, strict=True):
                assert mp.mpf(bound.lo) <= value <= mp.mpf(bound.hi)
        for i, row in enumerate(certificate.target_jacobian):
            for j, bound in enumerate(row):
                assert mp.mpf(bound.lo) <= jacobian[i, j] <= mp.mpf(bound.hi)


def test_affine_family_has_identity_scaled_jacobian_and_zero_diffusion_over_whole_box():
    model = GaussianHeatMap(((2, -3),), (F(1, 7),), time=F(3, 2))
    box = (Interval(-2.0, 3.0), Interval(-1.0, 4.0))
    certificate = certify_gaussian_inverse_evolution(model, box)
    assert certificate is not None and certificate.diffusion_norm_bound == 0
    for target in _points(box):
        _enclosed(np.asarray([2.0, -3.0]) + 1.5 * target, certificate.spatial_enclosure)
        _enclosed(1.5 * np.eye(2), certificate.target_jacobian)
        _enclosed(np.zeros(2), certificate.diffusion)
        _enclosed(target, certificate.time_derivative)


def test_common_bias_and_center_translation_preserve_derivative_enclosures():
    pytest.importorskip("mpmath")
    model, box = _model(), _box()
    reference = certify_gaussian_inverse_evolution(model, box)
    translation = (F(8), -F(4))
    shifted = GaussianHeatMap(tuple(tuple(value + offset for value, offset in zip(row, translation, strict=True))
                                    for row in model.centers),
                              tuple(value + 10**100 for value in model.biases),
                              viscosity=model.viscosity, time=model.time)
    certificate = certify_gaussian_inverse_evolution(shifted, box)
    assert reference is not None and certificate is not None
    centers, biases = np.asarray(model.centers, dtype=np.float64), np.asarray(model.biases, dtype=np.float64)
    target = np.asarray([axis.mid for axis in box])
    result = torch_inverse(torch.tensor(target), centers, biases, float(model.viscosity), float(model.time), tolerance=1e-13)
    point = result.value.detach().numpy()
    jacobian, derivative, diffusion, _ = _moments(point, centers, biases, float(model.viscosity), float(model.time), target)
    _enclosed(point + np.asarray(translation, dtype=np.float64), certificate.spatial_enclosure)
    _enclosed(jacobian, certificate.target_jacobian)
    _enclosed(derivative, certificate.time_derivative)
    _enclosed(diffusion, certificate.diffusion)
    assert certificate.diffusion_norm_bound == reference.diffusion_norm_bound


def test_outside_contraction_regime_refuses_the_same_family_as_backends():
    model = GaussianHeatMap(((-2,), (2,)), (0, 0), time=1)
    assert model.contraction_bound == 2
    assert certify_gaussian_inverse_evolution(model, (Interval(-0.1, 0.1),)) is None
    torch_result = torch_inverse(torch.tensor([0.0], dtype=torch.float64), [[-2.0], [2.0]], [0.0, 0.0])
    jax_result = jax_inverse(jnp.asarray([0.0], dtype=jnp.float64), [[-2.0], [2.0]], [0.0, 0.0])
    assert not bool(torch_result.converged) and not bool(jax_result.converged)
    assert bool(torch.isnan(torch_result.value).all()) and bool(jnp.isnan(jax_result.value).all())
