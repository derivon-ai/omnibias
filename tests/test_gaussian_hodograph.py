# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Forward inverse jets satisfy the coupled hodograph and dual-potential PDEs."""

import math

import numpy as np
import pytest
import torch
from omnibias.torch.gaussian_heat import gaussian_heat_inverse
from omnibias.torch.gaussian_jet import gaussian_heat_inverse_jet


def _moments(point, centers, biases, viscosity, age):
    """Independent physical-coordinate Gaussian sums, without inverse helpers."""
    distances = ((point - centers) ** 2).sum(axis=1)
    logits = biases - distances / (4 * viscosity * age)
    weights = np.exp(logits - logits.max())
    probabilities = weights / weights.sum()
    mean = probabilities @ centers
    centered = centers - mean
    covariance = (centered.T * probabilities) @ centered
    mean_time = (probabilities * distances / (4 * viscosity * age**2)) @ centered
    jacobian = np.eye(point.size) / age - covariance / (2 * viscosity * age**2)
    return mean, mean_time, jacobian


@pytest.mark.parametrize("age", [0.75, 1.5, 3.0])
@pytest.mark.parametrize("target", [(0.2, -0.35), (-0.4, 0.1)])
def test_coupled_hodograph_time_derivative_equals_forward_jet_diffusion(age, target):
    centers = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 2.0], [1.0, 1.0]])
    biases = np.array([0.0, 0.3, -0.4, 0.2])
    viscosity = 1.25
    time = torch.tensor(age, dtype=torch.float64, requires_grad=True)
    wanted = torch.tensor(target, dtype=torch.float64)
    inverse = gaussian_heat_inverse(wanted, centers, biases, viscosity, time, tolerance=1e-13)
    assert bool(inverse.converged) and float(inverse.contraction_bound) < 1
    time_derivative = np.array([
        torch.autograd.grad(component, time, retain_graph=True)[0].item()
        for component in inverse.value
    ])
    point = inverse.value.detach().numpy()
    mean, mean_time, jacobian = _moments(point, centers, biases, viscosity, age)
    actual_velocity = (point - mean) / age
    np.testing.assert_allclose(actual_velocity, target, rtol=0, atol=1e-13)
    assert abs(jacobian[0, 1]) > 1e-4  # Both spatial coordinates interact.
    independent_time = np.linalg.solve(jacobian, (actual_velocity + mean_time) / age)
    np.testing.assert_allclose(time_derivative, independent_time, rtol=2e-11, atol=2e-12)

    # Each path has a constant, frozen direction G[:, r]. Differentiating the
    # direction itself would add terms absent from the nondivergence-form PDE.
    paths = np.stack([np.broadcast_to(target, (2, 2)), jacobian.T, np.zeros((2, 2))])
    jets = gaussian_heat_inverse_jet(
        torch.tensor(paths, dtype=torch.float64), centers, biases,
        viscosity, age, tolerance=1e-13,
    )
    assert bool(jets.base_result.converged.all())
    assert float(jets.base_target_residual.abs().max()) < 1e-13
    # Coefficient two is one half of the raw second directional derivative.
    diffusion = 2 * viscosity * jets.coefficients[2].detach().numpy().sum(axis=0)
    assert np.linalg.norm(diffusion) > 1e-4
    np.testing.assert_allclose(time_derivative, np.array(target) + diffusion, rtol=2e-10, atol=2e-12)


@pytest.mark.parametrize("age", [0.75, 1.5, 3.0])
def test_legendre_time_equation_requires_normalized_heat_density(age):
    centers = torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 2.0], [1.0, 1.0]], dtype=torch.float64)
    biases = torch.tensor([0.0, 0.3, -0.4, 0.2], dtype=torch.float64)
    target = torch.tensor([0.2, -0.35], dtype=torch.float64, requires_grad=True)
    time = torch.tensor(age, dtype=torch.float64, requires_grad=True)
    viscosity = 1.25
    inverse = gaussian_heat_inverse(target, centers, biases, viscosity, time, tolerance=1e-13)
    assert bool(inverse.converged)
    point = inverse.value
    logits = biases - (point - centers).square().sum(-1) / (4 * viscosity * time)
    log_unnormalized = torch.logsumexp(logits, dim=0)
    log_heat = log_unnormalized - centers.shape[1] / 2 * torch.log(4 * math.pi * viscosity * time)
    dual = target @ point + 2 * viscosity * log_heat
    unnormalized_dual = target @ point + 2 * viscosity * log_unnormalized
    dual_time = torch.autograd.grad(dual, time, retain_graph=True)[0]
    unnormalized_time = torch.autograd.grad(unnormalized_dual, time, retain_graph=True)[0]
    dual_gradient = torch.autograd.grad(dual, target, create_graph=True)[0]
    dual_hessian = torch.stack([
        torch.autograd.grad(component, target, retain_graph=True)[0]
        for component in dual_gradient
    ])
    _, _, jacobian = _moments(
        point.detach().numpy(), centers.numpy(), biases.numpy(), viscosity, age,
    )
    np.testing.assert_allclose(dual_gradient.detach().numpy(), point.detach().numpy(), rtol=2e-11, atol=2e-12)
    np.testing.assert_allclose(dual_hessian.detach().numpy(), np.linalg.inv(jacobian), rtol=2e-10, atol=2e-12)
    expected = float(target.detach().square().sum()) / 2 - viscosity * np.trace(jacobian)
    assert float(dual_time) == pytest.approx(expected, rel=2e-11, abs=2e-12)
    assert float(unnormalized_time - dual_time) == pytest.approx(viscosity * 2 / age, rel=2e-12)
