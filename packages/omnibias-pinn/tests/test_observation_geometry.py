# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""General information/nuisance geometry and smooth design objectives."""

import numpy as np
import pytest
from omnibias.pinn.inverse.design import design_gradient, design_score, differentiable_design_score
from omnibias.pinn.inverse.observation import (
    ObservationModel,
    candidate_information,
    identifiability_report,
    observation_information,
    profile_information,
)


def test_correlated_observations_and_rotated_null_direction():
    x = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    # Only theta0+theta1 and theta2 are visible.
    j = np.column_stack((x[:, 0], x[:, 0], x[:, 1]))
    model = ObservationModel(lambda t, d: j @ t, lambda t, d: j, ("a", "b", "c"))
    covariance = np.eye(3) + 0.2 * np.ones((3, 3))
    info = observation_information(model, np.ones(3), x, covariance=covariance)
    np.testing.assert_allclose(info.fisher, j.T @ np.linalg.solve(covariance, j), atol=1e-12)
    report = identifiability_report(info)
    assert report.rank == 2 and report.weak_directions.shape == (1, 3)
    np.testing.assert_allclose(j @ report.weak_directions.T, 0, atol=1e-12)
    assert not report.certified
    prof = profile_information(info.whitened_jacobian, [2], [0, 1])
    assert prof.nuisance_rank == 1
    residual = (
        info.whitened_jacobian[:, 2]
        - info.whitened_jacobian[:, :2]
        @ np.linalg.lstsq(info.whitened_jacobian[:, :2], info.whitened_jacobian[:, 2], rcond=None)[
            0
        ]
    )
    assert prof.fisher[0, 0] == pytest.approx(residual @ residual)


def test_generic_decay_candidates_and_nuisance():
    def value(t, d):
        return t[0] * np.exp(-t[1] * d[:, 0]) + t[2]

    def jac(t, d):
        e = np.exp(-t[1] * d[:, 0])
        return np.column_stack((e, -t[0] * d[:, 0] * e, np.ones(len(d))))

    model = ObservationModel(value, jac, ("amplitude", "rate", "offset"))
    d = np.linspace(0, 4, 11)[:, None]
    theta = np.array([2.0, 0.7, 0.1])
    blocks = candidate_information(model, theta, d, covariance=0.04)
    info = observation_information(model, theta, d, covariance=0.04)
    np.testing.assert_allclose(np.sum(blocks, axis=0), info.fisher, atol=1e-12)
    assert identifiability_report(info).rank == 3
    prof = profile_information(info.whitened_jacobian, [1], [0, 2])
    assert 0 < prof.fisher[0, 0] < info.fisher[1, 1]


@pytest.mark.parametrize("criterion", ["D", "A"])
def test_design_gradient_and_backend_autodiff(criterion):
    jax = pytest.importorskip("jax")
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp

    torch = pytest.importorskip("torch")
    factors = np.array([[1.0, 0.0], [0.0, 2.0], [1.0, 1.0]])
    blocks = np.einsum("ci,cj->cij", factors, factors)
    prior = np.eye(2)
    weights = np.array([0.2, 0.3, 0.5])
    expected = design_gradient(blocks, weights, prior, criterion=criterion)

    def fn(w):
        return differentiable_design_score(
            jnp.asarray(blocks), w, jnp.asarray(prior), backend="jax", criterion=criterion
        )

    np.testing.assert_allclose(jax.grad(fn)(jnp.asarray(weights)), expected, rtol=1e-11)
    w = torch.tensor(weights, requires_grad=True)
    out = differentiable_design_score(
        torch.tensor(blocks), w, torch.tensor(prior), backend="torch", criterion=criterion
    )
    out.backward()
    np.testing.assert_allclose(w.grad.numpy(), expected, rtol=1e-11)
    assert float(out.detach()) == pytest.approx(
        design_score(blocks, weights, prior, criterion=criterion)
    )


def test_input_guards_and_zero_jacobian():
    model = ObservationModel(
        lambda t, d: t[0] ** 3 * np.ones(len(d)),
        lambda t, d: np.full((len(d), 1), 3 * t[0] ** 2),
        ("t",),
    )
    assert identifiability_report(observation_information(model, np.zeros(1), np.ones(2))).rank == 0
    with pytest.raises(ValueError, match="covariance"):
        observation_information(model, np.ones(1), np.ones(2), covariance=-1)
    with pytest.raises(ValueError, match="distinct"):
        profile_information(np.eye(2), [0], [0])
