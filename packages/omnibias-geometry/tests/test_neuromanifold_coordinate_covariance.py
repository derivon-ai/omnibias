# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Numerical covariance under a nonorthogonal regular chart change."""

import numpy as np
from omnibias.geometry.neuromanifold import (
    ObservationMetric,
    extrinsic_geometry,
    least_squares_hessian,
    quotient_step,
)


def test_regular_coordinate_change_preserves_represented_natural_step():
    # This graph is an injective regular chart because its first two outputs
    # are the coordinates themselves. S is an admissible invertible linear
    # chart change, not a sampled-nullspace claim about a nonlinear quotient.
    s = np.array([[1.5, 0.4], [-0.3, 0.8]])
    theta = np.array([0.4, -0.2])
    phi = np.linalg.solve(s, theta)
    target = np.array([0.1, 0.2, 0.3])
    metric = ObservationMetric(np.array([[2.0, 0.2, 0.0], [0.2, 1.4, 0.1], [0.0, 0.1, 0.8]]))

    def observation(q):
        return np.array([q[0], q[1], q[0] ** 2 + 0.5 * q[1] ** 2])

    def objective(q):
        residual = observation(q) - target
        return 0.5 * float(residual @ metric.weight @ residual)

    j = np.array([[1.0, 0.0], [0.0, 1.0], [2.0 * theta[0], theta[1]]])
    h = np.zeros((3, 2, 2))
    h[2] = np.diag([2.0, 1.0])
    j_phi = j @ s
    h_phi = np.einsum("ki,akl,lj->aij", s, h, s)
    residual = observation(theta) - target
    g = j.T @ metric.weight @ residual
    g_phi = j_phi.T @ metric.weight @ residual
    gram = metric.dense(j)
    gram_phi = metric.dense(j_phi)

    np.testing.assert_allclose(gram_phi, s.T @ gram @ s, rtol=1e-13, atol=1e-13)
    np.testing.assert_allclose(g_phi, s.T @ g, rtol=1e-13, atol=1e-13)
    np.testing.assert_allclose(
        least_squares_hessian(j_phi, h_phi, residual, metric),
        s.T @ least_squares_hessian(j, h, residual, metric) @ s,
        rtol=1e-13,
        atol=1e-13,
    )
    original = extrinsic_geometry(j, metric, h)
    transformed = extrinsic_geometry(j_phi, metric, h_phi)
    np.testing.assert_allclose(
        transformed.tangent_projection, original.tangent_projection, rtol=1e-13, atol=1e-13
    )
    np.testing.assert_allclose(
        transformed.second_fundamental,
        np.einsum("ki,akl,lj->aij", s, original.second_fundamental, s),
        rtol=1e-13,
        atol=1e-13,
    )

    # An isotropic coordinate damping term is deliberately zero: adding the
    # same lambda*I in each chart is not a covariant regularization tensor.
    direct = quotient_step(theta, g, gram, objective, damping=0.0)
    changed = quotient_step(phi, g_phi, gram_phi, lambda q: objective(s @ q), damping=0.0)
    assert direct.accepted and changed.accepted
    assert direct.step_size == changed.step_size
    np.testing.assert_allclose(s @ changed.coordinates, direct.coordinates, rtol=1e-13, atol=1e-13)
    np.testing.assert_allclose(
        j @ (direct.coordinates - theta),
        j_phi @ (changed.coordinates - phi),
        rtol=1e-13,
        atol=1e-13,
    )
    # The fixture is not an orthogonal change that also preserves Euclidean
    # gradients: the natural metric is necessary for this invariance.
    assert np.linalg.norm(j @ g - j_phi @ g_phi) > 0.1
