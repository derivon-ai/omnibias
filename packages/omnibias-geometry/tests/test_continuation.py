# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Continuation traverses singular projections and classifies analytic events."""

import numpy as np
import pytest
from omnibias.geometry.continuation import (
    ImplicitFamily,
    continue_branch,
    follow_model_boundary,
    locate_fold,
    locate_hopf,
)


def fold_family():
    return ImplicitFamily(
        lambda y: np.array([y[0] ** 2 - y[1]]),
        lambda y: np.array([[2 * y[0], -1.0]]),
        lambda y: np.array([[[2.0, 0.0], [0.0, 0.0]]]),
    )


def test_observation_geodesic_preserves_speed_and_domain_stop():
    def jac(t):
        return np.array([[1.0], [2 * t[0]]])

    def second(t, v):
        return np.array([0.0, 2 * v[0] ** 2])

    path = follow_model_boundary(
        jac, second, np.array([0.0]), direction=np.ones(1), step=0.01, n_steps=70
    )
    assert path.status == "complete" and path.points[-1, 0] > 0.5
    speeds = [np.linalg.norm(jac(t) @ v) for t, v in zip(path.points, path.velocities, strict=True)]
    np.testing.assert_allclose(speeds, 1.0, atol=1e-8)
    stopped = follow_model_boundary(
        jac, second, np.array([0.0]), direction=np.ones(1), admissible=lambda t: t[0] < 0.2
    )
    assert stopped.status == "domain_boundary"


def test_arclength_crosses_fold_and_preserves_residual():
    f = fold_family()
    branch = continue_branch(
        f, np.array([1.0, 1.0]), direction=np.array([-1.0, -2.0]), step=0.05, n_steps=65
    )
    assert branch.status == "complete"
    assert branch.points[-1, 0] < -0.5
    assert max(branch.residuals) < 1e-10
    assert np.min(branch.points[:, 1]) < 0.001
    event = locate_fold(f, np.array([0.08, 0.01]))
    assert event.nondegenerate and not event.certified
    np.testing.assert_allclose(event.point, [0, 0], atol=1e-10)


def hopf_family(a):
    def value(y):
        x, z, p = y
        return np.array([p * x - z + a * x * (x * x + z * z), x + p * z + a * z * (x * x + z * z)])

    def jac(y):
        x, z, p = y
        return np.array(
            [
                [p + a * (3 * x * x + z * z), -1 + 2 * a * x * z, x],
                [1 + 2 * a * x * z, p + a * (x * x + 3 * z * z), z],
            ]
        )

    def second(y):
        x, z, _ = y
        h = np.zeros((2, 3, 3))
        h[0, :2, :2] = [[6 * a * x, 2 * a * z], [2 * a * z, 2 * a * x]]
        h[1, :2, :2] = [[2 * a * z, 2 * a * x], [2 * a * x, 6 * a * z]]
        h[0, 0, 2] = h[0, 2, 0] = h[1, 1, 2] = h[1, 2, 1] = 1
        return h

    def third(y):
        c = np.zeros((2, 3, 3, 3))
        c[0, 0, 0, 0] = c[1, 1, 1, 1] = 6 * a
        for i, j, k in [(0, 1, 1), (1, 0, 1), (1, 1, 0)]:
            c[0, i, j, k] = 2 * a
        for i, j, k in [(1, 0, 0), (0, 1, 0), (0, 0, 1)]:
            c[1, i, j, k] = 2 * a
        return c

    return ImplicitFamily(value, jac, second, third)


@pytest.mark.parametrize("a", [-1.0, 1.0, 0.0])
def test_hopf_normal_form_and_degenerate_control(a):
    event = locate_hopf(hopf_family(a), np.array([0.0, 0.0, 0.05]))
    assert event.residual < 1e-9
    assert event.frequency == pytest.approx(1.0)
    assert event.transversality == pytest.approx(1.0)
    assert event.coefficient == pytest.approx(2 * a)
    assert event.nondegenerate == (a != 0)


def test_bratu_analytic_solution_curve_fold():
    # u(x)=2 log(cosh(a/2)/cosh(a(x-1/2))), lambda=2a²/cosh²(a/2).
    # This residual represents that analytic family, not a sampled PDE residual.
    def f(y):
        a, p = y
        return np.array([p * np.cosh(a / 2) ** 2 - 2 * a * a])

    def j(y):
        a, p = y
        return np.array([[p * np.sinh(a) / 2 - 4 * a, np.cosh(a / 2) ** 2]])

    def h(y):
        a, p = y
        return np.array([[[p * np.cosh(a) / 2 - 4, np.sinh(a) / 2], [np.sinh(a) / 2, 0.0]]])

    event = locate_fold(ImplicitFamily(f, j, h), np.array([2.4, 3.51]))
    assert event.nondegenerate
    assert event.point[0] == pytest.approx(2.3993572805, abs=1e-8)
    assert event.point[1] == pytest.approx(3.5138307191, abs=1e-8)


def test_missing_derivative_and_bad_start_refused():
    with pytest.raises(ValueError, match="start"):
        continue_branch(fold_family(), np.array([0.0, 1.0]))
    with pytest.raises(ValueError, match="third"):
        locate_hopf(fold_family(), np.array([0.0, 0.0]))
