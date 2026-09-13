# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Independent directional and dense references for residual-family adapters."""

import numpy as np
import pytest
from omnibias.geometry.continuation import continue_branch, locate_fold
from omnibias.geometry.continuation_directional import directional_residual_family


def test_mixed_directional_callbacks_match_independent_dense_polynomials():
    def value(y):
        x, z, mu = y
        return np.array([x * x + x * z - mu, z**3 + np.sin(mu)])

    def first(y, v):
        x, z, mu = y
        return np.array(
            [(2 * x + z) * v[0] + x * v[1] - v[2], 3 * z * z * v[1] + np.cos(mu) * v[2]]
        )

    def second(y, u, v):
        return np.array(
            [
                2 * u[0] * v[0] + u[0] * v[1] + u[1] * v[0],
                6 * y[1] * u[1] * v[1] - np.sin(y[2]) * u[2] * v[2],
            ]
        )

    def third(y, u, v, w):
        return np.array([0.0, 6 * u[1] * v[1] * w[1] - np.cos(y[2]) * u[2] * v[2] * w[2]])

    family = directional_residual_family(
        value, first, state_dimension=2, second=second, third=third
    )
    y = np.array([0.2, -0.4, 0.3])
    expected_h = np.zeros((2, 3, 3))
    expected_h[0, :2, :2] = [[2, 1], [1, 0]]
    expected_h[1, 1, 1], expected_h[1, 2, 2] = 6 * y[1], -np.sin(y[2])
    expected_t = np.zeros((2, 3, 3, 3))
    expected_t[1, 1, 1, 1], expected_t[1, 2, 2, 2] = 6, -np.cos(y[2])
    np.testing.assert_allclose(family.second(y), expected_h, rtol=1e-12, atol=1e-14)
    np.testing.assert_allclose(family.third(y), expected_t, rtol=1e-12, atol=1e-14)
    rng = np.random.default_rng(902)
    for _ in range(12):
        u, v, w = rng.normal(size=(3, 3))
        np.testing.assert_allclose(family.jacobian(y) @ v, first(y, v), rtol=1e-12, atol=1e-14)
        np.testing.assert_allclose(
            np.einsum("ijk,j,k->i", family.second(y), u, v), second(y, u, v), rtol=1e-12, atol=1e-14
        )
        np.testing.assert_allclose(
            np.einsum("ijkl,j,k,l->i", family.third(y), u, v, w),
            third(y, u, v, w),
            rtol=1e-12,
            atol=1e-14,
        )


def test_directional_fold_callbacks_drive_continuation_and_event_localization():
    family = directional_residual_family(
        lambda y: np.array([y[0] ** 2 - y[1]]),
        lambda y, v: np.array([2 * y[0] * v[0] - v[1]]),
        state_dimension=1,
        second=lambda _y, u, v: np.array([2 * u[0] * v[0]]),
        third=lambda _y, _u, _v, _w: np.zeros(1),
    )
    event = locate_fold(family, np.array([0.1, 0.02]))
    assert event.nondegenerate and event.residual < 1e-10
    path = continue_branch(
        family, np.array([-0.2, 0.04]), direction=np.array([1.0, 0.0]), step=0.03, n_steps=20
    )
    assert path.status == "complete" and path.points[-1, 0] > 0
    np.testing.assert_allclose(path.points[:, 1], path.points[:, 0] ** 2, atol=1e-10)


def test_derivative_budget_and_callback_shape_are_checked_before_solve():
    calls = []

    def value(y):
        calls.append(y)
        return y[:1]

    with pytest.raises(ValueError, match="budget"):
        directional_residual_family(
            value, lambda y, v: v[:1], state_dimension=10_000, max_coefficients=1_000
        )
    assert not calls
    family = directional_residual_family(value, lambda y, v: v, state_dimension=1)
    with pytest.raises(ValueError, match="residual vector"):
        family.jacobian(np.ones(2))
    with pytest.raises(ValueError, match="state-plus-parameter"):
        family.value(np.ones(3))
    with pytest.raises(ValueError, match="second derivative"):
        directional_residual_family(
            value, lambda y, v: v[:1], state_dimension=1, third=lambda y, u, v, w: y[:1]
        )
