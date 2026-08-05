# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
"""Temperature-collapse-geometry warm starts: vertex prediction + phase-1 skip."""

from __future__ import annotations

import numpy as np
import pytest
from omnibias.convex import (
    active_set_warm_start,
    geometry_warm_start,
    predicted_vertex,
)

jax = pytest.importorskip("jax")
import jax.numpy as jnp  # noqa: E402
from omnibias.convex.jax import solve_qp  # noqa: E402


def _arr(x: object) -> jnp.ndarray:
    return jnp.asarray(np.asarray(x, dtype=np.float64))


# box -2 <= x,y and x <= 2, y <= 3 ; origin strictly interior.
_A = np.array([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0]])
_B = np.array([2.0, 3.0, 2.0, 2.0])


def test_predicted_vertex_recovers_active_intersection() -> None:
    # optimum of min -x-y is the vertex x=2, y=3 (rows 0,1 active).
    scores = np.array([0.9, 0.8, 0.1, 0.05])
    vertex = predicted_vertex(_A, _B, scores)
    assert vertex is not None
    np.testing.assert_allclose(vertex, [2.0, 3.0], atol=1e-12)


def test_predicted_vertex_singular_selection_returns_none() -> None:
    # duplicate top-scored rows -> singular 2x2 system.
    A = np.array([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    b = np.array([1.0, 1.0, 1.0])
    scores = np.array([0.9, 0.89, 0.1])
    assert predicted_vertex(A, b, scores) is None


def test_geometry_warm_start_passthrough_when_feasible() -> None:
    x = np.array([0.0, 0.0])  # strictly interior
    out = geometry_warm_start(_A, _B, x)
    assert out is not None
    np.testing.assert_allclose(out, x)


def test_geometry_warm_start_backs_off_to_interior() -> None:
    out = geometry_warm_start(_A, _B, np.array([2.0, 3.0]))  # on the boundary
    assert out is not None
    slack = _B - _A @ out
    assert np.all(slack > 0.0)  # strictly feasible
    # backed off along the segment toward the origin, so still near the vertex.
    assert out[0] > 1.5 and out[1] > 2.0


def test_geometry_warm_start_none_without_feasible_anchor() -> None:
    # x >= 1 box: origin infeasible, so the default anchor fails -> None.
    A = np.array([[-1.0], [1.0]])
    b = np.array([-1.0, 5.0])
    assert geometry_warm_start(A, b, np.array([10.0])) is None


def test_active_set_warm_start_is_strictly_feasible() -> None:
    scores = np.array([0.9, 0.8, 0.1, 0.05])
    out = active_set_warm_start(_A, _B, scores)
    assert out is not None
    assert np.all(_B - _A @ out > 0.0)


def test_warm_start_skips_phase1_and_cuts_newton_iterations() -> None:
    # x in [1, 5]^2 : origin infeasible -> cold solve must run phase-1.
    n = 2
    A = np.vstack([np.eye(n), -np.eye(n)])
    b = np.array([5.0, 5.0, -1.0, -1.0])  # x <= 5 ; -x <= -1  (x >= 1)
    Q = np.eye(n)
    c = -3.0 * np.ones(n)  # min 1/2||x - 3||^2 -> x* = [3, 3] (interior)

    cold = solve_qp(_arr(Q), _arr(c), _arr(A), _arr(b))
    warm = solve_qp(_arr(Q), _arr(c), _arr(A), _arr(b), x0=_arr([3.0, 3.0]))

    assert cold.converged and warm.converged
    np.testing.assert_allclose(np.asarray(cold.x), [3.0, 3.0], atol=1e-6)
    np.testing.assert_allclose(np.asarray(warm.x), [3.0, 3.0], atol=1e-6)
    # cold pays for a phase-1 path-follow; the feasible warm start skips it.
    assert warm.newton_iterations < cold.newton_iterations
