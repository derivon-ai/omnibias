# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
"""Tempered temperature-collapse penalty (gradient-descent) LP/QP solver -- JAX.

The first-order homotopy solver is honestly a *tolerance* solver (not the Newton
interior point), so accuracy assertions use ``atol ~ 5e-3``; its selling points --
a closed-form temperature-collapse gradient, a nonnegative dual estimate, an
exterior/infeasible start, and full differentiability -- are checked exactly.
"""

from __future__ import annotations

import numpy as np
import pytest
from omnibias.convex import Certificate, certify_qp_optimum
from omnibias.convex.problem import PenaltyOptions

jax = pytest.importorskip("jax")
import jax.numpy as jnp  # noqa: E402
from omnibias.convex.jax import (  # noqa: E402
    penalty_descent,
    penalty_gradient,
    solve_lp,
    solve_lp_penalty,
    solve_qp_penalty,
)

scipy_linprog = pytest.importorskip("scipy.optimize").linprog


def _arr(x: object) -> jnp.ndarray:
    return jnp.asarray(np.asarray(x, dtype=np.float64))


def _lp_2d() -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    # max 3x + 2y == min -3x - 2y, s.t. x+y<=4, x+3y<=6, x>=0, y>=0 ; optimum (4, 0).
    c = _arr([-3.0, -2.0])
    A = _arr([[1.0, 1.0], [1.0, 3.0], [-1.0, 0.0], [0.0, -1.0]])
    b = _arr([4.0, 6.0, 0.0, 0.0])
    return c, A, b


def test_lp_matches_scipy_2d() -> None:
    c, A, b = _lp_2d()
    sol = solve_lp_penalty(c, A, b)
    ref = scipy_linprog(np.asarray(c), A_ub=np.asarray(A), b_ub=np.asarray(b), bounds=(None, None))
    assert sol.converged
    np.testing.assert_allclose(np.asarray(sol.x), ref.x, atol=5e-3)
    np.testing.assert_allclose(float(sol.obj), ref.fun, atol=5e-3)


def test_lp_matches_newton_interior_point() -> None:
    c, A, b = _lp_2d()
    sol = solve_lp_penalty(c, A, b)
    ref = solve_lp(c, A, b)
    np.testing.assert_allclose(np.asarray(sol.x), np.asarray(ref.x), atol=5e-3)
    np.testing.assert_allclose(float(sol.obj), float(ref.obj), atol=5e-3)


def test_qp_projection_onto_nonnegative_orthant() -> None:
    # min 1/2||x - p||^2  s.t.  -x <= 0   =>   x* = relu(p).
    p = _arr([1.5, -2.0, 0.7, -0.1])
    sol = solve_qp_penalty(jnp.eye(4), -p, -jnp.eye(4), jnp.zeros(4))
    np.testing.assert_allclose(np.asarray(sol.x), np.maximum(np.asarray(p), 0.0), atol=5e-3)


def test_dual_is_nonnegative_and_complementary() -> None:
    c, A, b = _lp_2d()
    sol = solve_lp_penalty(c, A, b)
    lam = np.asarray(sol.dual)
    # dual = mu * sigma(beta u) is a temperature-collapse activation -> nonnegative by construction.
    assert np.min(lam) >= 0.0
    # complementarity lambda_i * slack_i -> 0.
    assert np.max(lam * np.asarray(sol.slack)) < 5e-3
    # stationarity c + A^T lambda -> 0 (Q = 0).
    stat = np.asarray(c) + np.asarray(A).T @ lam
    assert np.max(np.abs(stat)) < 5e-2


def test_more_annealing_reduces_gap() -> None:
    # Growing the beta/mu homotopy drives the point toward the true optimum.
    c, A, b = _lp_2d()
    ref = float(solve_lp(c, A, b).obj)
    short = solve_lp_penalty(c, A, b, options=PenaltyOptions(stages=4, gd_steps=1000))
    long = solve_lp_penalty(c, A, b, options=PenaltyOptions(stages=10, gd_steps=1000))
    assert abs(float(long.obj) - ref) <= abs(float(short.obj) - ref) + 1e-9
    assert abs(float(long.obj) - ref) < 5e-3


def test_certificate_encloses_penalty_optimum() -> None:
    # Feed the GD primal + temperature-collapse dual into the verified certificate.
    p = np.array([1.5, -2.0, 0.7, -0.1])
    Q = np.eye(4)
    c = -p
    A = -np.eye(4)
    b = np.zeros(4)
    sol = solve_qp_penalty(_arr(Q), _arr(c), _arr(A), _arr(b))
    cert = certify_qp_optimum(Q, c, A, b, np.asarray(sol.x), np.asarray(sol.dual))
    x_star = np.maximum(p, 0.0)
    f_star = 0.5 * float(np.sum(x_star**2)) - float(p @ x_star)
    assert isinstance(cert, Certificate)
    assert cert.primal_feasible
    assert cert.enclosure.lo <= f_star <= cert.enclosure.hi
    assert cert.gap >= 0.0


def test_exterior_start_from_infeasible_point() -> None:
    # No phase-1: the exterior penalty solves from a wildly infeasible x0.
    c, A, b = _lp_2d()
    sol = solve_lp_penalty(c, A, b, x0=_arr([9.0, 9.0]))
    ref = solve_lp(c, A, b)
    assert sol.converged
    np.testing.assert_allclose(np.asarray(sol.x), np.asarray(ref.x), atol=5e-3)


def test_closed_form_gradient_equals_autograd() -> None:
    # penalty_gradient is exactly grad of F = c.x + 1/2 x'Qx + mu*sum softplus(beta u)/beta.
    _, A, b = _lp_2d()
    Q = jnp.eye(2)
    c = jnp.array([0.3, -1.2])
    x = jnp.array([0.7, 0.4])
    beta, mu = 5.3, 2.1

    def objective(z: jnp.ndarray) -> jnp.ndarray:
        u = A @ z - b
        return c @ z + 0.5 * z @ (Q @ z) + mu * jnp.sum(jax.nn.softplus(beta * u)) / beta

    g_auto = jax.grad(objective)(x)
    g_closed = penalty_gradient(Q, c, A, b, x, beta=beta, mu=mu)
    np.testing.assert_allclose(np.asarray(g_closed), np.asarray(g_auto), atol=1e-12)


def test_penalty_descent_is_differentiable() -> None:
    # jax.grad through a fixed-beta descent (strongly convex, prox=0) is finite/nonzero.
    _, A, b = _lp_2d()

    def solved_sum(cost: jnp.ndarray) -> jnp.ndarray:
        x = penalty_descent(
            jnp.eye(2), cost, A, b, jnp.zeros(2), beta=8.0, mu=6.0, steps=300, prox=0.0
        )
        return jnp.sum(x)

    grad = jax.grad(solved_sum)(jnp.array([-3.0, -2.0]))
    assert bool(jnp.all(jnp.isfinite(grad)))
    assert float(jnp.max(jnp.abs(grad))) > 1e-6


def test_negative_order_options_rejected() -> None:
    with pytest.raises(ValueError):
        PenaltyOptions(prox=-1.0)
    with pytest.raises(ValueError):
        PenaltyOptions(step_safety=2.0)
