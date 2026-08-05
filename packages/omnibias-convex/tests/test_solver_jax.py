# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
"""LP/QP interior-point solver (JAX) vs scipy / closed-form references."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

jax = pytest.importorskip("jax")
from omnibias.convex.jax import InfeasibleProblemError, solve_lp, solve_qp  # noqa: E402

scipy_linprog = pytest.importorskip("scipy.optimize").linprog


def _arr(x):
    return jnp.asarray(np.asarray(x, dtype=np.float64))


def test_lp_matches_scipy_2d() -> None:
    # max 3x + 2y  ==  min -3x - 2y, s.t. x+y<=4, x+3y<=6, x>=0, y>=0.
    c = _arr([-3.0, -2.0])
    A = _arr([[1.0, 1.0], [1.0, 3.0], [-1.0, 0.0], [0.0, -1.0]])
    b = _arr([4.0, 6.0, 0.0, 0.0])
    sol = solve_lp(c, A, b)
    ref = scipy_linprog(np.asarray(c), A_ub=np.asarray(A), b_ub=np.asarray(b), bounds=(None, None))
    assert sol.converged
    np.testing.assert_allclose(np.asarray(sol.x), ref.x, atol=1e-6)
    np.testing.assert_allclose(float(sol.obj), ref.fun, atol=1e-6)


def test_lp_matches_scipy_random_bounded() -> None:
    rng = np.random.default_rng(0)
    n, m = 3, 12
    c = rng.standard_normal(n)
    # A includes a box -I/+I block so the feasible region is bounded (full col rank).
    G = rng.standard_normal((m - 2 * n, n))
    A = np.vstack([G, np.eye(n), -np.eye(n)])
    b = np.concatenate([rng.uniform(1.0, 3.0, size=m - 2 * n), 5.0 * np.ones(2 * n)])
    sol = solve_lp(_arr(c), _arr(A), _arr(b))
    ref = scipy_linprog(c, A_ub=A, b_ub=b, bounds=(None, None))
    assert sol.converged and ref.status == 0
    np.testing.assert_allclose(float(sol.obj), ref.fun, atol=1e-5)
    np.testing.assert_allclose(np.asarray(sol.x), ref.x, atol=1e-4)


def test_qp_projection_onto_nonnegative_orthant() -> None:
    # min 1/2||x - p||^2  s.t.  -x <= 0   =>   x* = relu(p).
    p = _arr([1.5, -2.0, 0.7, -0.1])
    n = 4
    Q = jnp.eye(n)
    c = -p
    A = -jnp.eye(n)
    b = jnp.zeros(n)
    sol = solve_qp(Q, c, A, b)
    np.testing.assert_allclose(np.asarray(sol.x), np.maximum(np.asarray(p), 0.0), atol=1e-6)


def test_qp_projection_onto_halfspace() -> None:
    # min 1/2||x - 1||^2 s.t. sum(x) <= 1  (interior exists; constraint active)
    #   => x* = 1 - (sum(1) - 1)/n * 1 = (1/n) * 1.
    n = 5
    Q = jnp.eye(n)
    c = -jnp.ones(n)  # min 1/2 x^T x - 1^T x == 1/2||x-1||^2 + const
    A = jnp.ones((1, n))
    b = _arr([1.0])
    sol = solve_qp(Q, c, A, b)
    np.testing.assert_allclose(np.asarray(sol.x), np.full(n, 1.0 / n), atol=1e-6)
    np.testing.assert_allclose(float(jnp.sum(sol.x)), 1.0, atol=1e-6)


def test_kkt_stationarity_and_complementarity() -> None:
    c = _arr([-3.0, -2.0])
    A = _arr([[1.0, 1.0], [1.0, 3.0], [-1.0, 0.0], [0.0, -1.0]])
    b = _arr([4.0, 6.0, 0.0, 0.0])
    sol = solve_lp(c, A, b)
    lam = np.asarray(sol.dual)
    stat = np.asarray(c) + np.asarray(A).T @ lam  # Q=0 => c + A^T lambda = 0
    # Barrier dual estimate lambda = 1/(t s); stationarity holds to centering accuracy.
    assert np.max(np.abs(stat)) < 1e-3
    assert np.min(lam) > -1e-12  # dual feasibility (lambda >= 0 by construction)
    assert np.max(np.asarray(sol.slack) * lam) < 1e-6  # complementarity 1/t


def test_phase1_when_origin_infeasible() -> None:
    # x >= 1 and x <= 3  (origin x=0 is infeasible: -x <= -1 violated).
    c = _arr([1.0])
    A = _arr([[-1.0], [1.0]])
    b = _arr([-1.0, 3.0])
    sol = solve_lp(c, A, b)  # min x => x* = 1
    assert sol.converged
    np.testing.assert_allclose(float(sol.x[0]), 1.0, atol=1e-6)


def test_infeasible_raises() -> None:
    # x <= -1 and x >= 1 simultaneously: empty feasible set.
    c = _arr([1.0])
    A = _arr([[1.0], [-1.0]])
    b = _arr([-1.0, -1.0])
    with pytest.raises(InfeasibleProblemError):
        solve_lp(c, A, b)


def test_warm_start_strict_feasibility_check() -> None:
    c = _arr([1.0, 1.0])
    A = _arr([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0]])
    b = _arr([5.0, 5.0, 0.0, 0.0])
    with pytest.raises(InfeasibleProblemError):
        solve_lp(c, A, b, x0=_arr([10.0, 10.0]))  # outside the box


def _covering_lp(P: int, S: int, k: int, seed: int):
    """Fractional set-cover LP ``min sum x  s.t.  cover x >= 1, 0 <= x <= 1`` (interior exists)."""
    rng = np.random.default_rng(seed)
    cover = np.zeros((P, S))
    for i in range(P):
        cover[i, rng.choice(S, size=k, replace=False)] = 1.0
    eye = np.eye(S)
    A = np.concatenate([-cover, -eye, eye], axis=0)
    b = np.concatenate([-np.ones(P), np.zeros(S), np.ones(S)])
    return np.ones(S), A, b


def test_ill_conditioned_covering_lp_stays_finite() -> None:
    # Regression: a many-constraint covering LP drives the active slacks to ~m/t, so the
    # barrier Hessian A^T diag(1/s^2) A becomes numerically singular near the float64 limit.
    # The solver must keep the last centered iterate (finite, correct) instead of a NaN.
    c, A, b = _covering_lp(P=26, S=64, k=3, seed=0)
    sol = solve_lp(_arr(c), _arr(A), _arr(b))
    ref = scipy_linprog(c, A_ub=A, b_ub=b, bounds=(None, None))
    assert bool(jnp.isfinite(sol.x).all())
    assert np.isfinite(float(sol.obj))
    np.testing.assert_allclose(float(sol.obj), ref.fun, atol=1e-4)
    assert bool((sol.slack > 0.0).all())
    # The retained iterate respects the explicit 0 <= x <= 1 box (never escapes feasibility).
    assert bool((sol.x >= -1e-6).all()) and bool((sol.x <= 1.0 + 1e-6).all())


def _duplicated_cover_lp(P: int, reps: int, k: int, seed: int):
    """Covering LP whose ``k`` base columns are each duplicated ``reps`` times.

    The exact column degeneracy makes the barrier Hessian ``A^T diag(1/s^2) A`` singular, so the
    central path breaks down near the optimal face -- the case where a spurious tiny line-search
    step used to push the primal out of the ``0 <= x <= 1`` box while ``m/t`` still looked
    converged (the structured-image LP that returned ``x ~ 1e34`` as "converged").
    """
    rng = np.random.default_rng(seed)
    base = np.zeros((P, k))
    for i in range(P):
        base[i, rng.integers(0, k)] = 1.0
    for j in range(k):
        base[rng.choice(P, size=max(1, P // k), replace=False), j] = 1.0
    cols = np.concatenate([base] * reps, axis=1)
    s = cols.shape[1]
    A = np.concatenate([-cols, -np.eye(s), np.eye(s)], axis=0)
    b = np.concatenate([-np.ones(P), np.zeros(s), np.ones(s)])
    return np.ones(s), A, b


def test_degenerate_duplicate_columns_stays_primal_feasible() -> None:
    # Regression: duplicated columns make the barrier Hessian exactly singular, so the central
    # path breaks down. The solver must keep the last *primal-feasible* iterate, not a divergent
    # primal that escaped the 0 <= x <= 1 box while still reporting a tiny m/t gap as converged.
    c, A, b = _duplicated_cover_lp(P=20, reps=8, k=4, seed=0)
    At, bt = _arr(A), _arr(b)
    sol = solve_lp(_arr(c), At, bt)
    assert bool(jnp.isfinite(sol.x).all())
    # Every constraint (including the box) is respected: the primal never left the feasible set.
    assert bool((At @ sol.x - bt <= 1e-6).all())
    assert bool((sol.x >= -1e-6).all()) and bool((sol.x <= 1.0 + 1e-6).all())
    ref = scipy_linprog(c, A_ub=A, b_ub=b, bounds=(None, None))
    np.testing.assert_allclose(float(sol.obj), ref.fun, atol=1e-4)
