# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
"""LP/QP interior-point solver (torch): references + bit-parity with the JAX twin."""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")
from omnibias.convex.torch import InfeasibleProblemError, solve_lp, solve_qp  # noqa: E402

scipy_linprog = pytest.importorskip("scipy.optimize").linprog


def _t(x: object) -> torch.Tensor:
    return torch.as_tensor(np.asarray(x, dtype=np.float64))


def test_lp_matches_scipy_2d() -> None:
    c = np.array([-3.0, -2.0])
    A = np.array([[1.0, 1.0], [1.0, 3.0], [-1.0, 0.0], [0.0, -1.0]])
    b = np.array([4.0, 6.0, 0.0, 0.0])
    sol = solve_lp(_t(c), _t(A), _t(b))
    ref = scipy_linprog(c, A_ub=A, b_ub=b, bounds=(None, None))
    assert sol.converged
    np.testing.assert_allclose(sol.x.numpy(), ref.x, atol=1e-6)
    np.testing.assert_allclose(float(sol.obj), ref.fun, atol=1e-6)


def test_qp_projection_onto_nonnegative_orthant() -> None:
    p = np.array([1.5, -2.0, 0.7, -0.1])
    n = 4
    sol = solve_qp(_t(np.eye(n)), _t(-p), _t(-np.eye(n)), _t(np.zeros(n)))
    np.testing.assert_allclose(sol.x.numpy(), np.maximum(p, 0.0), atol=1e-6)


def test_phase1_when_origin_infeasible() -> None:
    c = np.array([1.0])
    A = np.array([[-1.0], [1.0]])
    b = np.array([-1.0, 3.0])
    sol = solve_lp(_t(c), _t(A), _t(b))
    assert sol.converged
    np.testing.assert_allclose(float(sol.x[0]), 1.0, atol=1e-6)


def test_infeasible_raises() -> None:
    c = np.array([1.0])
    A = np.array([[1.0], [-1.0]])
    b = np.array([-1.0, -1.0])
    with pytest.raises(InfeasibleProblemError):
        solve_lp(_t(c), _t(A), _t(b))


def _covering_lp(P: int, S: int, k: int, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
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
    # The solver must keep the last centered iterate (finite, correct) instead of reporting a
    # divergent NaN as "converged".
    c, A, b = _covering_lp(P=26, S=64, k=3, seed=0)
    sol = solve_lp(_t(c), _t(A), _t(b))
    ref = scipy_linprog(c, A_ub=A, b_ub=b, bounds=(None, None))
    assert bool(torch.isfinite(sol.x).all())
    assert np.isfinite(float(sol.obj))
    np.testing.assert_allclose(float(sol.obj), ref.fun, atol=1e-4)
    # Slacks stay strictly feasible for the retained iterate; dual uses the matching t.
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
    At, bt = _t(A), _t(b)
    sol = solve_lp(_t(c), At, bt)
    assert bool(torch.isfinite(sol.x).all())
    # Every constraint (including the box) is respected: the primal never left the feasible set.
    assert bool((At @ sol.x - bt <= 1e-6).all())
    assert bool((sol.x >= -1e-6).all()) and bool((sol.x <= 1.0 + 1e-6).all())
    ref = scipy_linprog(c, A_ub=A, b_ub=b, bounds=(None, None))
    np.testing.assert_allclose(float(sol.obj), ref.fun, atol=1e-4)


def test_torch_jax_solution_parity() -> None:
    jax = pytest.importorskip("jax")
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp
    from omnibias.convex.jax import solve_qp as solve_qp_jax

    rng = np.random.default_rng(3)
    n = 3
    M = rng.standard_normal((n, n))
    Q = M @ M.T + n * np.eye(n)
    c = rng.standard_normal(n)
    A = np.vstack([rng.standard_normal((2, n)), np.eye(n), -np.eye(n)])
    b = np.concatenate([rng.uniform(0.5, 1.5, size=2), 3.0 * np.ones(2 * n)])

    st = solve_qp(_t(Q), _t(c), _t(A), _t(b))
    sj = solve_qp_jax(jnp.asarray(Q), jnp.asarray(c), jnp.asarray(A), jnp.asarray(b))
    np.testing.assert_allclose(st.x.numpy(), np.asarray(sj.x), rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(st.dual.numpy(), np.asarray(sj.dual), rtol=1e-9, atol=1e-9)
    assert st.newton_iterations == sj.newton_iterations
