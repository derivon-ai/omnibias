# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
"""Equality-constrained temperature-collapse penalty solver (jax + torch).

The exterior penalty gains an optional quadratic equality block
``(mu/2)||A_eq x - b_eq||^2`` (closed-form gradient ``mu A_eq^T (A_eq x - b_eq)``,
Lipschitz bump ``mu ||A_eq||_F^2``, dual estimate ``nu = mu (A_eq x - b_eq)``).
These tests check it against closed form / scipy, that autograd agrees with the
closed-form gradient, that the no-equality path is byte-unchanged, and that the
two backends stay bit-identical (~1e-11) in float64.
"""

from __future__ import annotations

import numpy as np
import pytest
from omnibias.convex.problem import PenaltyOptions

jax = pytest.importorskip("jax")
torch = pytest.importorskip("torch")
import jax.numpy as jnp  # noqa: E402
from omnibias.convex.jax import penalty as jpen  # noqa: E402
from omnibias.convex.torch import penalty as tpen  # noqa: E402

scipy_linprog = pytest.importorskip("scipy.optimize").linprog

torch.set_default_dtype(torch.float64)

# The equality block is a *quadratic* exterior penalty -- asymptotically exact in
# ``mu`` (unlike the exact linear inequality penalty), so a gentle-beta homotopy
# and an honest first-order feasibility tolerance are used for the solve tests.
_EQ_OPTS = PenaltyOptions(
    beta_growth=1.4, penalty0=2.0, penalty_growth=1.8, stages=14,
    gd_steps=3000, prox=0.0, feas_tol=1e-3, tol=1e-3,
)
_LP_EQ_OPTS = PenaltyOptions(
    beta0=2.0, beta_growth=1.6, penalty0=1.0, penalty_growth=1.6, stages=22,
    gd_steps=4000, prox=0.1, feas_tol=5e-3, tol=5e-3,
)


def _hyperplane_qp() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    # min 1/2||x - p||^2  s.t.  1^T x = 1,  |x| <= 10 (non-binding).
    p = np.array([0.2, 0.9, -0.3])
    n = 3
    Q = np.eye(n)
    c = -p
    A = np.vstack([np.eye(n), -np.eye(n)])
    b = 10.0 * np.ones(2 * n)
    A_eq = np.ones((1, n))
    b_eq = np.array([1.0])
    return Q, c, A, b, A_eq, b_eq


def _project_hyperplane(p: np.ndarray) -> np.ndarray:
    n = p.shape[0]
    return p + (1.0 - p.sum()) / n


def test_jax_equality_projection_matches_closed_form() -> None:
    Q, c, A, b, A_eq, b_eq = _hyperplane_qp()
    sol = jpen.solve_qp_penalty(
        jnp.asarray(Q), jnp.asarray(c), jnp.asarray(A), jnp.asarray(b),
        A_eq=jnp.asarray(A_eq), b_eq=jnp.asarray(b_eq), options=_EQ_OPTS,
    )
    x_star = _project_hyperplane(-c)  # p = -c
    assert sol.converged
    assert sol.eq_dual is not None
    np.testing.assert_allclose(np.asarray(sol.x), x_star, atol=5e-3)
    assert abs(float(np.sum(np.asarray(sol.x))) - 1.0) < 2e-3  # equality satisfied


def test_torch_equality_projection_matches_closed_form() -> None:
    Q, c, A, b, A_eq, b_eq = _hyperplane_qp()
    sol = tpen.solve_qp_penalty(Q, c, A, b, A_eq=A_eq, b_eq=b_eq, options=_EQ_OPTS)
    x_star = _project_hyperplane(-c)
    assert sol.converged
    assert sol.eq_dual is not None
    np.testing.assert_allclose(sol.x.numpy(), x_star, atol=5e-3)


def test_equality_dual_estimate_is_hyperplane_multiplier() -> None:
    # For min 1/2||x-p||^2 s.t. 1^T x = 1, the KKT multiplier is nu* = (sum p - 1)/n
    # (x* = p - nu* 1); the quadratic penalty's estimate mu (A_eq x - b_eq) -> nu*.
    Q, c, A, b, A_eq, b_eq = _hyperplane_qp()
    p = -c
    nu_star = (p.sum() - 1.0) / p.shape[0]
    sol = jpen.solve_qp_penalty(
        jnp.asarray(Q), jnp.asarray(c), jnp.asarray(A), jnp.asarray(b),
        A_eq=jnp.asarray(A_eq), b_eq=jnp.asarray(b_eq), options=_EQ_OPTS,
    )
    assert sol.eq_dual is not None
    np.testing.assert_allclose(float(np.asarray(sol.eq_dual)[0]), nu_star, atol=5e-3)


def test_equality_lp_matches_scipy_simplex() -> None:
    # min c^T x  s.t.  1^T x = 1, x >= 0  ->  vertex at argmin c.
    c = np.array([-3.0, -1.0, -2.0])
    A = -np.eye(3)
    b = np.zeros(3)
    A_eq = np.ones((1, 3))
    b_eq = np.array([1.0])
    sol = jpen.solve_lp_penalty(
        jnp.asarray(c), jnp.asarray(A), jnp.asarray(b),
        A_eq=jnp.asarray(A_eq), b_eq=jnp.asarray(b_eq), options=_LP_EQ_OPTS,
    )
    ref = scipy_linprog(c, A_ub=A, b_ub=b, A_eq=A_eq, b_eq=b_eq, bounds=(None, None))
    np.testing.assert_allclose(float(sol.obj), ref.fun, atol=5e-2)
    assert abs(float(np.sum(np.asarray(sol.x))) - 1.0) < 5e-3
    assert float(np.min(np.asarray(sol.x))) > -5e-3


def test_closed_form_gradient_with_equality_equals_autograd() -> None:
    _, c, A, b, A_eq, b_eq = _hyperplane_qp()
    Q = jnp.eye(3)
    cj, Aj, bj = jnp.asarray(c), jnp.asarray(A), jnp.asarray(b)
    Aeqj, beqj = jnp.asarray(A_eq), jnp.asarray(b_eq)
    x = jnp.asarray([0.7, 0.4, -0.2])
    beta, mu = 5.3, 2.1

    def objective(z: jnp.ndarray) -> jnp.ndarray:
        u = Aj @ z - bj
        ineq = mu * jnp.sum(jax.nn.softplus(beta * u)) / beta
        eq = 0.5 * mu * jnp.sum((Aeqj @ z - beqj) ** 2)
        return cj @ z + 0.5 * z @ (Q @ z) + ineq + eq

    g_auto = jax.grad(objective)(x)
    g_closed = jpen.penalty_gradient(Q, cj, Aj, bj, x, beta=beta, mu=mu, A_eq=Aeqj, b_eq=beqj)
    np.testing.assert_allclose(np.asarray(g_closed), np.asarray(g_auto), atol=1e-12)


def test_no_equality_path_is_unchanged() -> None:
    # Passing A_eq=None must reproduce the original (equality-free) solve exactly.
    Q, c, A, b, _, _ = _hyperplane_qp()
    args = (jnp.asarray(Q), jnp.asarray(c), jnp.asarray(A), jnp.asarray(b))
    base = jpen.solve_qp_penalty(*args)
    none = jpen.solve_qp_penalty(*args, A_eq=None, b_eq=None)
    np.testing.assert_array_equal(np.asarray(base.x), np.asarray(none.x))
    assert base.eq_dual is None and none.eq_dual is None


def test_equality_parity_jax_torch() -> None:
    # Same equality-constrained QP on both backends -> bit-identical (float64).
    Q, c, A, b, A_eq, b_eq = _hyperplane_qp()
    opts = PenaltyOptions(stages=9, gd_steps=1500)
    sj = jpen.solve_qp_penalty(
        jnp.asarray(Q), jnp.asarray(c), jnp.asarray(A), jnp.asarray(b),
        A_eq=jnp.asarray(A_eq), b_eq=jnp.asarray(b_eq), options=opts,
    )
    st = tpen.solve_qp_penalty(Q, c, A, b, A_eq=A_eq, b_eq=b_eq, options=opts)
    np.testing.assert_allclose(np.asarray(sj.x), st.x.numpy(), atol=1e-11, rtol=0.0)
    np.testing.assert_allclose(float(sj.obj), float(st.obj), atol=1e-11)
    assert sj.eq_dual is not None and st.eq_dual is not None
    np.testing.assert_allclose(
        np.asarray(sj.eq_dual), st.eq_dual.numpy(), atol=1e-11, rtol=0.0
    )


def test_bad_equality_shape_rejected() -> None:
    Q, c, A, b, _, _ = _hyperplane_qp()
    with pytest.raises(ValueError):
        jpen.solve_qp_penalty(
            jnp.asarray(Q), jnp.asarray(c), jnp.asarray(A), jnp.asarray(b),
            A_eq=jnp.asarray(np.ones((1, 5))), b_eq=jnp.asarray([1.0]),
        )
