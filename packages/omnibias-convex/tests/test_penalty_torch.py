# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
"""Tempered temperature-collapse penalty (gradient-descent) LP/QP solver -- torch.

Mirrors ``test_penalty_jax.py`` and adds a torch/jax cross-backend parity check.
"""

from __future__ import annotations

import numpy as np
import pytest
from omnibias.convex import Certificate, certify_qp_optimum
from omnibias.convex.problem import PenaltyOptions

torch = pytest.importorskip("torch")
from omnibias.convex.torch import (  # noqa: E402
    penalty_descent,
    penalty_gradient,
    solve_lp,
    solve_lp_penalty,
    solve_qp_penalty,
)

scipy_linprog = pytest.importorskip("scipy.optimize").linprog


def _t(x: object) -> torch.Tensor:
    return torch.as_tensor(np.asarray(x, dtype=np.float64), dtype=torch.float64)


def _lp_2d() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # max 3x + 2y == min -3x - 2y, s.t. x+y<=4, x+3y<=6, x>=0, y>=0 ; optimum (4, 0).
    c = _t([-3.0, -2.0])
    A = _t([[1.0, 1.0], [1.0, 3.0], [-1.0, 0.0], [0.0, -1.0]])
    b = _t([4.0, 6.0, 0.0, 0.0])
    return c, A, b


def test_lp_matches_scipy_2d() -> None:
    c, A, b = _lp_2d()
    sol = solve_lp_penalty(c, A, b)
    ref = scipy_linprog(c.numpy(), A_ub=A.numpy(), b_ub=b.numpy(), bounds=(None, None))
    assert sol.converged
    np.testing.assert_allclose(sol.x.numpy(), ref.x, atol=5e-3)
    np.testing.assert_allclose(float(sol.obj), ref.fun, atol=5e-3)


def test_lp_matches_newton_interior_point() -> None:
    c, A, b = _lp_2d()
    sol = solve_lp_penalty(c, A, b)
    ref = solve_lp(c, A, b)
    np.testing.assert_allclose(sol.x.numpy(), ref.x.numpy(), atol=5e-3)
    np.testing.assert_allclose(float(sol.obj), float(ref.obj), atol=5e-3)


def test_qp_projection_onto_nonnegative_orthant() -> None:
    # min 1/2||x - p||^2  s.t.  -x <= 0   =>   x* = relu(p).
    p = _t([1.5, -2.0, 0.7, -0.1])
    sol = solve_qp_penalty(torch.eye(4, dtype=torch.float64), -p, -torch.eye(4, dtype=torch.float64), torch.zeros(4, dtype=torch.float64))
    np.testing.assert_allclose(sol.x.numpy(), np.maximum(p.numpy(), 0.0), atol=5e-3)


def test_dual_is_nonnegative_and_complementary() -> None:
    c, A, b = _lp_2d()
    sol = solve_lp_penalty(c, A, b)
    lam = sol.dual.numpy()
    assert np.min(lam) >= 0.0
    assert np.max(lam * sol.slack.numpy()) < 5e-3
    stat = c.numpy() + A.numpy().T @ lam
    assert np.max(np.abs(stat)) < 5e-2


def test_certificate_encloses_penalty_optimum() -> None:
    p = np.array([1.5, -2.0, 0.7, -0.1])
    Q = np.eye(4)
    c = -p
    A = -np.eye(4)
    b = np.zeros(4)
    sol = solve_qp_penalty(_t(Q), _t(c), _t(A), _t(b))
    cert = certify_qp_optimum(Q, c, A, b, sol.x.numpy(), sol.dual.numpy())
    x_star = np.maximum(p, 0.0)
    f_star = 0.5 * float(np.sum(x_star**2)) - float(p @ x_star)
    assert isinstance(cert, Certificate)
    assert cert.primal_feasible
    assert cert.enclosure.lo <= f_star <= cert.enclosure.hi
    assert cert.gap >= 0.0


def test_exterior_start_from_infeasible_point() -> None:
    c, A, b = _lp_2d()
    sol = solve_lp_penalty(c, A, b, x0=_t([9.0, 9.0]))
    ref = solve_lp(c, A, b)
    assert sol.converged
    np.testing.assert_allclose(sol.x.numpy(), ref.x.numpy(), atol=5e-3)


def test_closed_form_gradient_equals_autograd() -> None:
    _, A, b = _lp_2d()
    Q = torch.eye(2, dtype=torch.float64)
    c = _t([0.3, -1.2])
    x = _t([0.7, 0.4]).requires_grad_(True)
    beta, mu = 5.3, 2.1
    u = A @ x - b
    objective = c @ x + 0.5 * x @ (Q @ x) + mu * torch.sum(torch.nn.functional.softplus(beta * u)) / beta
    objective.backward()
    g_auto = x.grad.detach().numpy()
    g_closed = penalty_gradient(Q, c, A, b, x.detach(), beta=beta, mu=mu).numpy()
    np.testing.assert_allclose(g_closed, g_auto, atol=1e-12)


def test_penalty_descent_is_differentiable() -> None:
    _, A, b = _lp_2d()
    cost = _t([-3.0, -2.0]).requires_grad_(True)
    x = penalty_descent(
        torch.eye(2, dtype=torch.float64), cost, A, b,
        torch.zeros(2, dtype=torch.float64), beta=8.0, mu=6.0, steps=300, prox=0.0,
    )
    x.sum().backward()
    grad = cost.grad
    assert bool(torch.isfinite(grad).all())
    assert float(grad.abs().max()) > 1e-6


def test_torch_jax_parity() -> None:
    pytest.importorskip("jax")
    import jax.numpy as jnp
    from omnibias.convex.jax import solve_lp_penalty as jax_solve_lp_penalty
    from omnibias.convex.jax import solve_qp_penalty as jax_solve_qp_penalty

    opts = PenaltyOptions(stages=8, gd_steps=1000)
    c, A, b = _lp_2d()
    st = solve_lp_penalty(c, A, b, options=opts)
    sj = jax_solve_lp_penalty(jnp.asarray(c.numpy()), jnp.asarray(A.numpy()), jnp.asarray(b.numpy()), options=opts)
    np.testing.assert_allclose(st.x.numpy(), np.asarray(sj.x), atol=1e-6)
    np.testing.assert_allclose(st.dual.numpy(), np.asarray(sj.dual), atol=1e-6)
    np.testing.assert_allclose(float(st.obj), float(sj.obj), atol=1e-8)

    p = _t([1.5, -2.0, 0.7, -0.1])
    stq = solve_qp_penalty(torch.eye(4, dtype=torch.float64), -p, -torch.eye(4, dtype=torch.float64), torch.zeros(4, dtype=torch.float64), options=opts)
    sjq = jax_solve_qp_penalty(jnp.eye(4), -jnp.asarray(p.numpy()), -jnp.eye(4), jnp.zeros(4), options=opts)
    np.testing.assert_allclose(stq.x.numpy(), np.asarray(sjq.x), atol=1e-6)
