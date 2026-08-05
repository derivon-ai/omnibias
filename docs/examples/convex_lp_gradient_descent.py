# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Solve an LP by gradient descent -- the temperature-collapse penalty solver.

Run:

    pip install "omnibias-convex[jax]" scipy
    python docs/examples/convex_lp_gradient_descent.py

Every constraint ``a_i^T x <= b_i`` is a hyperplane. The omnibias *temperature-collapse*
unit ``sigma(beta (a_i^T x - b_i))`` has that hyperplane as its decision boundary,
and its integral is the smooth hinge ``softplus(beta u)/beta -> max(u, 0)`` as
``beta -> inf``. Summing those penalties turns the LP into a smooth objective whose
gradient is closed form -- a sum of temperature-collapse sigmoids -- so the LP becomes a
plain gradient-descent problem, annealed along a ``beta`` / ``mu`` homotopy.

This script (1) checks the closed-form temperature-collapse gradient against autodiff,
(2) prints the ``beta``-annealing convergence, (3) compares the recovered vertex to
the Newton interior point and scipy, and (4) certifies a strictly convex QP solved
the same way.
"""

from __future__ import annotations

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402  (after x64 config)
import numpy as np  # noqa: E402
from omnibias.convex import certify_qp_optimum  # noqa: E402
from omnibias.convex.jax import (  # noqa: E402
    penalty_gradient,
    solve_lp,
    solve_lp_penalty,
    solve_qp_penalty,
)
from omnibias.convex.problem import PenaltyOptions  # noqa: E402
from scipy.optimize import linprog  # noqa: E402


def main() -> None:
    # max 3x + 2y  ==  min -3x - 2y  s.t.  x + y <= 4, x + 3y <= 6, x >= 0, y >= 0.
    # The unique optimum is the vertex (4, 0) with objective -12.
    c = jnp.array([-3.0, -2.0])
    A = jnp.array([[1.0, 1.0], [1.0, 3.0], [-1.0, 0.0], [0.0, -1.0]])
    b = jnp.array([4.0, 6.0, 0.0, 0.0])

    # (1) The penalty gradient is closed form: c + mu * A^T sigma(beta (A x - b)),
    #     one temperature-collapse sigmoid per constraint hyperplane. Match it to autodiff.
    beta, mu = 4.0, 3.0
    x_probe = jnp.array([1.0, 1.0])

    def smooth_penalty(x: jnp.ndarray) -> jnp.ndarray:
        u = A @ x - b
        return c @ x + mu * jnp.sum(jax.nn.softplus(beta * u)) / beta

    g_closed = penalty_gradient(jnp.zeros((2, 2)), c, A, b, x_probe, beta=beta, mu=mu)
    g_auto = jax.grad(smooth_penalty)(x_probe)
    print("closed-form temperature-collapse grad :", np.asarray(g_closed))
    print("autodiff grad                  :", np.asarray(g_auto))
    assert jnp.allclose(g_closed, g_auto, atol=1e-12)

    # (2) beta / mu homotopy: more annealing stages -> the temperature-collapse edges
    #     sharpen onto the exact hyperplanes and the gap to the true optimum shrinks.
    newton = solve_lp(c, A, b)
    obj_star = float(newton.obj)
    print(f"\nbeta-annealing convergence (optimum obj = {obj_star:.6f}):")
    print("  stages   final beta        obj        |obj - obj*|")
    for stages in (2, 4, 6, 8, 11):
        opts = PenaltyOptions(stages=stages, gd_steps=1500)
        sol = solve_lp_penalty(c, A, b, options=opts)
        final_beta = opts.beta0 * opts.beta_growth ** (sol.iterations - 1)
        print(
            f"  {stages:>4d}   {final_beta:>10.1f}   {float(sol.obj):>10.6f}   "
            f"{abs(float(sol.obj) - obj_star):.2e}"
        )

    # (3) Full solve vs Newton interior point and scipy.
    sol = solve_lp_penalty(c, A, b)
    ref = linprog(np.asarray(c), A_ub=np.asarray(A), b_ub=np.asarray(b), bounds=(None, None))
    print("\ngradient-descent x :", np.asarray(sol.x))
    print("Newton x           :", np.asarray(newton.x))
    print("scipy x            :", ref.x)
    print("dual (mu*sigma)    :", np.asarray(sol.dual), " >= 0:", bool(jnp.all(sol.dual >= 0)))
    assert np.allclose(np.asarray(sol.x), ref.x, atol=5e-3)

    # (4) Solve a strictly convex QP the same way, then certify it rigorously.
    #     min 1/2||x - p||^2 s.t. x >= 0  =>  x* = relu(p).
    p = np.array([1.5, -2.0, 0.7, -0.1])
    Q, cq, Aq, bq = np.eye(4), -p, -np.eye(4), np.zeros(4)
    solq = solve_qp_penalty(jnp.asarray(Q), jnp.asarray(cq), jnp.asarray(Aq), jnp.asarray(bq))
    cert = certify_qp_optimum(Q, cq, Aq, bq, np.asarray(solq.x), np.asarray(solq.dual))
    f_star = 0.5 * float(np.sum(np.maximum(p, 0.0) ** 2)) - float(p @ np.maximum(p, 0.0))
    print("\nQP  x                :", np.asarray(solq.x), " (relu(p) =", np.maximum(p, 0.0), ")")
    print(f"certified enclosure  : [{cert.enclosure.lo:.6f}, {cert.enclosure.hi:.6f}]  f* = {f_star:.6f}")
    print(f"primal_feasible      : {cert.primal_feasible}  certified gap = {cert.gap:.2e}")
    assert cert.enclosure.lo <= f_star <= cert.enclosure.hi
    print("\nOK: LP solved by gradient descent matches Newton/scipy; QP certified.")


if __name__ == "__main__":
    main()
