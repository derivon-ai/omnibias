# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
r"""Differentiable ``argmin`` for LP/QP -- the NN-native convex layer (JAX).

``qp_layer(Q, c, A, b)`` returns the QP optimiser ``x*(Q, c, A, b)`` and is
differentiable in all of ``Q, c, A, b`` via the **KKT implicit-function theorem**
(OptNet / cvxpylayers style): the forward solve is the interior-point method from
:mod:`omnibias.convex.jax.solver`, and the backward pass solves the linearised
KKT system once for the adjoint -- no unrolling through the solver iterations.

At the optimum the KKT residual

.. math::
    r(x, \lambda) = \begin{bmatrix} Q x + c + A^\top \lambda \\
                     \operatorname{diag}(\lambda)(A x - b) \end{bmatrix} = 0

has Jacobian (w.r.t. ``(x, lambda)``)

.. math::
    K = \begin{bmatrix} Q & A^\top \\
        \operatorname{diag}(\lambda) A & \operatorname{diag}(A x - b) \end{bmatrix},

and for an upstream cotangent ``g = dL/dx`` we solve ``K^T [y_x; y_l] = -[g; 0]``
and read off the parameter gradients (Amos & Kolter, 2017).

The forward solve runs eagerly through :func:`jax.pure_callback`, so ``qp_layer``
is usable under ``jax.grad`` / ``jax.jacobian`` without making the (data-dependent,
Python-control-flow) interior-point loop itself trace-able.
"""

from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from omnibias.convex.jax.solver import solve_qp
from omnibias.convex.problem import BarrierOptions


def _host_solve(
    options: BarrierOptions | None,
    Q: np.ndarray,
    c: np.ndarray,
    A: np.ndarray,
    b: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    sol = solve_qp(jnp.asarray(Q), jnp.asarray(c), jnp.asarray(A), jnp.asarray(b), options=options)
    dtype = np.asarray(c).dtype
    return (
        np.asarray(sol.x, dtype=dtype),
        np.asarray(sol.dual, dtype=dtype),
        np.asarray(sol.slack, dtype=dtype),
    )


def _solve_callback(
    Q: Array, c: Array, A: Array, b: Array, options: BarrierOptions | None
) -> tuple[Array, Array, Array]:
    n = c.shape[0]
    m = b.shape[0]
    dtype = jnp.asarray(c).dtype
    out_shapes = (
        jax.ShapeDtypeStruct((n,), dtype),  # type: ignore[no-untyped-call]
        jax.ShapeDtypeStruct((m,), dtype),  # type: ignore[no-untyped-call]
        jax.ShapeDtypeStruct((m,), dtype),  # type: ignore[no-untyped-call]
    )
    result: tuple[Array, Array, Array] = jax.pure_callback(
        partial(_host_solve, options), out_shapes, Q, c, A, b
    )
    return result


@partial(jax.custom_vjp, nondiff_argnums=(4,))
def qp_layer(
    Q: Array, c: Array, A: Array, b: Array, options: BarrierOptions | None = None
) -> Array:
    """Differentiable QP optimiser ``x*`` of ``min 1/2 x^T Q x + c^T x s.t. A x <= b``."""
    x, _, _ = _solve_callback(Q, c, A, b, options)
    return x


def _qp_fwd(
    Q: Array, c: Array, A: Array, b: Array, options: BarrierOptions | None
) -> tuple[Array, tuple[Array, ...]]:
    x, dual, slack = _solve_callback(Q, c, A, b, options)
    return x, (Q, A, x, dual, slack)


def _qp_bwd(
    options: BarrierOptions | None, res: tuple[Array, ...], g: Array
) -> tuple[Array, Array, Array, Array]:
    Q, A, x, lam, slack = res
    n = x.shape[0]
    m = slack.shape[0]
    # K = [[Q, A^T], [diag(lam) A, diag(A x - b)]];  A x - b = -slack.
    top = jnp.concatenate([Q, A.T], axis=1)
    bottom = jnp.concatenate([lam[:, None] * A, jnp.diag(-slack)], axis=1)
    kkt = jnp.concatenate([top, bottom], axis=0)
    rhs = -jnp.concatenate([g, jnp.zeros((m,), dtype=g.dtype)])
    y = jnp.linalg.solve(kkt.T, rhs)
    y_x = y[:n]
    y_l = y[n:]
    grad_Q = 0.5 * (jnp.outer(y_x, x) + jnp.outer(x, y_x))
    grad_c = y_x
    grad_A = jnp.outer(lam, y_x) + jnp.outer(lam * y_l, x)
    grad_b = -lam * y_l
    return grad_Q, grad_c, grad_A, grad_b


qp_layer.defvjp(_qp_fwd, _qp_bwd)


def lp_layer(
    c: Array, A: Array, b: Array, options: BarrierOptions | None = None
) -> Array:
    """Differentiable LP optimiser ``x*`` of ``min c^T x s.t. A x <= b`` (``Q = 0``)."""
    c = jnp.asarray(c)
    n = c.shape[0]
    Q = jnp.zeros((n, n), dtype=c.dtype)
    return qp_layer(Q, c, A, b, options)


__all__ = ["lp_layer", "qp_layer"]
