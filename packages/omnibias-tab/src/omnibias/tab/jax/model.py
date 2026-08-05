# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
r"""Functional JAX soft-tree forward (bit-identical twin of the numpy / torch models).

``forward(params, X, beta)`` reproduces :func:`omnibias.tab._core.forward.forward_np`
bit-for-bit (float64, parity ``~1e-9``), so a model trained with the torch driver can be
served / differentiated under JAX transforms (``jit`` / ``grad`` / ``vmap``) unchanged.

Terminology: the split gate ``sigmoid(beta (W.x - t))`` hardens as ``beta -> inf`` -- the
feasibility / temperature sense of "collapse", distinct from the **founding bias
collapse** (the multi-bias ``delta -> 0`` limit to the closed-form derivative
``sigma^(K-1)``; see ``docs/theory.md``).
"""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
from omnibias.tab._core.params import TabParams, leaf_code_matrix


def forward_arrays(
    W: Any, t: Any, leaves: Any, b0: Any, X: Any, beta: float, depth: int
) -> Any:
    r"""Raw scores ``F`` ``(n, k)`` from raw arrays -- the ``jax``-traceable kernel."""
    codes = jnp.asarray(leaf_code_matrix(depth))  # (L, D)
    z = jnp.einsum("nd,mjd->nmj", X, W) - t[None, :, :]
    g = jax.nn.sigmoid(beta * z)  # (n, T, D)
    gexp = g[:, :, None, :]  # (n, T, 1, D)
    bexp = codes[None, None, :, :]  # (1, 1, L, D)
    factors = bexp * gexp + (1.0 - bexp) * (1.0 - gexp)  # (n, T, L, D)
    memberships = jnp.prod(factors, axis=-1)  # (n, T, L)
    return jnp.einsum("nml,mlk->nk", memberships, leaves) + b0[None, :]


def forward(params: TabParams, X: Any, beta: float) -> Any:
    r"""Raw ensemble scores ``F`` of shape ``(n, n_outputs)`` for a :class:`TabParams`."""
    Xv = jnp.asarray(X, dtype=jnp.float64)
    return forward_arrays(
        jnp.asarray(params.W),
        jnp.asarray(params.t),
        jnp.asarray(params.leaves),
        jnp.asarray(params.b0),
        Xv,
        float(beta),
        params.depth,
    )


def _score_derivs(F: Any, y: Any, task: str) -> tuple[Any, Any]:
    r"""Closed-form score-space grad ``g`` and (diagonal) Hessian ``h`` (see the torch loss)."""
    if task == "binary":
        p = jax.nn.sigmoid(F[:, 0])
        g = (p - jnp.asarray(y).reshape(-1))[:, None]
        h = jnp.clip(p * (1.0 - p), 1e-12, None)[:, None]
        return g, h
    if task == "multiclass":
        p = jax.nn.softmax(F, axis=-1)
        Y = jax.nn.one_hot(jnp.asarray(y).astype(jnp.int32), F.shape[1])
        return p - Y, jnp.clip(p * (1.0 - p), 1e-12, None)
    yv = jnp.asarray(y).reshape(F.shape)
    return 2.0 * (F - yv), jnp.full_like(F, 2.0)


def natural_gradient_step(
    params: TabParams,
    X: Any,
    y: Any,
    *,
    beta: float,
    lr: float = 1.0,
    damping: float = 1e-3,
) -> TabParams:
    r"""One functional **Gauss-Newton natural-gradient** step (jax twin of the torch driver).

    Steps ``theta <- theta - lr (F_GN + damping I)^{-1} g`` with the Gauss-Newton Fisher
    ``F_GN = J^T diag(h) J`` (``J`` the score Jacobian, ``h`` the closed-form loss
    curvature). Natural gradient == Fisher scoring, so on a GLM this is a Newton step; it
    is the documented conceptual parity to the exact-Hessian torch trainer (torch remains
    the primary second-order driver -- see :mod:`omnibias.tab.torch.train`).
    """
    import numpy as _np

    W, t, leaves, b0 = params.W, params.t, params.leaves, params.b0
    sizes = [W.size, t.size, leaves.size, b0.size]
    shapes = [W.shape, t.shape, leaves.shape, b0.shape]
    theta0 = jnp.concatenate([jnp.asarray(a).reshape(-1) for a in (W, t, leaves, b0)])
    Xv = jnp.asarray(X, dtype=jnp.float64)
    task, depth = params.config.task, params.depth
    n = Xv.shape[0]

    def unflatten(theta: Any) -> tuple[Any, Any, Any, Any]:
        parts, off = [], 0
        for size, shape in zip(sizes, shapes, strict=True):
            parts.append(theta[off : off + size].reshape(shape))
            off += size
        return parts[0], parts[1], parts[2], parts[3]

    def score_flat(theta: Any) -> Any:
        Wt, tt, lv, bb = unflatten(theta)
        return forward_arrays(Wt, tt, lv, bb, Xv, float(beta), depth).reshape(-1)

    F = score_flat(theta0).reshape(n, -1)
    g, h = _score_derivs(F, y, task)
    jac = jax.jacobian(score_flat)(theta0)  # (n*k, P)
    grad = jac.T @ g.reshape(-1)
    fisher = (jac * h.reshape(-1)[:, None]).T @ jac
    p = theta0.shape[0]
    delta = jnp.linalg.solve(fisher + damping * jnp.eye(p), grad)
    theta1 = theta0 - lr * delta

    Wt, tt, lv, bb = unflatten(theta1)
    return TabParams(
        params.config,
        _np.asarray(Wt),
        _np.asarray(tt),
        _np.asarray(lv),
        _np.asarray(bb),
    )


def fit_natural_gradient(
    params: TabParams,
    X: Any,
    y: Any,
    *,
    steps: int = 20,
    lr: float = 1.0,
    damping: float = 1e-3,
) -> TabParams:
    r"""Convenience loop of :func:`natural_gradient_step` with the config's ``beta`` ramp."""
    cur = params
    for step in range(steps):
        beta = params.config.beta_at(step)
        cur = natural_gradient_step(cur, X, y, beta=beta, lr=lr, damping=damping)
    return cur


__all__ = [
    "fit_natural_gradient",
    "forward",
    "forward_arrays",
    "natural_gradient_step",
]
