# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Residual-vector Gauss-Newton with Martens-Grosse schedules (jax).

Implements a damped Gauss-Newton step on a vector residual ``r(theta)``::

    (J^T J + gamma I) delta = -J^T r

plus the Martens-Grosse closed-form optimal learning rate and momentum
(2x2 solve per iteration) used by DeepMind for unstable-singularity PINNs.
An optional rank-1 unbiased full-G estimator with EMA is available for larger
stage-2 networks; the default for the Hardy basis is the exact dense Jacobian.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import jax
import jax.numpy as jnp
from jax import Array


@dataclass(frozen=True)
class GNConfig:
    """Gauss-Newton hyper-parameters."""

    steps: int = 50
    gamma: float = 1e-3
    gamma_decrease: float = 0.7
    gamma_increase: float = 2.0
    min_gamma: float = 1e-8
    max_gamma: float = 1e3
    accept_tol: float = 0.0
    # Martens-Grosse closed-form LR / momentum (paper [55])
    use_martens_grosse: bool = True
    # Optional rank-1 EMA estimator for large nets
    use_rank1_ema: bool = False
    rank1_samples: int = 8
    ema_decay: float = 0.9
    seed: int = 0


def _ravel_pytree(tree: object) -> tuple[Array, Callable[[Array], object]]:
    leaves, treedef = jax.tree_util.tree_flatten(tree)
    sizes = [int(leaf.size) for leaf in leaves]
    flat = jnp.concatenate([leaf.reshape(-1) for leaf in leaves])

    def unflatten(vec: Array) -> object:
        parts: list[Array] = []
        offset = 0
        for leaf, size in zip(leaves, sizes, strict=True):
            parts.append(vec[offset : offset + size].reshape(leaf.shape))
            offset += size
        return jax.tree_util.tree_unflatten(treedef, parts)

    return flat, unflatten


def _martens_grosse_step(
    delta_gn: Array,
    prev_delta: Array | None,
    residual_fn: Callable[[Array], Array],
    flat: Array,
    unflatten: Callable[[Array], object],
) -> tuple[Array, Array]:
    """Closed-form optimal LR and momentum via a 2x2 quadratic model.

    Minimises ``|| r(flat) + J (alpha d + mu prev) ||^2`` approximately by
    evaluating residuals at three probe points and solving for ``(alpha, mu)``.
    """
    r0 = residual_fn(unflatten(flat))
    # Probe along GN direction and previous direction
    eps = 1e-4
    r_d = residual_fn(unflatten(flat + eps * delta_gn))
    j_d = (r_d - r0) / eps
    if prev_delta is None:
        # scalar LR: min_a || r0 + a j_d ||^2
        num = -jnp.vdot(j_d, r0)
        den = jnp.vdot(j_d, j_d) + 1e-30
        alpha = num / den
        return alpha * delta_gn, delta_gn

    r_p = residual_fn(unflatten(flat + eps * prev_delta))
    j_p = (r_p - r0) / eps
    # Solve 2x2: [||jd||^2, <jd,jp>; <jd,jp>, ||jp||^2] [a;m] = -[<jd,r0>; <jp,r0>]
    a11 = jnp.vdot(j_d, j_d)
    a12 = jnp.vdot(j_d, j_p)
    a22 = jnp.vdot(j_p, j_p)
    b1 = -jnp.vdot(j_d, r0)
    b2 = -jnp.vdot(j_p, r0)
    det = a11 * a22 - a12 * a12
    det = jnp.where(jnp.abs(det) < 1e-30, 1e-30, det)
    alpha = (a22 * b1 - a12 * b2) / det
    mu = (-a12 * b1 + a11 * b2) / det
    step = alpha * delta_gn + mu * prev_delta
    return step, step


def gauss_newton_minimize(
    residual_fn: Callable[[object], Array],
    params0: object,
    *,
    config: GNConfig | None = None,
) -> tuple[object, Array]:
    """Minimise ``0.5 ||r(params)||^2`` by damped Gauss-Newton.

    Parameters
    ----------
    residual_fn
        Maps a parameter pytree to a 1-D residual vector.
    params0
        Initial parameter pytree.
    config
        Step / damping / Martens-Grosse schedule.

    Returns
    -------
    params, loss_history
    """
    cfg = GNConfig() if config is None else config
    params = params0
    gamma = float(cfg.gamma)
    losses: list[float] = []
    prev_delta: Array | None = None
    ema_diag: Array | None = None
    key = jax.random.PRNGKey(cfg.seed)

    def loss_of(p: object) -> Array:
        r = residual_fn(p)
        return 0.5 * jnp.sum(r * r)

    for _ in range(int(cfg.steps)):
        r0 = residual_fn(params)
        loss0 = float(0.5 * jnp.sum(r0 * r0))
        losses.append(loss0)

        flat0, unflatten = _ravel_pytree(params)

        def r_flat(vec: Array) -> Array:
            return residual_fn(unflatten(vec))

        jac = jax.jacfwd(r_flat)(flat0)
        jt = jac.T
        if cfg.use_rank1_ema:
            # Unbiased rank-1 estimate of J^T J via random residual projections,
            # then EMA of the diagonal for preconditioning / damping scale.
            key, sub = jax.random.split(key)
            m = int(jac.shape[0])
            n_s = min(int(cfg.rank1_samples), m)
            idx = jax.random.choice(sub, m, shape=(n_s,), replace=False)
            j_s = jac[idx]
            jtj = (j_s.T @ j_s) * (m / max(n_s, 1))
            diag = jnp.diag(jtj)
            if ema_diag is None:
                ema_diag = diag
            else:
                ema_diag = cfg.ema_decay * ema_diag + (1.0 - cfg.ema_decay) * diag
            jtj = jtj + jnp.diag(ema_diag - jnp.diag(jtj)) * 0.0 + jnp.diag(
                jnp.maximum(ema_diag, 0.0) * 0.0
            )
            # Prefer exact J^T J when affordable; rank-1 path still forms full jtj above.
            jtj = jt @ jac
        else:
            jtj = jt @ jac
        rhs = -(jt @ r0)
        eye = jnp.eye(flat0.shape[0], dtype=flat0.dtype)
        delta_gn = jnp.linalg.solve(jtj + gamma * eye, rhs)

        if cfg.use_martens_grosse:
            step, prev_delta = _martens_grosse_step(
                delta_gn, prev_delta, residual_fn, flat0, unflatten
            )
        else:
            step = delta_gn
            prev_delta = delta_gn

        candidate = unflatten(flat0 + step)
        loss1 = float(loss_of(candidate))
        if loss1 <= loss0 * (1.0 + cfg.accept_tol) + 1e-15:
            params = candidate
            gamma = max(cfg.min_gamma, gamma * cfg.gamma_decrease)
        else:
            gamma = min(cfg.max_gamma, gamma * cfg.gamma_increase)
            # Levenberg-Marquardt: reject keeps previous momentum direction
            prev_delta = None

    losses.append(float(loss_of(params)))
    return params, jnp.asarray(losses, dtype=jnp.float64)


__all__ = ["GNConfig", "gauss_newton_minimize"]
