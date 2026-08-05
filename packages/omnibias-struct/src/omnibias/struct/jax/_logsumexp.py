# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Closed-form log-sum-exp / softmax primitives and jets (jax).

Bit-identical twin of :mod:`omnibias.struct.torch._logsumexp` (float64 -- enable
``jax_enable_x64``). The soft-DP combine is ``lse_beta``; its exact derivatives are the
closed-form omnibias tower, **not** autodiff:

- ``grad lse_beta = softmax(beta a)`` (:func:`logsumexp_beta_jacobian`);
- ``hess lse_beta = beta (diag(p) - p p^T)`` (:func:`logsumexp_beta_hessian`);
- and, pairwise, ``lse_beta(a, b) = a + beta^-1 softplus(beta (b - a))`` whose whole
  Taylor jet is the beta-tempered ``softplus`` tower ``softplus^(n) = sigma^(n-1)`` from
  :mod:`omnibias.core`, propagated with :func:`omnibias.jax.jet.compose_jet`
  (:func:`pairwise_lse_jet`). This is the ``delta -> 0`` engine that differentiates the
  ``beta -> inf`` relaxation.
"""

from __future__ import annotations

import jax.numpy as jnp
from jax import Array
from omnibias.jax.activations import get_activation
from omnibias.jax.jet import compose_jet


def logsumexp_beta(a: Array, beta: float = 1.0, *, axis: int = -1) -> Array:
    r"""Stable ``lse_beta(a) = beta^-1 log sum_i exp(beta a_i)`` along ``axis``."""
    scaled = beta * a
    max_a = jnp.max(scaled, axis=axis, keepdims=True)
    log_sum = jnp.squeeze(max_a, axis=axis) + jnp.log(
        jnp.sum(jnp.exp(scaled - max_a), axis=axis),
    )
    return log_sum / beta


def softmax_beta(a: Array, beta: float = 1.0, *, axis: int = -1) -> Array:
    r"""Stable ``softmax(beta a)`` along ``axis`` -- the gradient of :func:`logsumexp_beta`."""
    scaled = beta * a
    max_a = jnp.max(scaled, axis=axis, keepdims=True)
    exp_a = jnp.exp(scaled - max_a)
    return exp_a / jnp.sum(exp_a, axis=axis, keepdims=True)


def logsumexp_beta_jacobian(a: Array, beta: float = 1.0, *, axis: int = -1) -> Array:
    r"""Closed-form gradient of ``lse_beta`` w.r.t. ``a``: ``softmax(beta a)``."""
    return softmax_beta(a, beta, axis=axis)


def logsumexp_beta_hessian(a: Array, beta: float = 1.0, *, axis: int = -1) -> Array:
    r"""Closed-form Hessian of ``lse_beta``: ``beta (diag(p) - p p^T)`` (``(..., n, n)``)."""
    p = softmax_beta(a, beta, axis=axis)
    outer = p[..., :, None] * p[..., None, :]
    n = p.shape[-1]
    diag = jnp.einsum("...i,ij->...ij", p, jnp.eye(n, dtype=p.dtype))
    return beta * (diag - outer)


def pairwise_lse(a: Array, b: Array, beta: float = 1.0) -> Array:
    r"""Elementwise soft-max combine ``lse_beta(a, b) = beta^-1 log(e^{beta a} + e^{beta b})``.

    Equal to ``a + beta^-1 softplus(beta (b - a))``; computed by the stable symmetric
    log-sum-exp of the stacked pair.
    """
    stacked = jnp.stack([a, b], axis=-1)
    return logsumexp_beta(stacked, beta, axis=-1)


def pairwise_lse_jet(
    a0: Array,
    b0: Array,
    db: Array,
    beta: float = 1.0,
    order: int = 1,
) -> Array:
    r"""Taylor jet (order ``order``) of ``t -> lse_beta(a0, b0 + t db)`` from the tower.

    Writing ``lse_beta(a0, b0 + t db) = a0 + beta^-1 softplus(u(t))`` with
    ``u(t) = beta (b0 - a0) + (beta db) t``, the jet is
    :func:`omnibias.jax.jet.compose_jet` of the linear ``u``-jet with the closed-form
    ``softplus`` derivative tower ``softplus^(k)(u_0)`` (``softplus^(k) = sigma^(k-1)``
    from :mod:`omnibias.core`) -- exact, no autodiff or finite differences. Returns
    coefficients ``c_k = f^(k)(0) / k!`` along a new leading axis of length ``order + 1``.
    """
    if order < 0:
        raise ValueError(f"order must be >= 0, got {order}")
    spec = get_activation("softplus")
    fastpath = spec.fastpath
    if fastpath is None:  # defensive: softplus always ships its closed-form tower
        raise NotImplementedError("softplus is missing its closed-form fastpath tower")
    a0 = jnp.asarray(a0)
    b0 = jnp.asarray(b0)
    db = jnp.asarray(db)
    u0 = beta * (b0 - a0)
    zero = jnp.zeros_like(u0)
    u_rows = [u0]
    if order >= 1:
        u_rows.append((beta * db) + zero)
    u_rows.extend(zero for _ in range(order - 1))
    u_jet = jnp.stack(u_rows[: order + 1], axis=0)
    tower_rows = [spec.forward(u0)]
    for k in range(1, order + 1):
        tower_rows.append(fastpath(u0, k))
    sigma_tower = jnp.stack(tower_rows, axis=0)
    sp_jet = compose_jet(u_jet, sigma_tower) / beta
    out0 = (sp_jet[0] + a0)[None, ...]
    return jnp.concatenate([out0, sp_jet[1:]], axis=0)


__all__ = [
    "logsumexp_beta",
    "logsumexp_beta_hessian",
    "logsumexp_beta_jacobian",
    "pairwise_lse",
    "pairwise_lse_jet",
    "softmax_beta",
]
