# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Gauss-Newton matrices of one-layer ridge residuals on tensor-product grids, without a Jacobian.

For ``u(x) = sum_h c_h act(w_h . x + b_h)`` with ``act`` in ``sin``, ``cos``,
``exp`` and a constant-coefficient operator ``L`` with symbol ``p``, every
column of the residual Jacobian is ``Im[(alpha + beta x_i) e^(i w_h.x)]``
(``sin``, ``cos``) or ``(alpha + beta x_i) e^(w_h.x)`` (``exp``): ``alpha``
depends on the parameter (``c_h``, ``b_h`` or ``w_(h,i)``) through ``p``, its
gradient and ``b_h``, and ``beta`` is non-zero only for ``w_(h,i)``.  On a grid
``X_1 x ... x X_D`` with product weights the Gauss-Newton matrix
``sum_x q(x) J(x)^T J(x)`` is therefore a combination of grid sums with at most
two coordinate factors, each a Hadamard product over axes of per-axis moment
Gram matrices ``sum_n q_n x_n^m e^(lam (w_(h,i) +- w_(k,i)) x_n)``, ``m = 0, 1, 2``.
The cost is ``O(D H^2 N + D^3 H^2)`` time and ``O(D^2 H^2)`` memory instead of
the ``O(N^D D^2 H^2)`` of forming ``J^T J`` from an ``N^D x (D + 2) H`` Jacobian.

The matrix does not depend on the target.  It is the energy (Gauss-Newton)
Gramian used by Levenberg-Marquardt and natural-gradient steps; how to damp and
solve with it, and whether such steps converge for a given PDE, is left to the
caller.  Entries are sums of products of per-axis sums; no cancellation beyond
that of the per-point Jacobian products is introduced, but the matrix is
typically ill-conditioned, so solves need damping or a pseudo-inverse.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from omnibias.jax.experimental.closed_ridge import _validate, derivative_closed_system
from omnibias.jax.experimental.grid_ridge import _LAMBDA, _axes, _stack, _symbol
from omnibias.jax.experimental.tanh_ridge import RidgeTerm

import jax.numpy as jnp
from jax import Array

__all__ = ["closed_ridge_grid_gauss_newton"]


def _symbol_grad(W: Array, terms: tuple[RidgeTerm, ...], lam: complex | float) -> Array:
    """``d p(lam w_h) / d w_(h,i)`` as an ``(H, D)`` array."""
    base = W * lam
    out = jnp.zeros(W.shape, dtype=base.dtype)
    for term in terms:
        if term.kind == "laplace":
            s = jnp.sum(base * base, axis=1)
            scale = term.coeff * term.order * s ** (term.order - 1) * 2.0 * lam * lam
            out = out + scale[:, None] * W
        else:
            col = term.coeff * term.order * lam * base[:, term.axis] ** (term.order - 1)
            out = out.at[:, term.axis].add(col)
    return out


def _moment_kernels(M: Array) -> tuple[Array, Array, Array]:
    """Grid sums with no, one (``x_i``) and two (``x_i x_j``) coordinate factors from per-axis moments."""
    dim = M.shape[0]

    def prod(moments: list[int]) -> Array:
        out = M[0, moments[0]]
        for i in range(1, dim):
            out = out * M[i, moments[i]]
        return out

    k0 = prod([0] * dim)
    k1 = jnp.stack([prod([int(a == i) for a in range(dim)]) for i in range(dim)])
    k2 = jnp.stack(
        [
            jnp.stack(
                [
                    prod([(2 if i == j else 1) if a in (i, j) else 0 for a in range(dim)])
                    for j in range(dim)
                ]
            )
            for i in range(dim)
        ]
    )
    return k0, k1, k2


def _blocks(alpha: Array, beta: Array, alpha2: Array, beta2: Array, M: Array) -> Array:
    """``sum_x q (alpha + beta x_i)(alpha2 + beta2 x_j) K`` for every parameter-kind pair, ``(D+2, D+2, H, H)``."""
    k0, k1, k2 = _moment_kernels(M)
    H = alpha.shape[1]
    k1x = jnp.concatenate([jnp.zeros((2, H, H), k1.dtype), k1], axis=0)
    k2x = jnp.pad(k2, ((2, 0), (2, 0), (0, 0), (0, 0)))
    return (
        jnp.einsum("ih,jk,hk->ijhk", alpha, alpha2, k0)
        + jnp.einsum("ih,jk,jhk->ijhk", alpha, beta2, k1x)
        + jnp.einsum("ih,jk,ihk->ijhk", beta, alpha2, k1x)
        + jnp.einsum("ih,jk,ijhk->ijhk", beta, beta2, k2x)
    )


def closed_ridge_grid_gauss_newton(
    W: Array,
    b: Array,
    c: Array,
    nodes: Sequence[Array],
    terms: Sequence[RidgeTerm],
    *,
    weights: Sequence[Array] | None = None,
    activation: str = "sin",
) -> Array:
    """``sum_x q(x) g(x) g(x)^T`` with ``g = d(sum_t coeff_t L_t u)(x) / d theta`` over the grid ``nodes``.

    ``theta`` is in ``jax.flatten_util.ravel_pytree((W, b, c))`` order (``W`` row-major, then ``b``,
    then ``c``), so the result is ``(P, P)`` with ``P = (D + 2) H``.  ``nodes``, ``weights``, ``terms``
    and ``activation`` are those of
    :func:`~omnibias.jax.experimental.grid_ridge.closed_ridge_grid_loss`; together with the gradient of
    that loss this gives Gauss-Newton or Levenberg-Marquardt steps without visiting the grid points.
    """
    if activation not in _LAMBDA:
        derivative_closed_system(activation)
        raise ValueError(f"{activation!r} is not an exponential-family activation (sin, cos, exp)")
    H, dim = W.shape
    terms_t = _validate(terms, dim)
    dtype = jnp.result_type(W, b, c)
    xs, qs, _ = _axes(nodes, weights, None, dim, dtype)
    lam = _LAMBDA[activation]
    n = max(x.shape[0] for x in xs)
    X = _stack(xs, n)
    qm = _stack(qs, n)[:, None, :] * X[:, None, :] ** jnp.arange(3, dtype=dtype)[None, :, None]
    t = W.T[:, :, None] * X[:, None, :]
    p = _symbol(W, terms_t, lam)
    dp = _symbol_grad(W, terms_t, lam)
    if activation == "exp":
        eb = jnp.exp(b)
        amp = c * p * eb
        alpha = jnp.concatenate([(p * eb)[None], amp[None], ((c * eb)[:, None] * dp).T], axis=0)
        beta = jnp.concatenate([jnp.zeros((2, H), dtype), jnp.broadcast_to(amp, (dim, H))], axis=0)
        E = jnp.exp(t)
        G = _blocks(alpha, beta, alpha, beta, jnp.einsum("dhn,dmn,dkn->dmhk", E, qm, E))
    else:
        kap = 1.0 if activation == "sin" else 1j
        eb = jnp.exp(1j * b)
        amp = kap * c * p * eb
        alpha = jnp.concatenate(
            [(kap * p * eb)[None], (1j * amp)[None], (kap * (c * eb)[:, None] * dp).T], axis=0
        )
        beta = jnp.concatenate(
            [jnp.zeros((2, H), amp.dtype), jnp.broadcast_to(1j * amp, (dim, H))], axis=0
        )
        CS = jnp.concatenate([jnp.cos(t), jnp.sin(t)], axis=1)
        R = jnp.einsum("dhn,dmn,dkn->dmhk", CS, qm, CS)
        CC, CSt, SC, SS = R[..., :H, :H], R[..., :H, H:], R[..., H:, :H], R[..., H:, H:]
        Mm = (CC + SS) + 1j * (SC - CSt)
        Mp = (CC - SS) + 1j * (SC + CSt)
        G = 0.5 * jnp.real(
            _blocks(alpha, beta, jnp.conj(alpha), jnp.conj(beta), Mm)
        ) - 0.5 * jnp.real(_blocks(alpha, beta, alpha, beta, Mp))
    P = (dim + 2) * H
    G = jnp.transpose(G, (0, 2, 1, 3)).reshape(P, P)
    kinds = np.arange(P).reshape(dim + 2, H)
    perm = np.concatenate([kinds[2:].T.reshape(-1), kinds[1], kinds[0]])
    return G[perm][:, perm]
