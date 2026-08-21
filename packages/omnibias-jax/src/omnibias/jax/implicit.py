# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Implicit / DEQ Newton (theory 08-08), JAX twin.

Solve ``u = sigma(W u + x)`` to a named residual, then differentiate
the root by the implicit-function theorem with exact ``sigma'``
(bias collapse, ``delta -> 0``). One linear solve replaces unrolled
backprop through a fixed-point iteration.

The default solver loop is ``lax.while_loop`` (G4). A Python ``while``
is used only when ``require_contraction=True`` so a bound ``>= 1`` can
raise instead of silently unrolling. Do not wrap a data-dependent
Python ``break`` in ``jax.jit``. IFT is the chain rule at a fixed
point, not an absence of the chain rule. Not a global min and not CCF
stretch. Enable 64-bit JAX before the first array for torch parity.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

from omnibias.core.implicit import (
    DEQConfig,
    DEQNotContractive,
    DEQSolverUnknown,
    honesty_payload,
    reject_anderson,
    reject_deq_contraction,
)
from omnibias.jax.activations import get_activation

import jax
import jax.numpy as jnp
from jax import Array


@dataclass(frozen=True)
class DEQResult:
    """Fixed point, residual, iteration count, and contraction bound."""

    u: Array
    residual: float
    n_iter: int
    spectral_radius_bound: float


def _as_W(W: Array) -> Array:
    w = jnp.asarray(W)
    if w.ndim == 0:
        w = jnp.reshape(w, (1, 1))
    if w.ndim != 2 or int(w.shape[0]) != int(w.shape[1]):
        raise ValueError(f"W must be a square matrix, got shape {tuple(w.shape)}")
    return w


def _as_x(x: Array, d: int) -> Array:
    t = jnp.asarray(x)
    if t.ndim == 0:
        t = jnp.reshape(t, (1,))
    if t.ndim == 1:
        if int(t.shape[0]) != d:
            raise ValueError(f"x dim {int(t.shape[0])} must equal W size {d}")
        return t
    if int(t.shape[-1]) != d:
        raise ValueError(f"x trailing dim {int(t.shape[-1])} must equal W size {d}")
    return t


def _affine(u: Array, W: Array) -> Array:
    return jnp.tensordot(u, W, axes=([-1], [-1]))


def _sigma(z: Array, spec: str) -> Array:
    return get_activation(spec).forward(z)


def _sigma_prime(z: Array, spec: str) -> Array:
    act = get_activation(spec)
    fp = act.fastpath
    if fp is None:
        raise ValueError(
            f"activation {spec!r} has no fastpath; exact sigma' is required"
        )
    return fp(z, 1)


def _residual(u: Array, W: Array, x: Array, spec: str) -> Array:
    return u - _sigma(_affine(u, W) + x, spec)


def _residual_norm(res: Array) -> Array:
    return cast(Array, jnp.linalg.norm(res.reshape(-1)))


def _spectral_bound(u: Array, W: Array, x: Array, spec: str) -> Array:
    z = _affine(u, W) + x
    sigp = jnp.abs(_sigma_prime(z, spec))
    row_l1 = jnp.sum(jnp.abs(W), axis=-1)
    per = sigp * row_l1
    return jnp.max(per.reshape(-1))


def _linear_map(u: Array, W: Array, spec: str, x: Array) -> Array:
    z = _affine(u, W) + x
    return _sigma_prime(z, spec)


def _solve_A(sigp: Array, W: Array, rhs: Array) -> Array:
    d = int(W.shape[0])
    eye = jnp.eye(d, dtype=W.dtype)
    if rhs.ndim == 1:
        a = eye - sigp.reshape((d, 1)) * W
        return cast(Array, jnp.linalg.solve(a, rhs))
    n = int(rhs.shape[0])
    a = eye[None, :, :] - sigp.reshape((n, d, 1)) * W[None, :, :]
    return cast(Array, jnp.linalg.solve(a, rhs.reshape((n, d))))


def _newton_step(u: Array, W: Array, x: Array, spec: str) -> Array:
    res = _residual(u, W, x, spec)
    sigp = _linear_map(u, W, spec, x)
    return u + _solve_A(sigp, W, -res)


def _iterate_step(u: Array, W: Array, x: Array, spec: str) -> Array:
    return _sigma(_affine(u, W) + x, spec)


def _maybe_contract(
    u: Array, W: Array, x: Array, spec: str, config: DEQConfig
) -> float:
    bound = float(jnp.asarray(_spectral_bound(u, W, x, spec)).reshape(-1)[0])
    reject_deq_contraction(bound, require=config.require_contraction)
    return bound


def _step(u: Array, W: Array, x: Array, spec: str, solver: str) -> Array:
    if solver == "newton":
        return _newton_step(u, W, x, spec)
    return _iterate_step(u, W, x, spec)


def _while_loop_solve(
    W: Array,
    x: Array,
    spec: str,
    solver: str,
    max_iter: int,
    tol: float,
    u0: Array,
) -> tuple[Array, Array, Array]:
    def cond(state: tuple[Array, Array, Array]) -> Array:
        _u, i, res = state
        return jnp.logical_and(res > tol, i < max_iter)

    def body(state: tuple[Array, Array, Array]) -> tuple[Array, Array, Array]:
        u, i, _res = state
        u_new = _step(u, W, x, spec, solver)
        return u_new, i + jnp.int32(1), _residual_norm(_residual(u_new, W, x, spec))

    return jax.lax.while_loop(
        cond,
        body,
        (u0, jnp.int32(0), _residual_norm(_residual(u0, W, x, spec))),
    )


def deq_solve(
    W: Array,
    x: Array,
    spec: str = "tanh",
    *,
    config: DEQConfig | None = None,
) -> DEQResult:
    """Solve ``u = sigma(W u + x)`` by Newton or Banach iteration.

    jax deq_solve uses lax.while_loop for the Newton / Banach loop
    (G4). require_contraction=True falls back to a Python while so a
    bound >= 1 can raise. The IFT VJP is one linear solve, not
    unrolled BPTT.
    """
    cfg = config if config is not None else DEQConfig()
    reject_anderson(cfg.solver)
    w = _as_W(W)
    x_t = _as_x(x, int(w.shape[0])).astype(w.dtype)
    u = jnp.zeros_like(x_t)
    if cfg.require_contraction:
        bound = _maybe_contract(u, w, x_t, spec, cfg)
        res = float(jnp.asarray(_residual_norm(_residual(u, w, x_t, spec))).reshape(-1)[0])
        n_iter = 0
        while res > cfg.tol and n_iter < cfg.max_iter:
            u = _step(u, w, x_t, spec, cfg.solver)
            bound = _maybe_contract(u, w, x_t, spec, cfg)
            res = float(
                jnp.asarray(_residual_norm(_residual(u, w, x_t, spec))).reshape(-1)[0]
            )
            n_iter += 1
        return DEQResult(
            u=u, residual=res, n_iter=n_iter, spectral_radius_bound=bound
        )
    u, n_i, res_a = _while_loop_solve(
        w, x_t, spec, cfg.solver, cfg.max_iter, cfg.tol, u
    )
    bound = _maybe_contract(u, w, x_t, spec, cfg)
    return DEQResult(
        u=u,
        residual=float(jnp.asarray(res_a).reshape(-1)[0]),
        n_iter=int(jnp.asarray(n_i).reshape(-1)[0]),
        spectral_radius_bound=bound,
    )


def deq_vjp(
    W: Array,
    x: Array,
    spec: str,
    g: Array,
    *,
    config: DEQConfig | None = None,
    u: Array | None = None,
) -> Array:
    """VJP of ``u*`` with respect to ``W`` through the IFT.

    ``g`` is the cotangent of ``u*`` (same shape as ``u``). The
    backward linear solve uses exact ``sigma'``, not unrolled BPTT.
    """
    cfg = config if config is not None else DEQConfig()
    w = _as_W(W)
    x_t = _as_x(x, int(w.shape[0])).astype(w.dtype)
    if u is None:
        u_star = deq_solve(w, x_t, spec, config=cfg).u
    else:
        u_star = jnp.asarray(u, dtype=w.dtype)
        if u_star.shape != x_t.shape:
            raise ValueError(
                f"u shape {tuple(u_star.shape)} must match x shape {tuple(x_t.shape)}"
            )
    g_t = jnp.asarray(g, dtype=w.dtype)
    if g_t.shape != u_star.shape:
        raise ValueError(
            f"g shape {tuple(g_t.shape)} must match u shape {tuple(u_star.shape)}"
        )
    sigp = _linear_map(u_star, w, spec, x_t)
    if u_star.ndim == 1:
        d = int(w.shape[0])
        eye = jnp.eye(d, dtype=w.dtype)
        at = eye - w.T * sigp.reshape((1, d))
        v = jnp.linalg.solve(at, g_t)
        return jnp.outer(v * sigp, u_star)
    n = int(u_star.shape[0])
    d = int(w.shape[0])
    eye = jnp.eye(d, dtype=w.dtype)
    at = eye[None, :, :] - w.T[None, :, :] * sigp.reshape((n, 1, d))
    v = jnp.linalg.solve(at, g_t)
    return jnp.einsum("bi,bj->ij", v * sigp, u_star)


def deq_du_dW(
    W: Array,
    x: Array,
    spec: str = "tanh",
    *,
    config: DEQConfig | None = None,
    u: Array | None = None,
) -> Array:
    """IFT Jacobian ``du*/dW`` with shape ``(..., d, d, d)``."""
    cfg = config if config is not None else DEQConfig()
    w = _as_W(W)
    x_t = _as_x(x, int(w.shape[0])).astype(w.dtype)
    if u is None:
        u_star = deq_solve(w, x_t, spec, config=cfg).u
    else:
        u_star = jnp.asarray(u, dtype=w.dtype)
    sigp = _linear_map(u_star, w, spec, x_t)
    d = int(w.shape[0])
    eye = jnp.eye(d, dtype=w.dtype)
    if u_star.ndim == 1:
        a = eye - sigp.reshape((d, 1)) * w
        ainv = jnp.linalg.inv(a)
        return cast(
            Array,
            ainv[..., None] * sigp.reshape((1, d, 1)) * u_star.reshape((1, 1, d)),
        )
    n = int(u_star.shape[0])
    a = eye[None, :, :] - sigp.reshape((n, d, 1)) * w[None, :, :]
    ainv = jnp.linalg.inv(a)
    return cast(
        Array,
        ainv[..., None]
        * sigp.reshape((n, 1, d, 1))
        * u_star.reshape((n, 1, 1, d)),
    )


__all__ = [
    "DEQConfig",
    "DEQNotContractive",
    "DEQResult",
    "DEQSolverUnknown",
    "deq_du_dW",
    "deq_solve",
    "deq_vjp",
    "honesty_payload",
]
