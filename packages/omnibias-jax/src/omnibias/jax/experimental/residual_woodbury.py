# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Linear solves with the Jacobian of residual MLP blocks through the Woodbury identity.

For a vector field ``s(x) = W_L a_n + b_L`` with ``a_l = sigma_l(z_l)``, ``z_l = W_l a_{l-1} + b_l`` and
``a_0 = x``, the Jacobian is ``J = W_L A W_1`` with ``A = S_n W_n ... W_2 S_1`` and ``S_l = diag sigma_l'(z_l)``.
The residual block ``x + s(x)`` has the Jacobian ``I_D + W_L A W_1``, and Woodbury's identity moves the ``D x D``
system to the narrowest layer:

    (I_D + W_L A W_1)^{-1} g = g - W_L (I_{H_n} + A C)^{-1} A W_1 g,        C = W_1 W_L  (H_1 x H_n),

    (I_D + W_L A W_1)^{-T} g = g - W_1^T A^T (I_{H_n} + A C)^{-T} W_L^T g.

Per point this needs the ``H_n x H_n`` matrix ``A C`` (the ``H_n`` columns of ``C`` through the hidden layers, the
same matrix as for the log-determinant), one ``H_n x H_n`` LU and passes of the right-hand side through ``A``
(``H^2`` per layer), instead of the ``D x D`` Jacobian, an LU at ``D^3 / 3`` and ``B D^2`` memory. Newton inversion
of ``y = x + s(x)`` repeats one such solve per step (quadratic convergence, no ``D x D`` matrix). When the field has
fewer outputs than hidden units (``D < H_n``) the ``D x D`` system is the smaller one and is solved directly.

Everything is ordinary JAX, so parameter gradients, input gradients and higher derivatives follow from autodiff
(the derivative of the solve is the implicit one, from ``jnp.linalg.solve``).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import numpy as np
from omnibias.jax.experimental._chain import (
    affine_readout,
    carry_columns,
    hidden_slopes,
    readout_value,
)

import jax
import jax.numpy as jnp
from jax import Array

if TYPE_CHECKING:  # pragma: no cover
    from omnibias.jax.experimental._chain import LayerSpec

__all__ = ["residual_inverse", "residual_inverse_newton", "residual_solve", "residual_solve_transpose"]


def _slopes(x: Array, layers: Sequence[LayerSpec]) -> tuple[list[Array], list[Array], Array, Array]:
    """``(weights, slopes, W_L, value)`` of one pass; ``x`` is ``(B, D)``, slopes unit-major ``(H_l, B)``,
    ``value`` is ``(m, B)``."""
    hidden, WL, bL = affine_readout(layers, require_hidden=True)
    weights, slopes, a = hidden_slopes(x.T, hidden)
    return weights, slopes, WL, readout_value(WL, bL, a, len(hidden))


def _apply_A(weights: list[Array], slopes: list[Array], v: Array) -> Array:
    """``A v`` for one column per point: ``v`` is ``(H_1, B)``, the result ``(H_n, B)``."""
    v = slopes[0] * v
    for W, s in zip(weights[1:], slopes[1:], strict=True):
        v = s * (W @ v)
    return v


def _apply_At(weights: list[Array], slopes: list[Array], q: Array) -> Array:
    """``A^T q`` for one column per point: ``q`` is ``(H_n, B)``, the result ``(H_1, B)``."""
    w = slopes[-1] * q
    for layer in range(len(weights) - 1, 0, -1):
        w = slopes[layer - 1] * (weights[layer].T @ w)
    return w


def _system(weights: list[Array], slopes: list[Array], WL: Array) -> Array:
    """``I + A C`` per point, ``(B, H_n, H_n)`` (``C = W_1 W_L``), or ``I + J``, ``(B, D, D)``, when ``D < H_n``."""
    m, width = WL.shape
    seeds = weights[0] if m < width else weights[0] @ WL
    T = carry_columns(weights, slopes, seeds)
    block = jnp.einsum("ji,ibd->bjd", WL, T) if m < width else jnp.transpose(T, (1, 0, 2))
    return jnp.eye(block.shape[-1], dtype=block.dtype) + block


def _solve(
    weights: list[Array], slopes: list[Array], WL: Array, K: Array, g: Array, transpose: bool
) -> Array:
    """The solution rows ``(B, D)`` of ``(I + J)^{(T)} y = g`` from the system matrix ``K``; ``g`` is ``(B, D)``."""
    if WL.shape[0] < WL.shape[1]:
        Km = jnp.swapaxes(K, -1, -2) if transpose else K
        return jnp.linalg.solve(Km, g[..., None])[..., 0]  # type: ignore[no-any-return]
    if not transpose:
        v = _apply_A(weights, slopes, weights[0] @ g.T)  # A W_1 g, (H_n, B)
        q = jnp.linalg.solve(K, v.T[..., None])[..., 0]  # (B, H_n)
        return g - (WL @ q.T).T  # type: ignore[no-any-return]
    t = WL.T @ g.T  # W_L^T g, (H_n, B)
    q = jnp.linalg.solve(jnp.swapaxes(K, -1, -2), t.T[..., None])[..., 0]
    w = _apply_At(weights, slopes, q.T)  # (H_1, B)
    return g - (weights[0].T @ w).T


def _check(x: Array, WL: Array) -> None:
    if WL.shape[0] != x.shape[1]:
        raise ValueError(f"the field has {WL.shape[0]} outputs for {x.shape[1]} inputs")


def residual_solve(
    x: Array, layers: Sequence[LayerSpec], rhs: Array, *, transpose: bool = False
) -> tuple[Array, Array]:
    r"""``(s(x), y)`` with ``(I_D + J_s(x))^{(T)} y = rhs`` for the residual block ``x + s(x)``.

    Parameters
    ----------
    x : Array of shape ``(D,)`` or ``(B, D)``.
    layers : sequence of ``(W, b, spec)``
        ``W`` has shape ``(out, in)``; ``spec=None`` is an affine layer. The last layer must be affine and
        map to ``D`` outputs.
    rhs : Array with the shape of ``x``.
    transpose : solve with ``I + J_s^T`` (adjoint systems, gradients through the inverse map).

    Returns
    -------
    value, solution : Arrays with the shape of ``x``.
    """
    x = jnp.asarray(x)
    rhs = jnp.asarray(rhs)
    if rhs.shape != x.shape:
        raise ValueError("rhs must have the shape of x")
    batched = x.ndim > 1
    a, g = (x, rhs) if batched else (x[None, :], rhs[None, :])
    if a.ndim != 2:
        raise ValueError("x must have shape (D,) or (B, D)")
    weights, slopes, WL, value = _slopes(a, layers)
    _check(a, WL)
    K = _system(weights, slopes, WL)
    y = _solve(weights, slopes, WL, K, g, transpose)
    if not batched:
        return value[:, 0], y[0]
    return value.T, y


def residual_solve_transpose(
    x: Array, layers: Sequence[LayerSpec], rhs: Array
) -> tuple[Array, Array]:
    """``residual_solve(x, layers, rhs, transpose=True)``."""
    return residual_solve(x, layers, rhs, transpose=True)


def residual_inverse_newton(
    y: Array,
    layers: Sequence[LayerSpec],
    *,
    tol: float = 1e-12,
    max_iter: int = 30,
    x0: Array | None = None,
) -> tuple[Array, Array, Array]:
    """Solve ``x + s(x) = y`` by Newton's method with one Woodbury solve per step.

    Returns ``(x, steps, residual)``: the iterate, the number of Newton steps taken (a scalar shared by the
    batch) and the per-point ``max |x + s(x) - y|`` of the returned iterate. Points that reach ``tol`` are frozen.
    The iteration is not differentiated through; differentiate the converged map with :func:`residual_solve`
    (implicit function theorem). Newton's method needs a handful of steps at any contraction factor of ``s`` but each
    step costs a Woodbury solve: it is faster than fixed-point iteration when that factor is near or above one and slower
    when it is small (:func:`residual_inverse` with ``method="auto"`` chooses; ``docs/performance.md`` has the regimes).
    """
    y = jnp.asarray(y)
    batched = y.ndim > 1
    target = y if batched else y[None, :]
    if target.ndim != 2:
        raise ValueError("y must have shape (D,) or (B, D)")
    start = target if x0 is None else jnp.asarray(x0)
    start = start if batched or x0 is None else start[None, :]

    def evaluate(x: Array) -> tuple[list[Array], list[Array], Array, Array, Array]:
        weights, slopes, WL, value = _slopes(x, layers)
        _check(x, WL)
        r = x + value.T - target
        return weights, slopes, WL, r, jnp.max(jnp.abs(r), axis=-1)

    weights, slopes0, WL, r0, res0 = evaluate(start)

    def cond(state: tuple) -> Array:  # type: ignore[type-arg]
        return (jnp.max(state[3]) > tol) & (state[4] < max_iter)  # type: ignore[no-any-return]

    def body(state: tuple) -> tuple:  # type: ignore[type-arg]
        x, slopes, r, res, it = state
        slopes = list(slopes)
        K = _system(weights, slopes, WL)
        delta = _solve(weights, slopes, WL, K, r, False)
        x_new = jnp.where((res > tol)[:, None], x - delta, x)
        _, new_slopes, _, r_new, res_new = evaluate(x_new)
        return x_new, tuple(new_slopes), r_new, res_new, it + 1

    x, _, _, res, it = jax.lax.while_loop(
        cond, body, (start, tuple(slopes0), r0, res0, jnp.zeros((), dtype=jnp.int32))
    )
    if not batched:
        return x[0], it, res[0]
    return x, it, res


def residual_inverse(
    y: Array,
    layers: Sequence[LayerSpec],
    *,
    method: str = "auto",
    tol: float = 1e-12,
    max_iter: int | None = None,
    x0: Array | None = None,
    switch_steps: float | None = None,
) -> tuple[Array, Array, Array]:
    """Solve ``x + s(x) = y`` by fixed-point iteration, Newton's method with Woodbury solves, or both.

    ``method="fixed_point"`` iterates ``x <- y - s(x)`` (one pass of the field per step, linear convergence at the
    contraction factor of ``s``). ``method="newton"`` is :func:`residual_inverse_newton` (one pass and one Woodbury
    solve per step, quadratic convergence). ``method="auto"`` starts with fixed-point steps and switches, per call, to
    Newton's steps from the current iterate when the contraction observed over the last step (the ratio of the largest
    residuals, ``rho``) predicts more than ``switch_steps`` further fixed-point steps to reach ``tol``
    (``log(tol / res) / log(rho)``), or when a step did not contract at all (``rho >= 1``; Newton then restarts from the
    starting point, because that iterate may have left the basin of the solution). The default
    ``switch_steps`` is ``4 H_n`` on a CPU and ``H_n`` on an accelerator: a Newton step carries ``H_n`` columns through the
    hidden layers and solves an ``H_n x H_n`` system, which costs about ``0.8 H_n`` fixed-point steps on one CPU thread and
    ``0.2`` to ``0.7 H_n`` on an A100 (measured), and Newton's method needs about four steps. A strong contraction never pays
    for a solve; a map with a factor near one or above does not wait for a slow iteration.

    Returns ``(x, steps, residual)`` like :func:`residual_inverse_newton`. ``max_iter`` bounds the steps of each phase
    (default 30 for ``"newton"``, 500 otherwise). The iteration is not differentiated through; differentiate
    the converged map with :func:`residual_solve`.
    """
    if method not in ("auto", "fixed_point", "newton"):
        raise ValueError(f"method must be 'auto', 'fixed_point' or 'newton', not {method!r}")
    if method == "newton":
        return residual_inverse_newton(y, layers, tol=tol, max_iter=30 if max_iter is None else max_iter, x0=x0)
    y = jnp.asarray(y)
    batched = y.ndim > 1
    target = y if batched else y[None, :]
    if target.ndim != 2:
        raise ValueError("y must have shape (D,) or (B, D)")
    start = target if x0 is None else jnp.asarray(x0)
    start = start if batched or x0 is None else start[None, :]
    limit = 500 if max_iter is None else max_iter
    auto = method == "auto"
    hidden = jnp.asarray(layers[-1][0]).shape[1]
    # a Newton step costs about 0.8 H_n fixed-point steps on a CPU thread and 0.2 to 0.7 H_n on an A100 (measured, W10)
    switch_cap = (4.0 if jax.default_backend() == "cpu" else 1.0) * hidden if switch_steps is None else float(switch_steps)

    def evaluate(x: Array) -> tuple[Array, Array]:
        _, _, WL, value = _slopes(x, layers)
        _check(x, WL)
        r = x + value.T - target
        return r, jnp.max(jnp.abs(r), axis=-1)

    r0, res0 = evaluate(start)

    def cond(state: tuple) -> Array:  # type: ignore[type-arg]
        return (jnp.max(state[2]) > tol) & (state[3] < limit) & ~state[4]  # type: ignore[no-any-return]

    def body(state: tuple) -> tuple:  # type: ignore[type-arg]
        x, r, res, it, _, _ = state
        x_new = jnp.where((res > tol)[:, None], x - r, x)
        r_new, res_new = evaluate(x_new)
        stalled = diverged = jnp.zeros((), dtype=bool)
        if auto:
            tiny = float(np.finfo(np.dtype(r_new.dtype)).tiny)
            before, after = jnp.max(res), jnp.max(res_new)
            rho = after / jnp.maximum(before, tiny)
            remaining = jnp.log(jnp.maximum(tol, tiny) / jnp.maximum(after, tiny)) / jnp.log(jnp.minimum(rho, 1.0 - 1e-12))
            diverged = (it >= 1) & (after > tol) & (rho >= 1.0)
            stalled = diverged | ((it >= 1) & (after > tol) & (remaining > switch_cap))
        return x_new, r_new, res_new, it + 1, stalled, diverged

    flag = jnp.zeros((), dtype=bool)
    x, _, res, it, _, diverged = jax.lax.while_loop(
        cond, body, (start, r0, res0, jnp.zeros((), dtype=jnp.int32), flag, flag)
    )
    if auto:  # a call that converged enters the second loop with nothing left to do
        # a step that did not contract may have left the basin of the solution: Newton restarts from the start point then
        x, steps, res = residual_inverse_newton(target, layers, tol=tol, max_iter=limit, x0=jnp.where(diverged, start, x))
        it = it + steps
    if not batched:
        return x[0], it, res[0]
    return x, it, res
