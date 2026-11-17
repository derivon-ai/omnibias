# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Second-order steps for residual (PINN) losses: Levenberg-Marquardt, ENGD, SPRING and Kronecker-factored curvature.

The loss is ``1/2 * sum(r(theta) ** 2)`` for a residual vector ``r``; the Gauss-Newton direction solves
``(J^T J + lam I) delta = J^T r`` (primal) or, with ``delta = J^T alpha``, ``(J J^T + lam I) alpha = r`` (dual).  This module
adds what the dense, fixed-damping step lacks:

* :func:`gn_row_budget` scales the number of Jacobian rows to the parameter count and the memory available, and
  :func:`select_rows` takes a reproducible subset when the residual has more.
* :func:`gauss_newton_solve` chooses *dense* (Cholesky of the smaller of the two systems) or *matrix-free* conjugate
  gradients on ``(J J^T + lam I)`` / ``(J^T J + lam I)`` from one ``jax.linearize``, so ``J`` is never formed when the row
  count is large.  An optional SPRING-style momentum term ``mu`` pulls the update towards the previous one.
* :func:`lm_step` is a Levenberg-Marquardt step whose acceptance is decided on a *held-out* residual set (a trust region
  that cannot be fooled by overfitting the current batch) and whose damping follows the ratio of actual to predicted
  reduction.  It uses ``jnp.where`` for every decision, so it runs inside ``jit``/``scan`` with no host synchronisation.
* :func:`grid_line_search` is the deterministic, ``vmap``-ed grid search of energy natural gradient descent (ENGD):
  ``theta - 2**-k * delta`` for ``k = 0..K-1`` evaluated in one batched program, plus the no-move option.
* :func:`lowrank_damped_solve` and :func:`kron_damped_direction` build the KFAC-expand direction when one factor is a sum
  of rank-one terms: the first layer's input factor, whose derivative-stream rows repeat across the batch, is applied
  through the Woodbury identity in the ``m``-dimensional basis of the distinct rows (``m = B + k``), never as an
  ``n_in x n_in`` matrix and never with a ``B``-fold broadcast of the projection rows.
* :func:`lbfgs_with_counts` runs optax's L-BFGS with a zoom line search and returns the number of line-search steps taken
  by every iteration.

Nothing here claims convergence for a PDE; the steps are the standard algorithms with their stopping rules made explicit.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, NamedTuple

import numpy as np

import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree

Array = jax.Array


def _tiny(dtype: Any) -> float:
    return float(np.finfo(np.dtype(dtype)).tiny)


__all__ = [
    "GNSolution",
    "LMConfig",
    "LMState",
    "cg_solve",
    "engd_step",
    "gauss_newton_solve",
    "gn_row_budget",
    "grid_line_search",
    "init_lm_state",
    "kron_damped_direction",
    "lbfgs_with_counts",
    "lm_step",
    "lowrank_damped_solve",
    "select_rows",
]

#: ``GNSolution.method`` codes (an int so the step stays traceable).
METHODS = {"dense_dual": 0, "dense_primal": 1, "cg_dual": 2, "cg_primal": 3}


# ----------------------------------------------------------------------------------------------------- row budget
def gn_row_budget(
    n_params: int,
    *,
    dtype: Any = jnp.float64,
    memory_bytes: float = 2.0 * 2**30,
    row_ratio: float = 2.0,
    minimum: int = 64,
    maximum: int | None = None,
) -> int:
    """How many residual rows a dense Gauss-Newton step should use.

    Rows beyond a few multiples of the parameter count ``P`` add no rank to ``J^T J``; the dense solve also stores ``J``
    (``R x P``), the Gram matrix and its Cholesky factor (two ``R x R`` arrays).  The budget is
    ``min(row_ratio * P, R_mem)`` with ``R_mem`` the positive root of ``2 R^2 + P R = memory / itemsize``, clipped to
    ``[minimum, maximum]``.  A residual with more rows than this should be subsampled (:func:`select_rows`) or solved
    matrix-free (:func:`gauss_newton_solve` with ``solver="cg"``).
    """
    if n_params < 1:
        raise ValueError("n_params must be positive")
    if memory_bytes <= 0 or row_ratio <= 0:
        raise ValueError("memory_bytes and row_ratio must be positive")
    itemsize = float(np.dtype(dtype).itemsize)
    budget = memory_bytes / itemsize
    p = float(n_params)
    r_mem = (-p + math.sqrt(p * p + 8.0 * budget)) / 4.0
    rows = int(min(row_ratio * p, r_mem))
    rows = max(int(minimum), rows)
    if maximum is not None:
        rows = min(rows, int(maximum))
    return max(1, rows)


def select_rows(key: Array, n_rows: int, budget: int) -> Array:
    """A reproducible sorted subset of ``budget`` row indices out of ``n_rows`` (all rows when ``budget >= n_rows``)."""
    if budget >= n_rows:
        return jnp.arange(n_rows)
    return jnp.sort(jax.random.permutation(key, n_rows)[:budget])


# ------------------------------------------------------------------------------------------------------------ CG
def cg_solve(
    matvec: Callable[[Array], Array],
    b: Array,
    *,
    x0: Array | None = None,
    tol: float = 1e-10,
    maxiter: int = 200,
) -> tuple[Array, Array]:
    """Conjugate gradients for a symmetric positive-definite ``matvec``; returns ``(x, iterations)``.

    Stops when the residual norm is below ``tol * ||b||`` or after ``maxiter`` matvecs; traceable (``lax.while_loop``).
    """
    x = jnp.zeros_like(b) if x0 is None else x0
    r = b - matvec(x)
    p = r
    rs = jnp.vdot(r, r)
    bnorm2 = jnp.maximum(jnp.vdot(b, b), _tiny(b.dtype))
    thresh = (tol * tol) * bnorm2

    def cond(state: tuple[Array, Array, Array, Array, Array]) -> Array:
        _, _, _, rs_, k = state
        return (rs_ > thresh) & (k < maxiter)

    def body(
        state: tuple[Array, Array, Array, Array, Array],
    ) -> tuple[Array, Array, Array, Array, Array]:
        x_, r_, p_, rs_, k = state
        ap = matvec(p_)
        alpha = rs_ / jnp.maximum(jnp.vdot(p_, ap), _tiny(b.dtype))
        x_ = x_ + alpha * p_
        r_ = r_ - alpha * ap
        rs_new = jnp.vdot(r_, r_)
        p_ = r_ + (rs_new / jnp.maximum(rs_, _tiny(b.dtype))) * p_
        return x_, r_, p_, rs_new, k + 1

    x, _, _, _, k = jax.lax.while_loop(cond, body, (x, r, p, rs, jnp.zeros((), jnp.int32)))
    return x, k


# ------------------------------------------------------------------------------------- Gauss-Newton direction
class GNSolution(NamedTuple):
    """``delta`` is the update to *subtract* (``theta - delta``); ``jdelta = J delta`` gives the predicted reduction."""

    delta: Array
    jdelta: Array
    method: Array
    iterations: Array


def _linear_maps(
    f: Callable[[Array], Array], theta: Array
) -> tuple[Array, Callable[[Array], Array], Callable[[Array], Array]]:
    r, jvp = jax.linearize(f, theta)
    transpose = jax.linear_transpose(jvp, theta)

    def vjp(u: Array) -> Array:
        out: Array = transpose(u)[0]
        return out

    return r, jvp, vjp


def _dense_jacobian(
    f: Callable[[Array], Array], theta: Array, n_rows: int, chunk: int | None
) -> Array:
    n = theta.shape[0]
    jac: Array
    if n_rows >= n:
        jac = jax.jacfwd(f)(theta)
        return jac
    _, vjp_fn = jax.vjp(f, theta)

    def row(e: Array) -> Array:
        out: Array = vjp_fn(e)[0]
        return out

    eye = jnp.eye(n_rows, dtype=theta.dtype)
    if chunk is None or chunk >= n_rows:
        jac = jax.vmap(row)(eye)
        return jac
    pad = -(-n_rows // chunk) * chunk
    eye = jnp.eye(pad, n_rows, dtype=theta.dtype).reshape(pad // chunk, chunk, n_rows)
    rows: Array = jax.lax.map(jax.vmap(row), eye)
    jac = rows.reshape(pad, n)[:n_rows]
    return jac


def gauss_newton_solve(
    f: Callable[[Array], Array],
    theta: Array,
    lam: Array | float,
    *,
    solver: str = "auto",
    momentum: float = 0.0,
    prev_delta: Array | None = None,
    dense_rows: int | None = None,
    cg_tol: float = 1e-8,
    cg_maxiter: int = 200,
    chunk: int | None = 64,
) -> tuple[Array, GNSolution]:
    """The damped Gauss-Newton update ``delta`` for the flat residual map ``f(theta)``; returns ``(r, solution)``.

    ``solver`` is ``"dense"`` (Cholesky of the smaller of ``J J^T + lam I`` (``R x R``) and ``J^T J + lam I``
    (``P x P``)), ``"cg"`` (matrix-free conjugate gradients on the same system from one linearization), or ``"auto"``
    (dense while ``R <= dense_rows``, default :func:`gn_row_budget` of ``P``).  ``momentum`` ``mu`` with ``prev_delta`` is
    the SPRING update: ``delta = J^T (J J^T + lam I)^{-1} (r - mu J d_prev) + mu d_prev``, the minimiser of
    ``||J delta - r||^2 + lam ||delta - mu d_prev||^2``; it is applied in the dual form for both solvers.
    """
    if solver not in ("auto", "dense", "cg"):
        raise ValueError("solver must be 'auto', 'dense' or 'cg'")
    lam = jnp.asarray(lam, theta.dtype)
    r0 = jax.eval_shape(f, theta)
    n_rows, n = int(r0.shape[0]), int(theta.shape[0])
    if momentum != 0.0 and prev_delta is None:
        raise ValueError("momentum needs prev_delta")
    use_spring = momentum != 0.0
    if solver == "auto":
        budget = dense_rows if dense_rows is not None else gn_row_budget(n, dtype=theta.dtype)
        solver = "dense" if n_rows <= budget else "cg"
    dual = use_spring or n_rows <= n

    if solver == "dense":
        jac = _dense_jacobian(f, theta, n_rows, chunk)
        r = f(theta)
        if dual:
            rhs = r if not use_spring else r - momentum * (jac @ prev_delta)  # type: ignore[operator]
            gram = jac @ jac.T + lam * jnp.eye(n_rows, dtype=theta.dtype)
            alpha = jax.scipy.linalg.cho_solve(jax.scipy.linalg.cho_factor(gram), rhs)
            delta = jac.T @ alpha
            if use_spring:
                delta = delta + momentum * prev_delta  # type: ignore[operator]
            code = METHODS["dense_dual"]
        else:
            gram = jac.T @ jac + lam * jnp.eye(n, dtype=theta.dtype)
            delta = jax.scipy.linalg.cho_solve(jax.scipy.linalg.cho_factor(gram), jac.T @ r)
            code = METHODS["dense_primal"]
        return r, GNSolution(
            delta, jac @ delta, jnp.asarray(code, jnp.int32), jnp.zeros((), jnp.int32)
        )

    r, jvp, vjp = _linear_maps(f, theta)
    if dual:
        rhs = r if not use_spring else r - momentum * jvp(prev_delta)  # type: ignore[arg-type]
        alpha, iters = cg_solve(
            lambda a: jvp(vjp(a)) + lam * a, rhs, tol=cg_tol, maxiter=cg_maxiter
        )
        delta = vjp(alpha)
        if use_spring:
            delta = delta + momentum * prev_delta  # type: ignore[operator]
        code = METHODS["cg_dual"]
    else:
        delta, iters = cg_solve(
            lambda v: vjp(jvp(v)) + lam * v, vjp(r), tol=cg_tol, maxiter=cg_maxiter
        )
        code = METHODS["cg_primal"]
    return r, GNSolution(delta, jvp(delta), jnp.asarray(code, jnp.int32), iters)


# ----------------------------------------------------------------------------------------- Levenberg-Marquardt
@dataclass(frozen=True)
class LMConfig:
    """Levenberg-Marquardt controls.

    ``up``/``down`` scale ``lam`` after a poor/good step (ratio ``rho`` of actual to predicted reduction below ``poor`` /
    above ``good``); a rejected step also multiplies by ``up``.  ``momentum`` > 0 selects the SPRING update.  ``solver``,
    ``dense_rows``, ``cg_tol`` and ``cg_maxiter`` go to :func:`gauss_newton_solve`.
    """

    lam_min: float = 1e-10
    lam_max: float = 1e10
    up: float = 4.0
    down: float = 3.0
    poor: float = 0.25
    good: float = 0.75
    momentum: float = 0.0
    solver: str = "auto"
    dense_rows: int | None = None
    cg_tol: float = 1e-8
    cg_maxiter: int = 200
    chunk: int | None = 64


class LMState(NamedTuple):
    lam: Array
    prev_delta: Array
    step: Array


def init_lm_state(params: Any, lam0: float = 1e-2) -> LMState:
    flat, _ = ravel_pytree(params)
    return LMState(jnp.asarray(lam0, flat.dtype), jnp.zeros_like(flat), jnp.zeros((), jnp.int32))


def _half_sq(r: Array) -> Array:
    return 0.5 * jnp.sum(jnp.square(r))


def lm_step(
    residual: Callable[[Any, Array], Array],
    params: Any,
    state: LMState,
    key: Array,
    *,
    heldout: Callable[[Any, Array], Array] | None = None,
    config: LMConfig | None = None,
) -> tuple[Any, LMState, dict[str, Array]]:
    """One Levenberg-Marquardt step with trust-region acceptance on a held-out residual.

    ``residual(params, key)`` is the training residual vector (a fresh batch per key); ``heldout(params, key)`` is the
    residual on points the step did not see (omit it to accept on the training batch, the classical rule).  A step is
    accepted when the held-out mean square decreases; ``rho`` is the actual held-out reduction over the predicted
    training reduction (both per row), which drives ``lam``.  Everything is branch-free, so the step can sit in a ``scan``.
    """
    config = LMConfig() if config is None else config
    flat, unravel = ravel_pytree(params)

    def f(th: Array) -> Array:
        return residual(unravel(th), key)

    r, sol = gauss_newton_solve(
        f,
        flat,
        state.lam,
        solver=config.solver,
        momentum=config.momentum,
        prev_delta=state.prev_delta if config.momentum != 0.0 else None,
        dense_rows=config.dense_rows,
        cg_tol=config.cg_tol,
        cg_maxiter=config.cg_maxiter,
        chunk=config.chunk,
    )
    cand = flat - sol.delta
    n_train = r.shape[0]
    pred = (
        _half_sq(r) - _half_sq(r - sol.jdelta)
    ) / n_train  # per-row predicted reduction on the training rows
    if heldout is None:
        r1 = f(cand)
        before, after, rows = _half_sq(r), _half_sq(r1), n_train
    else:
        k_h = jax.random.fold_in(key, 1)
        h0, h1 = heldout(unravel(flat), k_h), heldout(unravel(cand), k_h)
        before, after, rows = _half_sq(h0), _half_sq(h1), h0.shape[0]
    actual = (before - after) / rows
    accepted = jnp.isfinite(after) & (after < before)
    rho = actual / jnp.maximum(pred, _tiny(flat.dtype))
    scale = jnp.where(
        ~accepted | (rho < config.poor),
        config.up,
        jnp.where(rho > config.good, 1.0 / config.down, 1.0),
    )
    lam = jnp.clip(state.lam * scale, config.lam_min, config.lam_max)
    new_flat = jnp.where(accepted, cand, flat)
    prev = jnp.where(accepted, sol.delta, jnp.zeros_like(sol.delta))
    info = {
        "loss": _half_sq(r),
        "before": before,
        "after": after,
        "rho": rho,
        "accepted": accepted,
        "lam": lam,
        "method": sol.method,
        "cg_iterations": sol.iterations,
    }
    return unravel(new_flat), LMState(lam, prev, state.step + 1), info


# ------------------------------------------------------------------------------------------------------- ENGD
def grid_line_search(
    loss: Callable[[Any], Array],
    params: Any,
    delta: Any,
    *,
    n_scales: int = 21,
    chunk: int | None = None,
) -> tuple[Any, Array, Array]:
    """Best of ``params - 2**-k * delta`` (``k = 0..n_scales-1``) or no move, by ``loss``; deterministic.

    The candidates are evaluated by one ``vmap`` (or ``lax.map`` over ``chunk``-sized groups to bound memory).  Returns
    ``(new_params, k, evals)`` with ``k = n_scales`` when no candidate beats the starting loss and ``evals = n_scales + 1``
    loss evaluations.  Ties go to the larger step.
    """
    scales = jnp.asarray(2.0 ** -np.arange(n_scales), jax.tree.leaves(params)[0].dtype)

    def at(s: Array) -> Array:
        return loss(jax.tree.map(lambda p, d: p - s * d, params, delta))

    if chunk is None or chunk >= n_scales:
        fs = jax.vmap(at)(scales)
    else:
        pad = -(-n_scales // chunk) * chunk
        padded = jnp.concatenate([scales, jnp.zeros(pad - n_scales, scales.dtype)])
        fs = jax.lax.map(jax.vmap(at), padded.reshape(pad // chunk, chunk)).reshape(pad)[:n_scales]
    f0 = loss(params)
    j = jnp.argmin(fs)
    ok = fs[j] < f0
    s_best = jnp.where(ok, scales[j], 0.0)
    new = jax.tree.map(lambda p, d: p - s_best * d, params, delta)
    return new, jnp.where(ok, j, n_scales).astype(jnp.int32), jnp.asarray(n_scales + 1, jnp.int32)


def engd_step(
    residual: Callable[[Any, Array], Array],
    params: Any,
    key: Array,
    lam: float | Array,
    *,
    n_scales: int = 21,
    chunk: int | None = None,
    solver: str = "auto",
    dense_rows: int | None = None,
    cg_tol: float = 1e-8,
    cg_maxiter: int = 200,
) -> tuple[Any, dict[str, Array]]:
    """Energy natural gradient descent: the damped Gauss-Newton direction with a fixed small ``lam`` and the grid search."""
    flat, unravel = ravel_pytree(params)

    def f(th: Array) -> Array:
        return residual(unravel(th), key)

    r, sol = gauss_newton_solve(
        f, flat, lam, solver=solver, dense_rows=dense_rows, cg_tol=cg_tol, cg_maxiter=cg_maxiter
    )

    def objective(th: Array) -> Array:
        return _half_sq(f(th))

    new_flat, k, evals = grid_line_search(
        objective, flat, sol.delta, n_scales=n_scales, chunk=chunk
    )
    return unravel(new_flat), {
        "loss": _half_sq(r),
        "k": k,
        "evals": evals,
        "method": sol.method,
        "cg_iterations": sol.iterations,
    }


# --------------------------------------------------------------------------------------------------------- KFAC
def lowrank_damped_solve(u: Array, c: Array, damping: Array | float, rhs: Array) -> Array:
    """Solve ``(damping * I + U diag(c) U^T) Z = RHS`` through the ``m x m`` Woodbury system.

    ``U`` is ``(n, m)`` with ``m << n``, ``c > 0`` is ``(m,)``, ``RHS`` is ``(n, k)``.  Cost ``O(n m^2 + m^3)`` and no
    ``n x n`` matrix: this is how the first layer's KFAC input factor ``sum_i c_i u_i u_i^T`` is inverted when its terms are
    the batch inputs and the (batch-repeated) projection rows.
    """
    damping = jnp.asarray(damping, u.dtype)
    inner = jnp.diag(damping / c) + u.T @ u
    chol = jax.scipy.linalg.cho_factor(inner)
    return (rhs - u @ jax.scipy.linalg.cho_solve(chol, u.T @ rhs)) / damping


def kron_damped_direction(
    u: Array,
    c: Array,
    gram_out: Array,
    n_samples: float | Array,
    lam: float | Array,
    grad: Array,
) -> Array:
    """KFAC-expand direction ``N * A^{-1} grad G^{-1}`` for one layer with a low-rank input factor.

    ``A = U diag(c) U^T`` (``(n_in+1)`` square, the homogeneous bias coordinate included in ``U``), ``G`` the output factor
    ``gram_out`` (``out x out``), ``grad`` the exact ``(out, n_in+1)`` gradient ``J^T r`` of the layer.  The damping
    ``lam * N`` is split between the factors by their mean traces (the trace-balanced rule), so the Kronecker product
    approximates ``F + lam * N * I`` where ``F`` is the Fisher.
    """
    n_in = u.shape[0]
    dtype = u.dtype
    root = jnp.sqrt(jnp.asarray(lam, dtype) * n_samples)
    tr_a = jnp.sum(c * jnp.sum(jnp.square(u), axis=0)) / n_in
    tr_g = jnp.trace(gram_out) / gram_out.shape[0]
    pi = jnp.sqrt(jnp.maximum(tr_a, 1e-300) / jnp.maximum(tr_g, 1e-300))
    gd = gram_out + (root / pi) * jnp.eye(gram_out.shape[0], dtype=dtype)
    left = jax.scipy.linalg.cho_solve(
        jax.scipy.linalg.cho_factor(gd), grad
    )  # G^-1 grad: (out, n_in+1)
    right = lowrank_damped_solve(u, c, pi * root, left.T).T  # (A + pi*root I)^-1 on the input side
    return right * n_samples


# ------------------------------------------------------------------------------------------------------ L-BFGS
def lbfgs_with_counts(
    loss: Callable[[Any], Array],
    params: Any,
    *,
    steps: int,
    memory_size: int = 10,
    max_linesearch_steps: int = 20,
) -> tuple[Any, dict[str, Array]]:
    """optax L-BFGS (zoom line search) for ``steps`` iterations with the per-iteration line-search counts.

    Returns ``(params, history)`` with ``history`` holding ``loss``, ``linesearch_steps`` (function/gradient evaluations
    the line search spent in each iteration) and their ``total``.  Needs ``optax``.
    """
    try:
        import optax  # type: ignore[import-untyped]
        import optax.tree_utils as otu  # type: ignore[import-untyped]
    except ImportError as err:  # pragma: no cover - optional dependency
        raise ImportError("lbfgs_with_counts needs optax") from err
    solver = optax.lbfgs(
        memory_size=memory_size,
        linesearch=optax.scale_by_zoom_linesearch(max_linesearch_steps=max_linesearch_steps),
    )
    vg = optax.value_and_grad_from_state(loss)

    def body(carry: tuple[Any, Any], _: Any) -> tuple[tuple[Any, Any], tuple[Array, Array]]:
        p, st = carry
        value, g = vg(p, state=st)
        updates, st = solver.update(g, st, p, value=value, grad=g, value_fn=loss)
        info = otu.tree_get(st, "info")
        evals = jnp.asarray(info.num_linesearch_steps, jnp.int32)
        return (optax.apply_updates(p, updates), st), (value, evals)

    (p, _), (values, counts) = jax.lax.scan(body, (params, solver.init(params)), None, length=steps)
    return p, {"loss": values, "linesearch_steps": counts, "total": jnp.sum(counts)}


def stack_rows(parts: Sequence[Array]) -> Array:
    """Concatenate residual blocks into one vector (a convenience for ``residual`` callables)."""
    return jnp.concatenate([jnp.reshape(p, (-1,)) for p in parts])
