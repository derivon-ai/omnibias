# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""PyTorch twin of :mod:`omnibias.jax.optim_pinn`: row budget, dense/matrix-free Gauss-Newton, SPRING, Levenberg-Marquardt
with held-out acceptance, ENGD, KFAC with a low-rank input factor and L-BFGS line-search counts.

The loss is ``1/2 * sum(r(theta) ** 2)`` and the maps are the same as in the JAX module; this file only changes what the
framework requires:

* parameters are an arbitrary pytree of tensors, flattened with :func:`ravel_params` (the same order as
  ``jax.flatten_util.ravel_pytree`` for lists/tuples/dicts of arrays);
* ``residual(params)`` has no ``key`` argument (draw the batch inside, or close over it);
* ``J v`` is ``torch.func.jvp`` (one forward pass with a tangent) and ``J^T u`` is the ``torch.func.vjp`` of one forward
  pass;
* conjugate gradients run a fixed number of masked iterations and look at convergence only every ``check_every``
  iterations (``check_every=0`` never reads a device value, as CUDA-graph capture needs), where the JAX version uses a
  ``while_loop``;
* every accept/reject decision is a ``torch.where``, so :func:`lm_step` does not synchronise the host.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, NamedTuple

import torch
from torch import Tensor
from torch.utils import _pytree

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
    "ravel_params",
    "select_rows",
]

METHODS = {"dense_dual": 0, "dense_primal": 1, "cg_dual": 2, "cg_primal": 3}


def ravel_params(params: Any) -> tuple[Tensor, Callable[[Tensor], Any]]:
    """Flatten a pytree of tensors to one vector and return ``(flat, unravel)``; ``unravel`` is differentiable."""
    leaves, spec = _pytree.tree_flatten(params)
    shapes = [leaf.shape for leaf in leaves]
    sizes = [leaf.numel() for leaf in leaves]
    flat = torch.cat([leaf.reshape(-1) for leaf in leaves]) if leaves else torch.zeros(0)

    def unravel(vec: Tensor) -> Any:
        parts = torch.split(vec, sizes)
        return _pytree.tree_unflatten(
            [p.reshape(s) for p, s in zip(parts, shapes, strict=True)], spec
        )

    return flat, unravel


# ----------------------------------------------------------------------------------------------------- row budget
def gn_row_budget(
    n_params: int,
    *,
    dtype: torch.dtype = torch.float64,
    memory_bytes: float = 2.0 * 2**30,
    row_ratio: float = 2.0,
    minimum: int = 64,
    maximum: int | None = None,
) -> int:
    """Rows a dense Gauss-Newton step should use: ``min(row_ratio * P, R_mem)`` clipped to ``[minimum, maximum]``.

    ``R_mem`` is the positive root of ``2 R^2 + P R = memory / itemsize`` (the Jacobian, the Gram matrix and its factor).
    """
    if n_params < 1:
        raise ValueError("n_params must be positive")
    if memory_bytes <= 0 or row_ratio <= 0:
        raise ValueError("memory_bytes and row_ratio must be positive")
    itemsize = float(torch.empty((), dtype=dtype).element_size())
    p = float(n_params)
    r_mem = (-p + math.sqrt(p * p + 8.0 * memory_bytes / itemsize)) / 4.0
    rows = max(int(minimum), int(min(row_ratio * p, r_mem)))
    if maximum is not None:
        rows = min(rows, int(maximum))
    return max(1, rows)


def select_rows(
    n_rows: int, budget: int, *, generator: torch.Generator | None = None, device: Any = None
) -> Tensor:
    """A reproducible sorted subset of ``budget`` row indices out of ``n_rows`` (all rows when it fits)."""
    if budget >= n_rows:
        return torch.arange(n_rows, device=device)
    perm = torch.randperm(n_rows, generator=generator, device=device)
    return torch.sort(perm[:budget]).values


# ------------------------------------------------------------------------------------------------------------ CG
def cg_solve(
    matvec: Callable[[Tensor], Tensor],
    b: Tensor,
    *,
    tol: float = 1e-10,
    maxiter: int = 200,
    check_every: int = 10,
) -> tuple[Tensor, Tensor]:
    """Conjugate gradients for an SPD ``matvec``; returns ``(x, iterations)`` (``iterations`` a device tensor).

    Iterations after convergence are masked to no-ops.  With ``check_every > 0`` the loop stops early when a (host) read
    every ``check_every`` iterations finds convergence; with ``0`` it runs ``maxiter`` masked iterations and never reads a
    device value.
    """
    tiny = torch.finfo(b.dtype).tiny
    x = torch.zeros_like(b)
    r = b.clone()
    p = r.clone()
    rs = torch.dot(r, r)
    thresh = (tol * tol) * torch.clamp(torch.dot(b, b), min=tiny)
    iters = torch.zeros((), dtype=torch.int32, device=b.device)
    for k in range(maxiter):
        active = rs > thresh
        ap = matvec(p)
        alpha = torch.where(
            active, rs / torch.clamp(torch.dot(p, ap), min=tiny), torch.zeros_like(rs)
        )
        x = x + alpha * p
        r = r - alpha * ap
        rs_new = torch.dot(r, r)
        beta = torch.where(active, rs_new / torch.clamp(rs, min=tiny), torch.zeros_like(rs))
        p = torch.where(active, r + beta * p, p)
        rs = torch.where(active, rs_new, rs)
        iters = iters + active.to(torch.int32)
        if check_every and (k + 1) % check_every == 0 and bool(rs <= thresh):
            break
    return x, iters


# ------------------------------------------------------------------------------------- Gauss-Newton direction
class GNSolution(NamedTuple):
    delta: Tensor
    jdelta: Tensor
    method: Tensor
    iterations: Tensor


def _dense_jacobian(
    f: Callable[[Tensor], Tensor], theta: Tensor, n_rows: int, chunk: int | None
) -> Tensor:
    n = theta.shape[0]
    if n_rows >= n:
        full: Tensor = torch.func.jacfwd(f)(theta)
        return full
    _, vjp_fn = torch.func.vjp(f, theta)

    def row(e: Tensor) -> Tensor:
        out: Tensor = vjp_fn(e)[0]
        return out

    eye = torch.eye(n_rows, dtype=theta.dtype, device=theta.device)
    step = n_rows if chunk is None else min(chunk, n_rows)
    rows = [torch.func.vmap(row)(eye[i : i + step]) for i in range(0, n_rows, step)]
    return torch.cat(rows, dim=0)


def gauss_newton_solve(
    f: Callable[[Tensor], Tensor],
    theta: Tensor,
    lam: Tensor | float,
    *,
    solver: str = "auto",
    momentum: float = 0.0,
    prev_delta: Tensor | None = None,
    dense_rows: int | None = None,
    cg_tol: float = 1e-8,
    cg_maxiter: int = 200,
    check_every: int = 10,
    chunk: int | None = 64,
) -> tuple[Tensor, GNSolution]:
    """Damped Gauss-Newton update ``delta`` (subtract it) for the flat residual map ``f``; returns ``(r, solution)``.

    ``solver``: ``"dense"``, ``"cg"`` (matrix-free) or ``"auto"`` (dense while ``R <= dense_rows``).  ``momentum`` with
    ``prev_delta`` is the SPRING update ``delta = J^T (J J^T + lam I)^{-1} (r - mu J d_prev) + mu d_prev``.
    """
    if solver not in ("auto", "dense", "cg"):
        raise ValueError("solver must be 'auto', 'dense' or 'cg'")
    theta = theta.detach()
    lam_t = torch.as_tensor(lam, dtype=theta.dtype, device=theta.device)
    n_rows = int(f(theta).shape[0])
    n = theta.shape[0]
    if momentum != 0.0 and prev_delta is None:
        raise ValueError("momentum needs prev_delta")
    use_spring = momentum != 0.0
    if solver == "auto":
        budget = dense_rows if dense_rows is not None else gn_row_budget(n, dtype=theta.dtype)
        solver = "dense" if n_rows <= budget else "cg"
    dual = use_spring or n_rows <= n
    zero_iters = torch.zeros((), dtype=torch.int32, device=theta.device)

    if solver == "dense":
        jac = _dense_jacobian(f, theta, n_rows, chunk)
        r = f(theta)
        if dual:
            rhs = r if not use_spring else r - momentum * (jac @ prev_delta)  # type: ignore[operator]
            gram = jac @ jac.T + lam_t * torch.eye(n_rows, dtype=theta.dtype, device=theta.device)
            alpha = torch.cholesky_solve(rhs[:, None], torch.linalg.cholesky(gram))[:, 0]
            delta = jac.T @ alpha
            if use_spring:
                delta = delta + momentum * prev_delta  # type: ignore[operator]
            code = METHODS["dense_dual"]
        else:
            gram = jac.T @ jac + lam_t * torch.eye(n, dtype=theta.dtype, device=theta.device)
            delta = torch.cholesky_solve((jac.T @ r)[:, None], torch.linalg.cholesky(gram))[:, 0]
            code = METHODS["dense_primal"]
        return r, GNSolution(
            delta,
            jac @ delta,
            torch.tensor(code, dtype=torch.int32, device=theta.device),
            zero_iters,
        )

    r, vjp_fn = torch.func.vjp(f, theta)

    def jvp(
        v: Tensor,
    ) -> Tensor:  # forward mode re-runs the forward pass with a tangent: no stored linearization
        out: Tensor = torch.func.jvp(f, (theta,), (v,))[1]
        return out

    def vjp(u: Tensor) -> Tensor:
        out: Tensor = vjp_fn(u)[0]
        return out

    if dual:
        rhs = r if not use_spring else r - momentum * jvp(prev_delta)  # type: ignore[arg-type]
        alpha, iters = cg_solve(
            lambda a: jvp(vjp(a)) + lam_t * a,
            rhs,
            tol=cg_tol,
            maxiter=cg_maxiter,
            check_every=check_every,
        )
        delta = vjp(alpha)
        if use_spring:
            delta = delta + momentum * prev_delta  # type: ignore[operator]
        code = METHODS["cg_dual"]
    else:
        delta, iters = cg_solve(
            lambda v: vjp(jvp(v)) + lam_t * v,
            vjp(r),
            tol=cg_tol,
            maxiter=cg_maxiter,
            check_every=check_every,
        )
        code = METHODS["cg_primal"]
    return r, GNSolution(
        delta, jvp(delta), torch.tensor(code, dtype=torch.int32, device=theta.device), iters
    )


# ----------------------------------------------------------------------------------------- Levenberg-Marquardt
@dataclass(frozen=True)
class LMConfig:
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
    check_every: int = 10
    chunk: int | None = 64


class LMState(NamedTuple):
    lam: Tensor
    prev_delta: Tensor
    step: Tensor


def init_lm_state(params: Any, lam0: float = 1e-2) -> LMState:
    flat, _ = ravel_params(params)
    return LMState(
        torch.tensor(lam0, dtype=flat.dtype, device=flat.device),
        torch.zeros_like(flat),
        torch.zeros((), dtype=torch.int32, device=flat.device),
    )


def _half_sq(r: Tensor) -> Tensor:
    return 0.5 * torch.sum(torch.square(r))


def lm_step(
    residual: Callable[[Any], Tensor],
    params: Any,
    state: LMState,
    *,
    heldout: Callable[[Any], Tensor] | None = None,
    config: LMConfig | None = None,
) -> tuple[Any, LMState, dict[str, Tensor]]:
    """One Levenberg-Marquardt step; accepted when the held-out mean square decreases (the training batch if none).

    Returns the new parameter pytree (new tensors; the inputs are not modified), the new state and an info dict.
    """
    config = LMConfig() if config is None else config
    flat, unravel = ravel_params(params)
    flat = flat.detach()

    def f(th: Tensor) -> Tensor:
        return residual(unravel(th))

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
        check_every=config.check_every,
        chunk=config.chunk,
    )
    with torch.no_grad():
        cand = flat - sol.delta
        n_train = r.shape[0]
        pred = (_half_sq(r) - _half_sq(r - sol.jdelta)) / n_train
        if heldout is None:
            before, after, rows = _half_sq(r), _half_sq(f(cand)), n_train
        else:
            h0, h1 = heldout(unravel(flat)), heldout(unravel(cand))
            before, after, rows = _half_sq(h0), _half_sq(h1), h0.shape[0]
        actual = (before - after) / rows
        accepted = torch.isfinite(after) & (after < before)
        rho = actual / torch.clamp(pred, min=torch.finfo(flat.dtype).tiny)
        one = torch.ones_like(rho)
        scale = torch.where(
            ~accepted | (rho < config.poor),
            one * config.up,
            torch.where(rho > config.good, one / config.down, one),
        )
        lam = torch.clamp(state.lam * scale, config.lam_min, config.lam_max)
        new_flat = torch.where(accepted, cand, flat)
        prev = torch.where(accepted, sol.delta, torch.zeros_like(sol.delta))
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
    loss: Callable[[Tensor], Tensor],
    theta: Tensor,
    delta: Tensor,
    *,
    n_scales: int = 21,
    vectorize: bool = True,
) -> tuple[Tensor, Tensor, int]:
    """Best of ``theta - 2**-k * delta`` (``k = 0..n_scales-1``) or no move, by ``loss``; deterministic.

    The candidates run through one ``torch.func.vmap`` (``vectorize=True``; ``loss`` must then be vmappable) or a loop.
    Returns ``(new_theta, k, evals)`` with ``k = n_scales`` for no move.
    """
    scales = 2.0 ** -torch.arange(n_scales, dtype=theta.dtype, device=theta.device)
    with torch.no_grad():
        if vectorize:
            fs = torch.func.vmap(lambda s: loss(theta - s * delta))(scales)
        else:
            fs = torch.stack([loss(theta - s * delta) for s in scales])
        f0 = loss(theta)
        j = torch.argmin(fs)
        ok = fs[j] < f0
        s_best = torch.where(ok, scales[j], torch.zeros_like(scales[0]))
        k = torch.where(ok, j, torch.full_like(j, n_scales)).to(torch.int32)
    return theta - s_best * delta, k, n_scales + 1


def engd_step(
    residual: Callable[[Any], Tensor],
    params: Any,
    lam: float | Tensor,
    *,
    n_scales: int = 21,
    vectorize: bool = True,
    solver: str = "auto",
    dense_rows: int | None = None,
    cg_tol: float = 1e-8,
    cg_maxiter: int = 200,
    check_every: int = 10,
) -> tuple[Any, dict[str, Any]]:
    """Energy natural gradient descent: damped Gauss-Newton direction with a fixed ``lam`` and the grid line search."""
    flat, unravel = ravel_params(params)
    flat = flat.detach()

    def f(th: Tensor) -> Tensor:
        return residual(unravel(th))

    r, sol = gauss_newton_solve(
        f,
        flat,
        lam,
        solver=solver,
        dense_rows=dense_rows,
        cg_tol=cg_tol,
        cg_maxiter=cg_maxiter,
        check_every=check_every,
    )

    def objective(th: Tensor) -> Tensor:
        return _half_sq(f(th))

    new_flat, k, evals = grid_line_search(
        objective, flat, sol.delta, n_scales=n_scales, vectorize=vectorize
    )
    return unravel(new_flat), {
        "loss": _half_sq(r).detach(),
        "k": k,
        "evals": evals,
        "method": sol.method,
        "cg_iterations": sol.iterations,
    }


# --------------------------------------------------------------------------------------------------------- KFAC
def lowrank_damped_solve(u: Tensor, c: Tensor, damping: Tensor | float, rhs: Tensor) -> Tensor:
    """Solve ``(damping * I + U diag(c) U^T) Z = RHS`` through the ``m x m`` Woodbury system (``U`` is ``(n, m)``)."""
    damping = torch.as_tensor(damping, dtype=u.dtype, device=u.device)
    inner = torch.diag(damping / c) + u.T @ u
    chol = torch.linalg.cholesky(inner)
    return (rhs - u @ torch.cholesky_solve(u.T @ rhs, chol)) / damping


def kron_damped_direction(
    u: Tensor,
    c: Tensor,
    gram_out: Tensor,
    n_samples: float | Tensor,
    lam: float | Tensor,
    grad: Tensor,
) -> Tensor:
    """KFAC-expand direction ``N * A^{-1} grad G^{-1}`` with ``A = U diag(c) U^T`` (see the JAX twin for the rule)."""
    n_in, dtype = u.shape[0], u.dtype
    lam_t = torch.as_tensor(lam, dtype=dtype, device=u.device)
    root = torch.sqrt(lam_t * n_samples)
    tr_a = torch.sum(c * torch.sum(torch.square(u), dim=0)) / n_in
    tr_g = torch.trace(gram_out) / gram_out.shape[0]
    pi = torch.sqrt(torch.clamp(tr_a, min=1e-300) / torch.clamp(tr_g, min=1e-300))
    gd = gram_out + (root / pi) * torch.eye(gram_out.shape[0], dtype=dtype, device=u.device)
    left = torch.cholesky_solve(grad, torch.linalg.cholesky(gd))
    right = lowrank_damped_solve(u, c, pi * root, left.T).T
    return right * n_samples


# ------------------------------------------------------------------------------------------------------ L-BFGS
def lbfgs_with_counts(
    loss: Callable[[Tensor], Tensor],
    theta: Tensor,
    *,
    steps: int,
    history_size: int = 10,
    max_iter: int = 20,
) -> tuple[Tensor, dict[str, Any]]:
    """``torch.optim.LBFGS`` with a strong-Wolfe line search; records the function evaluations of every ``step`` call.

    ``loss(theta)`` takes the flat vector.  Returns ``(theta, history)`` with ``loss`` (value at the end of each step),
    ``evaluations`` (closure calls per step: the line-search cost) and their ``total``.
    """
    x = theta.detach().clone().requires_grad_(True)
    opt = torch.optim.LBFGS(
        [x],
        lr=1.0,
        max_iter=max_iter,
        history_size=history_size,
        line_search_fn="strong_wolfe",
        tolerance_grad=1e-14,
        tolerance_change=1e-16,
    )
    counter = {"n": 0}

    def closure() -> Tensor:
        counter["n"] += 1
        opt.zero_grad()
        value = loss(x)
        value.backward()  # type: ignore[no-untyped-call]
        return value

    values: list[float] = []
    evals: list[int] = []
    for _ in range(steps):
        counter["n"] = 0
        value = opt.step(closure)  # type: ignore[no-untyped-call]
        evals.append(counter["n"])
        values.append(float(value.detach()))
    return x.detach(), {"loss": values, "evaluations": evals, "total": int(sum(evals))}
