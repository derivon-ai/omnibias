# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Second-order optimisation for omnibias PINNs (JAX): Gauss-Newton + adaptive weights.

First-order optimisers (Adam/SGD) stall on PINN losses because the differential
operator squares the condition number of the problem; they typically plateau near
``1e-3``. omnibias makes a much stronger optimiser practical: the residual map
``theta |-> r(theta)`` is computed from the *exact* closed-form jets
(:meth:`omnibias.jax.architectures.JetMLP.value_grad_hessian`, ``partials``, ...), so
its parameter-Jacobian ``J = d r / d theta`` is a single clean outer autodiff -- no
nested-autodiff blow-up, no finite-difference noise in ``r`` or ``J``.

This module provides

* :func:`gauss_newton_direction` -- the (Levenberg-Marquardt damped) Gauss-Newton /
  natural-gradient direction solving ``(J^T J + mu I) delta = -J^T r``, automatically
  switching to the equivalent *dual* (kernel / NTK) form ``delta = -J^T (J J^T + mu
  I)^{-1} r`` when there are more parameters than residuals (the push-through identity
  ``(J^T J + mu I)^{-1} J^T = J^T (J J^T + mu I)^{-1}`` makes them equal, and the dual
  system is far better conditioned in the over-parameterised regime).
* :func:`lstsq_gauss_newton_direction` -- QR / SVD least-squares LM on the *augmented*
  ``J`` (never squares ``kappa(J)``); prefer this on stiff PINN Jacobians.
* :func:`cgls` / :func:`gauss_newton_direction_cgls` -- matrix-free CGLS twin of the
  torch solvers (accuracy scales with ``kappa(J)``, not ``kappa(J)^2``).
* :func:`martens_grosse_combine` / :func:`martens_grosse_gauss_newton_minimize` --
  damped GN plus Martens–Grosse closed-form LR / momentum via **exact**
  :func:`jax.jvp` (no finite-difference probes). Default solver is ``"qr"``.
* :func:`gauss_newton_step` / :func:`gauss_newton_minimize` -- an adaptive-damping LM
  loop driven by a ``residual_fn``.
* :func:`grad_norm_weights` -- self-adaptive loss weights that equalise the per-term
  gradient norms (Wang-Teng-Perdikaris 2021 gradient-pathology balancing).

For the standard L2 collocation PINN functional, the Gauss-Newton matrix ``J^T J``
*is* the empirical Sobolev Gram matrix, so :func:`gauss_newton_step` is exactly the
**empirical energy natural gradient** (Mueller-Zeinhofer 2023) -- the method that takes
PINNs from ``1e-3`` to near machine precision.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal, cast

import jax
import jax.numpy as jnp
from jax import Array
from jax.flatten_util import ravel_pytree

ResidualFn = Callable[[Array], Array]
MatVec = Callable[[Array], Array]
GNSolver = Literal["dense", "qr", "cgls"]


def gauss_newton_direction(jac: Array, res: Array, damping: float) -> Array:
    r"""Levenberg-Marquardt Gauss-Newton step ``delta`` for ``min 1/2 ||r||^2``.

    Solves ``(J^T J + mu I) delta = -J^T r`` (primal, ``P x P``) when ``P <= N`` and the
    equivalent dual form ``delta = -J^T (J J^T + mu I)^{-1} r`` (``N x N``) otherwise.

    Parameters
    ----------
    jac:
        Residual Jacobian ``J = d r / d theta`` of shape ``(N, P)``.
    res:
        Residual vector ``r`` of shape ``(N,)``.
    damping:
        Levenberg-Marquardt damping ``mu >= 0``.
    """
    if jac.ndim != 2:
        raise ValueError(f"jac must be 2-D (N, P), got shape {jac.shape}")
    n, p = jac.shape
    if res.shape != (n,):
        raise ValueError(f"res must have shape ({n},) matching jac rows, got {res.shape}")
    mu = jnp.asarray(damping, dtype=jac.dtype)
    if p <= n:
        a = jac.T @ jac + mu * jnp.eye(p, dtype=jac.dtype)
        delta: Array = jnp.linalg.solve(a, -(jac.T @ res))
    else:
        g = jac @ jac.T + mu * jnp.eye(n, dtype=jac.dtype)
        delta = -(jac.T @ jnp.linalg.solve(g, res))
    return delta


def lstsq_gauss_newton_direction(jac: Array, res: Array, damping: float) -> Array:
    r"""LM Gauss-Newton step via QR least-squares -- never squares the conditioning.

    Solves the same damped problem as :func:`gauss_newton_direction` as
    ``min_delta || [J; sqrt(mu) I] delta - [-r; 0] ||^2`` via :func:`jnp.linalg.lstsq`.
    Prefer this over forming ``J^T J`` whenever ``J`` is stiff (essentially all PINNs).
    Bit-identical twin of :func:`omnibias.torch.optim.lstsq_gauss_newton_direction`.
    """
    if jac.ndim != 2:
        raise ValueError(f"jac must be 2-D (N, P), got shape {jac.shape}")
    n, p = int(jac.shape[0]), int(jac.shape[1])
    if res.shape != (n,):
        raise ValueError(f"res must have shape ({n},) matching jac rows, got {res.shape}")
    mu = float(damping)
    if mu < 0.0:
        raise ValueError(f"damping must be >= 0, got {mu}")
    if mu == 0.0:
        a_aug = jac
        b_aug = -res
    else:
        eye = jnp.sqrt(jnp.asarray(mu, dtype=jac.dtype)) * jnp.eye(p, dtype=jac.dtype)
        a_aug = jnp.concatenate([jac, eye], axis=0)
        b_aug = jnp.concatenate([-res, jnp.zeros((p,), dtype=res.dtype)])
    sol, _, _, _ = jnp.linalg.lstsq(a_aug, b_aug, rcond=None)
    return sol.reshape(-1)


def cgls(
    a_matvec: MatVec,
    at_matvec: MatVec,
    b: Array,
    *,
    damp: float = 0.0,
    max_iter: int = 100,
    tol: float = 1e-8,
) -> Array:
    r"""Solve ``min_x ||A x - b||^2 + damp^2 ||x||^2`` by CGLS, matrix-free on ``A``.

    Twin of :func:`omnibias.torch.optim.cgls`. Accuracy scales with ``kappa(A)``, not
    ``kappa(A)^2``.
    """
    lam = float(damp) ** 2
    r = b
    s = at_matvec(r)
    x = jnp.zeros_like(s)
    p = s
    gamma = jnp.vdot(s, s)
    norm_s0 = float(jnp.sqrt(gamma))
    if norm_s0 == 0.0:
        return x
    thresh = tol * norm_s0
    # Eager Python loop (data-dependent stop) -- matches torch cgls control flow.
    for _ in range(max_iter):
        if float(jnp.sqrt(gamma)) <= thresh:
            break
        q = a_matvec(p)
        denom = float(jnp.vdot(q, q) + lam * jnp.vdot(p, p))
        if denom <= 0.0:
            break
        alpha = gamma / denom
        x = x + alpha * p
        r = r - alpha * q
        s = at_matvec(r) - lam * x
        gamma_new = jnp.vdot(s, s)
        if float(jnp.sqrt(gamma_new)) <= thresh:
            break
        p = s + (gamma_new / gamma) * p
        gamma = gamma_new
    return x


def _linearize_gn(residual_fn: ResidualFn, params: Array) -> tuple[Array, MatVec, MatVec]:
    """Return ``(res, jt, jvec)`` via a single VJP linearisation + JVP."""
    res, vjp_fn = jax.vjp(residual_fn, params)

    def jt(u: Array) -> Array:
        return cast(Array, vjp_fn(u)[0])

    def jvec(v: Array) -> Array:
        return cast(Array, jax.jvp(residual_fn, (params,), (v,))[1])

    return res, jt, jvec


def gauss_newton_direction_cgls(
    residual_fn: ResidualFn,
    params: Array,
    damping: float,
    *,
    cgls_max_iter: int = 100,
    cgls_tol: float = 1e-8,
) -> Array:
    r"""Matrix-free LM Gauss-Newton direction via CGLS.

    Twin of :func:`omnibias.torch.optim.gauss_newton_direction_cgls`.
    """
    res, jt, jvec = _linearize_gn(residual_fn, params)
    return cgls(
        jvec,
        jt,
        -res,
        damp=float(jnp.sqrt(jnp.asarray(damping, dtype=res.dtype))),
        max_iter=cgls_max_iter,
        tol=cgls_tol,
    )


def martens_grosse_combine(
    residual_fn: ResidualFn,
    params: Array,
    delta_gn: Array,
    prev_delta: Array | None,
) -> tuple[Array, Array]:
    r"""Martens–Grosse closed-form LR / momentum via **exact** JVPs.

    Minimises the local quadratic model
    ``|| r + J (alpha d + mu prev) ||^2`` for ``(alpha, mu)`` using
    ``Jd = jvp(r, d)`` and ``Jp = jvp(r, prev)`` -- no finite-difference probes.
    Returns ``(step, momentum_state)`` where ``momentum_state`` is the accepted
    combined step (fed back as ``prev_delta`` on the next iteration).

    Twin of :func:`omnibias.torch.optim.martens_grosse_combine`.
    """
    _, j_d = jax.jvp(residual_fn, (params,), (delta_gn,))
    r0 = residual_fn(params)
    if prev_delta is None:
        num = -jnp.vdot(j_d, r0)
        den = jnp.vdot(j_d, j_d) + 1e-30
        alpha = num / den
        step = alpha * delta_gn
        return step, step

    _, j_p = jax.jvp(residual_fn, (params,), (prev_delta,))
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


def _half_sum_sq(res: Array) -> float:
    return 0.5 * float(jnp.sum(res * res))


@dataclass(frozen=True)
class MartensGrosseGNConfig:
    """Hyper-parameters for :func:`martens_grosse_gauss_newton_minimize`."""

    steps: int = 50
    damping: float = 1e-3
    damping_decrease: float = 0.7
    damping_increase: float = 2.0
    min_damping: float = 1e-8
    max_damping: float = 1e3
    accept_tol: float = 0.0
    use_martens_grosse: bool = True
    solver: GNSolver = "qr"
    cgls_max_iter: int = 100
    cgls_tol: float = 1e-8


def martens_grosse_gauss_newton_minimize(
    residual_fn: ResidualFn,
    params0: Array,
    *,
    config: MartensGrosseGNConfig | None = None,
) -> tuple[Array, Array]:
    r"""Minimise ``0.5 ||r(params)||^2`` by damped GN + optional Martens–Grosse.

    Default ``solver="qr"`` (non-squaring) and ``use_martens_grosse=True`` (exact JVP
    2x2 LR/momentum). ``solver="dense"`` forms ``J^T J`` (legacy); ``"cgls"`` is
    matrix-free.
    """
    cfg = MartensGrosseGNConfig() if config is None else config
    if cfg.solver not in ("dense", "qr", "cgls"):
        raise ValueError(f"solver must be 'dense', 'qr', or 'cgls', got {cfg.solver!r}")
    params = params0
    gamma = float(cfg.damping)
    losses: list[float] = []
    prev_delta: Array | None = None

    for _ in range(int(cfg.steps)):
        r0 = residual_fn(params)
        loss0 = _half_sum_sq(r0)
        losses.append(loss0)

        if cfg.solver == "cgls":
            delta_gn = gauss_newton_direction_cgls(
                residual_fn,
                params,
                gamma,
                cgls_max_iter=cfg.cgls_max_iter,
                cgls_tol=cfg.cgls_tol,
            )
        else:
            jac = jax.jacfwd(residual_fn)(params)
            if cfg.solver == "qr":
                delta_gn = lstsq_gauss_newton_direction(jac, r0, gamma)
            else:
                delta_gn = gauss_newton_direction(jac, r0, gamma)

        if cfg.use_martens_grosse:
            step, prev_delta = martens_grosse_combine(
                residual_fn, params, delta_gn, prev_delta
            )
        else:
            step = delta_gn
            prev_delta = delta_gn

        candidate = params + step
        loss1 = _half_sum_sq(residual_fn(candidate))
        if loss1 <= loss0 * (1.0 + cfg.accept_tol) + 1e-15:
            params = candidate
            gamma = max(cfg.min_damping, gamma * cfg.damping_decrease)
        else:
            gamma = min(cfg.max_damping, gamma * cfg.damping_increase)
            prev_delta = None

    losses.append(_half_sum_sq(residual_fn(params)))
    return params, jnp.asarray(losses, dtype=jnp.float64)


def natural_gradient_direction(metric: Array, grad: Array, *, damping: float = 1e-3) -> Array:
    r"""Natural-gradient direction ``delta = (M + damping I)^{-1} grad``.

    The metric-preconditioned counterpart of the raw gradient: for a Riemannian metric ``M``
    on parameter space the *natural* gradient ``M^{-1} grad`` makes the descent step invariant
    to smooth reparametrisations. ``metric`` is a symmetric positive-(semi)definite ``(P, P)``
    matrix and ``grad`` a ``(P,)`` vector; ``damping >= 0`` (Tikhonov) keeps the solve
    well-posed when ``M`` is singular or ill-conditioned.

    Bit-identical twin of :func:`omnibias.torch.optim.natural_gradient_direction` (dense path)
    and identical in form to :func:`omnibias.curvature.natural_gradient.damped_solve`. Pair
    it with the closed-form :func:`gauss_newton_fisher` (Fisher scoring / Newton on a
    least-squares residual) or a geometry pullback metric
    (:func:`omnibias.geometry.jax.ops.pullback_metric`).
    """
    if metric.ndim != 2 or metric.shape[0] != metric.shape[1]:
        raise ValueError(f"metric must be a square (P, P) matrix, got {tuple(metric.shape)}")
    if grad.ndim != 1 or grad.shape[0] != metric.shape[0]:
        raise ValueError(
            f"grad must be (P,) with P = {metric.shape[0]}, got {tuple(grad.shape)}"
        )
    if damping < 0.0:
        raise ValueError(f"damping must be >= 0, got {damping}")
    p = metric.shape[0]
    damped = metric + jnp.asarray(damping, dtype=metric.dtype) * jnp.eye(p, dtype=metric.dtype)
    delta: Array = jnp.linalg.solve(damped, grad)
    return delta


def gauss_newton_fisher(residual_fn: ResidualFn, params: Array) -> tuple[Array, Array]:
    r"""Dense Gauss-Newton Fisher ``F = (1/N) J^T J`` and gradient ``g = (1/N) J^T r``.

    The closed-form Fisher metric of the least-squares objective ``0.5 mean(r^2)``: with
    ``J = d r / d theta`` (one :func:`jax.jacrev`), ``F`` is the Gauss-Newton (and, at a zero
    residual, exact) Hessian and ``g`` its gradient. Feed the pair to
    :func:`natural_gradient_direction` for a Fisher-scoring / natural-gradient step (which on a
    residual linear in ``theta`` equals Newton and recovers the least-squares minimiser in one
    step). Bit-identical twin of :func:`omnibias.torch.optim.gauss_newton_fisher`.

    Returns ``(F, g)`` of shape ``((P, P), (P,))``.
    """
    res = residual_fn(params)
    if res.ndim != 1:
        raise ValueError(f"residual_fn must return a 1-D vector, got shape {tuple(res.shape)}")
    n_res = res.shape[0]
    jac = jax.jacrev(residual_fn)(params)  # (N, P)
    fisher = (jac.T @ jac) / n_res
    g = (jac.T @ res) / n_res
    return fisher, g


def natural_gradient_step(
    params: Array,
    grad: Array,
    metric: Array,
    *,
    learning_rate: float = 1.0,
    damping: float = 1e-3,
) -> Array:
    r"""Preconditioned parameter update ``theta - lr (M + damping I)^{-1} grad``.

    The metric-aware (natural-gradient / Riemannian) counterpart of vanilla gradient descent:
    descending along ``M^{-1} grad`` instead of ``grad`` makes the step invariant to smooth
    reparametrisations. ``params`` and ``grad`` are flat ``(P,)`` vectors and ``metric`` the
    ``(P, P)`` Riemannian metric (Fisher via :func:`gauss_newton_fisher`, or a geometry
    pullback). Mirrors :func:`omnibias.curvature.natural_gradient.natural_gradient_step` and is
    the functional twin of the torch :class:`omnibias.torch.optim.NaturalGradient` step.
    """
    delta = natural_gradient_direction(metric, grad, damping=damping)
    out: Array = params - learning_rate * delta
    return out


def _half_mean_sq(res: Array) -> float:
    return 0.5 * float(jnp.mean(res**2))


@dataclass(frozen=True)
class GaussNewtonState:
    """Immutable LM optimiser state (functional update via :func:`gauss_newton_step`)."""

    params: Array
    damping: float
    loss: float
    accepted: bool
    n_iter: int


def init_gauss_newton_state(params: Array, *, damping: float = 1e-3) -> GaussNewtonState:
    """Seed a :class:`GaussNewtonState` from a flat parameter vector."""
    if damping <= 0.0:
        raise ValueError(f"damping must be > 0, got {damping}")
    return GaussNewtonState(params=params, damping=float(damping), loss=float("inf"), accepted=False, n_iter=0)


def gauss_newton_step(
    residual_fn: ResidualFn,
    state: GaussNewtonState,
    *,
    damping_increase: float = 3.0,
    damping_decrease: float = 0.5,
    min_damping: float = 1e-12,
    max_damping: float = 1e12,
    max_line_search: int = 8,
) -> GaussNewtonState:
    r"""One adaptive-damping Gauss-Newton (Levenberg-Marquardt) iteration.

    Computes ``r`` and ``J = d r / d theta`` (one :func:`jax.jacrev`), proposes the
    damped GN direction, and accepts it iff it decreases ``1/2 mean(r^2)``; on success
    the damping is multiplied by ``damping_decrease``, on failure by ``damping_increase``
    (retried up to ``max_line_search`` times). The step is eager (data-dependent control
    flow); pass a ``jax.jit``-compiled ``residual_fn`` for speed.
    """
    params = state.params
    res = residual_fn(params)
    jac = jax.jacrev(residual_fn)(params)
    loss0 = _half_mean_sq(res)
    mu = state.damping
    for _ in range(max_line_search):
        delta = gauss_newton_direction(jac, res, mu)
        new_params = params + delta
        res_new = residual_fn(new_params)
        loss1 = _half_mean_sq(res_new)
        if jnp.isfinite(jnp.asarray(loss1)) and loss1 < loss0:
            new_mu = max(mu * damping_decrease, min_damping)
            return GaussNewtonState(new_params, new_mu, loss1, True, state.n_iter + 1)
        mu = min(mu * damping_increase, max_damping)
    return GaussNewtonState(params, mu, loss0, False, state.n_iter + 1)


def gauss_newton_minimize(
    residual_fn: ResidualFn,
    params: Array,
    *,
    steps: int,
    damping: float = 1e-3,
    **step_kwargs: Any,
) -> tuple[GaussNewtonState, list[float]]:
    """Run :func:`gauss_newton_step` ``steps`` times; return ``(state, loss_history)``."""
    state = init_gauss_newton_state(params, damping=damping)
    history: list[float] = []
    for _ in range(steps):
        state = gauss_newton_step(residual_fn, state, **step_kwargs)
        history.append(state.loss)
    return state, history


def make_residual_fn(
    build_residual: Callable[[Any], Array], params_pytree: Any
) -> tuple[Array, ResidualFn]:
    """Bridge a pytree model to a flat ``residual_fn``.

    Returns ``(flat0, residual_fn)`` where ``flat0`` is the ravelled parameter vector and
    ``residual_fn(vec) = build_residual(unravel(vec))``. ``build_residual`` receives a
    reconstructed pytree (e.g. a :class:`omnibias.jax.architectures.JetMLP`) and returns
    the stacked residual vector.
    """
    flat0, unravel = ravel_pytree(params_pytree)

    def residual_fn(vec: Array) -> Array:
        return build_residual(unravel(vec))

    return flat0, residual_fn


def grad_norm_weights(
    loss_fns: tuple[Callable[[Any], Array], ...],
    params: Any,
    prev_weights: Array,
    *,
    alpha: float = 0.9,
    ref_index: int = 0,
    eps: float = 1e-12,
) -> Array:
    r"""Self-adaptive loss weights equalising per-term gradient norms.

    For each term ``L_k`` computes ``g_k = ||d L_k / d theta||`` and forms the target
    weight ``lhat_k = ||g_ref|| / (||g_k|| + eps)`` (Wang-Teng-Perdikaris 2021 gradient
    balancing), then returns the EMA ``alpha * prev + (1 - alpha) * lhat``. Applying the
    returned weights makes the weighted gradient norms ``lambda_k ||g_k||`` all equal to
    the reference term's, curing the gradient-pathology stiffness of multi-term PINN
    losses.
    """
    if not 0.0 <= alpha <= 1.0:
        raise ValueError(f"alpha must be in [0, 1], got {alpha}")
    if not 0 <= ref_index < len(loss_fns):
        raise ValueError(f"ref_index {ref_index} out of range for {len(loss_fns)} terms")
    norms = []
    for fn in loss_fns:
        grad = jax.grad(fn)(params)
        flat, _ = ravel_pytree(grad)
        norms.append(jnp.linalg.norm(flat))
    norm_vec = jnp.stack(norms)
    target = norm_vec[ref_index] / (norm_vec + eps)
    out: Array = alpha * prev_weights + (1.0 - alpha) * target
    return out


__all__ = [
    "GNSolver",
    "GaussNewtonState",
    "MartensGrosseGNConfig",
    "MatVec",
    "ResidualFn",
    "cgls",
    "gauss_newton_direction",
    "gauss_newton_direction_cgls",
    "gauss_newton_fisher",
    "gauss_newton_minimize",
    "gauss_newton_step",
    "grad_norm_weights",
    "init_gauss_newton_state",
    "lstsq_gauss_newton_direction",
    "make_residual_fn",
    "martens_grosse_combine",
    "martens_grosse_gauss_newton_minimize",
    "natural_gradient_direction",
    "natural_gradient_step",
]
