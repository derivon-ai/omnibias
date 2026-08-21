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
* :func:`cubic_regularized_gauss_newton_minimize` -- ARC on the PSD Gauss-Newton
  model (Lanczos cubic subproblem). Twin of
  :class:`omnibias.torch.optim.CubicRegularizedGaussNewton`.
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

import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal, cast

import jax
import jax.numpy as jnp
from jax import Array
from jax.flatten_util import ravel_pytree

from omnibias.jax.line_search import (
    JetLineSearchConfig,
    LineSearchResult,
    jet_line_search,
    jet_line_search_on_ray,
)
from omnibias.jax.optim_composed import (
    ComposedCurvatureConfig,
    ComposedCurvatureReport,
    composed_block_hessian,
    composed_curvature_step,
)

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


def lanczos_tridiag(
    matvec: MatVec, b: Array, k: int, *, tol: float = 1e-10
) -> tuple[Array, Array]:
    r"""``k``-step Lanczos on a symmetric operator with full reorthogonalisation.

    Twin of :func:`omnibias.torch.optim.lanczos_tridiag`. Returns ``(Q, T)`` with
    ``Q`` (``n x m``) orthonormal (first column ``b/||b||``) and ``T = Q^T A Q``
    symmetric tridiagonal (``m <= min(k, n)``).
    """
    n = int(b.shape[0])
    m_max = min(int(k), n)
    beta0 = float(jnp.linalg.norm(b))
    if beta0 == 0.0:
        q = jnp.zeros_like(b).at[0].set(1.0)
    else:
        q = b / beta0
    qs: list[Array] = [q]
    alphas: list[Array] = []
    betas: list[Array] = []
    q_prev = jnp.zeros_like(b)
    beta_prev = jnp.zeros((), dtype=b.dtype)
    for _j in range(m_max):
        w = matvec(qs[-1])
        alpha = jnp.vdot(w, qs[-1])
        alphas.append(alpha)
        w = w - alpha * qs[-1] - beta_prev * q_prev
        for qi in qs:
            w = w - jnp.vdot(w, qi) * qi
        beta = jnp.linalg.norm(w)
        if float(beta) <= tol:
            break
        betas.append(beta)
        q_prev = qs[-1]
        beta_prev = beta
        qs.append(w / beta)
    m = len(alphas)
    q_basis = jnp.stack(qs[:m], axis=1)
    tri = jnp.diag(jnp.stack(alphas))
    if m > 1:
        off = jnp.stack(betas[: m - 1])
        tri = tri + jnp.diag(off, 1) + jnp.diag(off, -1)
    return q_basis, tri


def _solve_cubic_subproblem(tri: Array, c: Array, sigma: float, *, iters: int = 100) -> Array:
    r"""Global minimiser of ``c^T y + 0.5 y^T T y + (sigma/3) ||y||^3``.

    Twin of :func:`omnibias.torch.optim._solve_cubic_subproblem`.
    """
    theta, vecs = jnp.linalg.eigh(tri)
    chat = vecs.T @ c
    lam_lo = max(0.0, -float(theta[0]))
    eps = 1e-12 + 1e-9 * max(1.0, abs(float(theta[-1])))

    def z_of(lam: float) -> Array:
        return -chat / (theta + lam)

    def phi(lam: float) -> float:
        return float(jnp.linalg.norm(z_of(lam))) - lam / sigma

    lo = lam_lo + eps
    if phi(lo) <= 0.0:
        return vecs @ z_of(lo)
    hi = max(2.0 * lo, 1.0)
    for _ in range(200):
        if phi(hi) < 0.0:
            break
        hi *= 2.0
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        if phi(mid) > 0.0:
            lo = mid
        else:
            hi = mid
    return vecs @ z_of(0.5 * (lo + hi))


def _cubic_arc_step(
    params: Array,
    g: Array,
    hess_matvec: MatVec,
    f0: float,
    eval_f: Callable[[Array], float],
    sigma: float,
    *,
    krylov_dim: int,
    max_line_search: int,
    eta_accept: float,
    eta_success: float,
    sigma_increase: float,
    sigma_decrease: float,
    min_sigma: float,
    max_sigma: float,
) -> tuple[Array, bool, float, float, float]:
    r"""Shared ARC acceptance loop (twin of the torch cubic-GN core)."""
    gnorm = float(jnp.linalg.norm(g))
    if gnorm == 0.0:
        return params, False, sigma, 0.0, f0
    q_basis, tri = lanczos_tridiag(hess_matvec, g, krylov_dim)
    c = jnp.zeros((tri.shape[0],), dtype=params.dtype).at[0].set(gnorm)
    rho = 0.0
    for _ in range(max_line_search):
        y = _solve_cubic_subproblem(tri, c, sigma)
        s = q_basis @ y
        model_dec = -(
            float(jnp.vdot(c, y))
            + 0.5 * float(jnp.vdot(y, tri @ y))
            + (sigma / 3.0) * float(jnp.linalg.norm(y)) ** 3
        )
        if model_dec <= 0.0:
            sigma = min(sigma * sigma_increase, max_sigma)
            continue
        f1 = eval_f(params + s)
        rho = (f0 - f1) / model_dec if math.isfinite(f1) else float("-inf")
        if math.isfinite(f1) and rho >= eta_accept:
            new_sigma = max(sigma / sigma_decrease, min_sigma) if rho >= eta_success else sigma
            return params + s, True, new_sigma, rho, f1
        sigma = min(sigma * sigma_increase, max_sigma)
    return params, False, sigma, rho, f0


HomotopyResidualFn = Callable[[Array, float], Array]


@dataclass(frozen=True)
class HomotopyGNConfig:
    """Stage-wise GN on ``r(theta, t)`` for a coupling homotopy ``t`` in ``[0, 1]``.

    Each stage freezes ``t`` and runs Martens–Grosse or cubic GN. Designed for
    residuals that split as ``L + t Quad`` (nonlocal quadratic coupling).
    Does not weaken any downstream residual gate.
    """

    stages: tuple[float, ...] = (0.0, 0.25, 0.5, 0.75, 1.0)
    steps_per_stage: int = 20
    method: Literal["martens_grosse", "cubic"] = "cubic"
    damping: float = 1e-3
    cubic_sigma: float = 1.0
    solver: GNSolver = "qr"
    krylov_dim: int | None = None

    def __post_init__(self) -> None:
        if not self.stages:
            raise ValueError("stages must be non-empty")
        if any(not 0.0 <= float(t) <= 1.0 for t in self.stages):
            raise ValueError(f"every stage t must lie in [0, 1], got {self.stages}")
        if self.steps_per_stage < 1:
            raise ValueError(f"steps_per_stage must be >= 1, got {self.steps_per_stage}")
        if self.method not in ("martens_grosse", "cubic"):
            raise ValueError(f"method must be martens_grosse or cubic, got {self.method!r}")


def peak_weighted_residual(res: Array, power: float) -> Array:
    """Emphasize residual peaks without changing the zero set.

    ``r |r|^p`` is still zero iff ``r`` is. Used as a differentiable L^∞ proxy
    so Gauss–Newton cannot hide a large max-norm behind a small L2.
    """
    if power == 0.0:
        return res
    return res * jnp.power(1.0 + jnp.abs(res), float(power))


def linearized_linf_direction(
    jac: Array,
    res: Array,
    *,
    box: float | None = None,
    irls_iters: int = 16,
) -> Array:
    """Minimax (Chebyshev) step on the linearized residual ``res + jac @ c``.

    One column is exact: bisection on the epigraph radius ``t`` with interval
    intersection. Several columns use Lawson IRLS (reweighted least squares),
    which is the L^∞ successor of :func:`gauss_newton_direction` when L2
    Newton raises ``max|r|``. Optional ``box`` clips the step.
    """
    if jac.ndim != 2:
        raise ValueError(f"jac must be 2-D (N, P), got shape {jac.shape}")
    n, p = jac.shape
    if res.shape != (n,):
        raise ValueError(f"res must have shape ({n},) matching jac rows, got {res.shape}")
    if p == 1:
        j = jac[:, 0]
        lo = jnp.array(0.0, dtype=res.dtype)
        hi = jnp.max(jnp.abs(res))
        c_star = jnp.array(0.0, dtype=res.dtype)

        def body(carry, _):
            lo_, hi_, c_ = carry
            t = 0.5 * (lo_ + hi_)
            # For each row, c in an interval so |r + j c| <= t.
            # Skip near-zero j: those rows only constrain t >= |r|.
            nz = jnp.abs(j) > 1e-15
            left = jnp.where(j >= 0.0, (-t - res) / (j + 1e-30), (t - res) / (j + 1e-30))
            right = jnp.where(j >= 0.0, (t - res) / (j + 1e-30), (-t - res) / (j + 1e-30))
            left = jnp.where(nz, left, -jnp.inf)
            right = jnp.where(nz, right, jnp.inf)
            c_lo = jnp.max(left)
            c_hi = jnp.min(right)
            feasible = (c_lo <= c_hi) & (jnp.max(jnp.abs(res) * (1.0 - nz.astype(res.dtype))) <= t)
            c_mid = 0.5 * (c_lo + c_hi)
            if box is not None:
                c_mid = jnp.clip(c_mid, -float(box), float(box))
                feasible = feasible & (c_lo <= float(box)) & (c_hi >= -float(box))
            lo_n = jnp.where(feasible, lo_, t)
            hi_n = jnp.where(feasible, t, hi_)
            c_n = jnp.where(feasible, c_mid, c_)
            return (lo_n, hi_n, c_n), None

        (_, _, c_star), _ = jax.lax.scan(body, (lo, hi, c_star), xs=None, length=48)
        step = c_star.reshape((1,))
    else:
        w = jnp.ones_like(res)
        c = jnp.zeros((p,), dtype=res.dtype)

        def irls(carry, _):
            w_, _c = carry
            sw = jnp.sqrt(w_)
            c_n, _, _, _ = jnp.linalg.lstsq(jac * sw[:, None], -res * sw, rcond=None)
            if box is not None:
                c_n = jnp.clip(c_n, -float(box), float(box))
            resid = res + jac @ c_n
            w_n = w_ * (jnp.abs(resid) + 1e-15)
            w_n = w_n / jnp.mean(w_n)
            return (w_n, c_n), None

        (_, c), _ = jax.lax.scan(irls, (w, c), xs=None, length=int(irls_iters))
        step = c
    return step


def champ_barrier_residual(
    res: Array,
    tau: float,
    *,
    barrier_weight: float = 20.0,
    peak_power: float = 4.0,
    peak_frac: float = 0.85,
) -> Array:
    """Active-set L^∞ residual that cannot raise already-small entries above ``tau``.

    Rows with ``|r| >= peak_frac * max|r|`` are peak-weighted so Gauss–Newton
    works the current spike. One-sided excess ``relu(|r| - tau)`` is a barrier
    against walking past a known champ. The peak block is zero wherever
    ``res`` is; the barrier block is zero whenever ``|r| <= tau``.
    """
    ar = jnp.abs(res)
    rmax = jnp.max(ar)
    peak_mask = ar >= (float(peak_frac) * rmax)
    peak = jnp.where(
        peak_mask,
        peak_weighted_residual(res, peak_power),
        jnp.zeros_like(res),
    )
    excess = jnp.maximum(ar - float(tau), 0.0) * jnp.sign(res)
    return jnp.concatenate([peak, float(barrier_weight) * excess])


def homotopy_gauss_newton_minimize(
    residual_fn: HomotopyResidualFn,
    params0: Array,
    *,
    config: HomotopyGNConfig | None = None,
) -> tuple[Array, dict[str, Any]]:
    """Minimise ``0.5 ||r(theta, t)||^2`` at increasing coupling ``t``.

    ``residual_fn(theta, t)`` must already include every constraint that
    belongs in the Jacobian (hard gauge, anti-ghost). Extra Newton on a
    residual that omits the gauge will collapse amplitude.
    """
    cfg = HomotopyGNConfig() if config is None else config
    params = params0
    history: list[dict[str, float]] = []
    for t in cfg.stages:

        def r_t(vec: Array, t_frozen: float = float(t)) -> Array:
            return residual_fn(vec, t_frozen)

        if cfg.method == "cubic":
            cubic_cfg = CubicRegularizedGNConfig(
                steps=int(cfg.steps_per_stage),
                sigma=float(cfg.cubic_sigma),
                krylov_dim=cfg.krylov_dim,
            )
            params, losses = cubic_regularized_gauss_newton_minimize(
                r_t, params, config=cubic_cfg
            )
        else:
            mg_cfg = MartensGrosseGNConfig(
                steps=int(cfg.steps_per_stage),
                damping=float(cfg.damping),
                solver=cfg.solver,
            )
            params, losses = martens_grosse_gauss_newton_minimize(
                r_t, params, config=mg_cfg
            )
        r_end = residual_fn(params, float(t))
        history.append(
            {
                "t": float(t),
                "loss0": float(losses[0]),
                "loss1": float(losses[-1]),
                "max_abs": float(jnp.max(jnp.abs(r_end))),
            }
        )
    return params, {"stages": history, "method": cfg.method}


@dataclass(frozen=True)
class CubicRegularizedGNConfig:
    """Hyper-parameters for :func:`cubic_regularized_gauss_newton_minimize`."""

    steps: int = 50
    sigma: float = 1.0
    eta_accept: float = 0.1
    eta_success: float = 0.9
    sigma_increase: float = 2.0
    sigma_decrease: float = 2.0
    min_sigma: float = 1e-8
    max_sigma: float = 1e16
    # None = full parameter dimension. A small default (e.g. 20) silently
    # truncates ARC on anything wider than a toy residual.
    krylov_dim: int | None = None
    max_line_search: int = 12

    def __post_init__(self) -> None:
        if self.sigma <= 0.0:
            raise ValueError(f"sigma must be > 0, got {self.sigma}")
        if not 0.0 < self.eta_accept <= self.eta_success < 1.0:
            raise ValueError(
                f"need 0 < eta_accept <= eta_success < 1, got "
                f"{self.eta_accept}, {self.eta_success}"
            )
        if self.sigma_increase <= 1.0:
            raise ValueError(f"sigma_increase must be > 1, got {self.sigma_increase}")
        if self.sigma_decrease <= 1.0:
            raise ValueError(f"sigma_decrease must be > 1, got {self.sigma_decrease}")
        if self.krylov_dim is not None and self.krylov_dim < 1:
            raise ValueError(f"krylov_dim must be >= 1 or None, got {self.krylov_dim}")
        if self.max_line_search < 1:
            raise ValueError(f"max_line_search must be >= 1, got {self.max_line_search}")


def cubic_regularized_gauss_newton_minimize(
    residual_fn: ResidualFn,
    params0: Array,
    *,
    config: CubicRegularizedGNConfig | None = None,
) -> tuple[Array, Array]:
    r"""Minimise ``0.5 ||r(params)||^2`` by cubic-regularised Gauss-Newton (ARC).

    Matrix-free twin of :class:`omnibias.torch.optim.CubicRegularizedGaussNewton`:
    each step minimises ``g^T s + 0.5 s^T (J^T J) s + (sigma/3)||s||^3`` with
    ``g = J^T r`` in a Lanczos subspace, then accepts by the ARC ratio test.
    Loss history matches :func:`martens_grosse_gauss_newton_minimize` (value at
    the start of each step, then the final value).
    """
    cfg = CubicRegularizedGNConfig() if config is None else config
    params = params0
    sigma = float(cfg.sigma)
    n_params = int(params0.size)
    krylov = n_params if cfg.krylov_dim is None else min(int(cfg.krylov_dim), n_params)
    losses: list[float] = []

    for _ in range(int(cfg.steps)):
        res, jt, jvec = _linearize_gn(residual_fn, params)
        g = jt(res)
        f0 = _half_sum_sq(res)
        losses.append(f0)

        def matvec(v: Array, jt_fn: MatVec = jt, jvec_fn: MatVec = jvec) -> Array:
            return jt_fn(jvec_fn(v))

        def eval_f(trial: Array) -> float:
            return _half_sum_sq(residual_fn(trial))

        params, _accepted, sigma, _rho, _f1 = _cubic_arc_step(
            params,
            g,
            matvec,
            f0,
            eval_f,
            sigma,
            krylov_dim=krylov,
            max_line_search=int(cfg.max_line_search),
            eta_accept=float(cfg.eta_accept),
            eta_success=float(cfg.eta_success),
            sigma_increase=float(cfg.sigma_increase),
            sigma_decrease=float(cfg.sigma_decrease),
            min_sigma=float(cfg.min_sigma),
            max_sigma=float(cfg.max_sigma),
        )

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
    "ComposedCurvatureConfig",
    "ComposedCurvatureReport",
    "CubicRegularizedGNConfig",
    "GNSolver",
    "GaussNewtonState",
    "HomotopyGNConfig",
    "HomotopyResidualFn",
    "JetLineSearchConfig",
    "LineSearchResult",
    "MartensGrosseGNConfig",
    "MatVec",
    "ResidualFn",
    "cgls",
    "champ_barrier_residual",
    "composed_block_hessian",
    "composed_curvature_step",
    "cubic_regularized_gauss_newton_minimize",
    "gauss_newton_direction",
    "gauss_newton_direction_cgls",
    "gauss_newton_fisher",
    "gauss_newton_minimize",
    "gauss_newton_step",
    "grad_norm_weights",
    "homotopy_gauss_newton_minimize",
    "init_gauss_newton_state",
    "jet_line_search",
    "jet_line_search_on_ray",
    "lanczos_tridiag",
    "linearized_linf_direction",
    "lstsq_gauss_newton_direction",
    "make_residual_fn",
    "martens_grosse_combine",
    "martens_grosse_gauss_newton_minimize",
    "natural_gradient_direction",
    "natural_gradient_step",
    "peak_weighted_residual",
]
