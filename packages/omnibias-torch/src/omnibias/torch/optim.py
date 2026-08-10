# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Second-order optimisation for omnibias PINNs (torch): Gauss-Newton + adaptive weights.

First-order optimisers (Adam/SGD) stall on PINN losses because the differential
operator squares the condition number of the problem; they typically plateau near
``1e-3``. omnibias makes a much stronger optimiser practical: the residual map
``theta |-> r(theta)`` is computed from the *exact* closed-form jets
(:meth:`omnibias.torch.architectures.JetMLP.value_grad_hessian`, ``partials``, ...), so
its parameter-Jacobian ``J = d r / d theta`` is a single clean outer autodiff
(:func:`torch.func.jacrev`) -- no nested-autodiff blow-up, no finite-difference noise.

This module provides

* :func:`gauss_newton_direction` -- the (Levenberg-Marquardt damped) Gauss-Newton /
  natural-gradient direction ``(J^T J + mu I) delta = -J^T r``, automatically switching
  to the equivalent dual (kernel / NTK) form ``delta = -J^T (J J^T + mu I)^{-1} r`` when
  there are more parameters than residuals (the push-through identity makes them equal,
  and the dual system is far better conditioned in the over-parameterised regime).
* :func:`conjugate_gradient` -- a matrix-free SPD conjugate-gradient solver.
* :func:`gauss_newton_direction_cg` -- the **matrix-free** Gauss-Newton direction: it
  solves ``(J^T J + mu I) delta = -J^T r`` with conjugate gradient, forming only
  Jacobian-vector (:func:`torch.func.jvp`) and vector-Jacobian (:func:`torch.func.vjp`)
  products. The dense ``N x P`` Jacobian is never materialised and no ``P x P`` (or
  ``N x N``) system is factorised, so a Gauss-Newton step costs ``O(k)`` residual-sized
  linearisations for ``k`` CG iterations instead of ``O(min(N, P)^3)``. This is what
  turns the (already large) per-iteration advantage of Gauss-Newton into a wall-clock win.
* :class:`GaussNewton` -- an adaptive-damping LM optimiser over a flat ``residual_fn``,
  with ``solver="dense"`` (default) or the matrix-free ``solver="cg"``.
* :func:`functional_residual_fn` -- bridges an :class:`torch.nn.Module` whose ``forward``
  returns the residual vector to a flat ``residual_fn`` for :class:`GaussNewton`.
* :func:`weighted_residual_fn` / :func:`quadrature_loss` -- the **integral (function-space)
  loss** surface. Scaling a pointwise residual ``r`` sampled at quadrature nodes by
  ``sqrt(w)`` turns the least-squares objective into the quadrature integral
  ``||sqrt(w) r||^2 = sum_q w_q r_q^2 ~ integral r^2 dx``, so training minimises a genuine
  ``L^2`` residual and the Gauss-Newton matrix ``J^T diag(w) J`` becomes the discretised
  ``L^2`` function-space (energy natural-gradient) metric rather than an empirical average.
* :class:`CubicRegularizedNewton` -- a matrix-free **cubic-regularised Newton** (ARC)
  optimiser over a scalar ``loss_fn``. Each step minimises ``g^T s + 0.5 s^T H s +
  (sigma/3)||s||^3`` with the *exact full* Hessian applied matrix-free (:func:`hvp`, exact
  because the residual jet is closed form) and the subproblem solved in a small Lanczos
  subspace (:func:`lanczos_tridiag`). Unlike Gauss-Newton it keeps the curvature term, and
  unlike Newton it is globally convergent and escapes saddles with no learning-rate / radius.
* :class:`CubicRegularizedGaussNewton` -- the least-squares sibling: the same ARC cubic model
  built on the PSD Gauss-Newton curvature ``J^T J`` (matrix-free ``J v`` / ``J^T u`` products),
  i.e. Levenberg-Marquardt with the ``mu I`` damping replaced by ``(sigma/3)||s||^3`` for
  automatic step control and global convergence. This is the recommended higher-order PINN
  optimiser (the GN metric beats the full Hessian on a least-squares objective).
* :func:`taylor_line_min` -- an **exact high-order line search**: the along-direction
  derivatives of ``phi(a)=loss(theta+a d)`` are taken exactly by nested higher-order autodiff,
  and the truncated Taylor model is minimised in closed form (a near-optimal step in one shot).
* :class:`JetLBFGS` -- **limited-memory BFGS lifted by the exact jets**: the classic two-loop
  recursion, but with the Wolfe backtrack replaced by the exact :func:`taylor_line_min` and the
  scalar initial inverse-Hessian scale replaced by the exact curvature ``<g,g>/<g,Hg>`` (one
  :func:`hvp`). The honest low-memory quasi-Newton counterpart to the second-order methods.
* :class:`GradNormBalancer` -- self-adaptive loss weights that equalise the per-term
  gradient norms (Wang-Teng-Perdikaris 2021 gradient-pathology balancing).

For the standard L2 collocation PINN functional the Gauss-Newton matrix ``J^T J`` *is*
the empirical Sobolev Gram matrix, so :class:`GaussNewton` is exactly the **empirical
energy natural gradient** (Mueller-Zeinhofer 2023) -- the method that takes PINNs from
``1e-3`` to near machine precision.
"""

from __future__ import annotations

import math
from collections import deque
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from typing import Any, Literal, cast

from omnibias.core.verified.conditioning import certified_damping, conditioning_certificate

import torch
import torch.nn as nn
from torch import Tensor
from torch.func import functional_call, grad, jacfwd, jacrev, jvp, vjp
from torch.utils.hooks import RemovableHandle

ResidualFn = Callable[[Tensor], Tensor]
ScalarFn = Callable[[Tensor], Tensor]
MatVec = Callable[[Tensor], Tensor]


def gauss_newton_direction(jac: Tensor, res: Tensor, damping: float) -> Tensor:
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
        raise ValueError(f"jac must be 2-D (N, P), got shape {tuple(jac.shape)}")
    n, p = int(jac.shape[0]), int(jac.shape[1])
    if res.shape != (n,):
        raise ValueError(f"res must have shape ({n},) matching jac rows, got {tuple(res.shape)}")
    mu = float(damping)
    delta: Tensor
    if p <= n:
        a = jac.T @ jac + mu * torch.eye(p, dtype=jac.dtype, device=jac.device)
        delta = torch.linalg.solve(a, -(jac.T @ res))
    else:
        g = jac @ jac.T + mu * torch.eye(n, dtype=jac.dtype, device=jac.device)
        delta = -(jac.T @ torch.linalg.solve(g, res))
    return delta


def lstsq_gauss_newton_direction(jac: Tensor, res: Tensor, damping: float) -> Tensor:
    r"""LM Gauss-Newton step via a QR least-squares solve -- never squares the conditioning.

    Solves the *same* damped problem as :func:`gauss_newton_direction`, but as the
    linear least squares ``min_delta || [J; sqrt(mu) I] delta - [-r; 0] ||^2`` (QR / SVD via
    :func:`torch.linalg.lstsq`) instead of factorising the normal matrix ``J^T J + mu I``.
    Forming ``J^T J`` **squares the condition number** (``kappa(J^T J) = kappa(J)^2``), which
    for the ill-conditioned Jacobians of a differential operator caps the attainable accuracy
    near ``kappa(J)^2 * eps``; the QR route works with ``J`` directly and attains
    ``kappa(J) * eps``. Prefer this over ``gauss_newton_direction`` whenever ``J`` is stiff
    (essentially all PINNs); it is the dense analogue of :func:`gauss_newton_direction_cgls`.

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
        raise ValueError(f"jac must be 2-D (N, P), got shape {tuple(jac.shape)}")
    n, p = int(jac.shape[0]), int(jac.shape[1])
    if res.shape != (n,):
        raise ValueError(f"res must have shape ({n},) matching jac rows, got {tuple(res.shape)}")
    mu = float(damping)
    if mu < 0.0:
        raise ValueError(f"damping must be >= 0, got {mu}")
    if mu == 0.0:
        a_aug = jac
        b_aug = -res
    else:
        eye = math.sqrt(mu) * torch.eye(p, dtype=jac.dtype, device=jac.device)
        a_aug = torch.cat([jac, eye], dim=0)
        b_aug = torch.cat([-res, torch.zeros(p, dtype=res.dtype, device=res.device)])
    sol = torch.linalg.lstsq(a_aug, b_aug.unsqueeze(1)).solution
    return cast(Tensor, sol.reshape(-1))


def martens_grosse_combine(
    residual_fn: ResidualFn,
    params: Tensor,
    delta_gn: Tensor,
    prev_delta: Tensor | None,
) -> tuple[Tensor, Tensor]:
    r"""Martens–Grosse closed-form LR / momentum via **exact** JVPs.

    Minimises ``|| r + J (alpha d + mu prev) ||^2`` for ``(alpha, mu)`` using
    ``Jd = jvp(r, d)`` and ``Jp = jvp(r, prev)`` -- no finite-difference probes.
    Returns ``(step, momentum_state)``. Twin of
    :func:`omnibias.jax.optim.martens_grosse_combine`.
    """
    r0 = residual_fn(params)
    # Stubs may type jvp as a 3-tuple; runtime returns 2 for this call shape.
    _, j_d, *_ = jvp(residual_fn, (params,), (delta_gn,))
    if prev_delta is None:
        num = -torch.dot(j_d, r0)
        den = torch.dot(j_d, j_d) + 1e-30
        alpha = num / den
        step = alpha * delta_gn
        return step, step

    _, j_p, *_ = jvp(residual_fn, (params,), (prev_delta,))
    a11 = torch.dot(j_d, j_d)
    a12 = torch.dot(j_d, j_p)
    a22 = torch.dot(j_p, j_p)
    b1 = -torch.dot(j_d, r0)
    b2 = -torch.dot(j_p, r0)
    det = a11 * a22 - a12 * a12
    det = torch.where(torch.abs(det) < 1e-30, torch.as_tensor(1e-30, dtype=det.dtype, device=det.device), det)
    alpha = (a22 * b1 - a12 * b2) / det
    mu = (-a12 * b1 + a11 * b2) / det
    step = alpha * delta_gn + mu * prev_delta
    return step, step


def conjugate_gradient(
    matvec: MatVec,
    b: Tensor,
    *,
    max_iter: int = 100,
    tol: float = 1e-8,
    x0: Tensor | None = None,
) -> Tensor:
    r"""Solve the SPD system ``A x = b`` with conjugate gradient, matrix-free.

    Only the linear operator ``matvec: v |-> A v`` is required; ``A`` is never formed.
    Iterates until the relative residual ``||A x - b|| <= tol * ||b||`` or ``max_iter``
    matvecs, whichever comes first. If a non-positive curvature ``p^T A p <= 0`` is
    encountered (``A`` not positive definite to numerical tolerance) the iteration stops
    and returns the best iterate so far.

    Parameters
    ----------
    matvec:
        The SPD operator ``v |-> A v``.
    b:
        Right-hand side, shape ``(P,)``.
    max_iter:
        Maximum CG iterations (each is one ``matvec``).
    tol:
        Relative residual tolerance.
    x0:
        Optional warm start (defaults to zeros).
    """
    b_norm = float(torch.linalg.vector_norm(b))
    if b_norm == 0.0:
        return torch.zeros_like(b)
    if x0 is None:
        x = torch.zeros_like(b)
        r = b.clone()
    else:
        x = x0.clone()
        r = b - matvec(x)
    p = r.clone()
    rs_old = torch.dot(r, r)
    thresh_sq = (tol * b_norm) ** 2
    for _ in range(max_iter):
        if float(rs_old) <= thresh_sq:
            break
        ap = matvec(p)
        p_ap = torch.dot(p, ap)
        if float(p_ap) <= 0.0:
            break
        alpha = rs_old / p_ap
        x = x + alpha * p
        r = r - alpha * ap
        rs_new = torch.dot(r, r)
        p = r + (rs_new / rs_old) * p
        rs_old = rs_new
    return x


def _to_boundary(z: Tensor, d: Tensor, radius: float) -> float:
    r"""Largest ``tau >= 0`` with ``||z + tau d|| = radius`` (the positive trust-region root).

    Solves the quadratic ``||d||^2 tau^2 + 2 (z . d) tau + (||z||^2 - radius^2) = 0``. Because
    ``z`` is inside the ball (``||z|| <= radius``) the constant term is ``<= 0``, so the
    discriminant is non-negative and the ``+sqrt`` root is the unique non-negative solution.
    """
    dd = float(torch.dot(d, d))
    if dd <= 0.0:
        return 0.0
    zd = float(torch.dot(z, d))
    zz = float(torch.dot(z, z))
    c = zz - radius * radius
    disc = zd * zd - dd * c
    disc = disc if disc > 0.0 else 0.0
    return (-zd + math.sqrt(disc)) / dd


def steihaug_cg(
    matvec: MatVec,
    g: Tensor,
    radius: float,
    *,
    max_iter: int = 50,
    tol: float = 1e-8,
) -> tuple[Tensor, bool]:
    r"""Steihaug-Toint truncated CG for the trust-region subproblem, matrix-free.

    Approximately solves ``min_p  g^T p + 0.5 p^T H p  s.t. ||p|| <= radius`` using only the
    symmetric operator ``matvec: v |-> H v`` (``H`` need **not** be positive definite). This is
    the standard inexact-Newton trust-region solver (Nocedal-Wright Alg. 7.2): conjugate
    gradient on ``H p = -g`` truncated the moment it (a) meets **negative curvature**
    ``d^T H d <= 0`` or (b) would **leave the trust region** ``||p|| >= radius`` -- in either
    case it follows the current direction to the ball boundary and returns.

    Returns ``(p, hit_boundary)`` where ``p`` is the step (so the update is ``theta + p``, and
    ``p ~ -H^{-1} g`` when the unconstrained Newton step lies inside the region) and
    ``hit_boundary`` is ``True`` iff the returned step lies on ``||p|| = radius`` (used by the
    caller to decide whether to grow the radius).

    Parameters
    ----------
    matvec:
        The symmetric curvature operator ``v |-> H v`` (exact Hessian, Gauss-Newton, ...).
    g:
        The gradient at the current point, shape ``(P,)``.
    radius:
        Trust-region radius ``> 0``.
    max_iter:
        Maximum CG iterations (each is one ``matvec``).
    tol:
        Relative tolerance on the residual ``||H p + g||`` (fraction of ``||g||``).
    """
    z = torch.zeros_like(g)
    gnorm = float(torch.linalg.vector_norm(g))
    if gnorm == 0.0 or not math.isfinite(gnorm):
        return z, False
    r = g.clone()
    d = -g
    rs_old = torch.dot(r, r)
    thresh_sq = (tol * gnorm) ** 2
    for _ in range(max_iter):
        hd = matvec(d)
        d_hd = float(torch.dot(d, hd))
        if d_hd <= 0.0:  # negative curvature: ride the direction to the boundary
            tau = _to_boundary(z, d, radius)
            return z + tau * d, True
        alpha = float(rs_old) / d_hd
        z_next = z + alpha * d
        if float(torch.linalg.vector_norm(z_next)) >= radius:  # step exits the region
            tau = _to_boundary(z, d, radius)
            return z + tau * d, True
        r = r + alpha * hd
        rs_new = torch.dot(r, r)
        z = z_next
        if float(rs_new) <= thresh_sq:  # converged inside the region
            return z, False
        d = -r + (rs_new / rs_old) * d
        rs_old = rs_new
    return z, False


def cgls(
    a_matvec: MatVec,
    at_matvec: MatVec,
    b: Tensor,
    *,
    damp: float = 0.0,
    max_iter: int = 100,
    tol: float = 1e-8,
) -> Tensor:
    r"""Solve ``min_x ||A x - b||^2 + damp^2 ||x||^2`` by CGLS, matrix-free on ``A``.

    CGLS (Conjugate Gradient on the Least Squares problem, Bjorck) needs only ``A v`` and
    ``A^T u`` products, yet -- unlike conjugate gradient applied to the normal operator
    ``A^T A`` -- it keeps the residual ``b - A x`` in the range space and recomputes the
    normal-equation residual ``s = A^T (b - A x) - damp^2 x`` each step. It therefore
    attains accuracy governed by ``kappa(A)`` rather than ``kappa(A)^2``, which is the whole
    point for stiff Jacobians. (LSQR / LSMR are more elaborate variants of the same idea.)

    Parameters
    ----------
    a_matvec, at_matvec:
        The operators ``v |-> A v`` (shape ``(P,) -> (N,)``) and ``u |-> A^T u``
        (``(N,) -> (P,)``).
    b:
        Right-hand side of shape ``(N,)``.
    damp:
        Tikhonov / Levenberg-Marquardt damping ``>= 0`` (this is ``sqrt(mu)``, so the
        normal system solved is ``(A^T A + damp^2 I) x = A^T b``).
    max_iter, tol:
        Iteration budget and relative tolerance on ``||A^T(b - A x) - damp^2 x||``.
    """
    lam = float(damp) ** 2
    r = b.clone()
    s = at_matvec(r)
    x = torch.zeros_like(s)
    p = s.clone()
    gamma = torch.dot(s, s)
    norm_s0 = float(gamma) ** 0.5
    if norm_s0 == 0.0:
        return x
    thresh = tol * norm_s0
    for _ in range(max_iter):
        q = a_matvec(p)
        denom = torch.dot(q, q) + lam * torch.dot(p, p)
        if float(denom) <= 0.0:
            break
        alpha = gamma / denom
        x = x + alpha * p
        r = r - alpha * q
        s = at_matvec(r) - lam * x
        gamma_new = torch.dot(s, s)
        if float(gamma_new) ** 0.5 <= thresh:
            break
        p = s + (gamma_new / gamma) * p
        gamma = gamma_new
    return x


def _linearize_gn(residual_fn: ResidualFn, params: Tensor) -> tuple[Tensor, MatVec, MatVec]:
    r"""Return ``(res, jt, jvec)``: the residual and reusable ``J^T(.)`` / ``J(.)`` maps.

    The reverse-mode closure ``jt = J^T(.)`` is built once (single :func:`torch.func.vjp`
    linearisation) and reused for every product -- across CG iterations *and* across the
    LM damping line search -- so an entire step shares one ``vjp``. ``jvec = J(.)`` is a
    forward-mode :func:`torch.func.jvp`. Nothing dense (``J``, ``J^T J``, ``J J^T``) is
    ever assembled.
    """
    _vjp_out = vjp(residual_fn, params)
    res, vjp_fn = _vjp_out[0], _vjp_out[1]

    def jt(u: Tensor) -> Tensor:
        return cast(Tensor, vjp_fn(u)[0])

    def jvec(v: Tensor) -> Tensor:
        return cast(Tensor, jvp(residual_fn, (params,), (v,))[1])

    return cast(Tensor, res), jt, jvec


def _gn_cg_direction(
    res: Tensor,
    jt: MatVec,
    jvec: MatVec,
    n_params: int,
    damping: float,
    cg_max_iter: int,
    cg_tol: float,
) -> Tensor:
    r"""LM Gauss-Newton direction from a prebuilt linearisation, primal *or* dual CG.

    Mirrors :func:`gauss_newton_direction`: solves the ``P``-dimensional primal
    ``(J^T J + mu I) delta = -J^T r`` when ``P <= N`` and the ``N``-dimensional dual
    ``(J J^T + mu I) y = r``, ``delta = -J^T y`` otherwise -- so CG always runs in the
    smaller ``min(P, N)`` dimension and converges in at most that many iterations.
    """
    mu = float(damping)
    n_res = int(res.shape[0])
    if n_params <= n_res:
        g = jt(res)

        def matvec_primal(v: Tensor) -> Tensor:
            return jt(jvec(v)) + mu * v

        return conjugate_gradient(matvec_primal, -g, max_iter=cg_max_iter, tol=cg_tol)

    def matvec_dual(v: Tensor) -> Tensor:
        return jvec(jt(v)) + mu * v

    y = conjugate_gradient(matvec_dual, res, max_iter=cg_max_iter, tol=cg_tol)
    return -jt(y)


def gauss_newton_direction_cg(
    residual_fn: ResidualFn,
    params: Tensor,
    damping: float,
    *,
    cg_max_iter: int = 100,
    cg_tol: float = 1e-8,
) -> Tensor:
    r"""Matrix-free Levenberg-Marquardt Gauss-Newton direction via conjugate gradient.

    Solves ``(J^T J + mu I) delta = -J^T r`` (or its dual when there are more parameters
    than residuals) with :func:`conjugate_gradient`, using only :func:`torch.func.jvp` /
    :func:`torch.func.vjp` products of ``residual_fn`` -- the Jacobian ``J = d r / d theta``
    is never materialised. Equivalent to :func:`gauss_newton_direction` up to CG tolerance,
    but with cost independent of assembling / factorising the normal equations.

    Parameters
    ----------
    residual_fn:
        Differentiable residual map ``theta |-> r(theta)`` (see :func:`functional_residual_fn`).
    params:
        Current flat parameter vector ``theta`` of shape ``(P,)``.
    damping:
        Levenberg-Marquardt damping ``mu > 0`` (keeps the normal operator SPD for CG).
    cg_max_iter, cg_tol:
        Conjugate-gradient budget and relative tolerance.
    """
    res, jt, jvec = _linearize_gn(residual_fn, params)
    return _gn_cg_direction(
        res, jt, jvec, int(params.shape[0]), damping, cg_max_iter, cg_tol
    )


def _gn_cgls_direction(
    res: Tensor, jt: MatVec, jvec: MatVec, damping: float, max_iter: int, tol: float
) -> Tensor:
    r"""LM Gauss-Newton direction from a prebuilt linearisation via CGLS (no ``kappa^2``).

    Solves ``(J^T J + mu I) delta = -J^T r`` as the damped least squares
    ``min_delta ||J delta + r||^2 + mu ||delta||^2`` on ``J`` directly (``A = J``,
    ``b = -r``, ``damp = sqrt(mu)``), so it needs no primal/dual switch and inherits CGLS's
    ``kappa(J)`` (rather than ``kappa(J)^2``) accuracy.
    """
    return cgls(jvec, jt, -res, damp=math.sqrt(float(damping)), max_iter=max_iter, tol=tol)


def gauss_newton_direction_cgls(
    residual_fn: ResidualFn,
    params: Tensor,
    damping: float,
    *,
    cgls_max_iter: int = 100,
    cgls_tol: float = 1e-8,
) -> Tensor:
    r"""Matrix-free LM Gauss-Newton direction via CGLS -- the ill-conditioning-safe solver.

    Like :func:`gauss_newton_direction_cg` it uses only :func:`torch.func.jvp` / ``vjp``
    products, but it runs :func:`cgls` on ``J`` instead of conjugate gradient on ``J^T J``,
    so its attainable accuracy scales with ``kappa(J)`` not ``kappa(J)^2``. This is the
    matrix-free method to reach machine precision on stiff PINN residuals.
    """
    res, jt, jvec = _linearize_gn(residual_fn, params)
    return _gn_cgls_direction(res, jt, jvec, damping, cgls_max_iter, cgls_tol)


def natural_gradient_direction(
    metric: Tensor | MatVec,
    grad: Tensor,
    *,
    damping: float = 1e-3,
    cg_max_iter: int = 100,
    cg_tol: float = 1e-8,
) -> Tensor:
    r"""Natural-gradient direction ``delta = (M + damping I)^{-1} grad``.

    The metric-preconditioned counterpart of the raw gradient: for a Riemannian metric
    ``M`` on parameter space the *natural* gradient is ``M^{-1} grad``, so descending along
    ``delta`` is invariant to smooth reparametrisations. ``M`` may be supplied either

    * **dense** -- a symmetric positive-(semi)definite ``(P, P)`` :class:`~torch.Tensor`
      (solved directly by :func:`torch.linalg.solve`), or
    * **matrix-free** -- a linear operator ``matvec: v |-> M v`` (a :data:`MatVec`), in which
      case the damped system ``(M + mu I) delta = grad`` is solved with
      :func:`conjugate_gradient`; nothing dense is assembled.

    ``damping >= 0`` (Tikhonov ``mu``) keeps the solve well-posed when ``M`` is singular or
    ill-conditioned. Mirrors :func:`omnibias.curvature.natural_gradient.damped_solve`
    (bit-identical on a shared dense metric) and its jax twin
    :func:`omnibias.jax.optim.natural_gradient_direction`.

    Parameters
    ----------
    metric:
        The metric ``M``: a dense ``(P, P)`` tensor or a matrix-free ``v |-> M v`` operator.
    grad:
        The (Euclidean) gradient ``grad`` of shape ``(P,)``.
    damping:
        Tikhonov damping ``mu >= 0``.
    cg_max_iter, cg_tol:
        Conjugate-gradient budget / relative tolerance (matrix-free path only).
    """
    if damping < 0.0:
        raise ValueError(f"damping must be >= 0, got {damping}")
    if grad.ndim != 1:
        raise ValueError(f"grad must be a 1-D (P,) vector, got shape {tuple(grad.shape)}")
    mu = float(damping)
    if isinstance(metric, Tensor):
        p = int(grad.shape[0])
        if metric.shape != (p, p):
            raise ValueError(
                f"dense metric must have shape ({p}, {p}) matching grad, got {tuple(metric.shape)}"
            )
        a = metric + mu * torch.eye(p, dtype=grad.dtype, device=grad.device)
        return cast(Tensor, torch.linalg.solve(a, grad))

    def damped_matvec(v: Tensor) -> Tensor:
        mv = metric(v)
        return mv + mu * v if mu != 0.0 else mv

    return conjugate_gradient(damped_matvec, grad, max_iter=cg_max_iter, tol=cg_tol)


def _certified_damping_floor(
    matrix: Tensor, target_condition: float
) -> tuple[float, dict[str, Any]]:
    r"""Certified Tikhonov damping ``eps`` s.t. ``kappa(matrix + eps I) <= target_condition``.

    Detaches the assembled *dense* metric to float64 and defers to the rigorous
    :func:`omnibias.core.verified.conditioning.certified_damping` (interval-eigenvalue
    bisection), also returning the sealed conditioning certificate. This is the
    ``eps -> 0`` rank/regularization collapse applied as a *conditioning floor* on the
    curvature solve; it backs the opt-in ``target_condition`` of :class:`NaturalGradient`
    and :class:`GaussNewton`. Dense paths only -- the matrix-free CG/CGLS solvers assemble
    no matrix to certify.
    """
    rows = cast("list[list[float]]", matrix.detach().to(torch.float64).cpu().tolist())
    eps = certified_damping(rows, target_condition=target_condition)
    cert = conditioning_certificate(rows, target_condition=target_condition, eps=eps)
    return eps, cert


def gauss_newton_fisher(residual_fn: ResidualFn, params: Tensor) -> tuple[Tensor, Tensor]:
    r"""Dense Gauss-Newton Fisher ``F = (1/N) J^T J`` and gradient ``g = (1/N) J^T r``.

    The closed-form Fisher metric of the least-squares objective ``0.5 mean(r^2)``: with
    ``J = d r / d theta`` (one :func:`torch.func.jacrev`), ``F`` is the Gauss-Newton (and,
    at a zero residual, exact) Hessian and ``g`` its gradient. Feed the pair to
    :func:`natural_gradient_direction` for a Fisher-scoring / natural-gradient step (which
    on a linear-in-``theta`` residual equals Newton and recovers the least-squares minimiser
    in one step). For a matrix-free equivalent see :func:`gauss_newton_fisher_matvec`.

    Returns ``(F, g)`` as detached tensors of shape ``((P, P), (P,))``.
    """
    res = residual_fn(params)
    if res.ndim != 1:
        raise ValueError(f"residual_fn must return a 1-D vector, got shape {tuple(res.shape)}")
    n_res = int(res.shape[0])
    jac = cast(Tensor, jacrev(residual_fn)(params))  # (N, P)
    jac_t = jac.transpose(0, 1)
    fisher = (jac_t @ jac) / n_res
    g = (jac_t @ res) / n_res
    return fisher.detach(), g.detach()


def gauss_newton_fisher_matvec(
    residual_fn: ResidualFn, params: Tensor
) -> tuple[Tensor, Tensor, MatVec]:
    r"""Matrix-free Gauss-Newton Fisher: ``(res, g = (1/N) J^T r, matvec: v |-> (1/N) J^T J v)``.

    The matrix-free twin of :func:`gauss_newton_fisher`: a single reverse-mode linearisation
    (:func:`torch.func.vjp`) is reused for ``g`` and for every Fisher product, so the dense
    ``(P, P)`` matrix ``J^T J`` is never assembled. Pass ``matvec`` straight to
    :func:`natural_gradient_direction` (which solves ``(F + mu I) delta = g`` by conjugate
    gradient) for an ``O(k)``-linearisation natural step.
    """
    res, jt, jvec = _linearize_gn(residual_fn, params)
    n_res = int(res.shape[0])
    g = jt(res) / n_res

    def matvec(v: Tensor) -> Tensor:
        return jt(jvec(v)) / n_res

    return res, g, matvec


def _half_mean_sq(res: Tensor) -> float:
    return 0.5 * float(torch.mean(res**2).item())


@dataclass
class GaussNewtonInfo:
    """Per-step diagnostics returned by :meth:`GaussNewton.step`."""

    loss: float
    damping: float
    accepted: bool
    n_iter: int


class GaussNewton:
    r"""Adaptive-damping Gauss-Newton (Levenberg-Marquardt) optimiser over a residual_fn.

    The optimiser is *functional* in the parameters: :meth:`step` takes the current flat
    parameter vector and returns a new one, so it never mutates a module behind your back
    (use :func:`functional_residual_fn` and :func:`torch.nn.utils.vector_to_parameters`
    to bridge an :class:`torch.nn.Module`). The damping is the only mutable state.

    Parameters
    ----------
    damping:
        Initial Levenberg-Marquardt damping ``mu > 0``.
    damping_increase, damping_decrease:
        Multipliers applied to ``mu`` on a rejected / accepted step.
    min_damping, max_damping:
        Clamp range for ``mu``.
    max_line_search:
        Max damping increases attempted within one :meth:`step` before giving up.
    solver:
        Linear solve for the LM direction. ``"dense"`` (default) and ``"cg"`` factorise /
        iterate the **normal** equations ``J^T J + mu I`` -- fast, but they square the
        conditioning (``kappa(J)^2``) and so plateau early on stiff differential operators.
        ``"qr"`` (dense :func:`lstsq_gauss_newton_direction`) and ``"cgls"`` (matrix-free
        :func:`gauss_newton_direction_cgls`) solve the equivalent least squares on ``J``
        directly (``kappa(J)``) -- prefer these to actually reach machine precision on PINNs.
        ``"dense"`` / ``"qr"`` build the Jacobian (:func:`torch.func.jacrev`); ``"cg"`` /
        ``"cgls"`` are matrix-free.
    damping_strategy:
        ``"classic"`` (default) multiplies ``mu`` by ``damping_decrease`` / ``damping_increase``
        on an accepted / rejected step. ``"nielsen"`` is the gain-ratio trust-region update
        (Nielsen 2003): ``mu <- mu * max(1/3, 1 - (2 rho - 1)^3)`` on acceptance and a
        doubling ``mu <- mu * nu`` (``nu <- 2 nu``) on rejection, where ``rho`` is the ratio of
        actual to model-predicted reduction. It anneals ``mu -> 0`` in the fast regime so GN
        can enter its quadratic convergence, and is recommended together with ``qr`` / ``cgls``.
        cg_max_iter, cg_tol:
        Krylov budget / relative tolerance for the iterative solvers (``"cg"`` and ``"cgls"``).
    target_condition:
        Optional certified-conditioning target (``> 1``); ``"dense"`` solver only. When set,
        each step floors the LM damping at the smallest ``eps`` for which the assembled
        Gauss-Newton normal matrix provably satisfies ``kappa(J^T J + eps I) <=
        target_condition`` (via :func:`omnibias.core.verified.conditioning.certified_damping`,
        the ``eps -> 0`` rank/regularization collapse), and stashes the sealed conditioning
        certificate on ``last_certificate``. Matrix-free solvers assemble no matrix to certify
        and ignore it.
    use_martens_grosse:
        If ``True``, after each LM direction apply :func:`martens_grosse_combine` (exact
        JVP 2x2 LR/momentum). Momentum state resets on rejected steps.
    """

    def __init__(
        self,
        *,
        damping: float = 1e-3,
        damping_increase: float = 3.0,
        damping_decrease: float = 0.5,
        min_damping: float = 1e-12,
        max_damping: float = 1e12,
        max_line_search: int = 8,
        solver: Literal["dense", "qr", "cg", "cgls"] = "dense",
        damping_strategy: Literal["classic", "nielsen"] = "classic",
        cg_max_iter: int = 100,
        cg_tol: float = 1e-8,
        target_condition: float | None = None,
        use_martens_grosse: bool = False,
    ) -> None:
        if damping <= 0.0:
            raise ValueError(f"damping must be > 0, got {damping}")
        if target_condition is not None and target_condition <= 1.0:
            raise ValueError(f"target_condition must be > 1, got {target_condition}")
        if damping_increase <= 1.0:
            raise ValueError(f"damping_increase must be > 1, got {damping_increase}")
        if not 0.0 < damping_decrease < 1.0:
            raise ValueError(f"damping_decrease must be in (0, 1), got {damping_decrease}")
        if max_line_search < 1:
            raise ValueError(f"max_line_search must be >= 1, got {max_line_search}")
        if solver not in ("dense", "qr", "cg", "cgls"):
            raise ValueError(f"solver must be 'dense', 'qr', 'cg' or 'cgls', got {solver!r}")
        if damping_strategy not in ("classic", "nielsen"):
            raise ValueError(
                f"damping_strategy must be 'classic' or 'nielsen', got {damping_strategy!r}"
            )
        if cg_max_iter < 1:
            raise ValueError(f"cg_max_iter must be >= 1, got {cg_max_iter}")
        if cg_tol <= 0.0:
            raise ValueError(f"cg_tol must be > 0, got {cg_tol}")
        self.damping = float(damping)
        self.damping_increase = float(damping_increase)
        self.damping_decrease = float(damping_decrease)
        self.min_damping = float(min_damping)
        self.max_damping = float(max_damping)
        self.max_line_search = int(max_line_search)
        self.solver = solver
        self.damping_strategy = damping_strategy
        self.cg_max_iter = int(cg_max_iter)
        self.cg_tol = float(cg_tol)
        self.target_condition = None if target_condition is None else float(target_condition)
        self.use_martens_grosse = bool(use_martens_grosse)
        self._prev_delta: Tensor | None = None
        self.last_certificate: dict[str, Any] | None = None
        self.n_iter = 0

    @torch.no_grad()
    def step(self, residual_fn: ResidualFn, params: Tensor) -> tuple[Tensor, GaussNewtonInfo]:
        """One LM iteration; returns ``(new_params, info)`` (params unchanged on failure).

        For the matrix-free solvers (``"cg"`` / ``"cgls"``) the reverse-mode linearisation
        ``J^T(.)`` is built once and reused for every damping value in the line search; only
        the Krylov solve is repeated.
        """
        matrix_free = self.solver in ("cg", "cgls")
        jac: Tensor | None = None
        jt: MatVec | None = None
        jvec: MatVec | None = None
        n_params = int(params.shape[0])
        if matrix_free:
            res, jt, jvec = _linearize_gn(residual_fn, params)
        else:
            res = residual_fn(params)
            jac = jacrev(residual_fn)(params)
        loss0 = _half_mean_sq(res)
        # gradient g = J^T r (sum-scale) is only needed for the Nielsen gain ratio
        grad: Tensor | None = None
        if self.damping_strategy == "nielsen":
            if matrix_free:
                assert jt is not None
                grad = jt(res)
            else:
                assert jac is not None
                grad = jac.T @ res
        sse0 = float(torch.dot(res, res))
        mu = self.damping
        nu = 2.0
        # Optional certified-conditioning floor on the assembled GN normal matrix
        # (the eps -> 0 collapse as a damping floor); dense solver only, computed once
        # per step since the normal matrix is fixed across the damping line search.
        cert_floor = 0.0
        if self.target_condition is not None and self.solver == "dense":
            assert jac is not None
            normal = jac.T @ jac if n_params <= int(jac.shape[0]) else jac @ jac.T
            cert_floor, self.last_certificate = _certified_damping_floor(
                normal, self.target_condition
            )
        for _ in range(self.max_line_search):
            if self.solver == "dense":
                assert jac is not None
                delta = gauss_newton_direction(jac, res, max(mu, cert_floor))
            elif self.solver == "qr":
                assert jac is not None
                delta = lstsq_gauss_newton_direction(jac, res, mu)
            elif self.solver == "cg":
                assert jt is not None and jvec is not None
                delta = _gn_cg_direction(res, jt, jvec, n_params, mu, self.cg_max_iter, self.cg_tol)
            else:  # cgls
                assert jt is not None and jvec is not None
                delta = _gn_cgls_direction(res, jt, jvec, mu, self.cg_max_iter, self.cg_tol)
            if self.use_martens_grosse:
                delta, mom = martens_grosse_combine(
                    residual_fn, params, delta, self._prev_delta
                )
            else:
                mom = delta
            new_params = params + delta
            res_new = residual_fn(new_params)
            loss1 = _half_mean_sq(res_new)
            if math.isfinite(loss1) and loss1 < loss0:
                self.damping = self._accept_damping(mu, delta, res_new, sse0, grad)
                self._prev_delta = mom.detach()
                self.n_iter += 1
                return new_params, GaussNewtonInfo(loss1, self.damping, True, self.n_iter)
            if self.damping_strategy == "nielsen":
                mu = min(mu * nu, self.max_damping)
                nu *= 2.0
            else:
                mu = min(mu * self.damping_increase, self.max_damping)
            self._prev_delta = None
        self.damping = mu
        self.n_iter += 1
        return params, GaussNewtonInfo(loss0, self.damping, False, self.n_iter)

    def _accept_damping(
        self, mu: float, delta: Tensor, res_new: Tensor, sse0: float, grad: Tensor | None
    ) -> float:
        """New ``mu`` after an accepted step (classic geometric or Nielsen gain-ratio)."""
        if self.damping_strategy == "classic" or grad is None:
            return max(mu * self.damping_decrease, self.min_damping)
        # Nielsen: rho = actual reduction / model-predicted reduction (sum-of-squares scale).
        actual = 0.5 * (sse0 - float(torch.dot(res_new, res_new)))
        predicted = 0.5 * (mu * float(torch.dot(delta, delta)) - float(torch.dot(grad, delta)))
        rho = actual / predicted if predicted > 0.0 else 1.0
        factor = max(1.0 / 3.0, 1.0 - (2.0 * rho - 1.0) ** 3)
        return min(max(mu * factor, self.min_damping), self.max_damping)

    def minimize(
        self, residual_fn: ResidualFn, params: Tensor, *, steps: int
    ) -> tuple[Tensor, list[float]]:
        """Run :meth:`step` ``steps`` times; return ``(final_params, loss_history)``."""
        history: list[float] = []
        for _ in range(steps):
            params, info = self.step(residual_fn, params)
            history.append(info.loss)
        return params, history


def martens_grosse_gauss_newton_minimize(
    residual_fn: ResidualFn,
    params0: Tensor,
    *,
    steps: int = 50,
    damping: float = 1e-3,
    solver: Literal["dense", "qr", "cg", "cgls"] = "qr",
    use_martens_grosse: bool = True,
    damping_decrease: float = 0.7,
    damping_increase: float = 2.0,
) -> tuple[Tensor, list[float]]:
    r"""Functional twin of :func:`omnibias.jax.optim.martens_grosse_gauss_newton_minimize`.

    Defaults to QR + Martens–Grosse (exact JVP). Prefer this for PINN residual maps
    when parity with the JAX earn path matters.
    """
    opt = GaussNewton(
        damping=damping,
        damping_decrease=damping_decrease,
        damping_increase=damping_increase,
        solver=solver,
        use_martens_grosse=use_martens_grosse,
    )
    return opt.minimize(residual_fn, params0, steps=steps)


# --- Cubic-regularised Newton (exact-jet second-order, saddle-escaping) ----


def hvp(loss_fn: ScalarFn, params: Tensor, v: Tensor) -> Tensor:
    r"""Hessian-vector product ``H v`` of a scalar ``loss_fn`` at ``params`` (matrix-free).

    Exact forward-over-reverse: the JVP of the gradient map in direction ``v`` equals
    ``H v = d/deps grad f(params + eps v)``. No dense Hessian is formed -- one call is one
    linearised backward pass. For an omnibias PINN loss the residual is the exact closed-form
    jet, so this full Hessian (unlike the Gauss-Newton ``J^T J`` surrogate) is clean and
    captures the curvature term ``sum_i r_i grad^2 r_i`` that matters far from the solution
    and for escaping saddles.
    """
    grad_fn = grad(loss_fn)
    return cast(Tensor, jvp(grad_fn, (params,), (v,))[1])


def lanczos_tridiag(matvec: MatVec, b: Tensor, k: int, *, tol: float = 1e-10) -> tuple[Tensor, Tensor]:
    r"""``k``-step Lanczos on a symmetric operator with full reorthogonalisation.

    Returns ``(Q, T)`` with ``Q`` (``n x m``) orthonormal and its first column ``b/||b||``,
    and ``T`` (``m x m``) symmetric tridiagonal equal to ``Q^T A Q`` (``m <= min(k, n)``,
    early exit on a tiny sub-diagonal). This is the matrix-free reduction that lets the cubic
    subproblem be solved exactly in a tiny ``m``-dimensional space, and it handles indefinite
    ``A`` gracefully (the regularisation lives in the reduced problem).
    """
    n = int(b.shape[0])
    m_max = min(int(k), n)
    beta0 = float(torch.linalg.vector_norm(b))
    if beta0 == 0.0:
        q = torch.zeros_like(b)
        q[0] = 1.0
    else:
        q = b / beta0
    qs: list[Tensor] = [q]
    alphas: list[Tensor] = []
    betas: list[Tensor] = []
    q_prev = torch.zeros_like(b)
    beta_prev = torch.zeros((), dtype=b.dtype, device=b.device)
    for _j in range(m_max):
        w = matvec(qs[-1])
        alpha = torch.dot(w, qs[-1])
        alphas.append(alpha)
        w = w - alpha * qs[-1] - beta_prev * q_prev
        for qi in qs:  # full reorthogonalisation (m is small)
            w = w - torch.dot(w, qi) * qi
        beta = torch.linalg.vector_norm(w)
        if float(beta) <= tol:
            break
        betas.append(beta)
        q_prev = qs[-1]
        beta_prev = beta
        qs.append(w / beta)
    m = len(alphas)
    q_basis = torch.stack(qs[:m], dim=1)
    tri = torch.diag(torch.stack(alphas))
    if m > 1:
        off = torch.stack(betas[: m - 1])
        tri = tri + torch.diag(off, 1) + torch.diag(off, -1)
    return q_basis, tri


def _solve_cubic_subproblem(tri: Tensor, c: Tensor, sigma: float, *, iters: int = 100) -> Tensor:
    r"""Global minimiser of ``c^T y + 0.5 y^T T y + (sigma/3) ||y||^3`` (the ARC subproblem).

    In the eigenbasis ``T = V diag(theta) V^T`` the minimiser is ``y = V z``,
    ``z_i = -chat_i / (theta_i + lam)`` with ``chat = V^T c`` and ``lam = sigma ||z|| >= 0``
    solving the secular equation ``||z(lam)|| = lam / sigma`` (safeguarded by bisection over
    ``lam > max(0, -theta_min)`` so ``T + lam I >= 0``).
    """
    theta, vecs = torch.linalg.eigh(tri)
    chat = vecs.T @ c
    lam_lo = max(0.0, -float(theta[0]))
    eps = 1e-12 + 1e-9 * max(1.0, abs(float(theta[-1])))

    def z_of(lam: float) -> Tensor:
        return cast(Tensor, -chat / (theta + lam))

    def phi(lam: float) -> float:
        return float(torch.linalg.vector_norm(z_of(lam))) - lam / sigma

    lo = lam_lo + eps
    if phi(lo) <= 0.0:  # tiny gradient / hard case: the safeguarded point is the solution
        return cast(Tensor, vecs @ z_of(lo))
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
    return cast(Tensor, vecs @ z_of(0.5 * (lo + hi)))


def cubic_regularized_newton_step(
    loss_fn: ScalarFn, params: Tensor, sigma: float, *, krylov_dim: int = 20
) -> Tensor:
    r"""One cubic-regularised Newton step (ARC), matrix-free via Lanczos on the exact Hessian.

    Minimises the cubic model ``m(s) = g^T s + 0.5 s^T H s + (sigma/3) ||s||^3`` over a
    Krylov subspace built from Hessian-vector products (:func:`hvp`). The cubic term makes the
    step well-defined even where ``H`` is indefinite, giving a globally convergent,
    saddle-escaping step without any trust-region radius or line-search tuning.
    """
    grad_fn = grad(loss_fn)
    g = cast(Tensor, grad_fn(params))
    gnorm = float(torch.linalg.vector_norm(g))
    if gnorm == 0.0:
        return torch.zeros_like(params)

    def matvec(v: Tensor) -> Tensor:
        return cast(Tensor, jvp(grad_fn, (params,), (v,))[1])

    q_basis, tri = lanczos_tridiag(matvec, g, krylov_dim)
    c = torch.zeros(tri.shape[0], dtype=params.dtype, device=params.device)
    c[0] = gnorm
    return q_basis @ _solve_cubic_subproblem(tri, c, sigma)


@dataclass
class CubicNewtonInfo:
    """Per-step diagnostics returned by the cubic-regularised optimisers' ``step``."""

    loss: float
    sigma: float
    accepted: bool
    rho: float
    grad_norm: float
    n_iter: int


def _cubic_arc_step(
    params: Tensor,
    g: Tensor,
    hess_matvec: MatVec,
    f0: float,
    eval_f: Callable[[Tensor], float],
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
) -> tuple[Tensor, bool, float, float, float]:
    r"""Shared ARC (adaptive cubic regularisation) acceptance loop.

    Given the gradient ``g`` and a symmetric curvature operator ``hess_matvec`` (the exact full
    Hessian for :class:`CubicRegularizedNewton`, the Gauss-Newton operator ``J^T J`` for
    :class:`CubicRegularizedGaussNewton`), the Lanczos reduction is built *once* and the cubic
    model ``g^T s + 0.5 s^T H s + (sigma/3)||s||^3`` is minimised for each ``sigma`` in the
    ratio-test loop. Returns ``(new_params, accepted, new_sigma, rho, new_loss)`` -- params and
    loss are unchanged (``f0``) if every trial step is rejected.
    """
    gnorm = float(torch.linalg.vector_norm(g))
    if gnorm == 0.0:
        return params, False, sigma, 0.0, f0
    q_basis, tri = lanczos_tridiag(hess_matvec, g, krylov_dim)
    c = torch.zeros(tri.shape[0], dtype=params.dtype, device=params.device)
    c[0] = gnorm
    rho = 0.0
    for _ in range(max_line_search):
        y = _solve_cubic_subproblem(tri, c, sigma)
        s = q_basis @ y
        model_dec = -(
            float(torch.dot(c, y))
            + 0.5 * float(torch.dot(y, tri @ y))
            + (sigma / 3.0) * float(torch.linalg.vector_norm(y)) ** 3
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


class _CubicArcOptimizer:
    r"""Shared state / hyper-parameters for the adaptive-cubic-regularisation optimisers.

    Subclasses implement :meth:`step` (the curvature operator differs); everything else -- the
    success-ratio ``sigma`` schedule and the :meth:`minimize` driver -- is common.

    Parameters
    ----------
    sigma:
        Initial cubic regularisation weight ``> 0`` (larger = smaller, safer steps).
    eta_accept, eta_success:
        Accept a step when the actual/predicted reduction ratio ``rho >= eta_accept``; treat it
        as very successful (shrink ``sigma``) when ``rho >= eta_success``.
    sigma_increase, sigma_decrease:
        Multiply / divide ``sigma`` on a rejected / very-successful step (both ``> 1``).
    min_sigma, max_sigma:
        Clamp range for ``sigma``.
    krylov_dim:
        Lanczos subspace dimension (number of curvature-vector products per step).
    max_line_search:
        Max ``sigma`` increases attempted within one :meth:`step` before giving up.
    """

    def __init__(
        self,
        *,
        sigma: float = 1.0,
        eta_accept: float = 0.1,
        eta_success: float = 0.9,
        sigma_increase: float = 2.0,
        sigma_decrease: float = 2.0,
        min_sigma: float = 1e-8,
        max_sigma: float = 1e16,
        krylov_dim: int = 20,
        max_line_search: int = 12,
    ) -> None:
        if sigma <= 0.0:
            raise ValueError(f"sigma must be > 0, got {sigma}")
        if not 0.0 < eta_accept <= eta_success < 1.0:
            raise ValueError(
                f"need 0 < eta_accept <= eta_success < 1, got {eta_accept}, {eta_success}"
            )
        if sigma_increase <= 1.0:
            raise ValueError(f"sigma_increase must be > 1, got {sigma_increase}")
        if sigma_decrease <= 1.0:
            raise ValueError(f"sigma_decrease must be > 1, got {sigma_decrease}")
        if krylov_dim < 1:
            raise ValueError(f"krylov_dim must be >= 1, got {krylov_dim}")
        if max_line_search < 1:
            raise ValueError(f"max_line_search must be >= 1, got {max_line_search}")
        self.sigma = float(sigma)
        self.eta_accept = float(eta_accept)
        self.eta_success = float(eta_success)
        self.sigma_increase = float(sigma_increase)
        self.sigma_decrease = float(sigma_decrease)
        self.min_sigma = float(min_sigma)
        self.max_sigma = float(max_sigma)
        self.krylov_dim = int(krylov_dim)
        self.max_line_search = int(max_line_search)
        self.n_iter = 0

    def _arc(
        self, params: Tensor, g: Tensor, hess_matvec: MatVec, f0: float, eval_f: Callable[[Tensor], float]
    ) -> tuple[Tensor, bool, float, float, float]:
        return _cubic_arc_step(
            params,
            g,
            hess_matvec,
            f0,
            eval_f,
            self.sigma,
            krylov_dim=self.krylov_dim,
            max_line_search=self.max_line_search,
            eta_accept=self.eta_accept,
            eta_success=self.eta_success,
            sigma_increase=self.sigma_increase,
            sigma_decrease=self.sigma_decrease,
            min_sigma=self.min_sigma,
            max_sigma=self.max_sigma,
        )

    def step(self, fn: Callable[[Tensor], Tensor], params: Tensor) -> tuple[Tensor, CubicNewtonInfo]:
        raise NotImplementedError

    def minimize(
        self, fn: Callable[[Tensor], Tensor], params: Tensor, *, steps: int
    ) -> tuple[Tensor, list[float]]:
        """Run :meth:`step` ``steps`` times; return ``(final_params, loss_history)``."""
        history: list[float] = []
        for _ in range(steps):
            params, info = self.step(fn, params)
            history.append(info.loss)
        return params, history


class CubicRegularizedNewton(_CubicArcOptimizer):
    r"""Adaptive cubic-regularised Newton (ARC) over a scalar ``loss_fn`` -- matrix-free.

    A true higher-order optimiser: each step minimises the cubic model
    ``g^T s + 0.5 s^T H s + (sigma/3) ||s||^3`` (Nesterov-Polyak / Cartis-Gould-Toint ARC),
    with the **exact full** Hessian applied matrix-free (:func:`hvp`) and the subproblem solved
    in a small Lanczos subspace. Unlike Gauss-Newton it uses the full Hessian (curvature term
    included) and unlike Newton it is globally convergent and escapes saddle points -- the
    cubic term is the principled step-size control, so there is no learning rate or
    trust-region radius. ``sigma`` is adapted by the success-ratio test.

    Best for *general* smooth nonconvex objectives (saddle-riddled landscapes). For a
    least-squares / PINN residual prefer :class:`CubicRegularizedGaussNewton`, whose PSD
    Gauss-Newton curvature is the better-suited metric.
    """

    @torch.no_grad()
    def step(self, fn: ScalarFn, params: Tensor) -> tuple[Tensor, CubicNewtonInfo]:
        """One ARC iteration on the full Hessian; ``(new_params, info)`` (unchanged on fail)."""
        grad_fn = grad(fn)
        g = cast(Tensor, grad_fn(params))

        def matvec(v: Tensor) -> Tensor:
            return cast(Tensor, jvp(grad_fn, (params,), (v,))[1])

        f0 = float(fn(params))
        new_params, accepted, self.sigma, rho, loss = self._arc(
            params, g, matvec, f0, lambda p: float(fn(p))
        )
        self.n_iter += 1
        gnorm = float(torch.linalg.vector_norm(g))
        return new_params, CubicNewtonInfo(loss, self.sigma, accepted, rho, gnorm, self.n_iter)


class CubicRegularizedGaussNewton(_CubicArcOptimizer):
    r"""Cubic-regularised Gauss-Newton (ARC on the GN model) over a ``residual_fn`` -- matrix-free.

    The least-squares sibling of :class:`CubicRegularizedNewton`: it minimises the Gauss-Newton
    cubic model ``g^T s + 0.5 s^T (J^T J) s + (sigma/3)||s||^3`` with ``g = J^T r``, using only
    ``J v`` / ``J^T u`` products (the exact omnibias jets, via :func:`_linearize_gn`) so nothing
    dense is ever formed. Versus Levenberg-Marquardt (:class:`GaussNewton`) it replaces the
    ``mu I`` damping with the cubic regulariser -- automatic step-size control and global
    ``O(1/k^2)`` convergence with no damping schedule; versus :class:`CubicRegularizedNewton` it
    uses the PSD GN curvature, the right metric for a least-squares (PINN) objective. This is
    the recommended higher-order optimiser for PINNs.

    The objective is the sum-of-squares ``0.5 ||r||^2`` (pass a ``sqrt(w)``-weighted residual
    via :func:`weighted_residual_fn` to minimise the ``L^2`` quadrature integral instead).
    """

    @torch.no_grad()
    def step(self, fn: ResidualFn, params: Tensor) -> tuple[Tensor, CubicNewtonInfo]:
        """One ARC iteration on the Gauss-Newton model; ``(new_params, info)``.

        The reverse-mode linearisation ``J^T(.)`` is built once and reused for the gradient, the
        curvature products, and (implicitly) the whole subproblem.
        """
        res, jt, jvec = _linearize_gn(fn, params)
        g = jt(res)

        def matvec(v: Tensor) -> Tensor:
            return jt(jvec(v))

        def eval_f(p: Tensor) -> float:
            r = fn(p)
            return 0.5 * float(torch.dot(r, r))

        f0 = 0.5 * float(torch.dot(res, res))
        new_params, accepted, self.sigma, rho, loss = self._arc(params, g, matvec, f0, eval_f)
        self.n_iter += 1
        gnorm = float(torch.linalg.vector_norm(g))
        return new_params, CubicNewtonInfo(loss, self.sigma, accepted, rho, gnorm, self.n_iter)


def taylor_line_min(
    loss_fn: ScalarFn,
    params: Tensor,
    direction: Tensor,
    *,
    order: int = 3,
    bracket: tuple[float, float] = (0.0, 1.0),
) -> tuple[float, Tensor]:
    r"""Exact directional-Taylor line search: minimise the degree-``order`` Taylor model of
    ``phi(a) = loss_fn(params + a * direction)``.

    The along-line derivatives ``phi'(0), phi''(0), phi'''(0)`` are computed *exactly* by
    nested :func:`torch.func.grad` (higher-order autodiff of the scalar restriction) -- for an
    omnibias PINN loss these are exact because the residual jet is closed form. The stationary
    points of the truncated model are found in closed form and the *true* loss is evaluated at
    each candidate (and the bracket ends) to pick the minimiser -- a near-optimal step in one
    shot, no Wolfe/Armijo iteration. Use ``order=2`` for the Newton-along-line step or
    ``order=3`` to exploit the exact third directional derivative.

    Returns ``(a_star, params + a_star * direction)``.
    """
    if order not in (2, 3):
        raise ValueError(f"order must be 2 or 3, got {order}")
    lo, hi = bracket
    if not lo < hi:
        raise ValueError(f"bracket must satisfy lo < hi, got {bracket}")

    def phi(a: Tensor) -> Tensor:
        return loss_fn(params + a * direction)

    a0 = torch.zeros((), dtype=params.dtype, device=params.device)
    d1 = grad(phi)
    c1 = float(cast(Tensor, d1(a0)))
    d2 = grad(d1)
    c2 = float(cast(Tensor, d2(a0)))
    c3 = float(cast(Tensor, grad(d2)(a0))) if order == 3 else 0.0

    candidates: list[float] = [lo, hi]
    if abs(c3) > 1e-30:  # phi'(a) = c1 + c2 a + (c3/2) a^2 = 0
        disc = c2 * c2 - 2.0 * c1 * c3
        if disc >= 0.0:
            root = math.sqrt(disc)
            candidates += [(-c2 + root) / c3, (-c2 - root) / c3]
    elif abs(c2) > 1e-30:
        candidates.append(-c1 / c2)

    best_a, best_f = lo, float("inf")
    for a in candidates:
        if a < lo or a > hi:
            continue
        fa = float(loss_fn(params + a * direction))
        if math.isfinite(fa) and fa < best_f:
            best_f, best_a = fa, a
    return best_a, params + best_a * direction


def taylor_subspace_model(
    loss_fn: ScalarFn,
    params: Tensor,
    basis: Tensor,
    *,
    order: int = 3,
) -> tuple[Tensor, Tensor, Tensor | None]:
    r"""Exact Taylor coefficients of the model restricted to a ``k``-dimensional subspace.

    Given an orthonormal basis ``Q`` (``basis``, shape ``(P, k)``) this returns the exact
    degree-``order`` Taylor model of the scalar restriction
    ``psi(a) = loss_fn(params + Q a)`` about ``a = 0`` -- the k-dimensional generalisation of
    :func:`taylor_line_min` (which is the ``k = 1`` case). The coefficients are

    * ``c = Q^T g`` (shape ``(k,)``) -- the reduced gradient ``grad psi(0)``,
    * ``H = Q^T (grad^2 loss) Q`` (shape ``(k, k)``) -- the reduced Hessian ``grad^2 psi(0)``,
    * ``T`` (shape ``(k, k, k)`` or ``None`` for ``order == 2``) -- the reduced third-derivative
      tensor ``grad^3 psi(0)``, ``T[i, j, l] = grad^3 loss[Q_i, Q_j, Q_l]``.

    All three are taken *exactly* by nested higher-order autodiff of the restriction (one outer
    reverse pass folded through forward-mode Jacobians), so for an omnibias PINN loss they are
    exact (the residual jet is closed form). The reduced model
    ``m(a) = c^T a + 1/2 a^T H a + 1/6 T[a, a, a]`` is what
    :func:`solve_subspace_trust_region` minimises. Because ``k`` is small the ``O(k^2)`` cost of
    the third-derivative tensor is a handful of extra passes, not the ``O(P)`` of a dense tensor.
    """
    if order not in (2, 3):
        raise ValueError(f"order must be 2 or 3, got {order}")
    if basis.ndim != 2 or int(basis.shape[0]) != int(params.shape[0]):
        raise ValueError(
            f"basis must be (P, k) with P = params dim {int(params.shape[0])}, got {tuple(basis.shape)}"
        )
    k = int(basis.shape[1])

    def psi(a: Tensor) -> Tensor:
        return loss_fn(params + basis @ a)

    a0 = torch.zeros(k, dtype=params.dtype, device=params.device)
    grad_psi = grad(psi)
    c = cast(Tensor, grad_psi(a0)).detach()
    hess = cast(Tensor, jacfwd(grad_psi)(a0)).detach()
    hess = 0.5 * (hess + hess.transpose(0, 1))  # symmetrise away round-off
    tensor3: Tensor | None = None
    if order >= 3:
        tensor3 = cast(Tensor, jacfwd(jacfwd(grad_psi))(a0)).detach()
    return c, hess, tensor3


def _solve_tr_dense(g: Tensor, hess: Tensor, radius: float, *, iters: int = 100) -> Tensor:
    r"""Exact dense trust-region step: ``argmin_s g^T s + 0.5 s^T H s s.t. ||s|| <= radius``.

    The Moere-Sorensen solution in the eigenbasis of ``H`` (dense ``eigh`` -- the subspace ``k``
    is tiny). Handles indefinite ``H``: the interior Newton step is taken when ``H`` is positive
    definite and it lies inside the ball, otherwise the secular equation ``||s(lam)|| = radius``
    is solved by bisection over ``lam >= max(0, -lambda_min)`` (so ``H + lam I >= 0``), with the
    hard case (no gradient component on the smallest-eigenvalue direction) resolved by riding that
    eigenvector to the boundary.
    """
    if float(torch.linalg.vector_norm(g)) == 0.0:
        return torch.zeros_like(g)
    evals, vecs = torch.linalg.eigh(hess)
    ghat = vecs.transpose(-1, -2) @ g
    lam_min = float(evals[0])
    eps = 1e-12 + 1e-9 * max(1.0, abs(float(evals[-1])))

    def s_of(lam: float) -> Tensor:
        return cast(Tensor, -(vecs @ (ghat / (evals + lam))))

    if lam_min > eps:  # positive definite: try the interior Newton step
        s_newton = s_of(0.0)
        if float(torch.linalg.vector_norm(s_newton)) <= radius:
            return s_newton
    lam_lo = max(0.0, -lam_min) + eps
    s_lo = s_of(lam_lo)
    if float(torch.linalg.vector_norm(s_lo)) <= radius:  # hard case: ride v_min to the boundary
        v_min = vecs[:, 0]
        tau = _to_boundary(s_lo, v_min, radius)
        return cast(Tensor, s_lo + tau * v_min)
    lo, hi = lam_lo, lam_lo
    for _ in range(200):  # bracket a radius-feasible lambda
        hi = max(2.0 * hi, 1.0)
        if float(torch.linalg.vector_norm(s_of(hi))) < radius:
            break
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        if float(torch.linalg.vector_norm(s_of(mid))) > radius:
            lo = mid
        else:
            hi = mid
    return s_of(0.5 * (lo + hi))


def solve_subspace_trust_region(
    c: Tensor,
    hess: Tensor,
    tensor3: Tensor | None = None,
    *,
    radius: float,
    iters: int = 50,
    tol: float = 1e-10,
) -> Tensor:
    r"""Minimise the subspace Taylor model over the trust region ``||a|| <= radius``.

    Minimises ``m(a) = c^T a + 0.5 a^T H a + (1/6) T[a, a, a]`` (the output of
    :func:`taylor_subspace_model`) subject to ``||a|| <= radius``. For a purely quadratic model
    (``tensor3 is None``) this is one exact dense trust-region solve (:func:`_solve_tr_dense`).
    With the exact third-order tensor the odd-degree model is unbounded below, so the ball
    constraint is essential; it is minimised by a short sequence of quadratic trust-region steps
    (re-linearising ``grad m`` / ``grad^2 m`` at each iterate) safeguarded by a backtracking line
    search on the *true* cubic model and projection back into the ball. ``k`` is tiny, so the whole
    solve is a handful of ``k x k`` eigendecompositions.
    """
    if radius <= 0.0:
        raise ValueError(f"radius must be > 0, got {radius}")
    if tensor3 is None:
        return _solve_tr_dense(c, hess, radius)

    def model(a: Tensor) -> float:
        quad = float(c @ a) + 0.5 * float(a @ (hess @ a))
        cubic = (1.0 / 6.0) * float(torch.einsum("ijl,i,j,l->", tensor3, a, a, a))
        return quad + cubic

    a = torch.zeros(int(c.shape[0]), dtype=c.dtype, device=c.device)
    m_cur = 0.0
    for _ in range(iters):
        ta = torch.einsum("ijl,l->ij", tensor3, a)  # T[:, :, a]
        g_m = c + hess @ a + 0.5 * torch.einsum("ij,j->i", ta, a)  # grad m(a)
        h_m = hess + ta  # grad^2 m(a)
        if float(torch.linalg.vector_norm(g_m)) < tol:
            break
        s = _solve_tr_dense(g_m, h_m, radius)
        a_trial = a + s
        if float(torch.linalg.vector_norm(a_trial)) > radius:  # keep the iterate in the ball
            tau = _to_boundary(a, s, radius)
            a_trial = a + tau * s
        step = a_trial - a
        t = 1.0
        improved = False
        for _ in range(30):  # backtrack on the true cubic model
            cand = a + t * step
            if model(cand) < m_cur - 1e-16:
                a, m_cur = cand, model(cand)
                improved = True
                break
            t *= 0.5
        if not improved:
            break
    return a


@dataclass
class LBFGSInfo:
    """Per-step diagnostics returned by :meth:`JetLBFGS.step`."""

    loss: float
    grad_norm: float
    step_size: float
    n_pairs: int
    n_iter: int


class JetLBFGS:
    r"""Limited-memory BFGS accelerated by omnibias's exact jets.

    Standard L-BFGS fakes second-order information from the history of gradient differences
    ``(s_k, y_k)`` because exact curvature is normally expensive. omnibias makes exact curvature
    *cheap*, so this variant injects it in the two places textbook L-BFGS approximates it, while
    keeping the low-memory two-loop recursion:

    * **Exact line search.** The step length is chosen by :func:`taylor_line_min` -- the exact
      degree-2/3 Taylor model of ``phi(a) = loss(theta + a d)`` along the search direction,
      minimised in closed form -- instead of a Wolfe/Armijo backtrack. (Exact because the
      omnibias residual jet is closed form.)
    * **Exact initial inverse-Hessian scale.** The initial ``H_0 = gamma I`` uses the exact
      curvature along the gradient, ``gamma = <g, g> / <g, H g>`` (one Hessian-vector product,
      :func:`hvp`), instead of the finite-difference scalar ``gamma = <s, y> / <y, y>``. With
      this scale the first step's exact line search lands at ``a = 1`` by construction.

    The rest is classic L-BFGS: the two-loop recursion over the last ``history_size`` curvature
    pairs, with the standard curvature-condition filter ``<s, y> > 0`` and a steepest-descent
    reset if the recursion ever returns a non-descent direction. Functional in the parameters
    (like :class:`GaussNewton` / :class:`CubicRegularizedNewton`); the only state is the pair
    history. This is the honest "can omnibias lift a quasi-Newton method?" baseline -- cheaper
    per step than the full second-order methods, dearer in memory-model fidelity.

    Parameters
    ----------
    history_size:
        Number of ``(s, y)`` curvature pairs retained (the ``m`` of L-BFGS).
    line_search_order:
        Taylor order for :func:`taylor_line_min` (``2`` = exact Newton-along-line, ``3`` adds
        the exact third directional derivative).
    max_step:
        Upper bracket for the line search (the natural quasi-Newton scale is ``1``).
    exact_h0:
        Use the exact-curvature initial scale ``<g,g>/<g,Hg>`` (one ``hvp``); if ``False`` fall
        back to the classic ``<s,y>/<y,y>`` scalar (no ``hvp``).
    curvature_tol:
        Relative threshold for the curvature condition ``<s, y> > curvature_tol * <y, y>`` that
        gates whether a pair is stored (keeps ``H`` positive definite).
    max_backtracks:
        The exact Taylor step is the *initial* trial; it is then Armijo-backtracked (halved)
        up to this many times to guarantee sufficient decrease on strongly non-quadratic
        objectives where the local Taylor model overshoots.
    """

    def __init__(
        self,
        *,
        history_size: int = 10,
        line_search_order: int = 2,
        max_step: float = 10.0,
        exact_h0: bool = True,
        curvature_tol: float = 1e-10,
        max_backtracks: int = 25,
    ) -> None:
        if history_size < 1:
            raise ValueError(f"history_size must be >= 1, got {history_size}")
        if line_search_order not in (2, 3):
            raise ValueError(f"line_search_order must be 2 or 3, got {line_search_order}")
        if max_step <= 0.0:
            raise ValueError(f"max_step must be > 0, got {max_step}")
        if curvature_tol < 0.0:
            raise ValueError(f"curvature_tol must be >= 0, got {curvature_tol}")
        if max_backtracks < 1:
            raise ValueError(f"max_backtracks must be >= 1, got {max_backtracks}")
        self.history_size = int(history_size)
        self.line_search_order = int(line_search_order)
        self.max_step = float(max_step)
        self.exact_h0 = bool(exact_h0)
        self.curvature_tol = float(curvature_tol)
        self.max_backtracks = int(max_backtracks)
        self._s: deque[Tensor] = deque(maxlen=self.history_size)
        self._y: deque[Tensor] = deque(maxlen=self.history_size)
        self._rho: deque[float] = deque(maxlen=self.history_size)
        self.n_iter = 0

    def _line_search(
        self, loss_fn: ScalarFn, params: Tensor, g: Tensor, direction: Tensor, f0: float
    ) -> tuple[float, Tensor]:
        """Exact Taylor step, then Armijo backtrack; returns ``(alpha, params + alpha d)``.

        The exact :func:`taylor_line_min` model minimiser is the initial trial (near-optimal in
        near-quadratic regions -- one shot, no iteration). A standard sufficient-decrease
        backtrack safeguards it where the truncated model overshoots, so a descent direction
        always makes progress; returns ``(0.0, params)`` only if backtracking exhausts.
        """
        g_dot_d = float(torch.dot(g, direction))
        if g_dot_d >= 0.0:
            return 0.0, params
        alpha, _ = taylor_line_min(
            loss_fn, params, direction, order=self.line_search_order, bracket=(0.0, self.max_step)
        )
        if not alpha > 0.0:
            alpha = 1.0
        for _ in range(self.max_backtracks):
            candidate = params + alpha * direction
            f_new = float(loss_fn(candidate))
            if math.isfinite(f_new) and f_new <= f0 + 1e-4 * alpha * g_dot_d:
                return alpha, candidate
            alpha *= 0.5
        return 0.0, params

    def _initial_scale(self, loss_fn: ScalarFn, params: Tensor, g: Tensor) -> float:
        """Initial inverse-Hessian scale ``gamma`` (exact Rayleigh quotient or classic scalar)."""
        if self.exact_h0:
            g_hess = hvp(loss_fn, params, g)
            g_hess_g = float(torch.dot(g, g_hess))
            if g_hess_g > 0.0:
                return float(torch.dot(g, g)) / g_hess_g
        if self._s:
            sy = float(torch.dot(self._s[-1], self._y[-1]))
            yy = float(torch.dot(self._y[-1], self._y[-1]))
            if yy > 0.0:
                return sy / yy
        return 1.0

    @torch.no_grad()
    def step(self, loss_fn: ScalarFn, params: Tensor) -> tuple[Tensor, LBFGSInfo]:
        """One L-BFGS iteration; returns ``(new_params, info)`` (unchanged if no progress)."""
        grad_fn = grad(loss_fn)
        g = cast(Tensor, grad_fn(params))
        gnorm = float(torch.linalg.vector_norm(g))
        self.n_iter += 1
        if gnorm == 0.0 or not math.isfinite(gnorm):
            return params, LBFGSInfo(float(loss_fn(params)), gnorm, 0.0, len(self._s), self.n_iter)

        # two-loop recursion: d = -H g over the stored (s, y) pairs
        s_list, y_list, rho_list = list(self._s), list(self._y), list(self._rho)
        m = len(s_list)
        q = g.clone()
        alpha = [0.0] * m
        for i in range(m - 1, -1, -1):
            alpha[i] = rho_list[i] * float(torch.dot(s_list[i], q))
            q = q - alpha[i] * y_list[i]
        r = self._initial_scale(loss_fn, params, g) * q
        for i in range(m):
            beta = rho_list[i] * float(torch.dot(y_list[i], r))
            r = r + s_list[i] * (alpha[i] - beta)
        direction = -r
        if float(torch.dot(g, direction)) >= 0.0:  # not a descent direction -> reset
            direction = -g
            self._s.clear()
            self._y.clear()
            self._rho.clear()

        f0 = float(loss_fn(params))
        step_size, new_params = self._line_search(loss_fn, params, g, direction, f0)
        if step_size == 0.0:  # line search found no improving step
            return params, LBFGSInfo(f0, gnorm, 0.0, len(self._s), self.n_iter)

        g_new = cast(Tensor, grad_fn(new_params))
        s_k = new_params - params
        y_k = g_new - g
        sy = float(torch.dot(s_k, y_k))
        yy = float(torch.dot(y_k, y_k))
        if sy > self.curvature_tol * yy and yy > 0.0:
            self._s.append(s_k)
            self._y.append(y_k)
            self._rho.append(1.0 / sy)
        loss1 = float(loss_fn(new_params))
        return new_params, LBFGSInfo(loss1, gnorm, step_size, len(self._s), self.n_iter)

    def minimize(
        self, loss_fn: ScalarFn, params: Tensor, *, steps: int
    ) -> tuple[Tensor, list[float]]:
        """Run :meth:`step` ``steps`` times; return ``(final_params, loss_history)``."""
        history: list[float] = []
        for _ in range(steps):
            params, info = self.step(loss_fn, params)
            history.append(info.loss)
        return params, history


def functional_residual_fn(
    module: nn.Module, *args: Any, **kwargs: Any
) -> tuple[Tensor, ResidualFn]:
    """Bridge a residual :class:`torch.nn.Module` to a flat ``residual_fn``.

    ``module.forward(*args, **kwargs)`` must return the stacked residual vector. Returns
    ``(flat0, residual_fn)`` where ``flat0`` is the current parameters as one vector and
    ``residual_fn(vec)`` evaluates the residual with the parameters replaced by ``vec``
    (via :func:`torch.func.functional_call`), so it is differentiable for
    :func:`torch.func.jacrev`.
    """
    names: list[str] = []
    shapes: list[torch.Size] = []
    numels: list[int] = []
    chunks: list[Tensor] = []
    for name, p in module.named_parameters():
        names.append(name)
        shapes.append(p.shape)
        numels.append(p.numel())
        chunks.append(p.detach().reshape(-1))
    flat0 = torch.cat(chunks) if chunks else torch.zeros(0)

    def residual_fn(vec: Tensor) -> Tensor:
        params: dict[str, Tensor] = {}
        offset = 0
        for name, shape, numel in zip(names, shapes, numels, strict=True):
            params[name] = vec[offset : offset + numel].reshape(shape)
            offset += numel
        return cast(Tensor, functional_call(module, params, args, kwargs))

    return flat0, residual_fn


def quadrature_loss(residual: Tensor, weights: Tensor) -> Tensor:
    r"""Integral (function-space) loss :math:`\int_\Omega r^2\,dx \approx \sum_q w_q r_q^2`.

    ``residual`` is a pointwise residual sampled at the nodes of a quadrature rule and
    ``weights`` the matching non-negative rule weights (e.g.
    ``omnibias.fields.gauss_legendre(bounds, n).weights`` as a tensor). Unlike the plain
    mean ``mean(r^2)``, this is a genuine discretised ``L^2`` norm: with a Gauss rule it is
    spectrally accurate and *node-count independent*, giving a smooth, deterministic loss
    landscape instead of the Monte-Carlo estimator's step-to-step jitter.

    Parameters
    ----------
    residual:
        Residual values ``r`` of shape ``(N,)`` at the quadrature nodes.
    weights:
        Quadrature weights ``w`` of shape ``(N,)`` (non-negative).
    """
    if residual.ndim != 1:
        raise ValueError(f"residual must be 1-D (N,), got shape {tuple(residual.shape)}")
    if weights.shape != residual.shape:
        raise ValueError(
            f"weights must match residual shape {tuple(residual.shape)}, "
            f"got {tuple(weights.shape)}"
        )
    return torch.dot(weights.to(residual.dtype), residual**2)


def weighted_residual_fn(residual_fn: ResidualFn, weights: Tensor) -> ResidualFn:
    r"""Turn a pointwise residual map into the ``sqrt(w)``-weighted (``L^2``) residual map.

    Wraps ``residual_fn`` so that its output is scaled elementwise by ``sqrt(weights)``.
    Then the least-squares objective becomes the quadrature integral,
    ``||sqrt(w) r||^2 = sum_q w_q r_q^2`` (see :func:`quadrature_loss`), and feeding the
    wrapped map to :class:`GaussNewton` realises the **integral-weighted natural gradient**:
    the Gauss-Newton matrix is ``J^T diag(w) J``, the *discretised* ``L^2`` function-space
    (energy) metric, rather than the empirical average ``(1/N) J^T J``. Directions are
    unchanged by an overall constant, so this is the principled fix to the metric, not a
    learning-rate rescale.

    Parameters
    ----------
    residual_fn:
        The base residual map ``theta |-> r(theta)`` sampled at the quadrature nodes.
    weights:
        Non-negative quadrature weights of shape ``(N,)`` matching the residual length.
    """
    if weights.ndim != 1:
        raise ValueError(f"weights must be 1-D (N,), got shape {tuple(weights.shape)}")
    if bool(torch.any(weights < 0.0)):
        raise ValueError("weights must be non-negative (they are quadrature weights)")
    sqrt_w = torch.sqrt(weights)

    def wrapped(vec: Tensor) -> Tensor:
        r = residual_fn(vec)
        return r * sqrt_w.to(r.dtype)

    return wrapped


class GradNormBalancer:
    r"""Self-adaptive loss weights equalising per-term gradient norms.

    Maintains an EMA of the balancing weights ``lambda_k``: for each term ``L_k`` it
    measures ``||d L_k / d theta||`` and targets ``lhat_k = ||g_ref|| / (||g_k|| + eps)``
    (Wang-Teng-Perdikaris 2021), so the *weighted* gradient norms ``lambda_k ||g_k||``
    all match the reference term's. This cures the gradient-pathology stiffness that makes
    multi-term PINN losses (PDE residual vs boundary / initial / data) fail to train.

    Parameters
    ----------
    n_terms:
        Number of loss terms.
    alpha:
        EMA retention of the previous weights in ``[0, 1]`` (``0`` = no smoothing).
    ref_index:
        Index of the reference term whose gradient norm the others are balanced to
        (usually the PDE-residual term).
    eps:
        Small constant guarding the division.
    """

    def __init__(
        self, n_terms: int, *, alpha: float = 0.9, ref_index: int = 0, eps: float = 1e-12
    ) -> None:
        if n_terms < 1:
            raise ValueError(f"n_terms must be >= 1, got {n_terms}")
        if not 0.0 <= alpha <= 1.0:
            raise ValueError(f"alpha must be in [0, 1], got {alpha}")
        if not 0 <= ref_index < n_terms:
            raise ValueError(f"ref_index {ref_index} out of range for {n_terms} terms")
        self.n_terms = n_terms
        self.alpha = float(alpha)
        self.ref_index = int(ref_index)
        self.eps = float(eps)
        self.weights = torch.ones(n_terms)

    def grad_norms(self, losses: Sequence[Tensor], params: Iterable[Tensor]) -> Tensor:
        """Per-term L2 norm of ``d L_k / d theta`` (one backward per term)."""
        param_list = [p for p in params if p.requires_grad]
        norms: list[Tensor] = []
        n = len(losses)
        for i, loss in enumerate(losses):
            grads = torch.autograd.grad(
                loss, param_list, retain_graph=(i < n - 1), allow_unused=True, create_graph=False
            )
            sq: Tensor | None = None
            for g in grads:
                if g is not None:
                    term = (g.detach() ** 2).sum()
                    sq = term if sq is None else sq + term
            norms.append(sq.sqrt() if sq is not None else torch.zeros((), dtype=loss.dtype))
        return torch.stack(norms)

    def update(self, losses: Sequence[Tensor], params: Iterable[Tensor]) -> Tensor:
        """Refresh and return the EMA weights from the current per-term gradient norms."""
        if len(losses) != self.n_terms:
            raise ValueError(f"expected {self.n_terms} losses, got {len(losses)}")
        norms = self.grad_norms(losses, params)
        target = (norms[self.ref_index] / (norms + self.eps)).to(self.weights.dtype)
        self.weights = self.alpha * self.weights + (1.0 - self.alpha) * target
        return self.weights.clone()


# --- Drop-in ``torch.optim.Optimizer`` front ends -------------------------
#
# The optimisers above are *functional*: they act on a flat parameter vector plus a
# closure ``fn(vec)`` built with :func:`torch.func` (see :func:`functional_residual_fn`).
# That is the honest scientific-computing interface, but it is not the one a PyTorch user
# reaches for -- they write ``opt = torch.optim.Adam(model.parameters()); opt.step()``.
#
# The classes below wrap the exact-curvature *math* (the ARC engine :func:`_cubic_arc_step`,
# the quasi-Newton recursion, and the diagonal preconditioner) behind the standard
# :class:`torch.optim.Optimizer` contract, so any exact-curvature method is a one-line,
# universal replacement for Adam (Adam is untouched and remains fully available). Curvature
# is obtained *matrix-free* by closure-based double-backward autograd -- exactly the pattern
# of ``omnibias.curvature.torch``: the closure recomputes the loss (or residual) from the
# live parameters, ``torch.autograd.grad(..., create_graph=True)`` gives the gradient, and a
# second reverse pass gives the Hessian-vector product. This works for *any* differentiable
# loss (PINN residual, CNF log-likelihood, a plain supervised objective), not just the
# closed-form omnibias jet -- though the omnibias jet is what makes the curvature *exact*.

Closure = Callable[[], Tensor]


class _CurvatureOptimizer(torch.optim.Optimizer):
    r"""Shared machinery for the drop-in curvature-aware optimisers.

    Gathers every trainable parameter across the param groups into one conceptual flat
    vector (second-order curvature couples all coordinates), and provides the flatten /
    unflatten / write-back plumbing plus the matrix-free gradient and Hessian-vector
    product built by double-backward autograd. Subclasses implement :meth:`step`.

    The closure contract (documented on each subclass' :meth:`step`) is: ``closure()``
    recomputes and returns the objective from the *current* parameters (a scalar loss, or a
    residual vector for the Gauss-Newton method) **with its autograd graph intact**; it does
    not need to call ``.backward()`` -- the optimiser owns differentiation. This mirrors the
    closure convention of :class:`torch.optim.LBFGS`.
    """

    def __init__(self, params: Iterable[Tensor], defaults: dict[str, Any]) -> None:
        super().__init__(params, defaults)
        self._params: list[Tensor] = [
            p for group in self.param_groups for p in group["params"] if p.requires_grad
        ]
        if not self._params:
            raise ValueError("optimizer requires at least one parameter with requires_grad=True")

    def _flat(self, tensors: Sequence[Tensor]) -> Tensor:
        """Concatenate a list of parameter-shaped tensors into one 1-D vector."""
        return torch.cat([t.reshape(-1) for t in tensors])

    def _clone_flat(self) -> Tensor:
        """Detached copy of the current parameters as one flat vector."""
        return self._flat([p.detach() for p in self._params])

    def _unflat(self, vec: Tensor) -> list[Tensor]:
        """Split a flat vector back into parameter-shaped tensors."""
        out: list[Tensor] = []
        offset = 0
        for p in self._params:
            n = p.numel()
            out.append(vec[offset : offset + n].reshape(p.shape))
            offset += n
        return out

    @torch.no_grad()
    def _write_flat(self, vec: Tensor) -> None:
        """Copy a flat vector into the live parameters in place."""
        offset = 0
        for p in self._params:
            n = p.numel()
            p.copy_(vec[offset : offset + n].reshape(p.shape))
            offset += n

    def _grad(self, loss: Tensor, *, create_graph: bool) -> tuple[Tensor, tuple[Tensor, ...]]:
        r"""Return ``(g_flat_detached, g_list)`` -- the gradient of ``loss`` w.r.t. params.

        ``g_list`` keeps the autograd graph iff ``create_graph`` so it can be reused as the
        outputs of a second reverse pass (the Hessian-vector product); ``g_flat_detached`` is
        the plain flat gradient for the linear algebra.
        """
        g_list = torch.autograd.grad(loss, self._params, create_graph=create_graph)
        return self._flat([g.detach() for g in g_list]), g_list

    def _hvp(self, g_list: Sequence[Tensor], v_flat: Tensor) -> Tensor:
        r"""Exact Hessian-vector product ``H v`` (matrix-free, one extra reverse pass).

        ``g_list`` must be the ``create_graph=True`` gradient; differentiating ``g^T v`` w.r.t.
        the parameters yields ``H v``. Returns a detached flat vector.
        """
        v_list = self._unflat(v_flat)
        hv = torch.autograd.grad(
            list(g_list), self._params, grad_outputs=v_list, retain_graph=True
        )
        return self._flat([h.detach() for h in hv])


class CubicNewton(_CurvatureOptimizer):
    r"""Drop-in adaptive cubic-regularised Newton (ARC) on the **exact full Hessian**.

    The :class:`torch.optim.Optimizer` front end for :class:`CubicRegularizedNewton`: each
    step minimises the cubic model ``g^T s + 0.5 s^T H s + (sigma/3)||s||^3`` in a Lanczos
    subspace, with ``H`` applied matrix-free by double-backward autograd and ``sigma`` adapted
    by the success-ratio test. Being a true higher-order method it is globally convergent and
    saddle-escaping -- there is no learning rate. Best for general smooth nonconvex objectives;
    for a least-squares / PINN residual prefer :class:`CubicGaussNewton`.

    Usage (a one-line swap for Adam)::

        opt = CubicNewton(model.parameters())
        def closure() -> torch.Tensor:
            return loss_fn(model)          # scalar loss, graph intact, no .backward()
        for _ in range(n_steps):
            opt.step(closure)
    """

    def __init__(
        self,
        params: Iterable[Tensor],
        *,
        sigma: float = 1.0,
        eta_accept: float = 0.1,
        eta_success: float = 0.9,
        sigma_increase: float = 2.0,
        sigma_decrease: float = 2.0,
        min_sigma: float = 1e-8,
        max_sigma: float = 1e16,
        krylov_dim: int = 20,
        max_line_search: int = 12,
    ) -> None:
        if sigma <= 0.0:
            raise ValueError(f"sigma must be > 0, got {sigma}")
        if not 0.0 < eta_accept <= eta_success < 1.0:
            raise ValueError(f"need 0 < eta_accept <= eta_success < 1, got {eta_accept}, {eta_success}")
        if sigma_increase <= 1.0 or sigma_decrease <= 1.0:
            raise ValueError("sigma_increase and sigma_decrease must be > 1")
        if krylov_dim < 1:
            raise ValueError(f"krylov_dim must be >= 1, got {krylov_dim}")
        if max_line_search < 1:
            raise ValueError(f"max_line_search must be >= 1, got {max_line_search}")
        super().__init__(params, {})
        self._sigma = float(sigma)
        self.eta_accept = float(eta_accept)
        self.eta_success = float(eta_success)
        self.sigma_increase = float(sigma_increase)
        self.sigma_decrease = float(sigma_decrease)
        self.min_sigma = float(min_sigma)
        self.max_sigma = float(max_sigma)
        self.krylov_dim = int(krylov_dim)
        self.max_line_search = int(max_line_search)
        self.n_iter = 0

    def step(self, closure: Closure | None = None) -> Tensor:  # type: ignore[override]
        """One ARC iteration on the exact full Hessian; ``closure`` returns the scalar loss."""
        if closure is None:
            raise ValueError("CubicNewton.step requires a closure returning the scalar loss")
        loss = closure()
        g, g_list = self._grad(loss, create_graph=True)
        params0 = self._clone_flat()

        def matvec(v: Tensor) -> Tensor:
            return self._hvp(g_list, v)

        def eval_f(trial: Tensor) -> float:
            self._write_flat(trial)
            return float(closure().detach())

        f0 = float(loss.detach())
        new_params, _accepted, self._sigma, _rho, _new_loss = _cubic_arc_step(
            params0,
            g,
            matvec,
            f0,
            eval_f,
            self._sigma,
            krylov_dim=self.krylov_dim,
            max_line_search=self.max_line_search,
            eta_accept=self.eta_accept,
            eta_success=self.eta_success,
            sigma_increase=self.sigma_increase,
            sigma_decrease=self.sigma_decrease,
            min_sigma=self.min_sigma,
            max_sigma=self.max_sigma,
        )
        self._write_flat(new_params)
        self.n_iter += 1
        return loss.detach()


class CubicGaussNewton(_CurvatureOptimizer):
    r"""Drop-in cubic-regularised Gauss-Newton (ARC on the PSD GN metric) for least squares.

    The :class:`torch.optim.Optimizer` front end for :class:`CubicRegularizedGaussNewton`, and
    the recommended higher-order optimiser for PINNs: it minimises the Gauss-Newton cubic model
    ``g^T s + 0.5 s^T (J^T J) s + (sigma/3)||s||^3`` with ``g = J^T r`` using only matrix-free
    ``J v`` / ``J^T u`` products (double-backward autograd on the residual). The objective is
    the mean-square residual ``0.5 * mean(r^2)``.

    The ``closure`` returns the **residual vector** ``r`` (not a scalar), with its graph intact::

        opt = CubicGaussNewton(model.parameters())
        def closure() -> torch.Tensor:
            return residual_vector(model)   # shape (N,), graph intact
        for _ in range(n_steps):
            opt.step(closure)
    """

    def __init__(
        self,
        params: Iterable[Tensor],
        *,
        sigma: float = 1.0,
        eta_accept: float = 0.1,
        eta_success: float = 0.9,
        sigma_increase: float = 2.0,
        sigma_decrease: float = 2.0,
        min_sigma: float = 1e-8,
        max_sigma: float = 1e16,
        krylov_dim: int = 20,
        max_line_search: int = 12,
    ) -> None:
        if sigma <= 0.0:
            raise ValueError(f"sigma must be > 0, got {sigma}")
        if not 0.0 < eta_accept <= eta_success < 1.0:
            raise ValueError(f"need 0 < eta_accept <= eta_success < 1, got {eta_accept}, {eta_success}")
        if sigma_increase <= 1.0 or sigma_decrease <= 1.0:
            raise ValueError("sigma_increase and sigma_decrease must be > 1")
        if krylov_dim < 1:
            raise ValueError(f"krylov_dim must be >= 1, got {krylov_dim}")
        if max_line_search < 1:
            raise ValueError(f"max_line_search must be >= 1, got {max_line_search}")
        super().__init__(params, {})
        self._sigma = float(sigma)
        self.eta_accept = float(eta_accept)
        self.eta_success = float(eta_success)
        self.sigma_increase = float(sigma_increase)
        self.sigma_decrease = float(sigma_decrease)
        self.min_sigma = float(min_sigma)
        self.max_sigma = float(max_sigma)
        self.krylov_dim = int(krylov_dim)
        self.max_line_search = int(max_line_search)
        self.n_iter = 0

    def step(self, closure: Closure | None = None) -> Tensor:  # type: ignore[override]
        """One ARC iteration on the GN metric; ``closure`` returns the residual vector."""
        if closure is None:
            raise ValueError("CubicGaussNewton.step requires a closure returning the residual vector")
        res = closure()
        if res.ndim != 1:
            raise ValueError(f"residual closure must return a 1-D vector, got shape {tuple(res.shape)}")
        n_res = res.shape[0]
        res_det = res.detach()
        # g = (1/N) J^T r
        jt_r = torch.autograd.grad(res, self._params, grad_outputs=res_det, retain_graph=True)
        g = self._flat([t.detach() for t in jt_r]) / n_res
        # J^T u as a function of a dummy u (create_graph) enables J v by a second reverse pass.
        u = torch.zeros(n_res, dtype=res.dtype, device=res.device, requires_grad=True)
        jt_u = self._flat(torch.autograd.grad(res, self._params, grad_outputs=u, create_graph=True))

        def matvec(v: Tensor) -> Tensor:
            jv = torch.autograd.grad(jt_u, u, grad_outputs=v, retain_graph=True)[0].detach()
            jtjv = torch.autograd.grad(res, self._params, grad_outputs=jv, retain_graph=True)
            return self._flat([t.detach() for t in jtjv]) / n_res

        params0 = self._clone_flat()

        def eval_f(trial: Tensor) -> float:
            self._write_flat(trial)
            r = closure().detach()
            return 0.5 * float((r**2).mean())

        f0 = 0.5 * float((res_det**2).mean())
        new_params, _accepted, self._sigma, _rho, _new_loss = _cubic_arc_step(
            params0,
            g,
            matvec,
            f0,
            eval_f,
            self._sigma,
            krylov_dim=self.krylov_dim,
            max_line_search=self.max_line_search,
            eta_accept=self.eta_accept,
            eta_success=self.eta_success,
            sigma_increase=self.sigma_increase,
            sigma_decrease=self.sigma_decrease,
            min_sigma=self.min_sigma,
            max_sigma=self.max_sigma,
        )
        self._write_flat(new_params)
        self.n_iter += 1
        return torch.as_tensor(f0, dtype=res.dtype, device=res.device)


MetricProvider = Callable[[Tensor], Tensor]


class NaturalGradient(_CurvatureOptimizer):
    r"""Drop-in metric-preconditioned (natural-gradient / Riemannian) gradient descent.

    Steps along the natural gradient ``delta = (M(theta) + damping I)^{-1} g`` rather than the
    raw gradient ``g``, with a **pluggable Riemannian metric** ``M``:

    * ``metric=None`` -- identity metric, i.e. ordinary backtracked gradient descent.
    * a dense provider ``metric(theta) -> (P, P)`` returning an SPD matrix at the current
      flat parameters. Two closed-form metrics drop straight in (built by the caller, so this
      module never imports ``geometry`` / ``curvature``):

      - **Fisher / Gauss-Newton** ``F = (1/N) J^T J`` via :func:`gauss_newton_fisher`
        (natural gradient == Fisher scoring; when ``F`` is the Hessian, e.g. a GLM or a
        residual linear in ``theta``, this is Newton and recovers the minimiser of a
        quadratic in a single step), and
      - **geometry pullback** ``g = J^T h J`` via
        ``omnibias.geometry.torch.ops.pullback_metric`` -- a learned-chart Riemannian metric,
        e.g. ``metric=lambda th: pullback_metric(th.reshape(1, -1), chart)[0]``.

    The metric is deliberately decoupled from the loss gradient (as a Riemannian metric
    should be), so any SPD ``M`` is admissible. A backtracking line search (halving from
    ``lr``) guarantees monotone descent; ``damping >= 0`` keeps the metric solve well-posed.
    Set ``target_condition`` (a dense metric only) to instead *floor* the damping at the
    smallest certified ``eps`` with ``kappa(M + eps I) <= target_condition`` -- the
    ``eps -> 0`` rank/regularization collapse via
    :func:`omnibias.core.verified.conditioning.certified_damping`; the sealed conditioning
    certificate lands on ``last_certificate``.
    The bit-identical functional twin lives in :mod:`omnibias.jax.optim`
    (:func:`~omnibias.jax.optim.natural_gradient_step`).

    The ``closure`` returns the **scalar loss** with its autograd graph intact::

        opt = NaturalGradient(model.parameters(), metric=my_metric_provider)
        def closure() -> torch.Tensor:
            return loss_fn(model)            # scalar, graph intact, no .backward()
        for _ in range(n_steps):
            opt.step(closure)
    """

    def __init__(
        self,
        params: Iterable[Tensor],
        *,
        metric: MetricProvider | None = None,
        lr: float = 1.0,
        damping: float = 1e-3,
        max_line_search: int = 20,
        cg_max_iter: int = 100,
        cg_tol: float = 1e-8,
        target_condition: float | None = None,
    ) -> None:
        if lr <= 0.0:
            raise ValueError(f"lr must be > 0, got {lr}")
        if damping < 0.0:
            raise ValueError(f"damping must be >= 0, got {damping}")
        if max_line_search < 1:
            raise ValueError(f"max_line_search must be >= 1, got {max_line_search}")
        if target_condition is not None and target_condition <= 1.0:
            raise ValueError(f"target_condition must be > 1, got {target_condition}")
        super().__init__(params, {})
        self.metric = metric
        self.lr = float(lr)
        self.damping = float(damping)
        self.max_line_search = int(max_line_search)
        self.cg_max_iter = int(cg_max_iter)
        self.cg_tol = float(cg_tol)
        self.target_condition = None if target_condition is None else float(target_condition)
        self.last_certificate: dict[str, Any] | None = None
        self.n_iter = 0

    def step(self, closure: Closure | None = None) -> Tensor:  # type: ignore[override]
        """One natural-gradient iteration; ``closure`` returns the scalar loss."""
        if closure is None:
            raise ValueError("NaturalGradient.step requires a closure returning the scalar loss")
        loss = closure()
        if loss.ndim != 0:
            raise ValueError(f"loss closure must return a scalar, got shape {tuple(loss.shape)}")
        g, _ = self._grad(loss, create_graph=False)
        params0 = self._clone_flat()
        f0 = float(loss.detach())

        if self.metric is None:
            delta = g
        else:
            m = self.metric(params0)
            damping = self.damping
            if self.target_condition is not None and isinstance(m, Tensor):
                # Certified-conditioning floor on the dense metric (eps -> 0 collapse).
                eps_cert, self.last_certificate = _certified_damping_floor(
                    m, self.target_condition
                )
                damping = max(self.damping, eps_cert)
            delta = natural_gradient_direction(
                m, g, damping=damping, cg_max_iter=self.cg_max_iter, cg_tol=self.cg_tol
            )

        alpha = self.lr
        for _ in range(self.max_line_search):
            self._write_flat(params0 - alpha * delta)
            with torch.no_grad():
                f1 = float(closure().detach())
            if math.isfinite(f1) and f1 < f0:
                self.n_iter += 1
                return torch.as_tensor(f0, dtype=loss.dtype, device=loss.device)
            alpha *= 0.5
        self._write_flat(params0)  # no decrease found on this ray: stay put
        self.n_iter += 1
        return torch.as_tensor(f0, dtype=loss.dtype, device=loss.device)


class JetLBFGSOptimizer(_CurvatureOptimizer):
    r"""Drop-in limited-memory BFGS with an exact-curvature initial scale and line search.

    The :class:`torch.optim.Optimizer` front end for :class:`JetLBFGS`: the classic two-loop
    recursion over the last ``history_size`` curvature pairs, but with the initial inverse-
    Hessian scale set by the *exact* curvature ``<g,g>/<g,Hg>`` (one Hessian-vector product)
    and the step chosen by the exact quadratic (Newton-along-line) model ``a* = -<g,d>/<d,Hd>``
    (one more product), safeguarded by an Armijo backtrack. ``closure`` returns the scalar loss.
    """

    def __init__(
        self,
        params: Iterable[Tensor],
        *,
        history_size: int = 10,
        max_step: float = 10.0,
        exact_h0: bool = True,
        curvature_tol: float = 1e-10,
        max_backtracks: int = 25,
    ) -> None:
        if history_size < 1:
            raise ValueError(f"history_size must be >= 1, got {history_size}")
        if max_step <= 0.0:
            raise ValueError(f"max_step must be > 0, got {max_step}")
        if curvature_tol < 0.0:
            raise ValueError(f"curvature_tol must be >= 0, got {curvature_tol}")
        if max_backtracks < 1:
            raise ValueError(f"max_backtracks must be >= 1, got {max_backtracks}")
        super().__init__(params, {})
        self.history_size = int(history_size)
        self.max_step = float(max_step)
        self.exact_h0 = bool(exact_h0)
        self.curvature_tol = float(curvature_tol)
        self.max_backtracks = int(max_backtracks)
        self._s: deque[Tensor] = deque(maxlen=self.history_size)
        self._y: deque[Tensor] = deque(maxlen=self.history_size)
        self._rho: deque[float] = deque(maxlen=self.history_size)
        self.n_iter = 0

    def _two_loop(self, g: Tensor, gamma: float) -> Tensor:
        """``d = -H g`` by the L-BFGS two-loop recursion over the stored ``(s, y)`` pairs."""
        s_list, y_list, rho_list = list(self._s), list(self._y), list(self._rho)
        m = len(s_list)
        q = g.clone()
        alpha = [0.0] * m
        for i in range(m - 1, -1, -1):
            alpha[i] = rho_list[i] * float(torch.dot(s_list[i], q))
            q = q - alpha[i] * y_list[i]
        r = gamma * q
        for i in range(m):
            beta = rho_list[i] * float(torch.dot(y_list[i], r))
            r = r + s_list[i] * (alpha[i] - beta)
        return -r

    def step(self, closure: Closure | None = None) -> Tensor:  # type: ignore[override]
        """One L-BFGS iteration; ``closure`` returns the scalar loss."""
        if closure is None:
            raise ValueError("JetLBFGSOptimizer.step requires a closure returning the scalar loss")
        loss = closure()
        g, g_list = self._grad(loss, create_graph=True)
        gnorm = float(torch.linalg.vector_norm(g))
        self.n_iter += 1
        if gnorm == 0.0 or not math.isfinite(gnorm):
            return loss.detach()

        # exact initial inverse-Hessian scale gamma = <g,g> / <g, H g>
        gamma = 1.0
        if self.exact_h0:
            g_hess = self._hvp(g_list, g)
            ghg = float(torch.dot(g, g_hess))
            if ghg > 0.0:
                gamma = float(torch.dot(g, g)) / ghg
        elif self._s:
            yy = float(torch.dot(self._y[-1], self._y[-1]))
            if yy > 0.0:
                gamma = float(torch.dot(self._s[-1], self._y[-1])) / yy

        direction = self._two_loop(g, gamma)
        gd = float(torch.dot(g, direction))
        if gd >= 0.0:  # not a descent direction -> reset to steepest descent
            direction = -g
            gd = -float(torch.dot(g, g))
            self._s.clear()
            self._y.clear()
            self._rho.clear()

        # exact quadratic line search: a* = -<g,d> / <d, H d>
        dhd = float(torch.dot(direction, self._hvp(g_list, direction)))
        alpha = -gd / dhd if dhd > 0.0 else 1.0
        if not alpha > 0.0:
            alpha = 1.0
        alpha = min(alpha, self.max_step)

        params0 = self._clone_flat()
        f0 = float(loss.detach())
        accepted = False
        for _ in range(self.max_backtracks):
            trial = params0 + alpha * direction
            self._write_flat(trial)
            f_new = float(closure().detach())
            if math.isfinite(f_new) and f_new <= f0 + 1e-4 * alpha * gd:
                accepted = True
                break
            alpha *= 0.5
        if not accepted:
            self._write_flat(params0)
            return loss.detach()

        new_params = params0 + alpha * direction
        g_new, _ = self._grad(closure(), create_graph=False)
        s_k = new_params - params0
        y_k = g_new - g
        sy = float(torch.dot(s_k, y_k))
        yy = float(torch.dot(y_k, y_k))
        if sy > self.curvature_tol * yy and yy > 0.0:
            self._s.append(s_k)
            self._y.append(y_k)
            self._rho.append(1.0 / sy)
        self._write_flat(new_params)
        return loss.detach()


class JetSubspaceTensor(_CurvatureOptimizer):
    r"""Drop-in **exact third-order tensor method in a Krylov subspace** -- matrix-free.

    Cubic-regularised Newton and trust-region Newton keep a *second*-order (Hessian) model and
    control the step with a cubic regulariser / trust radius. This optimiser instead builds the
    **exact third-order** Taylor model of the loss restricted to a small ``k``-dimensional Krylov
    subspace ``Q`` (a genuine tensor method, Nesterov high-order), which omnibias makes practical
    because the higher directional derivatives are exact and cheap (the residual jet is closed
    form). Each step:

    1. builds an orthonormal Krylov basis ``Q = span{g, H g, H^2 g, ...}`` by Lanczos on the exact
       Hessian applied matrix-free (:meth:`_hvp`) -- ``subspace_dim`` Hessian-vector products, which
       also yield the reduced Hessian ``H_sub = Q^T H Q`` (the Lanczos tridiagonal) for free;
    2. forms the reduced gradient ``c = Q^T g`` and, for ``order = 3``, the exact reduced third-order
       tensor ``T_sub[i, j, l] = grad^3 loss[Q_i, Q_j, Q_l]`` by one more level of autodiff
       (``O(subspace_dim^2)`` cheap reduced products, not the ``O(P)`` of a dense tensor);
    3. minimises the reduced cubic model ``c^T a + 0.5 a^T H_sub a + (1/6) T_sub[a, a, a]`` over the
       trust region ``||a|| <= radius`` (:func:`solve_subspace_trust_region`) and takes the step
       ``theta + Q a``, accepted / rejected by the usual actual-to-predicted reduction ratio with an
       adaptive radius.

    The thesis (see ``docs/benchmarks.md``): the per-step curvature win of the exact-Hessian methods
    only converts to a *wall-clock* win when the inner solve is cheap; a small fixed subspace with an
    exact third-order model needs far fewer steps than a long CG inner loop, at a fixed handful of
    products per step. Set ``order = 2`` to fall back to a pure subspace (Krylov) Newton trust-region
    step (no third-order tensor), which is the cheaper safe baseline.

    Usage (a one-line swap for Adam)::

        opt = JetSubspaceTensor(model.parameters())
        def closure() -> torch.Tensor:
            return loss_fn(model)          # scalar loss, graph intact, no .backward()
        for _ in range(n_steps):
            opt.step(closure)
    """

    def __init__(
        self,
        params: Iterable[Tensor],
        *,
        subspace_dim: int = 5,
        order: int = 3,
        radius: float = 1.0,
        max_radius: float = 1e3,
        min_radius: float = 1e-12,
        eta: float = 0.1,
    ) -> None:
        if subspace_dim < 1:
            raise ValueError(f"subspace_dim must be >= 1, got {subspace_dim}")
        if order not in (2, 3):
            raise ValueError(f"order must be 2 or 3, got {order}")
        if radius <= 0.0:
            raise ValueError(f"radius must be > 0, got {radius}")
        if max_radius < radius:
            raise ValueError(f"max_radius must be >= radius, got {max_radius} < {radius}")
        if min_radius <= 0.0:
            raise ValueError(f"min_radius must be > 0, got {min_radius}")
        if not 0.0 <= eta < 0.25:
            raise ValueError(f"eta (accept threshold) must be in [0, 0.25), got {eta}")
        super().__init__(params, {})
        self.subspace_dim = int(subspace_dim)
        self.order = int(order)
        self._radius = float(radius)
        self.max_radius = float(max_radius)
        self.min_radius = float(min_radius)
        self.eta = float(eta)
        self.n_iter = 0
        self.last_hvps = 0  # curvature products used by the most recent step (matrix-free)

    @property
    def radius(self) -> float:
        """The current trust-region radius (adapted by the reduction-ratio test)."""
        return self._radius

    def _subspace_third_order(self, g_list: Sequence[Tensor], basis: Tensor) -> Tensor:
        r"""Reduced third-order tensor ``T[i, j, l] = grad^3 loss[Q_i, Q_j, Q_l]``.

        ``g_list`` is the ``create_graph=True`` gradient. For each basis column ``q_j`` the exact
        ``H q_j`` is recomputed *with graph* (differentiating ``g^T q_j``), then differentiated once
        more against ``q_l`` to give ``grad^3 loss[., q_j, q_l]``; projecting onto ``Q`` yields the
        reduced tensor. Exploits full symmetry of the third derivative (only ``j <= l`` are formed).
        """
        m = int(basis.shape[1])
        cols = [self._unflat(basis[:, j].contiguous()) for j in range(m)]
        hcols_graph: list[tuple[Tensor, ...]] = []
        for j in range(m):
            hv = torch.autograd.grad(
                list(g_list), self._params, grad_outputs=cols[j], create_graph=True, retain_graph=True
            )
            hcols_graph.append(hv)
        tensor3 = torch.zeros(m, m, m, dtype=basis.dtype, device=basis.device)
        for j in range(m):
            # H q_j is linear in theta iff its blocks carry a graph; a block that does not (e.g. a
            # purely quadratic loss) has a zero third derivative and is simply dropped from the sum.
            live = [h.requires_grad for h in hcols_graph[j]]
            if not any(live):
                continue
            outs = [h for h, keep in zip(hcols_graph[j], live, strict=True) if keep]
            for el in range(j, m):
                gouts = [go for go, keep in zip(cols[el], live, strict=True) if keep]
                tjl = torch.autograd.grad(
                    outs, self._params, grad_outputs=gouts, retain_graph=True, allow_unused=True
                )
                tjl_flat = self._flat(
                    [t.detach() if t is not None else torch.zeros_like(p) for t, p in zip(tjl, self._params, strict=True)]
                )
                proj = basis.t() @ tjl_flat
                tensor3[:, j, el] = proj
                if el != j:
                    tensor3[:, el, j] = proj
        return tensor3

    def step(self, closure: Closure | None = None) -> Tensor:  # type: ignore[override]
        """One subspace tensor-method iteration; ``closure`` returns the scalar loss."""
        if closure is None:
            raise ValueError("JetSubspaceTensor.step requires a closure returning the scalar loss")
        loss = closure()
        g, g_list = self._grad(loss, create_graph=True)
        self.n_iter += 1
        gnorm = float(torch.linalg.vector_norm(g))
        if gnorm == 0.0 or not math.isfinite(gnorm):
            return loss.detach()

        hvps = 0

        def matvec(v: Tensor) -> Tensor:
            nonlocal hvps
            hvps += 1
            return self._hvp(g_list, v)

        # All curvature products happen *before* any trial writeback (the retained graph is tied to
        # the current parameters): Krylov basis + reduced Hessian (Lanczos), then the reduced tensor.
        basis, tri = lanczos_tridiag(matvec, g, self.subspace_dim)
        c = basis.t() @ g
        tensor3 = self._subspace_third_order(g_list, basis) if self.order >= 3 else None
        self.last_hvps = hvps

        a = solve_subspace_trust_region(c, tri, tensor3, radius=self._radius)
        p = basis @ a
        predicted = -(
            float(c @ a)
            + 0.5 * float(a @ (tri @ a))
            + ((1.0 / 6.0) * float(torch.einsum("ijl,i,j,l->", tensor3, a, a, a)) if tensor3 is not None else 0.0)
        )

        params0 = self._clone_flat()
        f0 = float(loss.detach())
        self._write_flat(params0 + p)
        f1 = float(closure().detach())
        actual = f0 - f1
        rho = actual / predicted if predicted > 0.0 and math.isfinite(f1) else float("-inf")

        hit_boundary = float(torch.linalg.vector_norm(a)) >= self._radius * (1.0 - 1e-6)
        if rho < 0.25:
            self._radius = max(0.25 * self._radius, self.min_radius)
        elif rho > 0.75 and hit_boundary:
            self._radius = min(2.0 * self._radius, self.max_radius)

        if not (rho > self.eta):  # reject: restore the pre-step parameters
            self._write_flat(params0)
        return loss.detach()


class _DiagonalCurvatureMixin(_CurvatureOptimizer):
    r"""Shared exact-curvature-*diagonal* machinery (Gauss-Newton / Hutchinson).

    Factors the exact curvature-diagonal estimators, their EMA accumulation, and the
    bias-corrected + condition-number-floored effective diagonal out of
    :class:`DiagonalCurvature` so :class:`ConformalSymplectic` can reuse the *identical*
    (exact, hardened) curvature signal -- as an Adam-style preconditioner in the former and
    as the diagonal mass matrix ``M`` of a Hamiltonian flow in the latter. Subclasses set
    the attributes declared below in their ``__init__`` and drive the estimators from their
    own ``step``.
    """

    # Attributes owned by the concrete subclass (declared here for the type checker only).
    beta2: float
    eps: float
    rel_floor: float
    bias_correction: bool
    hutchinson_samples: int
    curvature_subsample: int | None
    _d: Tensor | None
    _d_updates: int
    _gen: torch.Generator | None

    def _gn_diagonal(self, residual: Tensor, template: Tensor) -> Tensor:
        """Exact Gauss-Newton diagonal ``(1/m) sum_n (grad r_n)^2`` over ``m`` (sub)sampled rows.

        With ``curvature_subsample`` set, ``m`` random rows are drawn per refresh (a persistent
        generator so successive refreshes resample); the ``1/m`` normalisation keeps it an
        unbiased estimate of the full row-mean ``(1/N) sum_n``.
        """
        n_res = int(residual.shape[0])
        if self.curvature_subsample is not None and self.curvature_subsample < n_res:
            if self._gen is None:
                self._gen = torch.Generator().manual_seed(0)
            perm = torch.randperm(n_res, generator=self._gen)[: self.curvature_subsample]
            rows = [int(i) for i in perm.tolist()]
        else:
            rows = list(range(n_res))
        acc = torch.zeros_like(template)
        for n in rows:
            e = torch.zeros(n_res, dtype=residual.dtype, device=residual.device)
            e[n] = 1.0
            row = self._flat(torch.autograd.grad(residual, self._params, grad_outputs=e, retain_graph=True))
            acc = acc + row * row
        return acc / float(len(rows))

    def _hutchinson_diagonal(self, g_list: tuple[Tensor, ...], template: Tensor) -> Tensor:
        """Hutchinson diagonal ``E_z[z (.) H z]`` over Rademacher probes (exact HVPs)."""
        acc = torch.zeros_like(template)
        for _ in range(self.hutchinson_samples):
            z = torch.empty_like(template).bernoulli_(0.5).mul_(2.0).sub_(1.0)
            hz = self._hvp(g_list, z)
            acc = acc + z * hz
        return acc / self.hutchinson_samples

    def _update_diagonal_ema(self, h: Tensor) -> None:
        """EMA-accumulate the curvature diagonal ``h`` into ``self._d`` (zero- or warm-init)."""
        if self.bias_correction:
            self._d = (1.0 - self.beta2) * h if self._d is None else self.beta2 * self._d + (1.0 - self.beta2) * h
        else:
            self._d = h if self._d is None else self.beta2 * self._d + (1.0 - self.beta2) * h
        self._d_updates += 1

    def _effective_diagonal(self) -> Tensor:
        """Bias-corrected, condition-number-floored diagonal used to precondition the step."""
        assert self._d is not None
        d_hat = self._d
        if self.bias_correction and self._d_updates > 0:
            d_hat = d_hat / (1.0 - self.beta2**self._d_updates)
        if self.rel_floor > 0.0:
            d_hat = d_hat.clamp_min(self.rel_floor * float(d_hat.max()))
        return d_hat + self.eps


class DiagonalCurvature(_DiagonalCurvatureMixin):
    r"""Exact-curvature **diagonal preconditioner** -- the cheap, scalable Adam substitute.

    A first-order-cost optimiser that preconditions the (momentum) gradient by an EMA of a
    curvature *diagonal* ``d``:

    .. math::

        \theta \leftarrow \theta - \eta\, \operatorname{clip}\!\big(m / (|d| + \epsilon)\big),

    where ``m`` is the EMA gradient. Unlike Sophia / AdaHessian, whose diagonal is a
    mini-batch or purely stochastic estimate, omnibias supplies an **exact** curvature signal:

    * ``curvature="gauss_newton"`` -- the exact Gauss-Newton diagonal
      ``d_i = (1/N) sum_n (d r_n / d theta_i)^2`` from the closed-form residual Jacobian
      (PSD, the right metric for a least-squares / PINN objective). The ``closure`` returns the
      residual vector ``r``.
    * ``curvature="hutchinson"`` -- the Hutchinson diagonal ``d = E_z[z (.) H z]`` of the full
      Hessian using *exact* matrix-free Hessian-vector products (double backward). Unbiased in
      expectation; the HVP itself is exact. The ``closure`` returns the scalar loss.

    The curvature is refreshed every ``curvature_every`` steps and reused in between (the
    Sophia amortisation), so the amortised per-step cost is close to a first-order method.

    **Robustness (the hardened defaults).** A raw diagonal preconditioner wanders or diverges
    when the curvature is badly anisotropic (some coordinates near-zero curvature -> outsized
    steps). Four safeguards, on by default where they are free:

    * ``rel_floor`` -- floor the effective diagonal at ``rel_floor * max(d)``, capping the
      preconditioner's condition number (default ``1e-2`` = 100x). This is the key stabiliser;
      it only shrinks steps in near-flat directions and never touches the stored ``_d``.
    * ``bias_correction`` -- Adam-style cold-start correction on the ``m``/``d`` EMAs (default
      ``True``), so the first steps are not skewed by an empty running average.
    * ``curvature_subsample`` -- estimate the exact Gauss-Newton diagonal from a random subset
      of ``k`` residual rows per refresh (unbiased for the row-mean), turning the ``O(N)``
      refresh into ``O(k)`` -- this is what lets it scale to large residuals.
    * ``safeguard`` -- optionally reject / backtrack a step that would raise the loss (costs
      extra closure evaluations; off by default to preserve the first-order per-step cost).
    """

    def __init__(
        self,
        params: Iterable[Tensor],
        *,
        lr: float = 1e-1,
        curvature: Literal["gauss_newton", "hutchinson"] = "gauss_newton",
        beta1: float = 0.9,
        beta2: float = 0.99,
        eps: float = 1e-8,
        clip: float | None = None,
        curvature_every: int = 1,
        hutchinson_samples: int = 1,
        rel_floor: float = 1e-2,
        bias_correction: bool = True,
        curvature_subsample: int | None = None,
        safeguard: bool = False,
        safeguard_backtracks: int = 5,
    ) -> None:
        if lr <= 0.0:
            raise ValueError(f"lr must be > 0, got {lr}")
        if curvature not in ("gauss_newton", "hutchinson"):
            raise ValueError(f"curvature must be 'gauss_newton' or 'hutchinson', got {curvature!r}")
        if not 0.0 <= beta1 < 1.0:
            raise ValueError(f"beta1 must be in [0, 1), got {beta1}")
        if not 0.0 <= beta2 < 1.0:
            raise ValueError(f"beta2 must be in [0, 1), got {beta2}")
        if eps <= 0.0:
            raise ValueError(f"eps must be > 0, got {eps}")
        if clip is not None and clip <= 0.0:
            raise ValueError(f"clip must be > 0 or None, got {clip}")
        if curvature_every < 1:
            raise ValueError(f"curvature_every must be >= 1, got {curvature_every}")
        if hutchinson_samples < 1:
            raise ValueError(f"hutchinson_samples must be >= 1, got {hutchinson_samples}")
        if not 0.0 <= rel_floor < 1.0:
            raise ValueError(f"rel_floor must be in [0, 1), got {rel_floor}")
        if curvature_subsample is not None and curvature_subsample < 1:
            raise ValueError(f"curvature_subsample must be >= 1 or None, got {curvature_subsample}")
        if safeguard_backtracks < 1:
            raise ValueError(f"safeguard_backtracks must be >= 1, got {safeguard_backtracks}")
        super().__init__(params, {"lr": lr})
        self.lr = float(lr)
        self.curvature = curvature
        self.beta1 = float(beta1)
        self.beta2 = float(beta2)
        self.eps = float(eps)
        self.clip = clip
        self.curvature_every = int(curvature_every)
        self.hutchinson_samples = int(hutchinson_samples)
        self.rel_floor = float(rel_floor)
        self.bias_correction = bool(bias_correction)
        self.curvature_subsample = curvature_subsample
        self.safeguard = bool(safeguard)
        self.safeguard_backtracks = int(safeguard_backtracks)
        self._m: Tensor | None = None
        self._d: Tensor | None = None
        self._t = 0
        self._d_updates = 0
        self._gen: torch.Generator | None = None

    def step(self, closure: Closure | None = None) -> Tensor:  # type: ignore[override]
        """One preconditioned step; ``closure`` returns the loss (or residual for GN)."""
        if closure is None:
            raise ValueError("DiagonalCurvature.step requires a closure")
        refresh = (self._t % self.curvature_every) == 0
        out = closure()
        if self.curvature == "gauss_newton":
            if out.ndim != 1:
                raise ValueError(f"gauss_newton closure must return a 1-D residual, got {tuple(out.shape)}")
            loss = 0.5 * (out**2).mean()
            # retain the residual graph only when the diagonal will reuse it this step
            g_list = torch.autograd.grad(loss, self._params, retain_graph=refresh)
            g = self._flat([t.detach() for t in g_list])
            if refresh:
                h = self._gn_diagonal(out, g).abs()
        else:
            loss = out
            need_graph = refresh
            g, g_list = self._grad(loss, create_graph=need_graph)
            if refresh:
                h = self._hutchinson_diagonal(g_list, g).abs()

        # EMA updates: zero-init + Adam bias correction, or the plain warm-init average.
        if refresh:
            self._update_diagonal_ema(h)
        assert self._d is not None
        if self.bias_correction:
            self._m = (1.0 - self.beta1) * g if self._m is None else self.beta1 * self._m + (1.0 - self.beta1) * g
        else:
            self._m = g if self._m is None else self.beta1 * self._m + (1.0 - self.beta1) * g

        m_hat = self._m
        if self.bias_correction:
            m_hat = m_hat / (1.0 - self.beta1 ** (self._t + 1))
        precond = m_hat / self._effective_diagonal()
        if self.clip is not None:
            precond = precond.clamp(-self.clip, self.clip)

        base = self._clone_flat()
        if self.safeguard:

            def _trial_loss() -> float:
                o = closure()
                scalar = 0.5 * (o**2).mean() if self.curvature == "gauss_newton" else o
                return float(scalar.detach())

            f0 = float(loss.detach())
            scale = 1.0
            accepted = False
            for _ in range(self.safeguard_backtracks):
                self._write_flat(base - (self.lr * scale) * precond)
                f_trial = _trial_loss()
                if math.isfinite(f_trial) and f_trial <= f0:
                    accepted = True
                    break
                scale *= 0.5
            if not accepted:
                self._write_flat(base)  # reject: no step improved the loss
        else:
            self._write_flat(base - self.lr * precond)
        self._t += 1
        return loss.detach()


class ConformalSymplectic(_DiagonalCurvatureMixin):
    r"""**Conformal-Symplectic Descent (CSD)** -- optimisation as a dissipative Hamiltonian flow.

    Treats training as a *mechanical* system: a particle at position ``theta`` with velocity
    ``v`` and diagonal mass ``M`` moving in the potential ``L(theta)`` under linear friction
    ``gamma``,

    .. math::

        \dot\theta = v, \qquad M\,\dot v = -\nabla L(\theta) - \gamma\, M v,

    whose mechanical "energy" ``E = 1/2 v^T M v + L`` dissipates monotonically
    (``dE/dt = -gamma\, v^T M v <= 0``), so trajectories settle into a minimum. Classical
    momentum / heavy-ball -- and, loosely, Adam -- are *non-symplectic* discretisations of this
    flow. CSD instead uses the **conformal-symplectic** integrator: the friction is applied as the
    exact contraction ``mu = exp(-gamma * lr)`` and the gradient as a symplectic kick,

    .. math::

        v \leftarrow \mu\, v - M^{-1}\nabla L(\theta), \qquad
        \theta \leftarrow \theta + \mathrm{lr}\, v.

    The map contracts phase space at exactly the continuous rate ``mu`` with no secular energy
    drift, so it stays stable where an explicit (Euler-grade) discretisation of the same flow
    diverges. With ``mass="identity"`` this is exactly conformal-symplectic heavy-ball (and the
    ablation baseline); three omnibias-specific ingredients turn it into an Adam competitor at
    first-order cost:

    * **Exact-curvature mass ``M``** (``mass="hutchinson"`` / ``"gauss_newton"``) reuses the
      *identical* hardened curvature diagonal as :class:`DiagonalCurvature` (EMA, condition-number
      floor, bias correction, amortised refresh every ``curvature_every`` steps). ``M^{-1}`` makes
      the flow reparametrisation-aware -- the per-coordinate adaptivity Adam gets from its RMS
      denominator, but from *exact* curvature.
    * **Exact directional-jet line search** (``line_search=2`` / ``3``) sets the drift length by
      minimising the exact degree-2/3 directional Taylor model of the loss along ``v`` -- the
      coefficients ``phi'(0), phi''(0), phi'''(0)`` are exact directional derivatives (matrix-free
      HVPs on the closed-form tower), the candidate steps evaluated against the *true* loss. A
      derived step, no learning-rate schedule.
    * **Langevin thermostat** (``temperature>0``) adds fluctuation-dissipation-scaled noise so the
      dynamics explore rather than merely descend -- a flat-minima bias for the stochastic regime.

    The ``closure`` contract matches :class:`DiagonalCurvature`: it returns the scalar loss for
    ``mass in {"identity", "hutchinson"}`` and the 1-D residual vector for ``mass="gauss_newton"``
    (the closed-form Gauss-Newton mass of a least-squares / PINN objective), with its autograd
    graph intact.

    ``gamma`` and ``momentum`` are two views of the same friction: pass ``gamma`` (continuous
    friction ``>= 0``) or leave it ``None`` to set the per-step contraction directly via
    ``momentum = mu in [0, 1)``. ``gamma=0`` is the frictionless (energy-conserving) symplectic
    limit.

    Usage (a one-line swap for Adam)::

        opt = ConformalSymplectic(model.parameters(), lr=1e-1, mass="hutchinson")
        for _ in range(n_steps):
            opt.step(lambda: loss_fn(model))   # scalar loss, graph intact, no .backward()
    """

    def __init__(
        self,
        params: Iterable[Tensor],
        *,
        lr: float = 1e-1,
        momentum: float = 0.9,
        gamma: float | None = None,
        mass: Literal["identity", "hutchinson", "gauss_newton"] = "hutchinson",
        beta2: float = 0.99,
        eps: float = 1e-8,
        curvature_every: int = 1,
        hutchinson_samples: int = 1,
        rel_floor: float = 1e-2,
        bias_correction: bool = True,
        curvature_subsample: int | None = None,
        clip: float | None = None,
        line_search: int | None = None,
        line_search_max_scale: float = 4.0,
        temperature: float = 0.0,
        safeguard: bool = False,
        safeguard_backtracks: int = 5,
        seed: int = 0,
    ) -> None:
        if lr <= 0.0:
            raise ValueError(f"lr must be > 0, got {lr}")
        if gamma is not None and gamma < 0.0:
            raise ValueError(f"gamma must be >= 0 or None, got {gamma}")
        if gamma is None and not 0.0 <= momentum < 1.0:
            raise ValueError(f"momentum must be in [0, 1), got {momentum}")
        if mass not in ("identity", "hutchinson", "gauss_newton"):
            raise ValueError(f"mass must be 'identity', 'hutchinson' or 'gauss_newton', got {mass!r}")
        if clip is not None and clip <= 0.0:
            raise ValueError(f"clip must be > 0 or None, got {clip}")
        if not 0.0 <= beta2 < 1.0:
            raise ValueError(f"beta2 must be in [0, 1), got {beta2}")
        if eps <= 0.0:
            raise ValueError(f"eps must be > 0, got {eps}")
        if curvature_every < 1:
            raise ValueError(f"curvature_every must be >= 1, got {curvature_every}")
        if hutchinson_samples < 1:
            raise ValueError(f"hutchinson_samples must be >= 1, got {hutchinson_samples}")
        if not 0.0 <= rel_floor < 1.0:
            raise ValueError(f"rel_floor must be in [0, 1), got {rel_floor}")
        if curvature_subsample is not None and curvature_subsample < 1:
            raise ValueError(f"curvature_subsample must be >= 1 or None, got {curvature_subsample}")
        if line_search is not None and line_search not in (2, 3):
            raise ValueError(f"line_search must be None, 2 or 3, got {line_search}")
        if line_search_max_scale <= 0.0:
            raise ValueError(f"line_search_max_scale must be > 0, got {line_search_max_scale}")
        if temperature < 0.0:
            raise ValueError(f"temperature must be >= 0, got {temperature}")
        if safeguard_backtracks < 1:
            raise ValueError(f"safeguard_backtracks must be >= 1, got {safeguard_backtracks}")
        super().__init__(params, {"lr": lr})
        self.lr = float(lr)
        if gamma is not None:
            self.gamma = float(gamma)
            self._mu = math.exp(-self.gamma * self.lr)
        else:
            self._mu = float(momentum)
            self.gamma = math.inf if self._mu == 0.0 else -math.log(self._mu) / self.lr
        self.mass = mass
        self.beta2 = float(beta2)
        self.eps = float(eps)
        self.curvature_every = int(curvature_every)
        self.hutchinson_samples = int(hutchinson_samples)
        self.rel_floor = float(rel_floor)
        self.bias_correction = bool(bias_correction)
        self.curvature_subsample = curvature_subsample
        self.clip = clip
        self.line_search = line_search
        self.line_search_max_scale = float(line_search_max_scale)
        self.temperature = float(temperature)
        self.safeguard = bool(safeguard)
        self.safeguard_backtracks = int(safeguard_backtracks)
        self.seed = int(seed)
        self._v: Tensor | None = None
        self._d: Tensor | None = None
        self._t = 0
        self._d_updates = 0
        self._gen: torch.Generator | None = None
        self._noise_gen: torch.Generator | None = None

    @property
    def momentum(self) -> float:
        """The per-step conformal contraction ``mu = exp(-gamma * lr)`` (the momentum coefficient)."""
        return self._mu

    def _scalar_loss(self, out: Tensor) -> Tensor:
        """Scalar objective from a closure output (``0.5`` mean-square residual for Gauss-Newton)."""
        return 0.5 * (out**2).mean() if self.mass == "gauss_newton" else out

    def _mass_inverse(self, template: Tensor) -> Tensor:
        """Diagonal ``M^{-1}`` preconditioning the kick (all-ones for ``mass='identity'``)."""
        if self.mass == "identity":
            return torch.ones_like(template)
        return 1.0 / self._effective_diagonal()

    def _directional_coeffs(
        self, g: Tensor, g_list: tuple[Tensor, ...], v: Tensor, order: int
    ) -> tuple[float, float, float]:
        r"""Exact along-``v`` Taylor coefficients of ``phi(s) = L(theta + s v)`` at ``s = 0``.

        ``phi'(0) = grad L . v`` (from the detached gradient), ``phi''(0) = v^T H v`` (one exact
        matrix-free HVP), and for ``order == 3`` ``phi'''(0) = D^3 L[v, v, v]`` (one further
        reverse pass over the graph of ``v^T H v``). These are the coefficients of the exact
        directional Taylor model :func:`taylor_line_min` builds functionally.
        """
        c1 = float(g @ v)
        v_list = self._unflat(v)
        hv_list = torch.autograd.grad(
            list(g_list), self._params, grad_outputs=v_list, create_graph=(order == 3), retain_graph=True
        )
        hvv = torch.zeros((), dtype=v.dtype, device=v.device)
        for h_i, v_i in zip(hv_list, v_list, strict=True):
            hvv = hvv + (h_i * v_i).sum()
        c2 = float(hvv.detach())
        c3 = 0.0
        if order == 3 and hvv.requires_grad:  # a quadratic has no third derivative (no grad_fn)
            t_list = torch.autograd.grad(hvv, self._params, retain_graph=True, allow_unused=True)
            t_flat = self._flat(
                [t.detach() if t is not None else torch.zeros_like(p) for t, p in zip(t_list, self._params, strict=True)]
            )
            c3 = float(t_flat @ v)
        return c1, c2, c3

    def _line_search_candidates(self, c1: float, c2: float, c3: float) -> list[float]:
        """Stationary points of the degree-2/3 model within ``[0, max_scale*lr]``, plus endpoints."""
        lo, hi = 0.0, self.line_search_max_scale * self.lr
        candidates = [lo, hi, self.lr]
        if abs(c3) > 1e-30:  # phi'(s) = c1 + c2 s + (c3/2) s^2 = 0
            disc = c2 * c2 - 2.0 * c1 * c3
            if disc >= 0.0:
                root = math.sqrt(disc)
                candidates += [(-c2 + root) / c3, (-c2 - root) / c3]
        elif abs(c2) > 1e-30:
            candidates.append(-c1 / c2)
        return [s for s in candidates if lo <= s <= hi]

    def step(self, closure: Closure | None = None) -> Tensor:  # type: ignore[override]
        """One conformal-symplectic step; ``closure`` returns the loss (or residual for GN mass)."""
        if closure is None:
            raise ValueError("ConformalSymplectic.step requires a closure")
        refresh = self.mass != "identity" and (self._t % self.curvature_every == 0)
        ls_on = self.line_search is not None
        out = closure()

        if self.mass == "gauss_newton":
            if out.ndim != 1:
                raise ValueError(f"gauss_newton closure must return a 1-D residual, got {tuple(out.shape)}")
            loss = 0.5 * (out**2).mean()
            g_list = torch.autograd.grad(loss, self._params, create_graph=ls_on, retain_graph=refresh or ls_on)
            g = self._flat([t.detach() for t in g_list])
            if refresh:
                self._update_diagonal_ema(self._gn_diagonal(out, g).abs())
        else:
            loss = out
            g, g_list = self._grad(loss, create_graph=(ls_on or refresh))
            if refresh:
                self._update_diagonal_ema(self._hutchinson_diagonal(g_list, g).abs())

        minv = self._mass_inverse(g)

        # conformal-symplectic kick: v <- mu v - M^{-1} grad  (optional per-coordinate clip caps
        # the impulse in near-flat/high-M^{-1} directions -- the same stabiliser as DiagonalCurvature)
        kick = minv * g
        if self.clip is not None:
            kick = kick.clamp(-self.clip, self.clip)
        v_prev = torch.zeros_like(g) if self._v is None else self._v
        v = self._mu * v_prev - kick
        if self.temperature > 0.0:
            if self._noise_gen is None:
                self._noise_gen = torch.Generator(device=g.device).manual_seed(self.seed)
            noise = torch.randn(g.shape, dtype=g.dtype, device=g.device, generator=self._noise_gen)
            v = v + math.sqrt(max(1.0 - self._mu * self._mu, 0.0) * self.temperature) * minv.sqrt() * noise
        self._v = v

        base = self._clone_flat()
        f0 = float(loss.detach())

        # drift: theta <- theta + h v, with h = lr or an exact directional-Taylor line-searched scale.
        if ls_on:
            assert self.line_search is not None
            c1, c2, c3 = self._directional_coeffs(g, g_list, v, self.line_search)
            best_s, best_f = 0.0, f0
            for s in self._line_search_candidates(c1, c2, c3):
                self._write_flat(base + s * v)
                f = float(self._scalar_loss(closure()).detach())
                if math.isfinite(f) and f < best_f:
                    best_f, best_s = f, s
            self._write_flat(base + best_s * v)
        elif self.safeguard:
            scale = 1.0
            accepted = False
            for _ in range(self.safeguard_backtracks):
                self._write_flat(base + (self.lr * scale) * v)
                f = float(self._scalar_loss(closure()).detach())
                if math.isfinite(f) and f <= f0:
                    accepted = True
                    break
                scale *= 0.5
            if not accepted:  # reject: restore and reset the momentum buffer
                self._write_flat(base)
                self._v = torch.zeros_like(g)
        else:
            self._write_flat(base + self.lr * v)

        self._t += 1
        return loss.detach()


class FrugalCurvature(_DiagonalCurvatureMixin):
    r"""Memory-lean exact-curvature preconditioner -- Adam-class adaptivity at ~half the state.

    Adam keeps *two* ``O(P)`` buffers -- the smoothed gradient ``m`` and the per-coordinate
    second-moment ``v`` -- so its optimiser state doubles the model's memory, the binding cost at
    scale. FrugalCurvature keeps **one** ``O(P)`` momentum buffer and recovers per-coordinate
    adaptivity from an *exact* curvature probe stored at **coarse (per-tensor) granularity**:
    one scalar per parameter tensor, ``O(#tensors)`` -- negligible. The update is

    .. math::

        \theta \leftarrow \theta - \eta\, \operatorname{clip}\!\big(\hat m \,/\, (\sqrt{c_\ell} + \epsilon)\big),

    where ``c_ell`` is the (EMA of the) exact curvature diagonal *reduced over parameter tensor
    ``ell``* (``mean`` or ``rms``). The full ``O(P)`` diagonal is computed transiently every
    ``curvature_every`` steps and **immediately reduced to the per-tensor scalars, then
    discarded** -- it is never stored, which is what makes the state lean. As with
    :class:`DiagonalCurvature` the curvature is *exact*:

    * ``curvature="hutchinson"`` -- Hutchinson diagonal of the full Hessian via exact matrix-free
      HVPs (unbiased); the ``closure`` returns the scalar loss.
    * ``curvature="gauss_newton"`` -- the exact Gauss-Newton diagonal ``(1/N) sum_n (grad r_n)^2``;
      the ``closure`` returns the residual vector ``r`` (least-squares / PINN objective).

    ``sign_momentum=True`` descends ``sign(\hat m)`` (Lion-like) for the leanest, scale-free
    direction, still scaled per tensor by the exact curvature. Optional per-coordinate ``clip``
    and a monotone ``safeguard`` mirror :class:`DiagonalCurvature`.

    Honesty: per-tensor curvature is *coarser* than Adam's per-coordinate ``v`` -- the trade is
    memory for granularity -- so whether it matches Adam or lands between SGD and Adam is an
    empirical, problem-dependent question (see the ``mnist1d_double_descent`` optimizer axis).

    Usage (a one-line swap for Adam, at half the optimiser memory)::

        opt = FrugalCurvature(model.parameters(), lr=1e-2, curvature="hutchinson")
        for _ in range(n_steps):
            opt.step(lambda: loss_fn(model))   # scalar loss, graph intact, no .backward()
    """

    def __init__(
        self,
        params: Iterable[Tensor],
        *,
        lr: float = 1e-2,
        curvature: Literal["gauss_newton", "hutchinson"] = "hutchinson",
        beta1: float = 0.9,
        beta2: float = 0.99,
        eps: float = 1e-8,
        clip: float | None = None,
        curvature_every: int = 10,
        hutchinson_samples: int = 1,
        curvature_subsample: int | None = None,
        reduce: Literal["mean", "rms"] = "rms",
        sign_momentum: bool = False,
        bias_correction: bool = True,
        safeguard: bool = False,
        safeguard_backtracks: int = 5,
    ) -> None:
        if lr <= 0.0:
            raise ValueError(f"lr must be > 0, got {lr}")
        if curvature not in ("gauss_newton", "hutchinson"):
            raise ValueError(f"curvature must be 'gauss_newton' or 'hutchinson', got {curvature!r}")
        if not 0.0 <= beta1 < 1.0:
            raise ValueError(f"beta1 must be in [0, 1), got {beta1}")
        if not 0.0 <= beta2 < 1.0:
            raise ValueError(f"beta2 must be in [0, 1), got {beta2}")
        if eps <= 0.0:
            raise ValueError(f"eps must be > 0, got {eps}")
        if clip is not None and clip <= 0.0:
            raise ValueError(f"clip must be > 0 or None, got {clip}")
        if curvature_every < 1:
            raise ValueError(f"curvature_every must be >= 1, got {curvature_every}")
        if hutchinson_samples < 1:
            raise ValueError(f"hutchinson_samples must be >= 1, got {hutchinson_samples}")
        if curvature_subsample is not None and curvature_subsample < 1:
            raise ValueError(f"curvature_subsample must be >= 1 or None, got {curvature_subsample}")
        if reduce not in ("mean", "rms"):
            raise ValueError(f"reduce must be 'mean' or 'rms', got {reduce!r}")
        if safeguard_backtracks < 1:
            raise ValueError(f"safeguard_backtracks must be >= 1, got {safeguard_backtracks}")
        super().__init__(params, {"lr": lr})
        self.lr = float(lr)
        self.curvature = curvature
        self.beta1 = float(beta1)
        self.beta2 = float(beta2)
        self.eps = float(eps)
        self.clip = clip
        self.curvature_every = int(curvature_every)
        self.hutchinson_samples = int(hutchinson_samples)
        self.curvature_subsample = curvature_subsample
        self.reduce = reduce
        self.sign_momentum = bool(sign_momentum)
        self.bias_correction = bool(bias_correction)
        self.safeguard = bool(safeguard)
        self.safeguard_backtracks = int(safeguard_backtracks)
        self._m: Tensor | None = None
        self._c: Tensor | None = None  # per-tensor curvature scalars (O(#tensors))
        self._d: Tensor | None = None  # unused: the O(P) diagonal is never stored
        self._t = 0
        self._c_updates = 0
        self._d_updates = 0
        self._gen: torch.Generator | None = None

    def _reduce_per_tensor(self, h_flat: Tensor) -> Tensor:
        """Reduce the flat curvature diagonal to one scalar per parameter tensor."""
        parts = self._unflat(h_flat)
        if self.reduce == "rms":
            vals = [part.pow(2).mean().sqrt() for part in parts]
        else:
            vals = [part.mean() for part in parts]
        return torch.stack(vals)

    def _broadcast_per_tensor(self, per_tensor: Tensor) -> Tensor:
        """Expand the per-tensor scalars back to a flat, per-coordinate vector."""
        pieces = [per_tensor[i].expand(p.numel()) for i, p in enumerate(self._params)]
        return torch.cat(pieces)

    def _effective_per_tensor(self) -> Tensor:
        """Bias-corrected per-tensor curvature (its sqrt is the Adam-style denominator)."""
        assert self._c is not None
        c_hat = self._c
        if self.bias_correction and self._c_updates > 0:
            c_hat = c_hat / (1.0 - self.beta2**self._c_updates)
        return c_hat

    def step(self, closure: Closure | None = None) -> Tensor:  # type: ignore[override]
        """One lean preconditioned step; ``closure`` returns the loss (or residual for GN)."""
        if closure is None:
            raise ValueError("FrugalCurvature.step requires a closure")
        refresh = (self._t % self.curvature_every) == 0
        out = closure()
        if self.curvature == "gauss_newton":
            if out.ndim != 1:
                raise ValueError(f"gauss_newton closure must return a 1-D residual, got {tuple(out.shape)}")
            loss = 0.5 * (out**2).mean()
            g_list = torch.autograd.grad(loss, self._params, retain_graph=refresh)
            g = self._flat([t.detach() for t in g_list])
            if refresh:
                h = self._gn_diagonal(out, g).abs()
        else:
            loss = out
            g, g_list = self._grad(loss, create_graph=refresh)
            if refresh:
                h = self._hutchinson_diagonal(g_list, g).abs()

        # Reduce the transient O(P) diagonal to the O(#tensors) per-tensor scalars, then drop it.
        if refresh:
            c_new = self._reduce_per_tensor(h)
            if self.bias_correction:
                self._c = (1.0 - self.beta2) * c_new if self._c is None else self.beta2 * self._c + (1.0 - self.beta2) * c_new
            else:
                self._c = c_new if self._c is None else self.beta2 * self._c + (1.0 - self.beta2) * c_new
            self._c_updates += 1
        assert self._c is not None

        if self.bias_correction:
            self._m = (1.0 - self.beta1) * g if self._m is None else self.beta1 * self._m + (1.0 - self.beta1) * g
        else:
            self._m = g if self._m is None else self.beta1 * self._m + (1.0 - self.beta1) * g
        m_hat = self._m
        if self.bias_correction:
            m_hat = m_hat / (1.0 - self.beta1 ** (self._t + 1))

        denom = self._broadcast_per_tensor(self._effective_per_tensor().sqrt()) + self.eps
        direction = m_hat.sign() if self.sign_momentum else m_hat
        precond = direction / denom
        if self.clip is not None:
            precond = precond.clamp(-self.clip, self.clip)

        base = self._clone_flat()
        if self.safeguard:

            def _trial_loss() -> float:
                o = closure()
                scalar = 0.5 * (o**2).mean() if self.curvature == "gauss_newton" else o
                return float(scalar.detach())

            f0 = float(loss.detach())
            scale = 1.0
            accepted = False
            for _ in range(self.safeguard_backtracks):
                self._write_flat(base - (self.lr * scale) * precond)
                f_trial = _trial_loss()
                if math.isfinite(f_trial) and f_trial <= f0:
                    accepted = True
                    break
                scale *= 0.5
            if not accepted:
                self._write_flat(base)  # reject: no step improved the loss
        else:
            self._write_flat(base - self.lr * precond)
        self._t += 1
        return loss.detach()


class TrustRegionNewtonCG(_CurvatureOptimizer):
    r"""Drop-in **matrix-free trust-region Newton-CG** on the exact full Hessian.

    A classical inexact-Newton trust-region method (Steihaug-Toint), lifted by omnibias'
    exact curvature: each step approximately solves the trust-region subproblem
    ``min_p g^T p + 0.5 p^T H p  s.t. ||p|| <= radius`` with :func:`steihaug_cg`, where ``H``
    is the **exact full Hessian** applied matrix-free by double-backward autograd
    (:meth:`_hvp`) -- no dense ``P x P`` matrix, so it scales to large parameter counts at the
    cost of a handful of Hessian-vector products per step. The radius is expanded / contracted
    by the actual-to-predicted reduction ratio, and steps are accepted only on genuine descent,
    so the method is globally convergent, needs **no learning rate**, and -- because
    :func:`steihaug_cg` follows negative-curvature directions to the boundary -- escapes
    saddles.

    This is the scalable complement to :class:`CubicNewton`: cubic-ARC regularises the Newton
    step by ``(sigma/3)||s||^3``; here the same exact curvature is controlled by an explicit
    trust radius, which is the more familiar knob for very ill-conditioned problems.

    Usage (a one-line swap for Adam)::

        opt = TrustRegionNewtonCG(model.parameters())
        def closure() -> torch.Tensor:
            return loss_fn(model)          # scalar loss, graph intact, no .backward()
        for _ in range(n_steps):
            opt.step(closure)
    """

    def __init__(
        self,
        params: Iterable[Tensor],
        *,
        radius: float = 1.0,
        max_radius: float = 1e3,
        min_radius: float = 1e-12,
        eta: float = 0.1,
        cg_max_iter: int = 50,
        cg_tol: float = 1e-8,
    ) -> None:
        if radius <= 0.0:
            raise ValueError(f"radius must be > 0, got {radius}")
        if max_radius < radius:
            raise ValueError(f"max_radius must be >= radius, got {max_radius} < {radius}")
        if min_radius <= 0.0:
            raise ValueError(f"min_radius must be > 0, got {min_radius}")
        if not 0.0 <= eta < 0.25:
            raise ValueError(f"eta (accept threshold) must be in [0, 0.25), got {eta}")
        if cg_max_iter < 1:
            raise ValueError(f"cg_max_iter must be >= 1, got {cg_max_iter}")
        if cg_tol <= 0.0:
            raise ValueError(f"cg_tol must be > 0, got {cg_tol}")
        super().__init__(params, {})
        self._radius = float(radius)
        self.max_radius = float(max_radius)
        self.min_radius = float(min_radius)
        self.eta = float(eta)
        self.cg_max_iter = int(cg_max_iter)
        self.cg_tol = float(cg_tol)
        self.n_iter = 0
        self.last_cg_iters = 0  # CG iterations used by the most recent step (matrix-free HVP count)

    @property
    def radius(self) -> float:
        """The current trust-region radius (adapted by the reduction-ratio test)."""
        return self._radius

    def step(self, closure: Closure | None = None) -> Tensor:  # type: ignore[override]
        """One trust-region Newton-CG iteration; ``closure`` returns the scalar loss."""
        if closure is None:
            raise ValueError("TrustRegionNewtonCG.step requires a closure returning the scalar loss")
        loss = closure()
        g, g_list = self._grad(loss, create_graph=True)
        self.n_iter += 1
        gnorm = float(torch.linalg.vector_norm(g))
        if gnorm == 0.0 or not math.isfinite(gnorm):
            return loss.detach()

        cg_calls = 0

        def matvec(v: Tensor) -> Tensor:
            nonlocal cg_calls
            cg_calls += 1
            return self._hvp(g_list, v)

        # All Hessian-vector products happen *before* any trial writeback (the retained graph
        # is tied to the current parameters); the reduction ratio needs one extra product H p.
        p, hit_boundary = steihaug_cg(matvec, g, self._radius, max_iter=self.cg_max_iter, tol=self.cg_tol)
        self.last_cg_iters = cg_calls  # count only the products inside steihaug_cg (one per CG iter)
        hp = matvec(p)
        predicted = -(float(torch.dot(g, p)) + 0.5 * float(torch.dot(p, hp)))

        params0 = self._clone_flat()
        f0 = float(loss.detach())
        self._write_flat(params0 + p)
        f1 = float(closure().detach())
        actual = f0 - f1
        rho = actual / predicted if predicted > 0.0 and math.isfinite(f1) else float("-inf")

        if rho < 0.25:
            self._radius = max(0.25 * self._radius, self.min_radius)
        elif rho > 0.75 and hit_boundary:
            self._radius = min(2.0 * self._radius, self.max_radius)

        if not (rho > self.eta):  # reject: restore the pre-step parameters
            self._write_flat(params0)
        return loss.detach()


class StochasticNewtonCG(_CurvatureOptimizer):
    r"""Drop-in **subsampled / Levenberg-damped Newton-CG** for noisy (minibatch) objectives.

    Full-batch Newton-CG (:class:`TrustRegionNewtonCG`) assumes a deterministic loss; on a
    minibatch objective the gradient and curvature are noisy, so a raw Newton step overfits the
    sample. This variant is the standard **subsampled Newton** (Roosta-Khorasani-Mahoney,
    Xu et al.): the Newton system is regularised by a Levenberg term ``(H + lambda I)`` whose
    ``lambda`` is adapted by the actual-to-predicted reduction ratio (grow on a bad step, shrink
    on a good one), and the damped system is solved matrix-free with :func:`steihaug_cg`. Two
    subsampling hooks keep it honest and cheap:

    * ``resample`` -- an optional ``() -> None`` callback advanced **once per step** to draw the
      next minibatch; the loss/gradient/curvature and the accept/reject ratio are then all
      evaluated on that single fixed batch (so the ratio is valid).
    * ``curvature_closure`` -- an optional second closure returning the loss on a **smaller**
      curvature sample ``S_H`` (the gradient stays on the larger batch of ``closure``); the exact
      HVP is taken through ``S_H`` only. This is the ``|S_H| << |S_g|`` split that makes
      subsampled Newton scale.

    With both hooks ``None`` it is exactly a Levenberg-damped exact Newton-CG on the full batch.
    """

    def __init__(
        self,
        params: Iterable[Tensor],
        *,
        damping: float = 1.0,
        min_damping: float = 1e-8,
        max_damping: float = 1e12,
        eta: float = 0.1,
        max_step: float = 1e3,
        cg_max_iter: int = 50,
        cg_tol: float = 1e-8,
        resample: Callable[[], None] | None = None,
        curvature_closure: Closure | None = None,
    ) -> None:
        if damping <= 0.0:
            raise ValueError(f"damping must be > 0, got {damping}")
        if not 0.0 < min_damping <= damping <= max_damping:
            raise ValueError("need 0 < min_damping <= damping <= max_damping")
        if not 0.0 <= eta < 0.25:
            raise ValueError(f"eta (accept threshold) must be in [0, 0.25), got {eta}")
        if max_step <= 0.0:
            raise ValueError(f"max_step must be > 0, got {max_step}")
        if cg_max_iter < 1:
            raise ValueError(f"cg_max_iter must be >= 1, got {cg_max_iter}")
        if cg_tol <= 0.0:
            raise ValueError(f"cg_tol must be > 0, got {cg_tol}")
        super().__init__(params, {})
        self._lambda = float(damping)
        self.min_damping = float(min_damping)
        self.max_damping = float(max_damping)
        self.eta = float(eta)
        self.max_step = float(max_step)
        self.cg_max_iter = int(cg_max_iter)
        self.cg_tol = float(cg_tol)
        self.resample = resample
        self.curvature_closure = curvature_closure
        self.n_iter = 0
        self.last_cg_iters = 0  # CG iterations used by the most recent step (matrix-free HVP count)

    @property
    def damping(self) -> float:
        """The current Levenberg damping ``lambda`` (adapted by the reduction ratio)."""
        return self._lambda

    def step(self, closure: Closure | None = None) -> Tensor:  # type: ignore[override]
        """One subsampled Newton-CG iteration; ``closure`` returns the scalar loss."""
        if closure is None:
            raise ValueError("StochasticNewtonCG.step requires a closure returning the scalar loss")
        if self.resample is not None:
            self.resample()  # advance the minibatch once; everything below sees this fixed batch
        loss = closure()
        g, g_list = self._grad(loss, create_graph=True)
        self.n_iter += 1
        gnorm = float(torch.linalg.vector_norm(g))
        if gnorm == 0.0 or not math.isfinite(gnorm):
            return loss.detach()

        # curvature on the (optionally smaller) sample S_H; gradient stays on closure's batch
        if self.curvature_closure is not None:
            _c, c_list = self._grad(self.curvature_closure(), create_graph=True)
        else:
            c_list = g_list

        def matvec(v: Tensor) -> Tensor:
            return self._hvp(c_list, v)

        cg_calls = 0

        def damped_matvec(v: Tensor) -> Tensor:
            nonlocal cg_calls
            cg_calls += 1
            return self._hvp(c_list, v) + self._lambda * v

        p, _hit = steihaug_cg(damped_matvec, g, self.max_step, max_iter=self.cg_max_iter, tol=self.cg_tol)
        self.last_cg_iters = cg_calls  # count only the products inside steihaug_cg (one per CG iter)
        hp = matvec(p)  # undamped curvature for an honest model-reduction ratio
        predicted = -(float(torch.dot(g, p)) + 0.5 * float(torch.dot(p, hp)))

        params0 = self._clone_flat()
        f0 = float(loss.detach())
        self._write_flat(params0 + p)
        f1 = float(closure().detach())
        actual = f0 - f1
        rho = actual / predicted if predicted > 0.0 and math.isfinite(f1) else float("-inf")

        if rho < 0.25:  # noisy / inaccurate step -> damp harder (smaller, safer)
            self._lambda = min(self._lambda * 4.0, self.max_damping)
        elif rho > 0.75:  # reliable step -> relax damping toward the pure Newton step
            self._lambda = max(self._lambda / 4.0, self.min_damping)

        if not (rho > self.eta):  # reject
            self._write_flat(params0)
        return loss.detach()


class KFAC(torch.optim.Optimizer):
    r"""Kronecker-Factored Approximate Curvature (K-FAC) for ``nn.Linear`` stacks.

    The scalable block-diagonal curvature preconditioner (Martens-Grosse 2015). Per
    ``nn.Linear`` layer with augmented weight ``[W | b]``, K-FAC approximates that layer's
    Fisher / Gauss-Newton block as a Kronecker product ``A (x) G`` of two small factors:

    * ``A = E[a_aug a_aug^T]`` -- the (bias-augmented) **input-activation** covariance, captured
      by a forward hook.
    * ``G = E[g g^T]`` -- the covariance of the **pre-activation gradient** ``g = dL/ds``,
      captured by a full-backward hook.

    The natural-gradient step is then ``W <- W - lr * G^{-1} (grad W) A^{-1}``, computed
    matrix-free in the eigenbases of ``A`` and ``G`` (one ``eigh`` per factor per refresh) with
    the trace-normalised Tikhonov damping split ``(A + pi sqrt(lambda) I)``,
    ``(G + sqrt(lambda)/pi I)``. Factors are accumulated as an EMA and the eigendecompositions
    refreshed every ``refresh_every`` steps (the amortised cost), so a step is essentially two
    small mat-muls per layer -- far cheaper than a full ``P x P`` Newton solve.

    K-FAC captures **within-layer** coupling only; it is the scalable Adam-substitute for
    ordinary ``nn.Linear`` networks, not necessarily a match for the exact full-curvature
    methods on the stiffest problems.

    .. note::
       K-FAC reads its factors from **standard ``nn.Linear`` forward calls** via hooks. It does
       **not** support functional / jet forwards (e.g. a PINN residual assembled through
       :meth:`JetMLP.value_grad_hessian`, which reads the weights directly and never invokes
       ``Linear.__call__``); use a curvature method from the Newton-CG family there.

    ``precondition_modules`` restricts K-FAC to an explicit subset of ``nn.Linear`` layers
    (default: every ``nn.Linear`` in ``model``); ``other_optimizer`` hands the remaining
    parameters (e.g. the embeddings / LayerNorm of a transformer) to a standard optimiser such
    as ``AdamW`` instead of the plain-SGD fallback. Together they make K-FAC a drop-in **hybrid**
    preconditioner for mixed architectures: K-FAC on the big linear blocks, ``AdamW`` on the
    rest. ``other_optimizer`` must own exactly the parameters K-FAC does not precondition (the
    overlap is rejected, as it would double-update those tensors).

    **Stabilisation knobs** (all opt-in; defaults reproduce the classic Martens-Grosse step):

    * ``adaptive_damping`` -- guard each step with a Levenberg accept/reject (re-evaluate the loss,
      roll back + grow the damping on an increase, else relax it). Removes the divergence spikes a
      fixed damping can produce, at one extra forward per step.
    * ``max_step_norm`` -- a global trust-region cap on the norm of the whole natural-gradient step
      (scales every layer's update uniformly), distinct from the element-wise ``clip``.
    * ``solver`` -- ``"eigh"`` (default; one damping-independent eigendecomposition per factor) or
      ``"cholesky"`` (factor the damped matrices directly: cheaper factor + solve, re-factored when
      the working damping changes).

    Usage (a one-line swap for Adam; note it takes the **model**, to register hooks)::

        opt = KFAC(model)
        def closure() -> torch.Tensor:
            return loss_fn(model)          # scalar loss, graph intact, no .backward()
        for _ in range(n_steps):
            opt.step(closure)
    """

    def __init__(
        self,
        model: nn.Module,
        *,
        lr: float = 1e-1,
        damping: float = 1e-2,
        ema_decay: float = 0.95,
        refresh_every: int = 10,
        clip: float | None = None,
        precondition_modules: Iterable[nn.Linear] | None = None,
        other_optimizer: torch.optim.Optimizer | None = None,
        adaptive_damping: bool = False,
        damping_increase: float = 4.0,
        damping_decrease: float = 0.5,
        min_damping: float = 1e-8,
        max_damping: float = 1e3,
        max_step_norm: float | None = None,
        solver: Literal["eigh", "cholesky"] = "eigh",
    ) -> None:
        if lr <= 0.0:
            raise ValueError(f"lr must be > 0, got {lr}")
        if damping <= 0.0:
            raise ValueError(f"damping must be > 0, got {damping}")
        if not 0.0 <= ema_decay < 1.0:
            raise ValueError(f"ema_decay must be in [0, 1), got {ema_decay}")
        if refresh_every < 1:
            raise ValueError(f"refresh_every must be >= 1, got {refresh_every}")
        if clip is not None and clip <= 0.0:
            raise ValueError(f"clip must be > 0 or None, got {clip}")
        if damping_increase <= 1.0:
            raise ValueError(f"damping_increase must be > 1, got {damping_increase}")
        if not 0.0 < damping_decrease < 1.0:
            raise ValueError(f"damping_decrease must be in (0, 1), got {damping_decrease}")
        if not 0.0 < min_damping <= max_damping:
            raise ValueError(f"require 0 < min_damping <= max_damping, got {min_damping}, {max_damping}")
        if max_step_norm is not None and max_step_norm <= 0.0:
            raise ValueError(f"max_step_norm must be > 0 or None, got {max_step_norm}")
        if solver not in ("eigh", "cholesky"):
            raise ValueError(f"solver must be 'eigh' or 'cholesky', got {solver!r}")
        if precondition_modules is None:
            layers = [m for m in model.modules() if isinstance(m, nn.Linear)]
        else:
            layers = list(precondition_modules)
            if not all(isinstance(m, nn.Linear) for m in layers):
                raise ValueError("precondition_modules must all be nn.Linear")
            model_param_ids = {id(p) for p in model.parameters()}
            if any(id(lin.weight) not in model_param_ids for lin in layers):
                raise ValueError("each precondition_modules entry must be a submodule of `model`")
        if not layers:
            raise ValueError("KFAC requires at least one nn.Linear submodule in the model")
        super().__init__(list(model.parameters()), {"lr": lr})
        self.lr = float(lr)
        self.damping = float(damping)
        self.ema_decay = float(ema_decay)
        self.refresh_every = int(refresh_every)
        self.clip = clip
        self.other_optimizer = other_optimizer
        self.adaptive_damping = bool(adaptive_damping)
        self.damping_increase = float(damping_increase)
        self.damping_decrease = float(damping_decrease)
        self.min_damping = float(min_damping)
        self.max_damping = float(max_damping)
        self.max_step_norm = max_step_norm
        self.solver = solver
        self._damping = float(damping)  # mutable working damping (adapted iff adaptive_damping)
        self._layers = layers
        self._a_cov: dict[nn.Linear, Tensor] = {}
        self._g_cov: dict[nn.Linear, Tensor] = {}
        self._eig: dict[nn.Linear, tuple[Tensor, Tensor, Tensor, Tensor]] = {}
        self._chol: dict[nn.Linear, tuple[Tensor, Tensor, float]] = {}
        self._chol_damping = -1.0  # the damping baked into the current Cholesky factors
        self._a_in: dict[nn.Linear, Tensor] = {}
        self._g_out: dict[nn.Linear, Tensor] = {}
        self._t = 0
        self._handles: list[RemovableHandle] = []
        kfac_ids: set[int] = set()
        for lin in layers:
            self._handles.append(lin.register_forward_hook(self._fwd_hook))
            kfac_ids.add(id(lin.weight))
            if lin.bias is not None:
                kfac_ids.add(id(lin.bias))
        self._other = [
            p
            for group in self.param_groups
            for p in group["params"]
            if p.requires_grad and id(p) not in kfac_ids
        ]
        if other_optimizer is not None:
            other_ids = {id(p) for group in other_optimizer.param_groups for p in group["params"]}
            if not other_ids.isdisjoint(kfac_ids):
                raise ValueError(
                    "other_optimizer must not own any parameter that KFAC preconditions "
                    "(the overlap would double-update those tensors)"
                )

    def _fwd_hook(self, module: nn.Module, inp: tuple[Tensor, ...], out: Tensor) -> None:
        assert isinstance(module, nn.Linear)
        lin = module
        self._a_in[lin] = inp[0].detach()
        # capture dL/d(pre-activation) with a hook on the output *tensor* -- unlike a module
        # backward hook this never warns when the layer's input does not require grad.
        if out.requires_grad:

            def _capture(grad: Tensor) -> None:
                self._g_out[lin] = grad.detach()

            out.register_hook(_capture)  # type: ignore[no-untyped-call]

    def remove_hooks(self) -> None:
        """Detach every forward / backward hook this optimiser registered on the model."""
        for handle in self._handles:
            handle.remove()
        self._handles = []

    @property
    def current_damping(self) -> float:
        """The working Levenberg damping ``lambda`` (mutated only when ``adaptive_damping``)."""
        return self._damping

    @torch.no_grad()
    def _snapshot(self) -> list[Tensor]:
        """Clone every model parameter (for the adaptive-damping accept/reject)."""
        return [p.detach().clone() for group in self.param_groups for p in group["params"]]

    @torch.no_grad()
    def _restore(self, snap: list[Tensor]) -> None:
        params = [p for group in self.param_groups for p in group["params"]]
        for p, s in zip(params, snap, strict=True):
            p.copy_(s)

    @torch.no_grad()
    def _accumulate(self) -> None:
        """EMA-update the ``A`` and ``G`` Kronecker factors from the last forward/backward."""
        for lin in self._layers:
            if lin not in self._a_in or lin not in self._g_out:
                raise RuntimeError(
                    "KFAC captured no activations for an nn.Linear; it requires a standard "
                    "nn.Linear forward (not a functional / jet forward)."
                )
            a = self._a_in[lin].reshape(-1, self._a_in[lin].shape[-1])
            g = self._g_out[lin].reshape(-1, self._g_out[lin].shape[-1])
            batch = a.shape[0]
            if lin.bias is not None:
                ones = torch.ones(a.shape[0], 1, dtype=a.dtype, device=a.device)
                a = torch.cat([a, ones], dim=1)
            a_cov = (a.t() @ a) / batch
            g_cov = (g.t() @ g) / batch
            if lin in self._a_cov:
                self._a_cov[lin].mul_(self.ema_decay).add_(a_cov, alpha=1.0 - self.ema_decay)
                self._g_cov[lin].mul_(self.ema_decay).add_(g_cov, alpha=1.0 - self.ema_decay)
            else:
                self._a_cov[lin] = a_cov
                self._g_cov[lin] = g_cov

    @torch.no_grad()
    def _refresh(self) -> None:
        """Factor the current EMA factors (the amortised expensive part).

        ``eigh``: one symmetric eigendecomposition per factor (damping added at solve time, so the
        factorisation is damping-independent and refreshed only every ``refresh_every`` steps).
        ``cholesky``: factor the *damped* matrices ``(A + damp_a I)`` / ``(G + damp_g I)`` directly
        (~2x cheaper than ``eigh`` and a cheaper solve) -- these bake in the current damping, so they
        are re-factored whenever the working damping changes (see :meth:`step`).
        """
        if self.solver == "eigh":
            for lin in self._layers:
                eval_a, evec_a = torch.linalg.eigh(self._a_cov[lin])
                eval_g, evec_g = torch.linalg.eigh(self._g_cov[lin])
                self._eig[lin] = (evec_a, eval_a.clamp_min(0.0), evec_g, eval_g.clamp_min(0.0))
            return
        sqrt_damp = math.sqrt(self._damping)
        for lin in self._layers:
            a_cov, g_cov = self._a_cov[lin], self._g_cov[lin]
            tr_a = float(a_cov.diagonal().mean())
            tr_g = float(g_cov.diagonal().mean())
            pi = math.sqrt(tr_a / tr_g) if tr_a > 0.0 and tr_g > 0.0 else 1.0
            eye_a = torch.eye(a_cov.shape[0], dtype=a_cov.dtype, device=a_cov.device)
            eye_g = torch.eye(g_cov.shape[0], dtype=g_cov.dtype, device=g_cov.device)
            l_a = torch.linalg.cholesky(a_cov + (sqrt_damp * pi) * eye_a)
            l_g = torch.linalg.cholesky(g_cov + (sqrt_damp / pi) * eye_g)
            self._chol[lin] = (l_a, l_g, pi)
        self._chol_damping = self._damping

    def _natural_step(self, lin: nn.Linear, grad_aug: Tensor) -> Tensor:
        """Preconditioned (natural-gradient) update matrix ``G^{-1} grad A^{-1}`` for one layer."""
        if self.solver == "eigh":
            sqrt_damp = math.sqrt(self._damping)
            evec_a, eval_a, evec_g, eval_g = self._eig[lin]
            tr_a = float(eval_a.mean())
            tr_g = float(eval_g.mean())
            pi = math.sqrt(tr_a / tr_g) if tr_a > 0.0 and tr_g > 0.0 else 1.0
            v = evec_g.t() @ grad_aug @ evec_a
            denom = (eval_g.reshape(-1, 1) + sqrt_damp / pi) * (eval_a.reshape(1, -1) + sqrt_damp * pi)
            return evec_g @ (v / denom) @ evec_a.t()
        l_a, l_g, _pi = self._chol[lin]
        # nat = G_damped^{-1} grad_aug A_damped^{-1}; solve each Kronecker factor by cholesky_solve.
        y = torch.cholesky_solve(grad_aug, l_g)  # G_damped y = grad_aug
        return torch.cholesky_solve(y.t().contiguous(), l_a).t()  # nat A_damped = y  ->  nat = (A^{-1} y^T)^T

    @torch.no_grad()
    def _apply(self) -> None:
        """Take the K-FAC natural-gradient step on every hooked layer (+ SGD on the rest).

        Assembled in two passes so the optional global ``max_step_norm`` trust-region cap can scale
        the whole natural-gradient step uniformly (distinct from the element-wise ``clip``).
        """
        updates: list[tuple[Tensor, Tensor]] = []  # (param, delta) with delta already scaled by -lr
        for lin in self._layers:
            gw = lin.weight.grad
            if gw is None:
                continue
            bias = lin.bias
            gb = bias.grad if bias is not None else None
            grad_aug = torch.cat([gw, gb.reshape(-1, 1)], dim=1) if gb is not None else gw
            nat = self._natural_step(lin, grad_aug)
            in_features = int(gw.shape[1])
            nat_w = nat[:, :in_features]
            if self.clip is not None:
                nat_w = nat_w.clamp(-self.clip, self.clip)
            updates.append((lin.weight, nat_w.mul(-self.lr)))
            if gb is not None and bias is not None:
                nat_b = nat[:, in_features]
                if self.clip is not None:
                    nat_b = nat_b.clamp(-self.clip, self.clip)
                updates.append((bias, nat_b.mul(-self.lr)))
        if self.max_step_norm is not None and updates:
            total = math.sqrt(sum(float(torch.sum(d * d)) for _p, d in updates))
            if total > self.max_step_norm and total > 0.0:
                scale = self.max_step_norm / total
                updates = [(p, d.mul_(scale)) for p, d in updates]
        for p, d in updates:
            p.add_(d)
        if self.other_optimizer is not None:
            # Delegate the non-preconditioned params (e.g. embeddings / LayerNorm) to a standard
            # optimiser; KFAC.step already ran one loss.backward(), so their grads are populated.
            self.other_optimizer.step()
        else:
            for p in self._other:  # anything outside a hooked Linear falls back to plain SGD
                if p.grad is not None:
                    p.add_(p.grad, alpha=-self.lr)

    def step(self, closure: Closure | None = None) -> Tensor:  # type: ignore[override]
        """One K-FAC iteration; ``closure`` returns the scalar loss (graph intact).

        With ``adaptive_damping`` the natural-gradient step is guarded by a Levenberg accept/reject:
        the closure loss is re-evaluated after the update and, on a non-finite value or an increase,
        the parameters are rolled back and the damping grown (``damping_increase``); otherwise the
        step is accepted and the damping relaxed (``damping_decrease``). This turns the plain step
        into a monotone-descent step -- it removes the divergence spikes at one extra forward per step.
        """
        if closure is None:
            raise ValueError("KFAC.step requires a closure returning the scalar loss")
        self.zero_grad(set_to_none=True)
        loss = closure()
        loss.backward()  # type: ignore[no-untyped-call]
        self._accumulate()
        factored = self._eig if self.solver == "eigh" else self._chol
        need_refresh = (not factored) or (self._t % self.refresh_every == 0)
        if self.solver == "cholesky" and self._damping != self._chol_damping:
            need_refresh = True  # Cholesky factors bake in the damping -> re-factor when it changed
        if need_refresh:
            self._refresh()

        if not self.adaptive_damping:
            self._apply()
            self._t += 1
            return loss.detach()

        f0 = float(loss.detach())
        snap = self._snapshot()
        self._apply()
        with torch.no_grad():
            f1 = float(closure().detach())
        if not math.isfinite(f1) or f1 > f0:  # reject: roll back and damp harder
            self._restore(snap)
            self._damping = min(self._damping * self.damping_increase, self.max_damping)
        else:  # accept: relax damping toward the pure natural-gradient step
            self._damping = max(self._damping * self.damping_decrease, self.min_damping)
        self._t += 1
        return loss.detach()


__all__ = [
    "Closure",
    "ConformalSymplectic",
    "CubicGaussNewton",
    "CubicNewton",
    "CubicNewtonInfo",
    "CubicRegularizedGaussNewton",
    "CubicRegularizedNewton",
    "DiagonalCurvature",
    "FrugalCurvature",
    "GaussNewton",
    "GaussNewtonInfo",
    "GradNormBalancer",
    "JetLBFGS",
    "JetLBFGSOptimizer",
    "JetSubspaceTensor",
    "KFAC",
    "LBFGSInfo",
    "MatVec",
    "MetricProvider",
    "NaturalGradient",
    "ResidualFn",
    "ScalarFn",
    "StochasticNewtonCG",
    "TrustRegionNewtonCG",
    "cgls",
    "conjugate_gradient",
    "cubic_regularized_newton_step",
    "functional_residual_fn",
    "gauss_newton_direction",
    "gauss_newton_direction_cg",
    "gauss_newton_direction_cgls",
    "gauss_newton_fisher",
    "gauss_newton_fisher_matvec",
    "hvp",
    "lanczos_tridiag",
    "lstsq_gauss_newton_direction",
    "martens_grosse_combine",
    "martens_grosse_gauss_newton_minimize",
    "natural_gradient_direction",
    "quadrature_loss",
    "solve_subspace_trust_region",
    "steihaug_cg",
    "taylor_line_min",
    "taylor_subspace_model",
    "weighted_residual_fn",
]
