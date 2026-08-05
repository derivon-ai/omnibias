# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Exact-curvature sharpness for **multi-layer torch** networks.

This is the torch, arbitrary-depth counterpart of
:mod:`omnibias.curvature.sharpness` (which materialises the exact one-layer
parameter Hessian in JAX). For a deep network the full parameter Hessian is
too large to form, so the curvature functionals are computed **matrix-free**
from exact **Hessian-vector products** (HVPs):

* ``hvp`` -- one exact HVP :math:`Hv` via reverse-over-reverse autograd
  (:func:`torch.autograd.grad` with ``create_graph=True`` twice). Because the
  omnibias torch activations are plain differentiable ops (no custom
  ``autograd.Function`` with a hand-written backward), this double-backward is
  the *exact* Hessian action -- and where the net uses the Riccati
  activations, the :math:`\sigma', \sigma'', \sigma'''` that enter it are the
  bit-stable closed-form tower.
* ``top_eigenvalue`` -- :math:`\lambda_{\max}(H)` by (two-phase) power
  iteration on the HVP; converges to the algebraically largest eigenvalue
  even when :math:`H` is indefinite.
* ``hutchinson_trace`` / ``hutchinson_frobenius_sq`` -- unbiased Rademacher
  estimators of :math:`\operatorname{tr}(H)=\sum_i\lambda_i` and
  :math:`\lVert H\rVert_F^2=\sum_i\lambda_i^2`.
* ``sharpness_aware_loss`` / ``sam_objective`` -- differentiable training
  objectives; the penalty gradient rides on :math:`\sigma'''` in closed form
  (reverse-over-reverse-over-reverse), so it is exact -- no finite-difference
  sharpness, no ascent-step approximation.

The API takes a scalar ``loss`` (with an autograd graph) and the ``params`` it
depends on (e.g. ``list(model.parameters())``), so it drops into any training
loop -- an MLP, a PINN (`JetMLP`), or the `omnibias.score.flow` CNF velocity field.

Honesty
-------
* The HVP and ``dense_hessian`` are **exact** (autograd), matching
  ``torch.autograd.functional.hessian`` to floating point.
* ``top_eigenvalue`` is exact *in the power-iteration limit* (a few dozen
  iterations suffice for a well-separated top eigenvalue).
* ``hutchinson_*`` are **unbiased stochastic** estimators -- exact in
  expectation, with variance ~ ``1/n_samples``. For a *reported* number on a
  small net, use ``dense_hessian`` + the matrix helpers instead.
* This measures / regularises curvature; it does **not** assert that flat
  minima always generalise (that is problem-dependent).
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import Tensor

Params = Sequence[Tensor]
Vector = list[Tensor]


# ---------------------------------------------------------------------------
# Flat-space helpers over a list-of-tensors "vector"
# ---------------------------------------------------------------------------


def _flat_dot(a: Sequence[Tensor], b: Sequence[Tensor]) -> Tensor:
    zero = torch.zeros((), dtype=a[0].dtype, device=a[0].device)
    return sum(((x * y).sum() for x, y in zip(a, b, strict=True)), start=zero)


def _flat_norm(a: Sequence[Tensor]) -> Tensor:
    return torch.sqrt(_flat_dot(a, a))


def _rand_like(
    params: Sequence[Tensor], *,
    generator: torch.Generator | None = None,
    rademacher: bool = False,
) -> Vector:
    out: Vector = []
    for p in params:
        r = torch.randn(p.shape, dtype=p.dtype, device=p.device, generator=generator)
        if rademacher:
            r = torch.sign(r)
            r = torch.where(r == 0, torch.ones_like(r), r)
        out.append(r)
    return out


# ---------------------------------------------------------------------------
# Reusable exact Hessian-vector product operator
# ---------------------------------------------------------------------------


class HessianOperator:
    r"""Matrix-free exact Hessian of ``loss`` w.r.t. ``params``.

    Computes the first-order gradient once (with ``create_graph=True``) and
    exposes repeated HVPs :math:`v \mapsto Hv` against the retained graph.
    """

    def __init__(self, loss: Tensor, params: Params) -> None:
        self.params: list[Tensor] = [p for p in params]
        if not self.params:
            raise ValueError("params is empty")
        self.grads: tuple[Tensor, ...] = torch.autograd.grad(
            loss, self.params, create_graph=True,
        )

    def hvp(self, vec: Sequence[Tensor], *, create_graph: bool = False) -> Vector:
        """Exact ``H @ vec`` as a list matching ``params`` shapes."""
        dot = _flat_dot(self.grads, vec)
        hv = torch.autograd.grad(
            dot, self.params, retain_graph=True, create_graph=create_graph,
        )
        return list(hv)

    @property
    def grad_norm(self) -> Tensor:
        return _flat_norm(self.grads)


def hvp(loss: Tensor, params: Params, vec: Sequence[Tensor]) -> Vector:
    """One exact Hessian-vector product ``H @ vec`` (see :class:`HessianOperator`)."""
    return HessianOperator(loss, params).hvp(vec, create_graph=False)


def dense_hessian(loss: Tensor, params: Params) -> Tensor:
    r"""Materialise the exact ``(P, P)`` parameter Hessian by HVPs on the basis.

    Only for **small** models (``P`` up to a few hundred): it runs ``P`` backward
    passes. Handy as an oracle for the matrix-free estimators and for exact
    sharpness on tiny nets.
    """
    op = HessianOperator(loss, params)
    sizes = [p.numel() for p in op.params]
    total = int(sum(sizes))
    rows = []
    for i in range(total):
        # basis vector e_i as a list-of-tensors
        e: Vector = []
        j = i
        for p, n in zip(op.params, sizes, strict=True):
            block = torch.zeros(n, dtype=p.dtype, device=p.device)
            if 0 <= j < n:
                block[j] = 1.0
            j -= n
            e.append(block.reshape(p.shape))
        hv = op.hvp(e, create_graph=False)
        rows.append(torch.cat([h.reshape(-1) for h in hv]).detach())
    return torch.stack(rows)


# ---------------------------------------------------------------------------
# Matrix-level curvature functionals (parity with the JAX one-layer module)
# ---------------------------------------------------------------------------


def hessian_trace(H: Tensor) -> Tensor:
    r""":math:`\operatorname{tr}(H)=\sum_i\lambda_i`."""
    return torch.trace(H)


def hessian_frobenius_sq(H: Tensor) -> Tensor:
    r""":math:`\lVert H\rVert_F^2=\sum_i\lambda_i^2`."""
    return (H * H).sum()


def hessian_top_eigenvalue(H: Tensor) -> Tensor:
    r""":math:`\lambda_{\max}(H)` via a symmetric eigendecomposition."""
    Hs = 0.5 * (H + H.transpose(-1, -2))
    return torch.linalg.eigvalsh(Hs)[-1]


# ---------------------------------------------------------------------------
# Matrix-free estimators over (loss, params)
# ---------------------------------------------------------------------------


def _power_eigvec(
    op: HessianOperator, *, iters: int, generator: torch.Generator | None,
    shift: float, x0: Vector | None,
) -> Vector:
    """Power iteration on ``(H - shift*I)``; returns the dominant eigenvector."""
    v = x0 if x0 is not None else _rand_like(op.params, generator=generator)
    nrm = _flat_norm(v)
    v = [vi / nrm for vi in v]
    for _ in range(iters):
        hv = op.hvp(v, create_graph=False)
        hv = [h.detach() for h in hv]
        if shift != 0.0:
            hv = [h - shift * vi for h, vi in zip(hv, v, strict=True)]
        nrm = _flat_norm(hv)
        if float(nrm) == 0.0:
            break
        v = [h / nrm for h in hv]
    return v


def _rayleigh(op: HessianOperator, v: Vector) -> Tensor:
    """``v^T H v`` for a unit-norm ``v`` (detached)."""
    hv = op.hvp(v, create_graph=False)
    return _flat_dot(v, hv).detach()


def _top_eigvec_and_value(
    op: HessianOperator, *, iters: int, generator: torch.Generator | None,
) -> tuple[Vector, Tensor]:
    """Eigenvector + value of the algebraically-largest eigenvalue of ``H``.

    Two-phase: phase 1 finds the largest-magnitude eigenpair ``mu``; phase 2
    power-iterates ``H - mu*I`` to reach the opposite spectral extreme ``eta``.
    ``lambda_max = max(mu, eta)`` with its eigenvector.
    """
    v1 = _power_eigvec(op, iters=iters, generator=generator, shift=0.0, x0=None)
    mu = _rayleigh(op, v1)
    v2 = _power_eigvec(op, iters=iters, generator=generator, shift=float(mu), x0=None)
    eta = _rayleigh(op, v2)
    if float(mu) >= float(eta):
        return v1, mu
    return v2, eta


def hessian_eigenvalue_extremes(
    loss: Tensor, params: Params, *,
    iters: int = 30, generator: torch.Generator | None = None,
) -> tuple[float, float]:
    """``(lambda_min, lambda_max)`` of the parameter Hessian by power iteration."""
    op = HessianOperator(loss, params)
    v1 = _power_eigvec(op, iters=iters, generator=generator, shift=0.0, x0=None)
    mu = float(_rayleigh(op, v1))
    v2 = _power_eigvec(op, iters=iters, generator=generator, shift=mu, x0=None)
    eta = float(_rayleigh(op, v2))
    return (min(mu, eta), max(mu, eta))


def top_eigenvalue(
    loss: Tensor, params: Params, *,
    iters: int = 30, generator: torch.Generator | None = None,
    differentiable: bool = False,
) -> Tensor:
    r""":math:`\lambda_{\max}(H)` (algebraically largest) by power iteration.

    With ``differentiable=True`` returns the Rayleigh quotient at the (detached)
    top eigenvector, whose gradient is the exact Hellmann-Feynman derivative
    :math:`v^\top (\partial_\theta H)\, v`.
    """
    op = HessianOperator(loss, params)
    v_top, lam = _top_eigvec_and_value(op, iters=iters, generator=generator)
    if not differentiable:
        return lam
    hv = op.hvp(v_top, create_graph=True)
    return _flat_dot(v_top, hv)


def hutchinson_trace(
    loss: Tensor, params: Params, *,
    n_samples: int = 8, generator: torch.Generator | None = None,
    differentiable: bool = False,
) -> Tensor:
    r"""Unbiased Rademacher estimator of :math:`\operatorname{tr}(H)`."""
    op = HessianOperator(loss, params)
    acc = torch.zeros((), dtype=op.params[0].dtype, device=op.params[0].device)
    for _ in range(n_samples):
        v = _rand_like(op.params, generator=generator, rademacher=True)
        hv = op.hvp(v, create_graph=differentiable)
        acc = acc + _flat_dot(v, hv)
    return acc / n_samples


def hutchinson_frobenius_sq(
    loss: Tensor, params: Params, *,
    n_samples: int = 8, generator: torch.Generator | None = None,
    differentiable: bool = False,
) -> Tensor:
    r"""Unbiased Rademacher estimator of :math:`\lVert H\rVert_F^2`.

    Uses :math:`\mathbb E\lVert Hv\rVert^2 = \operatorname{tr}(H^\top H)` for a
    Rademacher probe ``v``.
    """
    op = HessianOperator(loss, params)
    acc = torch.zeros((), dtype=op.params[0].dtype, device=op.params[0].device)
    for _ in range(n_samples):
        v = _rand_like(op.params, generator=generator, rademacher=True)
        hv = op.hvp(v, create_graph=differentiable)
        acc = acc + _flat_dot(hv, hv)
    return acc / n_samples


# ---------------------------------------------------------------------------
# Sharpness-aware objectives
# ---------------------------------------------------------------------------

_MEASURES = ("trace", "frobenius", "top_eig")


def curvature_sharpness(
    loss: Tensor, params: Params, *,
    measure: str = "frobenius",
    n_samples: int = 8, iters: int = 30,
    generator: torch.Generator | None = None,
    differentiable: bool = False,
) -> Tensor:
    """Matrix-free curvature sharpness of ``loss`` at the current ``params``.

    ``measure`` is ``"trace"`` / ``"frobenius"`` (Hutchinson) or ``"top_eig"``
    (power iteration).
    """
    if measure == "trace":
        return hutchinson_trace(loss, params, n_samples=n_samples,
                                generator=generator, differentiable=differentiable)
    if measure == "frobenius":
        return hutchinson_frobenius_sq(loss, params, n_samples=n_samples,
                                       generator=generator, differentiable=differentiable)
    if measure == "top_eig":
        return top_eigenvalue(loss, params, iters=iters,
                              generator=generator, differentiable=differentiable)
    raise ValueError(
        f"unknown sharpness measure {measure!r}; choose from {list(_MEASURES)}"
    )


def sharpness_aware_loss(
    loss: Tensor, params: Params, *,
    lam: float = 1e-2, measure: str = "frobenius",
    n_samples: int = 8, iters: int = 30,
    generator: torch.Generator | None = None,
) -> Tensor:
    r"""Curvature-regularised loss :math:`L + \lambda\,\mathcal S(\nabla^2 L)`.

    Differentiable end-to-end; the penalty gradient rides on the closed-form
    :math:`\sigma'''` via reverse-over-reverse-over-reverse autograd.
    """
    penalty = curvature_sharpness(
        loss, params, measure=measure, n_samples=n_samples, iters=iters,
        generator=generator, differentiable=True,
    )
    return loss + lam * penalty


def sam_sharpness_gap(
    loss: Tensor, params: Params, *,
    rho: float = 0.05, iters: int = 30,
    generator: torch.Generator | None = None,
    differentiable: bool = False,
) -> Tensor:
    r"""Exact second-order SAM inner-max gap
    :math:`\rho\lVert\nabla L\rVert + \tfrac12\rho^2\max(\lambda_{\max},0)`.

    Upper-bounds :math:`\max_{\lVert\varepsilon\rVert\le\rho}L(\theta+\varepsilon)-L(\theta)`
    to :math:`O(\rho^3)`; the linear term is what classic SAM estimates with one
    ascent step, the curvature term is the exact-Hessian correction.
    """
    op = HessianOperator(loss, params)
    gnorm = op.grad_norm if differentiable else op.grad_norm.detach()
    v_top, lam = _top_eigvec_and_value(op, iters=iters, generator=generator)
    if differentiable:
        hv = op.hvp(v_top, create_graph=True)
        lam = _flat_dot(v_top, hv)
    lam_pos = torch.clamp(lam, min=0.0)
    return rho * gnorm + 0.5 * rho * rho * lam_pos


def sam_objective(
    loss: Tensor, params: Params, *,
    rho: float = 0.05, iters: int = 30,
    generator: torch.Generator | None = None,
) -> Tensor:
    r"""Ascent-free "SAM done right": :math:`L + \text{gap}_\rho`."""
    return loss + sam_sharpness_gap(
        loss, params, rho=rho, iters=iters, generator=generator, differentiable=True,
    )


__all__ = [
    "HessianOperator",
    "curvature_sharpness",
    "dense_hessian",
    "hessian_eigenvalue_extremes",
    "hessian_frobenius_sq",
    "hessian_top_eigenvalue",
    "hessian_trace",
    "hutchinson_frobenius_sq",
    "hutchinson_trace",
    "hvp",
    "sam_objective",
    "sam_sharpness_gap",
    "sharpness_aware_loss",
    "top_eigenvalue",
]
