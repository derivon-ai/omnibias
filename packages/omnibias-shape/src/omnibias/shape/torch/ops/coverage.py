# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Soft-coverage (soft-OR / log-sum-exp union) operators and closed-form curvature (torch).

The union of ``K`` soft shapes with existence gates ``alpha`` is the soft-OR
``C = 1 - prod_k (1 - alpha_k m_k)``. It is multilinear in each membership ``m_k``
(so ``d^2 C / d m_k^2 = 0``), which is what makes the coverage-energy Hessian clean.
The full derivation of ``coverage_energy_grad`` / ``coverage_energy_hessian`` is in
``examples/min_square_cover/HESSIAN.md``.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch
from omnibias.shape.torch.ops.occupancy import soft_box, soft_box_grad, soft_box_hessian
from torch import Tensor

__all__ = [
    "CoverageCache",
    "coverage_energy",
    "coverage_energy_grad",
    "coverage_energy_hessian",
    "coverage_residual",
    "lse_coverage",
    "soft_or_coverage",
]

_EPS = 1e-12
_LOSSES = ("softplus", "sq_hinge")


@dataclass
class CoverageCache:
    """Reusable products for the soft-OR gradient / Hessian."""

    coverage: Tensor  # C, shape (*grid)
    product: Tensor  # P = prod_k (1 - alpha_k m_k), shape (*grid)
    leave_one_out: Tensor  # P_{\k}, shape (K, *grid)
    one_minus: Tensor  # 1 - alpha_k m_k, shape (K, *grid)


def _gate_broadcast(gates: Tensor, ndim: int) -> Tensor:
    return gates.reshape((-1,) + (1,) * (ndim - 1))


def soft_or_coverage(occupancy: Tensor, gates: Tensor) -> tuple[Tensor, CoverageCache]:
    r"""Probabilistic-OR union ``C = 1 - prod_k (1 - gates_k * occupancy_k)``.

    Returns ``(C, CoverageCache)``; the cache holds ``P``, the leave-one-out products
    ``P_{\k}``, and ``1 - alpha_k m_k`` for the closed-form curvature.
    """
    g = _gate_broadcast(gates, occupancy.dim())
    one_minus = 1.0 - g * occupancy
    product = one_minus.prod(dim=0)
    coverage = 1.0 - product
    leave_one_out = product.unsqueeze(0) / one_minus.clamp_min(_EPS)
    return coverage, CoverageCache(
        coverage=coverage, product=product, leave_one_out=leave_one_out, one_minus=one_minus
    )


def lse_coverage(occupancy: Tensor, gates: Tensor, beta: float | Tensor) -> Tensor:
    r"""Log-sum-exp smooth-max union of the memberships ``s_k = gates_k * occupancy_k``.

    ``(1/beta) log sum_k exp(beta s_k)``; the same kernel as ``omnibias.hopfield``
    ``logsumexp_value`` and an upper smooth envelope of ``max_k s_k``.
    """
    g = _gate_broadcast(gates, occupancy.dim())
    s = g * occupancy
    return torch.logsumexp(beta * s, dim=0) / beta


def _loss_derivs(coverage: Tensor, loss: str, kappa: float) -> tuple[Tensor, Tensor, Tensor]:
    r"""Per-pixel penalty ``ell(C)`` and its first/second derivatives in ``C`` (under-coverage)."""
    if loss not in _LOSSES:
        raise ValueError(f"loss must be one of {_LOSSES}, got {loss!r}")
    under = 1.0 - coverage
    if loss == "sq_hinge":
        ell = 0.5 * under * under
        ellp = -under
        ellpp = torch.ones_like(coverage)
        return ell, ellp, ellpp
    a = kappa * under
    sa = torch.sigmoid(a)
    ell = torch.nn.functional.softplus(a)
    ellp = -kappa * sa
    ellpp = (kappa * kappa) * sa * (1.0 - sa)
    return ell, ellp, ellpp


def _over_derivs(coverage: Tensor, loss: str, kappa: float) -> tuple[Tensor, Tensor, Tensor]:
    r"""Background over-coverage penalty ``g(C)`` and its first/second derivatives in ``C``.

    Mirror of :func:`_loss_derivs` on the over-coverage ``C`` (want ``C -> 0`` on
    background pixels): ``sq_hinge`` gives ``1/2 C^2`` and ``softplus`` gives
    ``softplus(kappa C)``.
    """
    if loss not in _LOSSES:
        raise ValueError(f"loss must be one of {_LOSSES}, got {loss!r}")
    if loss == "sq_hinge":
        ell = 0.5 * coverage * coverage
        ellp = coverage
        ellpp = torch.ones_like(coverage)
        return ell, ellp, ellpp
    a = kappa * coverage
    sa = torch.sigmoid(a)
    ell = torch.nn.functional.softplus(a)
    ellp = kappa * sa
    ellpp = (kappa * kappa) * sa * (1.0 - sa)
    return ell, ellp, ellpp


def _potential_derivs(
    coverage: Tensor,
    ones_mask: Tensor,
    loss: str,
    kappa: float,
    bg_mask: Tensor | None,
    mu: float,
) -> tuple[Tensor, Tensor]:
    r"""Combined per-pixel potential derivatives ``Phi'(C)``, ``Phi''(C)``.

    ``Phi = ones_mask * ell(C) + mu * bg_mask * g(C)`` (under-coverage on the ones,
    over-coverage on the background), so both the closed-form gradient and Hessian are
    the same union machinery driven by these two aggregates.
    """
    _, ellp, ellpp = _loss_derivs(coverage, loss, kappa)
    phi_p = ellp * ones_mask
    phi_pp = ellpp * ones_mask
    if mu and bg_mask is not None:
        _, gp, gpp = _over_derivs(coverage, loss, kappa)
        phi_p = phi_p + mu * gp * bg_mask
        phi_pp = phi_pp + mu * gpp * bg_mask
    return phi_p, phi_pp


def coverage_energy(
    occupancy: Tensor,
    gates: Tensor,
    ones_mask: Tensor,
    *,
    loss: str = "softplus",
    kappa: float = 1.0,
    lam: float = 0.0,
    bg_mask: Tensor | None = None,
    mu: float = 0.0,
    union: str = "soft_or",
    beta_u: float = 10.0,
) -> Tensor:
    r"""Scalar coverage energy ``sum_{ones} ell(C) + mu sum_{bg} g(C) + lam sum_k gates_k``.

    ``ell`` is ``"softplus"`` (smooth hinge on the under-coverage ``1 - C``) or
    ``"sq_hinge"`` (``1/2 (1 - C)^2``). ``lam`` is the L0 count surrogate. The optional
    background-tightness term (``bg_mask`` + ``mu > 0``) penalises coverage of the
    0-pixels with the over-coverage penalty ``g(C)``; it is off by default. ``union``
    selects the soft-OR (default) or the log-sum-exp union (with sharpness ``beta_u``).
    """
    if union == "lse":
        coverage = lse_coverage(occupancy, gates, beta_u)
    elif union == "soft_or":
        coverage, _ = soft_or_coverage(occupancy, gates)
    else:
        raise ValueError(f"union must be 'soft_or' or 'lse', got {union!r}")
    ell, _, _ = _loss_derivs(coverage, loss, kappa)
    energy = (ell * ones_mask).sum()
    if mu and bg_mask is not None:
        g, _, _ = _over_derivs(coverage, loss, kappa)
        energy = energy + mu * (g * bg_mask).sum()
    if lam:
        energy = energy + lam * gates.sum()
    return energy


def coverage_residual(
    occupancy: Tensor, gates: Tensor, ones_mask: Tensor, *, weight: float = 1.0
) -> Tensor:
    r"""Under-coverage residual ``sqrt(weight) (1 - C)`` at the 1-pixels (for Gauss-Newton).

    The objective ``1/2 ||r||^2`` equals the ``sq_hinge`` coverage energy.
    """
    coverage, _ = soft_or_coverage(occupancy, gates)
    residual = (weight**0.5) * (1.0 - coverage)
    idx = ones_mask.reshape(-1).to(torch.bool)
    selected: Tensor = residual.reshape(-1)[idx]
    return selected


def _lse_membership(occ: Tensor, gates: Tensor, beta_u: float) -> Tensor:
    r"""Per-shape softmax weight ``p_k = softmax_k(beta_u * gates_k * occ_k)``, shape ``(K, *grid)``."""
    g = _gate_broadcast(gates, occ.dim())
    return torch.softmax(beta_u * (g * occ), dim=0)


def _lse_energy_grad(
    axes: Sequence[Tensor],
    centers: Tensor,
    side: float | Tensor,
    beta: float | Tensor,
    gates: Tensor,
    ones_mask: Tensor,
    beta_u: float,
    loss: str,
    kappa: float,
    lam: float,
    bg_mask: Tensor | None,
    mu: float,
    wrt: str,
) -> Tensor:
    r"""Closed-form gradient of the log-sum-exp-union coverage energy (``HESSIAN.md`` section 8)."""
    occ = soft_box(axes, centers, side, beta)
    grad_m = soft_box_grad(axes, centers, side, beta)
    coverage = lse_coverage(occ, gates, beta_u)
    ellp, _ = _potential_derivs(coverage, ones_mask, loss, kappa, bg_mask, mu)
    k, d = centers.shape
    p = _lse_membership(occ, gates, beta_u)
    g = _gate_broadcast(gates, occ.dim())
    dm = p * g  # dC/dm_k = p_k alpha_k
    grad_c = centers.new_zeros((k * d,))
    for kk in range(k):
        weight = ellp * dm[kk]
        for dd in range(d):
            grad_c[kk * d + dd] = (weight * grad_m[kk, dd]).sum()
    if wrt == "centers":
        return grad_c
    sp = gates * (1.0 - gates)
    dalpha = p * occ  # dC/dalpha_k = p_k m_k
    grad_g = centers.new_zeros((k,))
    for kk in range(k):
        grad_g[kk] = sp[kk] * ((ellp * dalpha[kk]).sum() + lam)
    return torch.cat([grad_c, grad_g])


def _lse_energy_hessian(
    axes: Sequence[Tensor],
    centers: Tensor,
    side: float | Tensor,
    beta: float | Tensor,
    gates: Tensor,
    ones_mask: Tensor,
    beta_u: float,
    loss: str,
    kappa: float,
    lam: float,
    bg_mask: Tensor | None,
    mu: float,
    gauss_newton: bool,
    wrt: str,
) -> Tensor:
    r"""Closed-form dense Hessian of the log-sum-exp-union coverage energy (``HESSIAN.md`` section 8).

    Same assembly as the soft-OR Hessian, but the union second derivatives use the
    softmax structure ``d^2 C / d s_k d s_l = beta_u (p_k delta_kl - p_k p_l)`` (so the
    same-square block picks up a nonzero ``d^2 C / d m_k^2`` diagonal term), chained
    through ``s_k = alpha_k m_k`` and ``alpha_k = sigmoid(a_k)``.
    """
    occ = soft_box(axes, centers, side, beta)
    grad_m = soft_box_grad(axes, centers, side, beta)
    hess_m = soft_box_hessian(axes, centers, side, beta)
    coverage = lse_coverage(occ, gates, beta_u)
    ellp, ellpp = _potential_derivs(coverage, ones_mask, loss, kappa, bg_mask, mu)
    k, d = centers.shape
    p = _lse_membership(occ, gates, beta_u)
    dm = p * _gate_broadcast(gates, occ.dim())  # dC/dm_k
    dalpha = p * occ  # dC/dalpha_k
    n = k * d + (k if wrt == "all" else 0)
    hess = centers.new_zeros((n, n))
    for kk in range(k):
        d2diag = beta_u * p[kk] * (1.0 - p[kk]) * (gates[kk] * gates[kk])  # d^2 C / d m_k^2
        for a in range(d):
            for b in range(d):
                val = ellpp * dm[kk] * dm[kk] * grad_m[kk, a] * grad_m[kk, b]
                if not gauss_newton:
                    val = val + ellp * (
                        d2diag * grad_m[kk, a] * grad_m[kk, b] + dm[kk] * hess_m[kk, a, b]
                    )
                hess[kk * d + a, kk * d + b] += val.sum()
        for ll in range(k):
            if ll == kk:
                continue
            d2off = -beta_u * p[kk] * p[ll] * gates[kk] * gates[ll]  # d^2 C / d m_k d m_l
            coef = ellpp * dm[kk] * dm[ll]
            if not gauss_newton:
                coef = coef + ellp * d2off
            for a in range(d):
                for b in range(d):
                    hess[kk * d + a, ll * d + b] += (coef * grad_m[kk, a] * grad_m[ll, b]).sum()
    if wrt == "centers":
        return hess

    sp = gates * (1.0 - gates)
    spp = sp * (1.0 - 2.0 * gates)
    base = k * d
    for kk in range(k):
        d2aa_diag = beta_u * p[kk] * (1.0 - p[kk]) * (occ[kk] * occ[kk])
        h_alpha = (ellpp * dalpha[kk] * dalpha[kk]).sum()
        if not gauss_newton:
            h_alpha = h_alpha + (ellp * d2aa_diag).sum()
        diag = sp[kk] * sp[kk] * h_alpha
        if not gauss_newton:
            diag = diag + spp[kk] * ((ellp * dalpha[kk]).sum() + lam)
        hess[base + kk, base + kk] += diag
        for ll in range(k):
            if ll == kk:
                continue
            d2aa_off = -beta_u * p[kk] * p[ll] * occ[kk] * occ[ll]
            coef = ellpp * dalpha[kk] * dalpha[ll]
            if not gauss_newton:
                coef = coef + ellp * d2aa_off
            hess[base + kk, base + ll] += sp[kk] * sp[ll] * coef.sum()
    for kk in range(k):
        for a in range(d):
            inner = ellpp * dm[kk] * dalpha[kk]
            if not gauss_newton:
                d2ma_same = beta_u * p[kk] * (1.0 - p[kk]) * gates[kk] * occ[kk] + p[kk]
                inner = inner + ellp * d2ma_same
            same = sp[kk] * (grad_m[kk, a] * inner).sum()
            hess[kk * d + a, base + kk] += same
            hess[base + kk, kk * d + a] += same
            for ll in range(k):
                if ll == kk:
                    continue
                coef = ellpp * dm[kk] * dalpha[ll]
                if not gauss_newton:
                    d2ma_diff = -beta_u * p[kk] * p[ll] * gates[kk] * occ[ll]
                    coef = coef + ellp * d2ma_diff
                mixed = sp[ll] * (grad_m[kk, a] * coef).sum()
                hess[kk * d + a, base + ll] += mixed
                hess[base + ll, kk * d + a] += mixed
    return hess


def coverage_energy_grad(
    axes: Sequence[Tensor],
    centers: Tensor,
    side: float | Tensor,
    beta: float | Tensor,
    gates: Tensor,
    ones_mask: Tensor,
    *,
    loss: str = "softplus",
    kappa: float = 1.0,
    lam: float = 0.0,
    bg_mask: Tensor | None = None,
    mu: float = 0.0,
    union: str = "soft_or",
    beta_u: float = 10.0,
    wrt: str = "centers",
) -> Tensor:
    r"""Closed-form gradient of ``coverage_energy``.

    With ``wrt="centers"`` (default) the result is the center gradient of shape
    ``(K*D,)``, flattened row-major over ``(k, d)`` so entry ``k*D + d`` is
    ``dE/d centers[k, d]``. With ``wrt="all"`` the gate-logit gradient is appended
    (``alpha_k = sigmoid(a_k)``, so the tail entry ``K*D + k`` is ``dE/d a_k``),
    including the ``lam`` count term; ``gates`` is read as ``alpha`` and the
    ``sigmoid'`` chain is applied. The optional ``bg_mask`` + ``mu`` background term
    enters through the per-pixel potential ``Phi'(C)``. ``union="lse"`` (with sharpness
    ``beta_u``) uses the log-sum-exp union instead of the soft-OR. Matches ``HESSIAN.md``
    sections 7-8.
    """
    if wrt not in ("centers", "all"):
        raise ValueError(f"wrt must be 'centers' or 'all', got {wrt!r}")
    if union not in ("soft_or", "lse"):
        raise ValueError(f"union must be 'soft_or' or 'lse', got {union!r}")
    if union == "lse":
        return _lse_energy_grad(
            axes, centers, side, beta, gates, ones_mask, beta_u, loss, kappa, lam, bg_mask, mu, wrt
        )
    occ = soft_box(axes, centers, side, beta)
    grad_m = soft_box_grad(axes, centers, side, beta)
    coverage, cache = soft_or_coverage(occ, gates)
    ellp, _ = _potential_derivs(coverage, ones_mask, loss, kappa, bg_mask, mu)
    k, d = centers.shape
    loo = cache.leave_one_out
    g = _gate_broadcast(gates, occ.dim())
    d_cov_dm = g * loo  # dC/dm_k = alpha_k P_{\k}
    grad_c = centers.new_zeros((k * d,))
    for kk in range(k):
        weight = ellp * d_cov_dm[kk]
        for dd in range(d):
            grad_c[kk * d + dd] = (weight * grad_m[kk, dd]).sum()
    if wrt == "centers":
        return grad_c
    sp = gates * (1.0 - gates)  # sigma'(a_k) as a function of alpha_k
    grad_g = centers.new_zeros((k,))
    for kk in range(k):
        # dC/dalpha_k = m_k P_{\k}; add the count term lam, then chain sigma'(a_k)
        s_alpha = (ellp * occ[kk] * loo[kk]).sum() + lam
        grad_g[kk] = sp[kk] * s_alpha
    return torch.cat([grad_c, grad_g])


def coverage_energy_hessian(
    axes: Sequence[Tensor],
    centers: Tensor,
    side: float | Tensor,
    beta: float | Tensor,
    gates: Tensor,
    ones_mask: Tensor,
    *,
    loss: str = "softplus",
    kappa: float = 1.0,
    lam: float = 0.0,
    bg_mask: Tensor | None = None,
    mu: float = 0.0,
    union: str = "soft_or",
    beta_u: float = 10.0,
    gauss_newton: bool = False,
    wrt: str = "centers",
) -> Tensor:
    r"""Closed-form dense Hessian of ``coverage_energy``.

    With ``wrt="centers"`` (default) the result is the center Hessian of shape
    ``(K*D, K*D)`` (the same-square and cross-square blocks of ``HESSIAN.md``
    section 5). With ``wrt="all"`` the gate-logit blocks and center-gate mixed
    blocks (section 7) are appended, giving a ``(K*D + K, K*D + K)`` matrix ordered
    ``[centers row-major, gate logits]``; ``gates`` is read as ``alpha`` and the
    ``sigmoid'`` / ``sigmoid''`` chain is applied. The ``lam`` count term enters
    only the gate diagonal; the optional ``bg_mask`` + ``mu`` background term enters
    through the per-pixel potential ``Phi''(C)``. ``union="lse"`` (with sharpness
    ``beta_u``) uses the log-sum-exp union (section 8) instead of the soft-OR.

    With ``gauss_newton=True`` only the PSD ``J^T J`` (``ell''``) terms are kept;
    the residual-curvature (``ell'``) and gate-map curvature (``sigmoid''``, ``lam``)
    terms are dropped.
    """
    if wrt not in ("centers", "all"):
        raise ValueError(f"wrt must be 'centers' or 'all', got {wrt!r}")
    if union not in ("soft_or", "lse"):
        raise ValueError(f"union must be 'soft_or' or 'lse', got {union!r}")
    if union == "lse":
        return _lse_energy_hessian(
            axes, centers, side, beta, gates, ones_mask, beta_u, loss, kappa, lam,
            bg_mask, mu, gauss_newton, wrt,
        )
    occ = soft_box(axes, centers, side, beta)
    grad_m = soft_box_grad(axes, centers, side, beta)
    hess_m = soft_box_hessian(axes, centers, side, beta)
    coverage, cache = soft_or_coverage(occ, gates)
    ellp, ellpp = _potential_derivs(coverage, ones_mask, loss, kappa, bg_mask, mu)
    k, d = centers.shape
    loo = cache.leave_one_out
    one_minus = cache.one_minus
    dcov = _gate_broadcast(gates, occ.dim()) * loo  # dC/dm_k = alpha_k P_{\k}
    n = k * d + (k if wrt == "all" else 0)
    hess = centers.new_zeros((n, n))
    for kk in range(k):
        # same-square block (k, k)
        for a in range(d):
            for b in range(d):
                val = ellpp * dcov[kk] * dcov[kk] * grad_m[kk, a] * grad_m[kk, b]
                if not gauss_newton:
                    val = val + ellp * dcov[kk] * hess_m[kk, a, b]
                hess[kk * d + a, kk * d + b] += val.sum()
        # cross-square blocks (k, l), l != k
        for ll in range(k):
            if ll == kk:
                continue
            leave_two = loo[kk] / one_minus[ll].clamp_min(_EPS)  # P_{\{k,l}}
            coef = ellpp * dcov[kk] * dcov[ll]
            if not gauss_newton:
                coef = coef - ellp * (gates[kk] * gates[ll]) * leave_two
            for a in range(d):
                for b in range(d):
                    hess[kk * d + a, ll * d + b] += (coef * grad_m[kk, a] * grad_m[ll, b]).sum()
    if wrt == "centers":
        return hess

    sp = gates * (1.0 - gates)  # sigma'(a_k)
    spp = sp * (1.0 - 2.0 * gates)  # sigma''(a_k)
    base = k * d
    for kk in range(k):
        dalpha_k = occ[kk] * loo[kk]  # dC/dalpha_k = m_k P_{\k}
        # gate diagonal (k, k)
        diag = (ellpp * (sp[kk] * dalpha_k) ** 2).sum()
        if not gauss_newton:
            diag = diag + spp[kk] * ((ellp * dalpha_k).sum() + lam)
        hess[base + kk, base + kk] += diag
        # gate off-diagonal (k, l), l != k
        for ll in range(k):
            if ll == kk:
                continue
            dalpha_l = occ[ll] * loo[ll]
            leave_two = loo[kk] / one_minus[ll].clamp_min(_EPS)
            coef = ellpp * dalpha_k * dalpha_l
            if not gauss_newton:
                coef = coef - ellp * occ[kk] * occ[ll] * leave_two
            hess[base + kk, base + ll] += sp[kk] * sp[ll] * coef.sum()
    # center-gate mixed blocks
    for kk in range(k):
        for a in range(d):
            inner_same = ellpp * gates[kk] * loo[kk] * occ[kk]
            if not gauss_newton:
                inner_same = inner_same + ellp
            same = sp[kk] * (loo[kk] * grad_m[kk, a] * inner_same).sum()
            hess[kk * d + a, base + kk] += same
            hess[base + kk, kk * d + a] += same
            for ll in range(k):
                if ll == kk:
                    continue
                leave_two = loo[kk] / one_minus[ll].clamp_min(_EPS)
                inner = ellpp * loo[kk] * loo[ll]
                if not gauss_newton:
                    inner = inner - ellp * leave_two
                mixed = sp[ll] * gates[kk] * (grad_m[kk, a] * occ[ll] * inner).sum()
                hess[kk * d + a, base + ll] += mixed
                hess[base + ll, kk * d + a] += mixed
    return hess
