# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Soft-coverage operators and closed-form curvature (jax).

Bit-identical algorithm to :mod:`omnibias.shape.torch.ops.coverage`.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import jax.numpy as jnp
from jax import Array, nn
from jax.scipy.special import logsumexp
from omnibias.shape.jax.ops.occupancy import soft_box, soft_box_grad, soft_box_hessian

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

    coverage: Array
    product: Array
    leave_one_out: Array
    one_minus: Array


def _gate_broadcast(gates: Array, ndim: int) -> Array:
    return gates.reshape((-1,) + (1,) * (ndim - 1))


def soft_or_coverage(occupancy: Array, gates: Array) -> tuple[Array, CoverageCache]:
    r"""Probabilistic-OR union ``C = 1 - prod_k (1 - gates_k * occupancy_k)``."""
    g = _gate_broadcast(gates, occupancy.ndim)
    one_minus = 1.0 - g * occupancy
    product = jnp.prod(one_minus, axis=0)
    coverage = 1.0 - product
    leave_one_out = product.reshape((1, *product.shape)) / jnp.maximum(one_minus, _EPS)
    return coverage, CoverageCache(
        coverage=coverage, product=product, leave_one_out=leave_one_out, one_minus=one_minus
    )


def lse_coverage(occupancy: Array, gates: Array, beta: float | Array) -> Array:
    r"""Log-sum-exp smooth-max union of the memberships ``s_k = gates_k * occupancy_k``."""
    g = _gate_broadcast(gates, occupancy.ndim)
    s = g * occupancy
    return logsumexp(beta * s, axis=0) / beta


def _loss_derivs(coverage: Array, loss: str, kappa: float) -> tuple[Array, Array, Array]:
    if loss not in _LOSSES:
        raise ValueError(f"loss must be one of {_LOSSES}, got {loss!r}")
    under = 1.0 - coverage
    if loss == "sq_hinge":
        ell = 0.5 * under * under
        ellp = -under
        ellpp = jnp.ones_like(coverage)
        return ell, ellp, ellpp
    a = kappa * under
    sa = nn.sigmoid(a)
    ell = nn.softplus(a)
    ellp = -kappa * sa
    ellpp = (kappa * kappa) * sa * (1.0 - sa)
    return ell, ellp, ellpp


def _over_derivs(coverage: Array, loss: str, kappa: float) -> tuple[Array, Array, Array]:
    r"""Background over-coverage penalty ``g(C)`` and its derivatives (want ``C -> 0``)."""
    if loss not in _LOSSES:
        raise ValueError(f"loss must be one of {_LOSSES}, got {loss!r}")
    if loss == "sq_hinge":
        ell = 0.5 * coverage * coverage
        ellp = coverage
        ellpp = jnp.ones_like(coverage)
        return ell, ellp, ellpp
    a = kappa * coverage
    sa = nn.sigmoid(a)
    ell = nn.softplus(a)
    ellp = kappa * sa
    ellpp = (kappa * kappa) * sa * (1.0 - sa)
    return ell, ellp, ellpp


def _potential_derivs(
    coverage: Array,
    ones_mask: Array,
    loss: str,
    kappa: float,
    bg_mask: Array | None,
    mu: float,
) -> tuple[Array, Array]:
    r"""Combined per-pixel potential derivatives ``Phi'(C)``, ``Phi''(C)``."""
    _, ellp, ellpp = _loss_derivs(coverage, loss, kappa)
    phi_p = ellp * ones_mask
    phi_pp = ellpp * ones_mask
    if mu and bg_mask is not None:
        _, gp, gpp = _over_derivs(coverage, loss, kappa)
        phi_p = phi_p + mu * gp * bg_mask
        phi_pp = phi_pp + mu * gpp * bg_mask
    return phi_p, phi_pp


def coverage_energy(
    occupancy: Array,
    gates: Array,
    ones_mask: Array,
    *,
    loss: str = "softplus",
    kappa: float = 1.0,
    lam: float = 0.0,
    bg_mask: Array | None = None,
    mu: float = 0.0,
    union: str = "soft_or",
    beta_u: float = 10.0,
) -> Array:
    r"""Scalar coverage energy ``sum_{ones} ell(C) + mu sum_{bg} g(C) + lam sum_k gates_k``."""
    if union == "lse":
        coverage = lse_coverage(occupancy, gates, beta_u)
    elif union == "soft_or":
        coverage, _ = soft_or_coverage(occupancy, gates)
    else:
        raise ValueError(f"union must be 'soft_or' or 'lse', got {union!r}")
    ell, _, _ = _loss_derivs(coverage, loss, kappa)
    energy = jnp.sum(ell * ones_mask)
    if mu and bg_mask is not None:
        g, _, _ = _over_derivs(coverage, loss, kappa)
        energy = energy + mu * jnp.sum(g * bg_mask)
    if lam:
        energy = energy + lam * jnp.sum(gates)
    return energy


def coverage_residual(
    occupancy: Array, gates: Array, ones_mask: Array, *, weight: float = 1.0
) -> Array:
    r"""Under-coverage residual ``sqrt(weight) (1 - C)`` at the 1-pixels (for Gauss-Newton)."""
    coverage, _ = soft_or_coverage(occupancy, gates)
    residual = (weight**0.5) * (1.0 - coverage)
    idx = ones_mask.reshape(-1) > 0.5
    selected: Array = residual.reshape(-1)[idx]
    return selected


def _lse_membership(occ: Array, gates: Array, beta_u: float) -> Array:
    r"""Per-shape softmax weight ``p_k = softmax_k(beta_u * gates_k * occ_k)``."""
    g = _gate_broadcast(gates, occ.ndim)
    return nn.softmax(beta_u * (g * occ), axis=0)


def _lse_energy_grad(
    axes: Sequence[Array],
    centers: Array,
    side: float | Array,
    beta: float | Array,
    gates: Array,
    ones_mask: Array,
    beta_u: float,
    loss: str,
    kappa: float,
    lam: float,
    bg_mask: Array | None,
    mu: float,
    wrt: str,
) -> Array:
    r"""Closed-form gradient of the log-sum-exp-union coverage energy (bit-identical twin)."""
    occ = soft_box(axes, centers, side, beta)
    grad_m = soft_box_grad(axes, centers, side, beta)
    coverage = lse_coverage(occ, gates, beta_u)
    ellp, _ = _potential_derivs(coverage, ones_mask, loss, kappa, bg_mask, mu)
    k, d = centers.shape
    p = _lse_membership(occ, gates, beta_u)
    g = _gate_broadcast(gates, occ.ndim)
    dm = p * g
    grad_c = jnp.zeros((k * d,), dtype=centers.dtype)
    for kk in range(k):
        weight = ellp * dm[kk]
        for dd in range(d):
            grad_c = grad_c.at[kk * d + dd].set(jnp.sum(weight * grad_m[kk, dd]))
    if wrt == "centers":
        return grad_c
    sp = gates * (1.0 - gates)
    dalpha = p * occ
    grad_g = jnp.zeros((k,), dtype=centers.dtype)
    for kk in range(k):
        grad_g = grad_g.at[kk].set(sp[kk] * (jnp.sum(ellp * dalpha[kk]) + lam))
    return jnp.concatenate([grad_c, grad_g])


def _lse_energy_hessian(
    axes: Sequence[Array],
    centers: Array,
    side: float | Array,
    beta: float | Array,
    gates: Array,
    ones_mask: Array,
    beta_u: float,
    loss: str,
    kappa: float,
    lam: float,
    bg_mask: Array | None,
    mu: float,
    gauss_newton: bool,
    wrt: str,
) -> Array:
    r"""Closed-form dense Hessian of the log-sum-exp-union coverage energy (bit-identical twin)."""
    occ = soft_box(axes, centers, side, beta)
    grad_m = soft_box_grad(axes, centers, side, beta)
    hess_m = soft_box_hessian(axes, centers, side, beta)
    coverage = lse_coverage(occ, gates, beta_u)
    ellp, ellpp = _potential_derivs(coverage, ones_mask, loss, kappa, bg_mask, mu)
    k, d = centers.shape
    p = _lse_membership(occ, gates, beta_u)
    dm = p * _gate_broadcast(gates, occ.ndim)
    dalpha = p * occ
    n = k * d + (k if wrt == "all" else 0)
    hess = jnp.zeros((n, n), dtype=centers.dtype)
    for kk in range(k):
        d2diag = beta_u * p[kk] * (1.0 - p[kk]) * (gates[kk] * gates[kk])
        for a in range(d):
            for b in range(d):
                val = ellpp * dm[kk] * dm[kk] * grad_m[kk, a] * grad_m[kk, b]
                if not gauss_newton:
                    val = val + ellp * (
                        d2diag * grad_m[kk, a] * grad_m[kk, b] + dm[kk] * hess_m[kk, a, b]
                    )
                hess = hess.at[kk * d + a, kk * d + b].add(jnp.sum(val))
        for ll in range(k):
            if ll == kk:
                continue
            d2off = -beta_u * p[kk] * p[ll] * gates[kk] * gates[ll]
            coef = ellpp * dm[kk] * dm[ll]
            if not gauss_newton:
                coef = coef + ellp * d2off
            for a in range(d):
                for b in range(d):
                    hess = hess.at[kk * d + a, ll * d + b].add(
                        jnp.sum(coef * grad_m[kk, a] * grad_m[ll, b])
                    )
    if wrt == "centers":
        return hess

    sp = gates * (1.0 - gates)
    spp = sp * (1.0 - 2.0 * gates)
    base = k * d
    for kk in range(k):
        d2aa_diag = beta_u * p[kk] * (1.0 - p[kk]) * (occ[kk] * occ[kk])
        h_alpha = jnp.sum(ellpp * dalpha[kk] * dalpha[kk])
        if not gauss_newton:
            h_alpha = h_alpha + jnp.sum(ellp * d2aa_diag)
        diag = sp[kk] * sp[kk] * h_alpha
        if not gauss_newton:
            diag = diag + spp[kk] * (jnp.sum(ellp * dalpha[kk]) + lam)
        hess = hess.at[base + kk, base + kk].add(diag)
        for ll in range(k):
            if ll == kk:
                continue
            d2aa_off = -beta_u * p[kk] * p[ll] * occ[kk] * occ[ll]
            coef = ellpp * dalpha[kk] * dalpha[ll]
            if not gauss_newton:
                coef = coef + ellp * d2aa_off
            hess = hess.at[base + kk, base + ll].add(sp[kk] * sp[ll] * jnp.sum(coef))
    for kk in range(k):
        for a in range(d):
            inner = ellpp * dm[kk] * dalpha[kk]
            if not gauss_newton:
                d2ma_same = beta_u * p[kk] * (1.0 - p[kk]) * gates[kk] * occ[kk] + p[kk]
                inner = inner + ellp * d2ma_same
            same = sp[kk] * jnp.sum(grad_m[kk, a] * inner)
            hess = hess.at[kk * d + a, base + kk].add(same)
            hess = hess.at[base + kk, kk * d + a].add(same)
            for ll in range(k):
                if ll == kk:
                    continue
                coef = ellpp * dm[kk] * dalpha[ll]
                if not gauss_newton:
                    d2ma_diff = -beta_u * p[kk] * p[ll] * gates[kk] * occ[ll]
                    coef = coef + ellp * d2ma_diff
                mixed = sp[ll] * jnp.sum(grad_m[kk, a] * coef)
                hess = hess.at[kk * d + a, base + ll].add(mixed)
                hess = hess.at[base + ll, kk * d + a].add(mixed)
    return hess


def coverage_energy_grad(
    axes: Sequence[Array],
    centers: Array,
    side: float | Array,
    beta: float | Array,
    gates: Array,
    ones_mask: Array,
    *,
    loss: str = "softplus",
    kappa: float = 1.0,
    lam: float = 0.0,
    bg_mask: Array | None = None,
    mu: float = 0.0,
    union: str = "soft_or",
    beta_u: float = 10.0,
    wrt: str = "centers",
) -> Array:
    r"""Closed-form gradient of ``coverage_energy``.

    ``wrt="centers"`` returns the ``(K*D,)`` center gradient; ``wrt="all"`` appends
    the gate-logit gradient (``alpha_k = sigmoid(a_k)``) including the ``lam`` count
    term. The optional ``bg_mask`` + ``mu`` background term enters via ``Phi'(C)``.
    ``union="lse"`` uses the log-sum-exp union. Bit-identical to the torch twin;
    see ``HESSIAN.md`` sections 7-8.
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
    g = _gate_broadcast(gates, occ.ndim)
    d_cov_dm = g * loo
    grad_c = jnp.zeros((k * d,), dtype=centers.dtype)
    for kk in range(k):
        weight = ellp * d_cov_dm[kk]
        for dd in range(d):
            grad_c = grad_c.at[kk * d + dd].set(jnp.sum(weight * grad_m[kk, dd]))
    if wrt == "centers":
        return grad_c
    sp = gates * (1.0 - gates)
    grad_g = jnp.zeros((k,), dtype=centers.dtype)
    for kk in range(k):
        s_alpha = jnp.sum(ellp * occ[kk] * loo[kk]) + lam
        grad_g = grad_g.at[kk].set(sp[kk] * s_alpha)
    return jnp.concatenate([grad_c, grad_g])


def coverage_energy_hessian(
    axes: Sequence[Array],
    centers: Array,
    side: float | Array,
    beta: float | Array,
    gates: Array,
    ones_mask: Array,
    *,
    loss: str = "softplus",
    kappa: float = 1.0,
    lam: float = 0.0,
    bg_mask: Array | None = None,
    mu: float = 0.0,
    union: str = "soft_or",
    beta_u: float = 10.0,
    gauss_newton: bool = False,
    wrt: str = "centers",
) -> Array:
    r"""Closed-form dense Hessian of ``coverage_energy``.

    ``wrt="centers"`` returns the ``(K*D, K*D)`` center Hessian; ``wrt="all"``
    appends the gate-logit and center-gate mixed blocks (``HESSIAN.md`` section 7),
    giving ``(K*D + K, K*D + K)`` ordered ``[centers, gate logits]``. The ``lam``
    count term enters only the gate diagonal; the optional ``bg_mask`` + ``mu``
    background term enters via ``Phi''(C)``. ``union="lse"`` uses the log-sum-exp
    union (section 8). Bit-identical to the torch twin.
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
    dcov = _gate_broadcast(gates, occ.ndim) * loo
    n = k * d + (k if wrt == "all" else 0)
    hess = jnp.zeros((n, n), dtype=centers.dtype)
    for kk in range(k):
        for a in range(d):
            for b in range(d):
                val = ellpp * dcov[kk] * dcov[kk] * grad_m[kk, a] * grad_m[kk, b]
                if not gauss_newton:
                    val = val + ellp * dcov[kk] * hess_m[kk, a, b]
                hess = hess.at[kk * d + a, kk * d + b].add(jnp.sum(val))
        for ll in range(k):
            if ll == kk:
                continue
            leave_two = loo[kk] / jnp.maximum(one_minus[ll], _EPS)
            coef = ellpp * dcov[kk] * dcov[ll]
            if not gauss_newton:
                coef = coef - ellp * (gates[kk] * gates[ll]) * leave_two
            for a in range(d):
                for b in range(d):
                    hess = hess.at[kk * d + a, ll * d + b].add(
                        jnp.sum(coef * grad_m[kk, a] * grad_m[ll, b])
                    )
    if wrt == "centers":
        return hess

    sp = gates * (1.0 - gates)
    spp = sp * (1.0 - 2.0 * gates)
    base = k * d
    for kk in range(k):
        dalpha_k = occ[kk] * loo[kk]
        diag = jnp.sum(ellpp * (sp[kk] * dalpha_k) ** 2)
        if not gauss_newton:
            diag = diag + spp[kk] * (jnp.sum(ellp * dalpha_k) + lam)
        hess = hess.at[base + kk, base + kk].add(diag)
        for ll in range(k):
            if ll == kk:
                continue
            dalpha_l = occ[ll] * loo[ll]
            leave_two = loo[kk] / jnp.maximum(one_minus[ll], _EPS)
            coef = ellpp * dalpha_k * dalpha_l
            if not gauss_newton:
                coef = coef - ellp * occ[kk] * occ[ll] * leave_two
            hess = hess.at[base + kk, base + ll].add(sp[kk] * sp[ll] * jnp.sum(coef))
    for kk in range(k):
        for a in range(d):
            inner_same = ellpp * gates[kk] * loo[kk] * occ[kk]
            if not gauss_newton:
                inner_same = inner_same + ellp
            same = sp[kk] * jnp.sum(loo[kk] * grad_m[kk, a] * inner_same)
            hess = hess.at[kk * d + a, base + kk].add(same)
            hess = hess.at[base + kk, kk * d + a].add(same)
            for ll in range(k):
                if ll == kk:
                    continue
                leave_two = loo[kk] / jnp.maximum(one_minus[ll], _EPS)
                inner = ellpp * loo[kk] * loo[ll]
                if not gauss_newton:
                    inner = inner - ellp * leave_two
                mixed = sp[ll] * gates[kk] * jnp.sum(grad_m[kk, a] * occ[ll] * inner)
                hess = hess.at[kk * d + a, base + ll].add(mixed)
                hess = hess.at[base + ll, kk * d + a].add(mixed)
    return hess
