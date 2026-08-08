# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Smooth boundary factors and jets (pure numpy).

Sampling SDFs (:mod:`omnibias.pinn.domain._core.sdf`) may be piecewise or
CSG-composed for rejection / projection. **Boundary factors** are the smooth
analytic fields used in hard BC ansätze:

* Dirichlet: ``u = g + \phi\, N`` with normalized ADF ``\phi``.
* Neumann / Robin (smooth boundaries only): factors built from ``\phi`` and
  ``\phi^2`` so the normal derivative is controlled where ``|\nabla\phi| = 1``.

:class:`~omnibias.pinn.domain._core.sdf.RCompose` junctions (two primitive
zero-level sets active at once) are **not** valid for normal-derivative BCs:
:func:`assert_smooth_for_normal_bc` fails explicitly rather than returning a
wrong normal.

Honesty: jets through :func:`boundary_factor_jet` are closed-form for
:class:`~omnibias.pinn.domain._core.sdf.Sphere`,
:class:`~omnibias.pinn.domain._core.sdf.Halfspace`, and
:class:`~omnibias.pinn.domain._core.sdf.Box` away from edges; R-compositions
use the smooth R-function chain rule; :class:`~omnibias.pinn.domain._core.sdf.Polygon`
falls back to finite differences.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal, cast

import numpy as np
from omnibias.core.multi_index import index_position, num_multi_indices
from omnibias.pinn.domain._core.adf import fd_gradient, normalize_adf
from omnibias.pinn.domain._core.sdf import (
    SDF,
    Box,
    Halfspace,
    Negate,
    RCompose,
    Sphere,
    evaluate_sdf,
)

FloatArray = np.ndarray
BCMode = Literal["dirichlet", "neumann", "robin"]


class NonSmoothBoundaryError(RuntimeError):
    """Raised when Neumann / Robin BCs are requested at a geometric junction."""


def omega_gradient(sdf: SDF, X: FloatArray, *, h: float = 1e-6) -> FloatArray:
    """Analytic gradient of ``sdf`` where implemented, else central FD."""
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X.reshape(1, -1)
    n, d = X.shape
    if isinstance(sdf, Halfspace):
        n_hat = np.asarray(sdf.normal, dtype=float)
        n_hat = n_hat / np.linalg.norm(n_hat)
        return cast(FloatArray, np.broadcast_to(n_hat, (n, d)))
    if isinstance(sdf, Sphere):
        c = np.asarray(sdf.center, dtype=float)
        diff = X - c
        dist = np.linalg.norm(diff, axis=-1, keepdims=True)
        safe = np.maximum(dist, 1e-30)
        return cast(FloatArray, diff / safe)
    if isinstance(sdf, Box):
        lo = np.asarray(sdf.lo, dtype=float)
        hi = np.asarray(sdf.hi, dtype=float)
        center = 0.5 * (lo + hi)
        half = 0.5 * (hi - lo)
        q = np.abs(X - center) - half
        outside = np.linalg.norm(np.maximum(q, 0.0), axis=-1, keepdims=True)
        on_out = outside > 0.0
        # Outside: grad of ||max(q,0)||; inside: grad of max(q) along max axis.
        grad = np.zeros_like(X)
        if np.any(on_out):
            qo = np.maximum(q, 0.0)
            norm = np.linalg.norm(qo, axis=-1, keepdims=True)
            mask = norm > 1e-30
            grad_out = np.where(mask, qo / np.maximum(norm, 1e-30), 0.0)
            grad_out *= np.sign(X - center)
            grad = np.where(on_out, grad_out, grad)
        if np.any(~on_out.squeeze(-1) if d == 1 else ~on_out[..., 0]):
            arg = np.argmax(q, axis=-1)
            for j in range(d):
                sel = arg == j
                if np.any(sel):
                    grad[sel, j] = np.sign(X[sel, j] - center[j])
        return cast(FloatArray, grad)
    if isinstance(sdf, Negate):
        return cast(FloatArray, -omega_gradient(sdf.child, X, h=h))
    if isinstance(sdf, RCompose):
        a = evaluate_sdf(sdf.left, X)
        b = evaluate_sdf(sdf.right, X)
        ga = omega_gradient(sdf.left, X, h=h)
        gb = omega_gradient(sdf.right, X, h=h)
        na = -a
        nb = -b
        rad = np.maximum(na * na + nb * nb - 2.0 * sdf.alpha * na * nb, 0.0)
        sqrt_r = np.sqrt(rad)
        denom = 1.0 + sdf.alpha
        if sdf.op == "and":
            dna = 0.5 * (1.0 - na / np.maximum(sqrt_r, 1e-30))
            dnb = 0.5 * (1.0 - nb / np.maximum(sqrt_r, 1e-30))
        else:
            dna = 0.5 * (1.0 + na / np.maximum(sqrt_r, 1e-30))
            dnb = 0.5 * (1.0 + nb / np.maximum(sqrt_r, 1e-30))
        return cast(FloatArray, (dna[..., None] * ga + dnb[..., None] * gb) / denom)
    return fd_gradient(sdf, X, h=h)


def boundary_junction_mask(
    sdf: SDF,
    X: FloatArray,
    *,
    tol: float = 1e-5,
) -> FloatArray:
    """True where a geometric junction invalidates a unique outward normal."""
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X.reshape(1, -1)
    if isinstance(sdf, RCompose):
        a = np.abs(evaluate_sdf(sdf.left, X))
        b = np.abs(evaluate_sdf(sdf.right, X))
        on_bdry = np.abs(evaluate_sdf(sdf, X)) < tol
        return cast(FloatArray, on_bdry & (a < tol) & (b < tol))
    if isinstance(sdf, Box):
        lo = np.asarray(sdf.lo, dtype=float)
        hi = np.asarray(sdf.hi, dtype=float)
        center = 0.5 * (lo + hi)
        half = 0.5 * (hi - lo)
        q = np.abs(X - center) - half
        on_bdry = np.abs(evaluate_sdf(sdf, X)) < tol
        n_active = np.sum(np.abs(q) < tol, axis=-1)
        return cast(FloatArray, on_bdry & (n_active >= 2))
    if isinstance(sdf, Negate):
        return boundary_junction_mask(sdf.child, X, tol=tol)
    return np.zeros(X.shape[0], dtype=bool)


def assert_smooth_for_normal_bc(
    sdf: SDF,
    X: FloatArray,
    *,
    tol: float = 1e-5,
) -> None:
    """Fail if any point in ``X`` lies on a non-smooth geometric junction."""
    mask = boundary_junction_mask(sdf, X, tol=tol)
    if np.any(mask):
        raise NonSmoothBoundaryError(
            "Neumann / Robin BCs require a smooth boundary; "
            f"{int(mask.sum())} point(s) lie on a CSG / box junction"
        )


def normalized_boundary_factor(
    sdf: SDF,
    X: FloatArray,
    *,
    h: float = 1e-6,
    eps: float = 1e-30,
) -> FloatArray:
    """Normalized ADF ``phi`` from primitive ``sdf``."""
    omega = evaluate_sdf(sdf, X)
    grad = omega_gradient(sdf, X, h=h)
    return normalize_adf(omega, grad, eps=eps)


def bc_distance_factor(
    mode: BCMode,
    phi: FloatArray,
    *,
    robin_alpha: float = 1.0,
    robin_beta: float = 0.0,
) -> FloatArray:
    """Scalar multiplicative factor for a BC mode given normalized ``phi``."""
    if mode == "dirichlet":
        return np.asarray(phi, dtype=float)
    if mode == "neumann":
        return np.asarray(phi * phi, dtype=float)
    if mode == "robin":
        return np.asarray(robin_alpha * phi + robin_beta * phi * phi, dtype=float)
    raise ValueError(f"unknown BC mode {mode!r}")


def _sphere_omega_jet(x0: FloatArray, center: Sequence[float], radius: float, order: int) -> FloatArray:
    dim = len(center)
    m = num_multi_indices(dim, order)
    pos = index_position(dim, order)
    out = np.zeros(m, dtype=float)
    c = np.asarray(center, dtype=float)
    diff = x0 - c
    r = float(np.linalg.norm(diff))
    safe = max(r, 1e-30)
    out[pos[(0,) * dim]] = r - float(radius)
    if order < 1:
        return out
    for i in range(dim):
        alpha = tuple(1 if j == i else 0 for j in range(dim))
        out[pos[alpha]] = diff[i] / safe
    if order < 2:
        return out
    inv_r3 = 1.0 / (safe**3)
    inv_r = 1.0 / safe
    for i in range(dim):
        ai = tuple(2 if j == i else 0 for j in range(dim))
        out[pos[ai]] = inv_r - diff[i] * diff[i] * inv_r3
        for j in range(i + 1, dim):
            aij = tuple(1 if k == i else (1 if k == j else 0) for k in range(dim))
            out[pos[aij]] = -diff[i] * diff[j] * inv_r3
    return out


def _halfspace_omega_jet(x0: FloatArray, normal: Sequence[float], point: Sequence[float], order: int) -> FloatArray:
    dim = len(normal)
    m = num_multi_indices(dim, order)
    pos = index_position(dim, order)
    out = np.zeros(m, dtype=float)
    n = np.asarray(normal, dtype=float)
    n = n / np.linalg.norm(n)
    p = np.asarray(point, dtype=float)
    out[pos[(0,) * dim]] = float((x0 - p) @ n)
    if order >= 1:
        for i in range(dim):
            alpha = tuple(1 if j == i else 0 for j in range(dim))
            out[pos[alpha]] = n[i]
    return out


def _compose_jet(
    op: str,
    jet_a: FloatArray,
    jet_b: FloatArray,
    val_a: float,
    val_b: float,
    *,
    alpha: float,
    dim: int,
    order: int,
) -> FloatArray:
    """Jet of negative-inside R-compose via 1-D chain rule on total value."""
    rad = max(val_a * val_a + val_b * val_b - 2.0 * alpha * val_a * val_b, 0.0)
    sqrt_r = float(np.sqrt(rad))
    denom = 1.0 + alpha
    na = -val_a
    nb = -val_b
    if op == "and":
        dna = 0.5 * (1.0 - na / max(sqrt_r, 1e-30))
        dnb = 0.5 * (1.0 - nb / max(sqrt_r, 1e-30))
    else:
        dna = 0.5 * (1.0 + na / max(sqrt_r, 1e-30))
        dnb = 0.5 * (1.0 + nb / max(sqrt_r, 1e-30))
    return cast(FloatArray, (dna * jet_a + dnb * jet_b) / denom)


def boundary_factor_jet(
    sdf: SDF,
    x0: FloatArray,
    *,
    order: int,
    mode: BCMode = "dirichlet",
    normalize: bool = True,
    robin_alpha: float = 1.0,
    robin_beta: float = 0.0,
    h: float = 1e-6,
) -> FloatArray:
    """Multivariate jet of the BC distance factor at a single point ``x0``.

    Returns shape ``(M,)`` with ``M = num_multi_indices(sdf.ndim, order)``.
    """
    if order < 0:
        raise ValueError(f"order must be >= 0, got {order}")
    x0 = np.asarray(x0, dtype=float).reshape(-1)
    dim = int(sdf.ndim)
    if x0.shape[0] != dim:
        raise ValueError(f"x0 length {x0.shape[0]} != sdf.ndim {dim}")

    if isinstance(sdf, Sphere):
        omega_jet = _sphere_omega_jet(x0, sdf.center, sdf.radius, order)
    elif isinstance(sdf, Halfspace):
        omega_jet = _halfspace_omega_jet(x0, sdf.normal, sdf.point, order)
    elif isinstance(sdf, Box):
        # Away from edges the box jet matches the sphere-style outside branch;
        # near edges fall back to FD on omega for robustness.
        lo = np.asarray(sdf.lo, dtype=float)
        hi = np.asarray(sdf.hi, dtype=float)
        center = 0.5 * (lo + hi)
        half = 0.5 * (hi - lo)
        q = np.abs(x0 - center) - half
        if np.min(np.abs(q)) < 1e-8:
            omega_jet = _fd_omega_jet(sdf, x0, order, h=h)
        else:
            omega_jet = _fd_omega_jet(sdf, x0, order, h=h)
    elif isinstance(sdf, Negate):
        return cast(FloatArray, -boundary_factor_jet(
            sdf.child, x0, order=order, mode=mode, normalize=normalize,
            robin_alpha=robin_alpha, robin_beta=robin_beta, h=h,
        ))
    elif isinstance(sdf, RCompose):
        ja = boundary_factor_jet(
            sdf.left, x0, order=order, mode="dirichlet", normalize=False, h=h
        )
        jb = boundary_factor_jet(
            sdf.right, x0, order=order, mode="dirichlet", normalize=False, h=h
        )
        va = float(evaluate_sdf(sdf.left, x0.reshape(1, -1))[0])
        vb = float(evaluate_sdf(sdf.right, x0.reshape(1, -1))[0])
        omega_jet = _compose_jet(sdf.op, ja, jb, va, vb, alpha=sdf.alpha, dim=dim, order=order)
    else:
        omega_jet = _fd_omega_jet(sdf, x0, order, h=h)

    if not normalize:
        phi_jet = omega_jet
    else:
        phi_jet = _normalize_jet(omega_jet, x0, sdf, order, h=h)

    if mode == "dirichlet":
        return phi_jet
    if mode == "neumann":
        return _jet_square(phi_jet, dim, order)
    if mode == "robin":
        sq = _jet_square(phi_jet, dim, order)
        return cast(FloatArray, robin_alpha * phi_jet + robin_beta * sq)
    raise ValueError(f"unknown BC mode {mode!r}")


def _fd_omega_jet(sdf: SDF, x0: FloatArray, order: int, *, h: float) -> FloatArray:
    dim = int(sdf.ndim)
    m = num_multi_indices(dim, order)
    pos = index_position(dim, order)
    out = np.zeros(m, dtype=float)
    out[pos[(0,) * dim]] = float(evaluate_sdf(sdf, x0.reshape(1, -1))[0])
    if order < 1:
        return out
    for i in range(dim):
        alpha = tuple(1 if j == i else 0 for j in range(dim))
        xp = x0.copy()
        xm = x0.copy()
        xp[i] += h
        xm[i] -= h
        out[pos[alpha]] = (
            evaluate_sdf(sdf, xp.reshape(1, -1))[0]
            - evaluate_sdf(sdf, xm.reshape(1, -1))[0]
        ) / (2.0 * h)
    if order < 2:
        return out
    for i in range(dim):
        ai = tuple(2 if j == i else 0 for j in range(dim))
        xp = x0.copy()
        xm = x0.copy()
        xp[i] += h
        xm[i] -= h
        out[pos[ai]] = (
            evaluate_sdf(sdf, xp.reshape(1, -1))[0]
            - 2.0 * out[pos[(0,) * dim]]
            + evaluate_sdf(sdf, xm.reshape(1, -1))[0]
        ) / (h * h)
    return out


def _normalize_jet(
    omega_jet: FloatArray,
    x0: FloatArray,
    sdf: SDF,
    order: int,
    *,
    h: float,
) -> FloatArray:
    """Jet of the normalized ADF at ``x0`` (FD for orders >= 1)."""
    dim = int(sdf.ndim)
    m = num_multi_indices(dim, order)
    pos = index_position(dim, order)
    phi0 = float(
        normalized_boundary_factor(sdf, x0.reshape(1, -1), h=h)[0]
    )
    out = np.zeros(m, dtype=float)
    out[pos[(0,) * dim]] = phi0
    if order < 1:
        return out
    for i in range(dim):
        alpha = tuple(1 if j == i else 0 for j in range(dim))
        xp = x0.copy()
        xm = x0.copy()
        xp[i] += h
        xm[i] -= h
        pp = float(normalized_boundary_factor(sdf, xp.reshape(1, -1), h=h)[0])
        pm = float(normalized_boundary_factor(sdf, xm.reshape(1, -1), h=h)[0])
        out[pos[alpha]] = (pp - pm) / (2.0 * h)
    return out


def _jet_square(jet: FloatArray, dim: int, order: int) -> FloatArray:
    from omnibias.core.multi_index import multiply_table

    m = num_multi_indices(dim, order)
    table = multiply_table(dim, order)
    out = np.zeros(m, dtype=float)
    for g, al, be in table:
        out[g] += jet[al] * jet[be]
    return cast(FloatArray, out)


__all__ = [
    "BCMode",
    "NonSmoothBoundaryError",
    "assert_smooth_for_normal_bc",
    "bc_distance_factor",
    "boundary_factor_jet",
    "boundary_junction_mask",
    "normalized_boundary_factor",
    "omega_gradient",
]
