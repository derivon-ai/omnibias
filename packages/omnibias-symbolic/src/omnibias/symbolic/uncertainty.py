# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Coefficient uncertainty for discovered sparse equations.

:func:`omnibias.symbolic.discovery.fit_sparse_equation` returns *point*
coefficients.  For a discovery to be trustworthy you also need to know how
*sure* those numbers are.  This module adds three complementary layers, from
cheap-and-statistical to rigorous-and-certified:

* :func:`bootstrap_coefficients` -- nonparametric (resampling) confidence
  intervals **and** per-term *selection frequency* (how often each term survives
  the sparsity threshold across resamples; the basis of stability selection).
* :func:`ridge_coefficient_covariance` -- the analytic ridge/OLS covariance
  ``sigma^2 (XᵀX + alpha I)^{-1} XᵀX (XᵀX + alpha I)^{-1}`` and its standard
  errors (fast, parametric, assumes homoscedastic noise).
* :func:`certified_coefficient_intervals` -- *rigorous* enclosures of the exact
  normal-equation solution via outward-rounded interval arithmetic
  (:mod:`omnibias.core.verified.linalg`).  This is the bridge from discovery to
  the verified substrate: the returned box provably contains the true least
  squares / ridge coefficient vector.

:func:`attach_uncertainty` runs the chosen layers and returns a new
:class:`~omnibias.symbolic.discovery.SparseEquation` carrying the results.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import numpy as np
from omnibias.core.verified.interval import Interval
from omnibias.core.verified.linalg import (
    inf_norm_vector,
    matvec,
    neumann_inverse_norm_bound,
    to_interval_matrix,
)
from omnibias.symbolic.discovery import SparseEquation, fit_sparse_equation


def _centered(design: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(design, dtype=float)
    y = np.asarray(target, dtype=float)
    if x.ndim != 2:
        raise ValueError(f"design must be 2D, got shape {x.shape}")
    if y.ndim != 1 or y.shape[0] != x.shape[0]:
        raise ValueError("target must be 1D matching the number of design rows")
    return x - x.mean(axis=0), y - float(y.mean())


def bootstrap_coefficients(
    design: np.ndarray,
    target: np.ndarray,
    term_names: list[str],
    *,
    alpha: float = 1e-8,
    threshold: float = 1e-4,
    max_iter: int = 8,
    n_boot: int = 200,
    ci_level: float = 0.95,
    seed: int = 0,
) -> dict[str, Any]:
    """Nonparametric bootstrap CIs + selection frequency for an STLSQ fit.

    Resamples ``(design, target)`` rows with replacement ``n_boot`` times,
    refits :func:`fit_sparse_equation` on each resample, and summarises the
    coefficient distribution per term.

    Returns a dict aligned to ``term_names`` with ``mean``, ``std``,
    ``ci_lower`` / ``ci_upper`` (``ci_level`` percentile interval), and
    ``selection_frequency`` (fraction of resamples in which the term was
    active -- the stability-selection score).
    """
    x = np.asarray(design, dtype=float)
    y = np.asarray(target, dtype=float)
    n, p = x.shape
    if p != len(term_names):
        raise ValueError("term_names must match design width")
    if not 0.0 < ci_level < 1.0:
        raise ValueError("ci_level must be in (0, 1)")
    rng = np.random.default_rng(seed)
    coefs = np.zeros((n_boot, p))
    active = np.zeros((n_boot, p), dtype=bool)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        eq = fit_sparse_equation(
            x[idx], y[idx], list(term_names),
            alpha=alpha, threshold=threshold, max_iter=max_iter,
        )
        coefs[b] = eq.coefficients
        active[b] = eq.active_mask
    lo_q = 0.5 * (1.0 - ci_level)
    hi_q = 1.0 - lo_q
    return {
        "term_names": tuple(term_names),
        "mean": coefs.mean(axis=0),
        "std": coefs.std(axis=0),
        "ci_lower": np.quantile(coefs, lo_q, axis=0),
        "ci_upper": np.quantile(coefs, hi_q, axis=0),
        "selection_frequency": active.mean(axis=0),
        "ci_level": float(ci_level),
        "n_boot": int(n_boot),
        "samples": coefs,
    }


def ridge_coefficient_covariance(
    design: np.ndarray,
    target: np.ndarray,
    *,
    alpha: float = 1e-8,
) -> dict[str, Any]:
    r"""Analytic ridge/OLS coefficient covariance and standard errors.

    With centered columns and ``A = XᵀX + alpha I``, the ridge estimator
    ``c = A^{-1} Xᵀy`` has covariance
    ``sigma^2 A^{-1} (XᵀX) A^{-1}`` under homoscedastic noise, with
    ``sigma^2`` estimated from the residual sum of squares over ``n - p``
    degrees of freedom.  For ``alpha = 0`` this is the ordinary least squares
    covariance ``sigma^2 (XᵀX)^{-1}``.
    """
    xc, yc = _centered(design, target)
    n, p = xc.shape
    a = xc.T @ xc + alpha * np.eye(p)
    a_inv = np.linalg.inv(a)
    coef = a_inv @ (xc.T @ yc)
    resid = yc - xc @ coef
    dof = max(n - p, 1)
    sigma2 = float(resid @ resid / dof)
    xtx = xc.T @ xc
    cov = sigma2 * (a_inv @ xtx @ a_inv)
    std_errors = np.sqrt(np.clip(np.diag(cov), 0.0, None))
    return {
        "coefficients": coef,
        "covariance": cov,
        "std_errors": std_errors,
        "sigma2": sigma2,
        "dof": int(dof),
    }


def certified_coefficient_intervals(
    design: np.ndarray,
    target: np.ndarray,
    *,
    alpha: float = 0.0,
) -> dict[str, Any]:
    r"""Rigorous interval enclosures of the normal-equation coefficient vector.

    Solves the (centered) normal equations ``A c = b`` with
    ``A = XᵀX + alpha I`` and ``b = Xᵀy`` and returns a **certified** box that
    provably contains the exact solution ``c*``.  The bound is the verified
    Neumann estimate: with a float approximate inverse ``B`` and approximate
    solution ``c0``,

    .. math::

        \|c^* - c_0\|_\infty \le \|A^{-1}\|_\infty\, \|b - A c_0\|_\infty,

    where ``\|A^{-1}\|_\infty`` is certified by
    :func:`omnibias.core.verified.linalg.neumann_inverse_norm_bound` and the
    residual norm is evaluated in outward-rounded interval arithmetic.  Hence
    each coefficient lies in ``[c0_i - r, c0_i + r]`` with ``r`` the certified
    radius.  For ``alpha = 0`` and exact (noiseless, full-rank) data ``c*`` is
    the true generating coefficient vector, so the box contains the ground truth.

    The ``certified`` flag is ``True`` only when the Neumann condition
    ``kappa = ||I - BA||_inf < 1`` holds (otherwise the radius is ``inf``).
    """
    xc, yc = _centered(design, target)
    _, p = xc.shape
    a = xc.T @ xc + alpha * np.eye(p)
    b = xc.T @ yc
    b_inv = np.linalg.inv(a)
    c0 = b_inv @ b

    bound = neumann_inverse_norm_bound(a.tolist(), b_inv.tolist())
    inv_norm = float(bound["inverse_norm_bound"])

    a_iv = to_interval_matrix(a.tolist())
    c0_iv = [Interval.point(float(v)) for v in c0]
    ac0 = matvec(a_iv, c0_iv)
    resid_iv = [Interval.point(float(b[i])) - ac0[i] for i in range(p)]
    resid_norm = inf_norm_vector(resid_iv)

    certified = bool(bound["certified"] and bool(np.isfinite(inv_norm)))
    if certified:
        radius = (Interval.point(inv_norm) * Interval.point(resid_norm)).hi
    else:
        radius = float("inf")
    intervals = tuple((float(v) - radius, float(v) + radius) for v in c0)
    return {
        "coefficients": c0,
        "intervals": intervals,
        "radius": float(radius),
        "inverse_norm_bound": inv_norm,
        "residual_inf_norm": float(resid_norm),
        "kappa": float(bound["kappa"]),
        "certified": certified,
    }


def attach_uncertainty(
    equation: SparseEquation,
    design: np.ndarray,
    target: np.ndarray,
    *,
    bootstrap: bool = True,
    certified: bool = True,
    n_boot: int = 200,
    ci_level: float = 0.95,
    seed: int = 0,
) -> SparseEquation:
    """Return a copy of ``equation`` carrying coefficient uncertainty.

    ``design`` / ``target`` are the *full-library* fit inputs (same column order
    as ``equation.term_names``).  Bootstrap CIs and selection frequency are
    computed over the full library; certified intervals are computed on the
    **active** sub-design via ordinary least squares (``alpha = 0``) and are
    ``(0.0, 0.0)`` for inactive terms.
    """
    x = np.asarray(design, dtype=float)
    names = list(equation.term_names)
    coefficient_ci: tuple[tuple[float, float], ...] | None = None
    selection_frequency: tuple[float, ...] | None = None
    coefficient_intervals: tuple[tuple[float, float], ...] | None = None

    if bootstrap:
        boot = bootstrap_coefficients(
            x, target, names,
            alpha=equation.alpha, threshold=equation.threshold,
            n_boot=n_boot, ci_level=ci_level, seed=seed,
        )
        coefficient_ci = tuple(
            (float(lo), float(hi))
            for lo, hi in zip(boot["ci_lower"], boot["ci_upper"], strict=True)
        )
        selection_frequency = tuple(float(v) for v in boot["selection_frequency"])

    if certified:
        active = np.asarray(equation.active_mask, dtype=bool)
        boxes: list[tuple[float, float]] = [(0.0, 0.0)] * len(names)
        if np.any(active):
            cols = np.flatnonzero(active)
            cert = certified_coefficient_intervals(x[:, cols], target, alpha=0.0)
            for slot, interval in zip(cols, cert["intervals"], strict=True):
                boxes[int(slot)] = (float(interval[0]), float(interval[1]))
        coefficient_intervals = tuple(boxes)

    return replace(
        equation,
        coefficient_ci=coefficient_ci,
        selection_frequency=selection_frequency,
        coefficient_intervals=coefficient_intervals,
    )


__all__ = [
    "attach_uncertainty",
    "bootstrap_coefficients",
    "certified_coefficient_intervals",
    "ridge_coefficient_covariance",
]
