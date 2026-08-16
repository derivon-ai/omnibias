# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Finite multi-spacing continuum *fit*, not an ``a → 0`` theorem.

``continuum_claim`` on :class:`ContinuumFitResult` is earned when the
hold-out skill versus a constant-in-``a`` baseline is positive. It is
not a Yang-Mills continuum limit and must not be copied onto sealed
transfer / Wilson-character certificates. ``yang_mills_claim`` stays false.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

CONTINUUM_FIT_SCOPE = "finite multi-spacing table; not a→0 theorem"


@dataclass(frozen=True)
class ContinuumFitResult:
    """Linear / quadratic ``a²`` extrapolation of a finite table."""

    intercept: float
    slope: float
    skill: float
    model_rmse: float
    baseline_rmse: float
    multi_beta_gate_passed: bool
    continuum_claim: bool
    yang_mills_claim: bool = False
    scope: str = CONTINUUM_FIT_SCOPE
    order: int = 1
    a_sqrt_sigma: float | None = None


def _rmse(target: np.ndarray, pred: np.ndarray) -> float:
    target = np.asarray(target, dtype=float).reshape(-1)
    pred = np.asarray(pred, dtype=float).reshape(-1)
    return float(np.sqrt(np.mean((target - pred) ** 2)))


def scale_from_string_tension(sigma_lat: float, *, spacing: float = 1.0) -> float:
    """Bookkeeping ``a √σ``. Not a physical string tension."""
    return float(spacing) * float(np.sqrt(max(float(sigma_lat), 0.0)))


def extrapolate_in_a2(
    spacing: np.ndarray,
    values: np.ndarray,
    errors: np.ndarray | None = None,
    *,
    order: int = 1,
    holdout: int = 2,
) -> ContinuumFitResult:
    """Fit ``y = c0 + c1 a² [+ c2 a⁴]`` and score against a constant baseline.

    ``continuum_claim`` is True only when hold-out skill is positive and finite.
    """
    if order not in (1, 2):
        raise ValueError(f"order must be 1 or 2, got {order}")
    a = np.asarray(spacing, dtype=float).reshape(-1)
    y = np.asarray(values, dtype=float).reshape(-1)
    if a.shape[0] < 4:
        raise ValueError("need at least 4 spacings")
    a2 = a * a
    n_hold = min(int(holdout), a.shape[0] // 3)
    n_hold = max(n_hold, 1)
    train = slice(0, a.shape[0] - n_hold)
    test = slice(a.shape[0] - n_hold, a.shape[0])
    cols = [np.ones_like(a2), a2]
    if order == 2:
        cols.append(a2 * a2)
    design = np.column_stack(cols)
    weights = None
    if errors is not None:
        err = np.asarray(errors, dtype=float).reshape(-1)
        weights = 1.0 / np.maximum(err, 1e-12)
    if weights is None:
        coef, *_ = np.linalg.lstsq(design[train], y[train], rcond=None)
    else:
        w = weights[train]
        coef, *_ = np.linalg.lstsq(design[train] * w[:, None], y[train] * w, rcond=None)
    pred_test = design[test] @ coef
    model = _rmse(y[test], pred_test)
    baseline = _rmse(y[test], np.full(y[test].shape, float(np.mean(y[train]))))
    skill = 0.0 if baseline <= 0.0 else 1.0 - model / baseline
    passed = bool(skill > 0.0 and np.isfinite(skill))
    return ContinuumFitResult(
        intercept=float(coef[0]),
        slope=float(coef[1]),
        skill=float(skill),
        model_rmse=model,
        baseline_rmse=baseline,
        multi_beta_gate_passed=passed,
        continuum_claim=passed,
        yang_mills_claim=False,
        order=order,
    )


__all__ = [
    "CONTINUUM_FIT_SCOPE",
    "ContinuumFitResult",
    "extrapolate_in_a2",
    "scale_from_string_tension",
]
