# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""SDF-aware interior / boundary sampling (pure numpy).

Mirrors the API shape of
:mod:`omnibias.pinn.solver._core.sampling` so residual-adaptive refinement
(:func:`~omnibias.pinn.solver._core.sampling.select_refinement_points`) works
over an SDF domain without changes -- score candidates, then keep.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from omnibias.pinn.domain._core.adf import fd_gradient
from omnibias.pinn.domain._core.sdf import SDF, evaluate_sdf

FloatArray = np.ndarray


def interior_points_sdf(
    sdf: SDF,
    bounds: Sequence[tuple[float, float]],
    *,
    n: int,
    seed: int = 0,
    max_trials: int | None = None,
    interior_sign: float = -1.0,
) -> FloatArray:
    """Rejection-sample ``n`` interior points where ``sign * sdf(x) > 0``.

    Default ``interior_sign=-1`` matches the graphics convention (SDF negative
    inside). Returns shape ``(n, d)``.
    """
    if n < 1:
        raise ValueError(f"n must be >= 1, got {n}")
    d = len(bounds)
    if sdf.ndim != d:
        raise ValueError(f"sdf.ndim={sdf.ndim} != len(bounds)={d}")
    trials = int(max_trials) if max_trials is not None else max(100 * n, 1000)
    rng = np.random.default_rng(seed)
    lo = np.array([a for a, _ in bounds], dtype=float)
    hi = np.array([b for _, b in bounds], dtype=float)
    kept: list[FloatArray] = []
    attempted = 0
    while sum(len(k) for k in kept) < n and attempted < trials:
        batch = min(n - sum(len(k) for k in kept), 256)
        X = rng.uniform(lo, hi, size=(batch, d))
        vals = evaluate_sdf(sdf, X)
        mask = interior_sign * vals > 0.0
        if np.any(mask):
            kept.append(X[mask])
        attempted += batch
    if not kept:
        raise RuntimeError(
            f"failed to sample any interior points in {trials} trials; "
            "check that the SDF has negative (interior) volume inside bounds"
        )
    out = np.concatenate(kept, axis=0)[:n]
    if out.shape[0] < n:
        raise RuntimeError(
            f"only sampled {out.shape[0]} / {n} interior points in {trials} trials"
        )
    return out


def boundary_points_sdf(
    sdf: SDF,
    bounds: Sequence[tuple[float, float]],
    *,
    n: int,
    seed: int = 0,
    n_newton: int = 8,
    h: float = 1e-6,
    tol: float = 1e-8,
) -> FloatArray:
    """Sample boundary points by Newton projection onto the zero level set.

    Draws uniform candidates in ``bounds``, then iterates
    ``x <- x - sdf(x) * grad / |grad|^2`` (the SDF Newton step). Points that
    fail to reach ``|sdf| < tol`` are discarded and redrawn.
    """
    if n < 1:
        raise ValueError(f"n must be >= 1, got {n}")
    d = len(bounds)
    if sdf.ndim != d:
        raise ValueError(f"sdf.ndim={sdf.ndim} != len(bounds)={d}")
    rng = np.random.default_rng(seed)
    lo = np.array([a for a, _ in bounds], dtype=float)
    hi = np.array([b for _, b in bounds], dtype=float)
    kept: list[FloatArray] = []
    attempts = 0
    max_attempts = max(50 * n, 500)
    while len(kept) < n and attempts < max_attempts:
        need = n - len(kept)
        X = rng.uniform(lo, hi, size=(need, d))
        for _ in range(n_newton):
            vals = evaluate_sdf(sdf, X)
            g = fd_gradient(sdf, X, h=h)
            g2 = np.sum(g * g, axis=-1, keepdims=True) + 1e-30
            X = X - (vals.reshape(-1, 1) * g) / g2
            X = np.clip(X, lo, hi)
        vals = evaluate_sdf(sdf, X)
        ok = np.abs(vals) < tol
        if np.any(ok):
            kept.extend(list(X[ok]))
        attempts += need
    if len(kept) < n:
        raise RuntimeError(
            f"only projected {len(kept)} / {n} boundary points; "
            "relax tol or enlarge bounds"
        )
    return np.asarray(kept[:n], dtype=float)


__all__ = [
    "boundary_points_sdf",
    "interior_points_sdf",
]
