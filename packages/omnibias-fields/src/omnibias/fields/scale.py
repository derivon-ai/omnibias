# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Field-side scale schedule and grid-free V-cycle (theory 03-07).

``alpha`` is a tempering scale, not a collapse parameter. The
founding bias collapse is ``delta -> 0``. Temperature collapse is
``beta -> inf`` (feasibility). Do not conflate the two. Scale flow
is a third axis.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np
from omnibias.core.scale import (
    ScaleBand,
    ScaledPack,
    stiffness_matrix,
)
from omnibias.core.spectral_design import alpha_for_peak


def scale_schedule(
    *,
    target_band: Callable[[float], float],
    steps: int,
    base: str = "gaussian",
    order: int = 1,
) -> tuple[float, ...]:
    """Cutoff schedule from order-as-frequency, not a hand-tuned window list."""
    n = int(steps)
    if n < 1:
        raise ValueError("steps must be >= 1")
    out: list[float] = []
    for i in range(n):
        t = 0.0 if n == 1 else i / (n - 1)
        xi = float(target_band(t))
        if xi <= 0.0:
            raise ValueError("target_band must return a positive frequency")
        out.append(float(alpha_for_peak(base, int(order), xi)))
    return tuple(out)


def _jacobi(a: np.ndarray, rhs: np.ndarray, x: np.ndarray, *, omega: float, sweeps: int) -> np.ndarray:
    diag = np.diag(a).copy()
    diag = np.where(np.abs(diag) < 1e-18, 1.0, diag)
    off = a - np.diag(np.diag(a))
    y = x.copy()
    for _ in range(int(sweeps)):
        y = (1.0 - omega) * y + omega * (rhs - off @ y) / diag
    return np.asarray(y, dtype=np.float64)


def grid_free_vcycle(
    packs: Sequence[ScaledPack],
    rhs: Sequence[float],
    *,
    bands: Sequence[ScaleBand],
    u0: Sequence[float] | None = None,
    pre_sweeps: int = 2,
    post_sweeps: int = 2,
    omega: float = 0.6,
) -> tuple[float, ...]:
    """One V-cycle on scale bands. Restriction / prolongation are exact rescales."""
    if len(bands) < 2:
        raise ValueError("V-cycle needs at least two scale bands")
    packs_t = tuple(packs)
    # Poisson ``-d^2/dx^2`` so the diagonal is positive for gaussian bumps.
    a = -stiffness_matrix(packs_t, derivative_order=2)
    b = np.asarray(rhs, dtype=np.float64)
    u = np.zeros(len(packs_t), dtype=np.float64) if u0 is None else np.asarray(u0, dtype=np.float64)
    u = _jacobi(a, b, u, omega=omega, sweeps=pre_sweeps)
    residual = b - a @ u
    coarse = bands[0]
    fine = bands[-1]
    coarse_idx = [i for i, p in enumerate(packs_t) if coarse.contains(p.alpha)]
    fine_idx = [i for i, p in enumerate(packs_t) if fine.contains(p.alpha) or coarse.contains(p.alpha)]
    if not coarse_idx:
        raise ValueError("coarse band contains no packs")
    a_c = a[np.ix_(coarse_idx, coarse_idx)]
    r_c = residual[coarse_idx]
    try:
        e_c = np.linalg.solve(a_c + 1e-12 * np.eye(len(coarse_idx)), r_c)
    except np.linalg.LinAlgError:
        e_c = np.linalg.lstsq(a_c, r_c, rcond=None)[0]
    e = np.zeros_like(u)
    e[coarse_idx] = e_c
    # Prolongation onto the fine band is the identity on shared slow modes.
    _ = fine_idx
    u = u + e
    u = _jacobi(a, b, u, omega=omega, sweeps=post_sweeps)
    return tuple(float(v) for v in u)


def tanh_feature_matrix(
    x: Sequence[float],
    alphas: Sequence[float],
    means: Sequence[float],
) -> np.ndarray:
    """``tanh(alpha (x - mu))`` bank used by the derived curriculum."""
    xv = np.asarray(x, dtype=np.float64).reshape(-1)
    cols = [np.tanh(float(a) * (xv - float(m))) for a in alphas for m in means]
    return np.column_stack(cols)


def lstsq_readout_mse(
    x: Sequence[float],
    y: Sequence[float],
    alphas: Sequence[float],
    *,
    n_means: int = 24,
) -> float:
    """One-shot least-squares readout on a scale bank. No gradient descent."""
    means = tuple(float(v) for v in np.linspace(0.0, 1.0, int(n_means)))
    phi = tanh_feature_matrix(x, alphas, means)
    target = np.asarray(y, dtype=np.float64).reshape(-1)
    coef, *_ = np.linalg.lstsq(phi, target, rcond=None)
    pred = phi @ coef
    return float(np.mean((pred - target) ** 2))


def residual_norm(packs: Sequence[ScaledPack], rhs: Sequence[float], coeffs: Sequence[float]) -> float:
    a = -stiffness_matrix(tuple(packs), derivative_order=2)
    r = np.asarray(rhs, dtype=np.float64) - a @ np.asarray(coeffs, dtype=np.float64)
    return float(np.linalg.norm(r))


__all__ = [
    "ScaleBand",
    "grid_free_vcycle",
    "lstsq_readout_mse",
    "residual_norm",
    "scale_schedule",
    "tanh_feature_matrix",
]
