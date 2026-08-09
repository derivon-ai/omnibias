# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Funnel / secant inference for admissible self-similar ``lambda``.

DeepMind (arXiv:2509.14185) observes that near an admissible ``lambda*`` the
signed max residual (near the origin) is approximately linear in ``lambda``,
so a secant update finds the zero. This module is the omnibias first-class
implementation of that playbook.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field


@dataclass
class FunnelState:
    """History for secant updates of ``lambda``."""

    lambdas: list[float] = field(default_factory=list)
    residuals: list[float] = field(default_factory=list)

    def record(self, lam: float, signed_max_residual: float) -> None:
        self.lambdas.append(float(lam))
        self.residuals.append(float(signed_max_residual))


def funnel_next_lambda(
    state: FunnelState,
    *,
    delta_lambda: float = 1e-3,
) -> float:
    """Return the next ``lambda`` via secant (or a small perturbation if n < 2)."""
    n = len(state.lambdas)
    if n == 0:
        raise ValueError("FunnelState is empty; record an initial (lam, residual) first")
    if n == 1:
        return float(state.lambdas[-1] + delta_lambda)
    lam_a, lam_b = state.lambdas[-2], state.lambdas[-1]
    r_a, r_b = state.residuals[-2], state.residuals[-1]
    denom = r_a - r_b
    if abs(denom) < 1e-30:
        return float(lam_b + delta_lambda)
    return float(lam_b - r_b * (lam_a - lam_b) / denom)


def signed_max_residual_near_origin(
    y: object,
    residual: object,
    *,
    radius: float = 0.5,
) -> float:
    """Signed residual with largest |R| among samples with ``|y| <= radius``."""
    import numpy as np

    y_arr = np.asarray(y, dtype=float).reshape(-1)
    r_arr = np.asarray(residual, dtype=float).reshape(-1)
    if y_arr.shape != r_arr.shape:
        raise ValueError(f"shape mismatch: y {y_arr.shape} vs residual {r_arr.shape}")
    mask = np.abs(y_arr) <= float(radius)
    if not np.any(mask):
        idx = int(np.argmin(np.abs(y_arr)))
        return float(r_arr[idx])
    r_loc = r_arr[mask]
    return float(r_loc[int(np.argmax(np.abs(r_loc)))])


def run_funnel_loop(
    *,
    lam0: float,
    train_and_residual: Callable[[float], tuple[float, object, object]],
    n_updates: int = 5,
    delta_lambda: float = 1e-3,
    radius: float = 0.5,
) -> FunnelState:
    """Run ``n_updates`` funnel iterations.

    ``train_and_residual(lam)`` must train (or evaluate) at fixed ``lam`` and
    return ``(lam_used, y, residual_vector)``.
    """
    state = FunnelState()
    lam = float(lam0)
    for _ in range(max(int(n_updates), 1)):
        lam_used, y, residual = train_and_residual(lam)
        signed = signed_max_residual_near_origin(y, residual, radius=radius)
        state.record(float(lam_used), signed)
        lam = funnel_next_lambda(state, delta_lambda=delta_lambda)
    return state


__all__ = [
    "FunnelState",
    "funnel_next_lambda",
    "run_funnel_loop",
    "signed_max_residual_near_origin",
]
