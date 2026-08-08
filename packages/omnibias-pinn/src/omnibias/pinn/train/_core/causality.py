# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Causality diagnostics for time-dependent PINN training (pure numpy).

These are *measurements*, not proofs of temporal consistency. A causally
trained solution should not fit a late time bin better than an earlier one;
:func:`causality_index` quantifies that discordance as an inversion fraction.
The Wang-Perdikaris advance criterion is exposed as :func:`unlocked_fraction`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class CausalityReport:
    """Snapshot of temporal-ordering diagnostics for one residual batch.

    Attributes
    ----------
    causality_index
        Inversion fraction in ``[0, 1]`` (related to Kendall ``tau`` by
        ``tau = 1 - 2 * causality_index``). ``0`` means per-bin residuals are
        non-decreasing in time; ``1`` means every pairwise comparison is
        inverted.
    unlocked_fraction
        ``min_i w_i`` of the causal weights -- Wang et al.'s advance signal.
    n_bins
        Number of time bins the report was computed over.
    mean_per_bin
        Mean squared residual of each time bin (detached measurement).
    """

    causality_index: float
    unlocked_fraction: float
    n_bins: int
    mean_per_bin: tuple[float, ...]


def causality_index(L_per_bin: np.ndarray | list[float]) -> float:
    """Inversion fraction of per-bin residual magnitudes.

    For a sequence ``L_0, ..., L_{n-1}`` of non-negative per-bin losses, count
    the fraction of pairs ``(i, j)`` with ``i < j`` where ``L_j < L_i`` (a late
    bin is fitted *better* than an earlier one). Returns ``0.0`` for a
    non-decreasing sequence and ``1.0`` when every pair is inverted.
    Related to Kendall ``tau`` by ``tau = 1 - 2 * causality_index``.

    A single bin (or empty) returns ``0.0`` -- there is nothing to discord.
    """
    L = np.asarray(L_per_bin, dtype=float).reshape(-1)
    n = int(L.size)
    if n < 2:
        return 0.0
    if np.any(~np.isfinite(L)):
        raise ValueError("L_per_bin must be finite")
    if np.any(L < 0.0):
        raise ValueError("L_per_bin must be non-negative")
    inversions = 0
    total = 0
    for i in range(n):
        for j in range(i + 1, n):
            total += 1
            if L[j] < L[i]:
                inversions += 1
    return float(inversions) / float(total)


def unlocked_fraction(weights: np.ndarray | list[float]) -> float:
    """Wang-Perdikaris advance signal: ``min_i w_i``.

    Causal weights are non-increasing, so the last bin's weight is the smallest.
    Once it rises above a schedule tolerance the whole window is unlocked.
    """
    w = np.asarray(weights, dtype=float).reshape(-1)
    if w.size == 0:
        raise ValueError("weights must be non-empty")
    if np.any(~np.isfinite(w)):
        raise ValueError("weights must be finite")
    if np.any(w < 0.0):
        raise ValueError("weights must be non-negative")
    return float(w.min())


def report_causality(
    L_per_bin: np.ndarray | list[float],
    weights: np.ndarray | list[float] | None = None,
) -> CausalityReport:
    """Build a :class:`CausalityReport` from per-bin losses and optional weights.

    When ``weights`` is omitted the unlocked fraction is reported as ``1.0``
    (no causal filter was applied).
    """
    L = np.asarray(L_per_bin, dtype=float).reshape(-1)
    idx = causality_index(L)
    if weights is None:
        unlocked = 1.0
    else:
        unlocked = unlocked_fraction(weights)
    return CausalityReport(
        causality_index=idx,
        unlocked_fraction=unlocked,
        n_bins=int(L.size),
        mean_per_bin=tuple(float(x) for x in L),
    )


__all__ = [
    "CausalityReport",
    "causality_index",
    "report_causality",
    "unlocked_fraction",
]
