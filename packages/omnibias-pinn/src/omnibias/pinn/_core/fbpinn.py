# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Backend-free FBPINN window geometry helpers."""

from __future__ import annotations


def window_centers_1d(
    lo: float, hi: float, n_windows: int, *, overlap: float = 0.5
) -> tuple[tuple[float, ...], float]:
    """Equally spaced 1-D window centers and shared half-width.

    ``overlap`` in ``(0, 1)`` controls how much neighbouring windows share;
    ``0.5`` recovers the classical FBPINN tiling where raised-cosines sum to 1.
    """
    if n_windows < 1:
        raise ValueError(f"n_windows must be >= 1, got {n_windows}")
    if not 0.0 < overlap < 1.0:
        raise ValueError(f"overlap must be in (0, 1), got {overlap}")
    if hi <= lo:
        raise ValueError(f"need hi > lo, got {(lo, hi)}")
    stride = (hi - lo) / n_windows
    half_width = 0.5 * stride / (1.0 - overlap)
    centers = tuple(lo + (i + 0.5) * stride for i in range(n_windows))
    return centers, float(half_width)


__all__ = ["window_centers_1d"]
