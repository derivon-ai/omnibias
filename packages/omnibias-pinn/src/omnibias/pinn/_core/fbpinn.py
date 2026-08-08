# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Backend-free FBPINN window geometry helpers."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass


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


@dataclass(frozen=True)
class FBPINNLevelSpec:
    """One level of a fixed multilevel FBPINN hierarchy.

    Each level tiles the domain with ``n_windows`` overlapping raised-cosine
    windows. Sub-networks on that level see locally normalised coordinates
    scaled by ``frequency_scales`` (one per window when provided).
    """

    n_windows: int
    overlap: float = 0.5
    frequency_scales: tuple[float, ...] | None = None

    def __post_init__(self) -> None:
        if self.n_windows < 1:
            raise ValueError(f"n_windows must be >= 1, got {self.n_windows}")
        if not 0.0 < self.overlap < 1.0:
            raise ValueError(f"overlap must be in (0, 1), got {self.overlap}")
        if self.frequency_scales is not None:
            if len(self.frequency_scales) != self.n_windows:
                raise ValueError(
                    f"frequency_scales length {len(self.frequency_scales)} "
                    f"!= n_windows={self.n_windows}"
                )


def default_multilevel_specs(
    n_levels: int = 3,
    *,
    base_windows: int = 1,
    overlap: float = 0.5,
    freq_base: float = 1.0,
) -> tuple[FBPINNLevelSpec, ...]:
    """Geometric multilevel ladder: window count and scales double per level."""
    if n_levels < 1:
        raise ValueError(f"n_levels must be >= 1, got {n_levels}")
    if base_windows < 1:
        raise ValueError(f"base_windows must be >= 1, got {base_windows}")
    specs: list[FBPINNLevelSpec] = []
    for level in range(n_levels):
        n_w = base_windows * (2**level)
        level_freq = freq_base * (2.0**level)
        scales = tuple(level_freq * (2.0**i) for i in range(n_w))
        specs.append(
            FBPINNLevelSpec(
                n_windows=n_w,
                overlap=overlap,
                frequency_scales=scales,
            )
        )
    return tuple(specs)


def resolve_level_specs(
    *,
    n_windows: int | None = None,
    overlap: float = 0.5,
    frequency_scales: Sequence[float] | None = None,
    level_specs: Sequence[FBPINNLevelSpec] | None = None,
    n_levels: int | None = None,
) -> tuple[FBPINNLevelSpec, ...]:
    """Normalise single-level and multilevel constructor arguments."""
    if level_specs is not None:
        if n_windows is not None or n_levels is not None:
            raise ValueError("pass either level_specs or n_windows/n_levels, not both")
        return tuple(level_specs)
    if n_levels is not None:
        return default_multilevel_specs(n_levels, overlap=overlap)
    if n_windows is None:
        n_windows = 4
    if frequency_scales is None:
        frequency_scales = tuple(1.0 for _ in range(n_windows))
    return (
        FBPINNLevelSpec(
            n_windows=n_windows,
            overlap=overlap,
            frequency_scales=tuple(float(s) for s in frequency_scales),
        ),
    )


__all__ = [
    "FBPINNLevelSpec",
    "default_multilevel_specs",
    "resolve_level_specs",
    "window_centers_1d",
]
