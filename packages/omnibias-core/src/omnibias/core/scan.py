# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Bias-scan bank algebra (theory 01-02).

A scan shares one pack template across a bank of bias offsets. Offsets live
on the transverse coordinate ``z = w . x``; there is no pixel grid. Pure
Python -- tensor evaluation is in the torch / jax twins.

The template's internal limit is founding bias collapse (``delta -> 0``).
The soft-argmax sharpness ``gamma`` is a softmax readout; driving it to
infinity would be temperature collapse (``beta -> inf``), a different limit.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass


def _as_float_tuple(values: Sequence[float], *, name: str) -> tuple[float, ...]:
    out = tuple(float(v) for v in values)
    if not out:
        raise ValueError(f"{name} must be non-empty")
    if not all(math.isfinite(v) for v in out):
        raise ValueError(f"{name} must be finite")
    return out


@dataclass(frozen=True)
class BankSpec:
    """A 1-D offset bank, optionally with a scale (tempering) axis."""

    offsets: tuple[float, ...]
    scales: tuple[float, ...] = (1.0,)

    def __post_init__(self) -> None:
        offsets = _as_float_tuple(self.offsets, name="offsets")
        scales = _as_float_tuple(self.scales, name="scales")
        if any(s <= 0.0 for s in scales):
            raise ValueError("scales must be positive")
        object.__setattr__(self, "offsets", offsets)
        object.__setattr__(self, "scales", scales)

    @classmethod
    def uniform(cls, lo: float, hi: float, n: int, *, scales: Sequence[float] = (1.0,)) -> BankSpec:
        if n < 2:
            raise ValueError(f"uniform bank needs n >= 2, got {n}")
        if not (math.isfinite(lo) and math.isfinite(hi)):
            raise ValueError("lo and hi must be finite")
        if hi == lo:
            raise ValueError("lo and hi must differ")
        step = (hi - lo) / (n - 1)
        offsets = tuple(lo + i * step for i in range(n))
        return cls(offsets, tuple(float(s) for s in scales))

    @property
    def n_offsets(self) -> int:
        return len(self.offsets)

    @property
    def n_scales(self) -> int:
        return len(self.scales)

    @property
    def spacing(self) -> float | None:
        """Uniform spacing, or ``None`` when the bank is irregular."""
        if len(self.offsets) < 2:
            return None
        diffs = [self.offsets[i + 1] - self.offsets[i] for i in range(len(self.offsets) - 1)]
        ref = diffs[0]
        if ref == 0.0:
            return None
        if any(abs(d - ref) > 1e-12 * max(abs(ref), 1.0) for d in diffs[1:]):
            return None
        return float(ref)

    def min_separation(self) -> float:
        """Minimum adjacent spacing after sorting (collapse diagnostic)."""
        ordered = sorted(self.offsets)
        if len(ordered) < 2:
            return float("inf")
        return min(ordered[i + 1] - ordered[i] for i in range(len(ordered) - 1))


__all__ = ["BankSpec"]
