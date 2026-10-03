# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
"""Slow-fast entry-exit machinery for Huzak-type quadratic graphics."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from typing import Callable

from omnibias.core.verified.interval import Interval
from omnibias.dynamics.return_maps import StoppedEventResult

__all__ = [
    "EntryExitCertificate",
    "SlowDivergenceIntegral",
    "SlowFastGraphic",
    "certify_entry_exit",
    "slow_divergence_integral",
]


@dataclass(frozen=True)
class SlowFastGraphic:
    """A declared slow-fast graphic with separate slow and fast coordinates."""

    name: str
    slow_parameter_count: int
    fast_parameter_count: int


@dataclass(frozen=True)
class SlowDivergenceIntegral:
    """Exact-Q slow divergence integral on a compact height set."""

    graphic: SlowFastGraphic
    height_lo: Fraction
    height_hi: Fraction
    value_enclosure: tuple[float, float]

    def to_payload(self) -> dict[str, object]:
        return {
            "graphic": self.graphic.name,
            "height_lo": [self.height_lo.numerator, self.height_lo.denominator],
            "height_hi": [self.height_hi.numerator, self.height_hi.denominator],
            "value_enclosure": [self.value_enclosure[0], self.value_enclosure[1]],
        }


def slow_divergence_integral(
    graphic: SlowFastGraphic,
    integrand: Callable[[Fraction], Fraction],
    *,
    height_lo: Fraction,
    height_hi: Fraction,
    steps: int = 16,
) -> SlowDivergenceIntegral:
    """Riemann enclosure of the slow divergence integral on ``[height_lo, height_hi]``."""
    if steps < 1 or height_lo <= 0 or height_hi <= height_lo:
        raise ValueError("invalid slow divergence integral domain")
    width = (height_hi - height_lo) / steps
    total_lo = 0.0
    total_hi = 0.0
    for step in range(steps):
        left = height_lo + width * step
        right = left + width
        mid = (left + right) / 2
        value = integrand(mid)
        mag = abs(float(value))
        total_lo += -mag * float(width)
        total_hi += mag * float(width)
    return SlowDivergenceIntegral(graphic, height_lo, height_hi, (total_lo, total_hi))


@dataclass(frozen=True)
class EntryExitCertificate:
    """Certified entry-exit balance between two stopped events."""

    graphic: SlowFastGraphic
    entry: StoppedEventResult
    exit: StoppedEventResult
    balance_enclosure: tuple[float, float]
    certified: bool

    def to_payload(self) -> dict[str, object]:
        return {
            "graphic": self.graphic.name,
            "entry_status": self.entry.status,
            "exit_status": self.exit.status,
            "balance_enclosure": [self.balance_enclosure[0], self.balance_enclosure[1]],
            "certified": self.certified,
        }


def certify_entry_exit(
    graphic: SlowFastGraphic,
    entry: StoppedEventResult,
    exit: StoppedEventResult,
    *,
    balance_fn: Callable[[Interval], Interval],
    box: tuple[float, float],
) -> EntryExitCertificate:
    """Certify entry-exit balance using interval enclosures on a parameter box."""
    if not entry.certified or not exit.certified:
        raise ValueError("entry and exit events must be certified")
    lo, hi = box
    balance = balance_fn(Interval(lo, hi))
    certified = balance.lo <= 0.0 <= balance.hi
    return EntryExitCertificate(graphic, entry, exit, (balance.lo, balance.hi), certified)

