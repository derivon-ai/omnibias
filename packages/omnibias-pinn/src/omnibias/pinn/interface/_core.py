# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Parallel-interface transmission algebra (theory 02-05).

``alpha -> inf`` is **interface sharpening**, neither bias collapse
(``delta -> 0``) nor temperature collapse (``beta -> inf``). Conditions hold
to a stated smoothing tolerance, not exactly, at finite ``alpha``. Only
**parallel** interfaces (a shared normal) are in scope.

This module is not :mod:`omnibias.pinn._core.interface` (the XPINN penalty
glue). Import :class:`Interface` from :mod:`omnibias.pinn.interface` or the
:class:`TransmissionInterface` alias.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from math import inf, log, nextafter
from typing import Literal

from omnibias.core.verified.interval import Interval

Condition = Literal["value", "flux", "curvature", "imperfect", "free"]

_ORDER: dict[str, int] = {
    "value": 0,
    "flux": 1,
    "curvature": 2,
    "imperfect": 0,
    "free": 0,
}


def order_for_condition(condition: str) -> int:
    """Pack/profile order: value→0, flux→1, curvature→2."""
    key = str(condition).lower()
    if key not in _ORDER:
        raise ValueError(f"unknown interface condition {condition!r}")
    return _ORDER[key]


@dataclass(frozen=True)
class Interface:
    """One parallel interface ``{ w · x + offset = 0 }``.

    Not the XPINN :class:`~omnibias.pinn._core.interface.Interface`.
    """

    normal: tuple[float, ...]
    offset: float
    condition: Condition = "flux"
    order: int | None = None
    sharpness: float = 50.0
    parameter: float | None = None
    jump: float = 0.0

    def __post_init__(self) -> None:
        if not self.normal:
            raise ValueError("normal must be non-empty")
        if float(self.sharpness) <= 0.0:
            raise ValueError("sharpness (alpha) must be positive")
        n = self.order if self.order is not None else order_for_condition(self.condition)
        object.__setattr__(self, "order", int(n))

    @property
    def alpha(self) -> float:
        return float(self.sharpness)

    def trans_coord(self, x: tuple[float, ...] | list[float]) -> float:
        return sum(float(w) * float(xi) for w, xi in zip(self.normal, x, strict=True)) + float(
            self.offset
        )


TransmissionInterface = Interface


def _log2_iv() -> Interval:
    x = log(2.0)
    return Interval(nextafter(x, -inf), nextafter(x, inf))


def smoothing_error_bound(iface: Interface, *, coeff: float | None = None) -> Interval:
    """Certified ``O(1/alpha)`` bound; worked flux example is ``|C| log 2 / alpha``.

    ``coeff`` defaults to ``iface.jump / 2`` (the outer pack weight for a
    tanh-family profile whose sharp jump is 2).
    """
    c = float(iface.jump) / 2.0 if coeff is None else float(coeff)
    alpha = Interval.point(float(iface.alpha))
    return Interval.point(abs(c)) * _log2_iv() / alpha


def log_cosh(z: float) -> float:
    az = abs(z)
    return az + math.log1p(math.exp(-2.0 * az)) - math.log(2.0)


def int_log_cosh(z: float, *, terms: int = 24) -> float:
    """``int_0^z log(cosh t) dt``, odd, numerically stable."""
    s = 1.0 if z >= 0.0 else -1.0
    az = abs(z)
    acc = 0.5 * az * az - math.log(2.0) * az
    for k in range(1, terms + 1):
        sign = 1.0 if k % 2 == 1 else -1.0
        acc += sign * (1.0 - math.exp(-2.0 * k * az)) / (2.0 * k * k)
    return s * acc


def profile(order: int, z: float, alpha: float) -> float:
    """Smoothed jump profile: value ``tanh``, flux ``logcosh/alpha``, curvature double integral."""
    n = int(order)
    az = alpha * z
    if n == 0:
        return math.tanh(az)
    if n == 1:
        return log_cosh(az) / alpha
    if n == 2:
        return int_log_cosh(az) / (alpha * alpha)
    raise ValueError(f"unsupported profile order {order}")


__all__ = [
    "Interface",
    "TransmissionInterface",
    "int_log_cosh",
    "log_cosh",
    "order_for_condition",
    "profile",
    "smoothing_error_bound",
]
