# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Enclosure Collapse algebra and the Width Law (theory 01-14).

The founding bias collapse (``delta -> 0``) yields a smooth
``sigma^(K-1)``. Temperature collapse (``beta -> inf``, feasibility)
yields a 0/1 step. Enclosure Collapse is the third limit:
``width -> 0`` of a *sound enclosure*. The output is a point plus a proof,
not a derivative and not a 0/1 step. Do not conflate the three.

There is no operator ``Collapse([lo, hi]) -> point``. Identifying
endpoints (``lo := hi``) is unsound and is not an API.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

from omnibias.core.verified.interval import Interval
from omnibias.core.verified.jet_flow import WidthBudget
from omnibias.core.verified.sigma import sigma_tower_interval

WidthLawKind = Literal["mean_value", "taylor_remainder", "critical_point"]
DominantName = Literal["truncation", "jacobian", "wrapping", "rounding"]
ActionName = Literal["raise_order", "subdivide", "shrink_step", "stop_floor"]

MEAN_VALUE_FACTOR = 2
CRITICAL_POINT_FACTOR = 1

_ACTIONS: dict[DominantName, ActionName] = {
    "truncation": "raise_order",
    "wrapping": "subdivide",
    "jacobian": "shrink_step",
    "rounding": "stop_floor",
}

_REASONS: dict[DominantName, str] = {
    "truncation": "raise jet / Taylor order; remainder is the leading piece",
    "wrapping": "subdivide the box or switch to affine / Taylor models",
    "jacobian": "shrink the validated-flow step; DF wrapping dominates",
    "rounding": "stop; Lemma Floor, further squeeze is unsound",
}


def honesty_payload() -> dict[str, object]:
    return {
        "founding_bias_collapse": True,
        "temperature_collapse": False,
        "enclosure_collapse": True,
        "width_to_zero": True,
        "not_a_derivative": True,
        "not_a_01_step": True,
        "theorem_prover_verified": False,
        "mathlib_verified": False,
    }


@dataclass(frozen=True)
class WidthLaw:
    """Closed-form leading coefficient of enclosure width."""

    center: float
    order: int
    predicted_leading: Interval
    exponent: int
    activation: str
    kind: WidthLawKind


@dataclass(frozen=True)
class RecommendedAction:
    """What to do when a named ``WidthBudget`` piece dominates."""

    dominant: DominantName
    action: ActionName
    reason: str

    def to_payload(self) -> dict[str, str]:
        return {
            "dominant": self.dominant,
            "action": self.action,
            "reason": self.reason,
        }


def _factorial(n: int) -> int:
    if n < 0:
        raise ValueError("factorial is undefined for negative n")
    acc = 1
    for k in range(2, n + 1):
        acc *= k
    return acc


def _require_radius(r: float) -> float:
    radius = float(r)
    if radius <= 0.0:
        raise ValueError(f"r must be positive, got {r!r}")
    return radius


def _box(center: float, r: float) -> tuple[Interval, Interval]:
    c = Interval.point(float(center))
    radius = _require_radius(r)
    return c, c + Interval(-radius, radius)


def _abs_at_center(activation: str, center: float, order: int) -> Interval:
    if order < 0:
        raise ValueError("order must be >= 0")
    tower = sigma_tower_interval(activation, Interval.point(float(center)), order)
    return tower[order].abs()


def width_law(
    activation: str,
    n: int,
    center: float,
    *,
    kind: WidthLawKind = "mean_value",
    order: int | None = None,
) -> WidthLaw:
    """Leading coefficient of ``w(r)`` from the closed-form tower.

    ``n`` is the order of ``f = sigma^(n)``. Mean-value uses
    ``|f'(c)| = |sigma^(n+1)(c)|`` with exponent 1. Critical-point uses
    ``|f''(c)|`` with exponent 2 and factor ``CRITICAL_POINT_FACTOR``.
    Taylor remainder of order ``N`` uses ``|f^{(N+1)}(c)| / (N+1)!``.
    """
    if n < 0:
        raise ValueError("n must be >= 0")
    name = str(activation)
    if kind == "mean_value":
        leading = _abs_at_center(name, center, n + 1)
        return WidthLaw(float(center), 0, leading, 1, name, kind)
    if kind == "critical_point":
        first = sigma_tower_interval(name, Interval.point(float(center)), n + 2)
        if not first[n + 1].contains_zero() and abs(first[n + 1].mid) > 1e-9:
            raise ValueError("critical_point requires f'(c) near 0")
        leading = first[n + 2].abs()
        return WidthLaw(float(center), 1, leading, 2, name, kind)
    if kind == "taylor_remainder":
        n_order = 2 if order is None else int(order)
        if n_order < 0:
            raise ValueError("Taylor order N must be >= 0")
        deriv_order = n + n_order + 1
        leading = _abs_at_center(name, center, deriv_order) / float(_factorial(n_order + 1))
        return WidthLaw(float(center), n_order, leading, n_order + 1, name, kind)
    raise ValueError(f"unknown WidthLaw kind {kind!r}")


def predicted_width(law: WidthLaw, r: float) -> Interval:
    """Enclosure of the predicted width ``C * r^exponent`` (with locked factors)."""
    radius = _require_radius(r)
    scale = Interval.point(radius) ** law.exponent
    if law.kind == "mean_value":
        return law.predicted_leading * scale * MEAN_VALUE_FACTOR
    if law.kind == "critical_point":
        return law.predicted_leading * scale * CRITICAL_POINT_FACTOR
    factor = 2 if law.exponent % 2 == 1 else 1
    return law.predicted_leading * scale * factor


def measured_mean_value_width(activation: str, n: int, center: float, r: float) -> Interval:
    """Mean-value enclosure ``f(c) + f'(I)·(I-c)`` of ``sigma^(n)`` on ``I(r)``."""
    if n < 0:
        raise ValueError("n must be >= 0")
    c, box = _box(center, r)
    f_c = sigma_tower_interval(activation, c, n)[n]
    f_prime = sigma_tower_interval(activation, box, n + 1)[n + 1]
    return f_c + f_prime * (box - c)


def measured_critical_width(activation: str, n: int, center: float, r: float) -> Interval:
    """Second-order form ``f(c) + f''(I)·(I-c)^2`` (factor locked to 1)."""
    if n < 0:
        raise ValueError("n must be >= 0")
    c, box = _box(center, r)
    f_c = sigma_tower_interval(activation, c, n)[n]
    f_pp = sigma_tower_interval(activation, box, n + 2)[n + 2]
    return f_c + f_pp * (box - c) ** 2


def measured_taylor_remainder(activation: str, n: int, center: float, r: float, order: int) -> Interval:
    """Lagrange remainder enclosure of ``sigma^(n)`` after order ``N``."""
    if n < 0 or order < 0:
        raise ValueError("n and order must be >= 0")
    _c, box = _box(center, r)
    deriv = sigma_tower_interval(activation, box, n + order + 1)[n + order + 1]
    return deriv * (box - Interval.point(float(center))) ** (order + 1) / float(_factorial(order + 1))


def diagnose_width(budget: WidthBudget) -> RecommendedAction:
    """Recommend an action from an existing ``WidthBudget``. Does not fork the type."""
    match budget.dominant:
        case "truncation" | "jacobian" | "wrapping" | "rounding" as name:
            return RecommendedAction(name, _ACTIONS[name], _REASONS[name])
        case other:
            raise ValueError(f"unknown WidthBudget dominant {other!r}")


def rounding_floor_witness() -> Interval:
    """Lemma Floor: a non-exact evaluation whose width is at least ``2 ulp``."""
    return Interval.point(1.0) / 3.0


def two_ulp(x: float) -> float:
    """Two outward steps from ``x`` in the ``+inf`` direction."""
    return math.nextafter(math.nextafter(float(x), math.inf), math.inf)


__all__ = [
    "CRITICAL_POINT_FACTOR",
    "MEAN_VALUE_FACTOR",
    "RecommendedAction",
    "WidthBudget",
    "WidthLaw",
    "diagnose_width",
    "honesty_payload",
    "measured_critical_width",
    "measured_mean_value_width",
    "measured_taylor_remainder",
    "predicted_width",
    "rounding_floor_witness",
    "two_ulp",
    "width_law",
]
