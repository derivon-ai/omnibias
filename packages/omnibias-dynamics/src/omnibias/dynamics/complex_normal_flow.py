# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
"""Complex fixed-time enclosure of one Hilbert-XVI normal-form cell.

The cubic ``(V,h)`` normal field is complexified together with a frozen
``epsilon`` coordinate.  The result rigorously bounds a fixed-real-time flow
for every epsilon in a declared complex rectangle.  It is a prerequisite for
a complex return-map argument, but it is not a parameter-dependent first-hit,
an exact Log-Noetherian chain, or a discharge of G3.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from fractions import Fraction

from omnibias.core.verified.complex_interval import ComplexInterval
from omnibias.core.verified.complex_ode import (
    ComplexTaylorSeries,
    integrate_complex_ivp,
)
from omnibias.core.verified.interval import Interval
from omnibias.dynamics.honesty_inventory import build_honesty

__all__ = [
    "ComplexNormalFlowReport",
    "complex_epsilon_cell",
    "complex_normal_field",
    "enclose_complex_normal_flow",
    "report",
]

_EPSILON = Fraction(1, 16)
_RADIUS = Fraction(1, 10_000)
_L = Fraction(9, 25)
_LAMBDA1 = Fraction(-2)
_FINAL_TIME = Fraction(1, 2)


def _rational_interval(lo: Fraction, hi: Fraction) -> Interval:
    return Interval.hull(Interval.from_rational(lo), Interval.from_rational(hi))


def complex_epsilon_cell(
    *,
    center: Fraction = _EPSILON,
    real_radius: Fraction = _RADIUS,
    imag_radius: Fraction = _RADIUS,
) -> ComplexInterval:
    """Closed complex epsilon rectangle around a positive interior point."""
    if real_radius <= 0 or imag_radius <= 0:
        raise ValueError("complex epsilon radii must be positive")
    if center - real_radius <= 0:
        raise ValueError("the epsilon rectangle must stay in Re(epsilon) > 0")
    return ComplexInterval(
        _rational_interval(center - real_radius, center + real_radius),
        _rational_interval(-imag_radius, imag_radius),
    )


def complex_normal_field(
    series: list[ComplexTaylorSeries],
) -> list[ComplexTaylorSeries]:
    """Complexified cubic normal field with frozen epsilon as state 3."""
    if len(series) != 3:
        raise ValueError("complex normal field expects (V, h, epsilon)")
    v_coord, height, epsilon = series
    v_squared = v_coord * v_coord
    field_f = (
        -_L * epsilon * epsilon * epsilon
        + _LAMBDA1 * epsilon * epsilon * v_coord
        - epsilon * v_squared
        + Fraction(1, 3) * epsilon * v_squared * v_coord
    )
    field_g = -1 + epsilon * (v_coord - 1)
    return [
        field_f + height * field_g,
        -v_coord * height,
        0 * epsilon,
    ]


def enclose_complex_normal_flow(
    *,
    epsilon: ComplexInterval | None = None,
    final_time: Fraction = _FINAL_TIME,
    order: int = 12,
    n_steps: int = 8,
) -> tuple[ComplexInterval, ComplexInterval, ComplexInterval]:
    """Enclose the complex normal flow from ``V=-epsilon, h=4 epsilon^3``."""
    if final_time <= 0:
        raise ValueError("final_time must be positive")
    epsilon_box = complex_epsilon_cell() if epsilon is None else epsilon
    initial_v = -epsilon_box
    initial_h = 4 * epsilon_box * epsilon_box * epsilon_box
    result = integrate_complex_ivp(
        complex_normal_field,
        (initial_v, initial_h, epsilon_box),
        0.0,
        float(final_time),
        order=order,
        n_steps=n_steps,
    )
    return result[0], result[1], result[2]


def _complex_payload(value: ComplexInterval) -> dict[str, list[float]]:
    return {
        "re": [value.re.lo, value.re.hi],
        "im": [value.im.lo, value.im.hi],
    }


@dataclass(frozen=True)
class ComplexNormalFlowReport:
    """One complex fixed-time flow enclosure; no event-map or LN claim."""

    epsilon_cell: ComplexInterval
    final_box: tuple[ComplexInterval, ComplexInterval, ComplexInterval]
    fixed_time_complex_flow_enclosed: bool
    flow_sup_upper: float
    honesty: Mapping[str, object]

    def to_payload(self) -> dict[str, object]:
        return {
            "schema": "hilbert16-complex-normal-flow-v1",
            "epsilon_cell": _complex_payload(self.epsilon_cell),
            "final_box": [_complex_payload(value) for value in self.final_box],
            "final_time": str(_FINAL_TIME),
            "fixed_time_complex_flow_enclosed": self.fixed_time_complex_flow_enclosed,
            "flow_sup_upper": self.flow_sup_upper,
            "complex_first_hit_holomorphic": False,
            "actual_return_ln_membership_proved": False,
            "g3_passed": False,
            "full_hilbert16_solved": False,
            "honesty": dict(self.honesty),
            "scope": (
                "Validated fixed-real-time complex flow of the cubic Hilbert-XVI "
                "normal-form comparison field on one interior epsilon rectangle. "
                "Not a complex first-hit map, not the full physical quadratic "
                "return family, not Log-Noetherian membership, not G3, and not "
                "Hilbert XVI."
            ),
        }


def report() -> ComplexNormalFlowReport:
    """Enclose the declared complex cell and retain every parent refusal."""
    epsilon = complex_epsilon_cell()
    final_box = enclose_complex_normal_flow(epsilon=epsilon)
    finite = all(
        math.isfinite(endpoint)
        for value in final_box
        for endpoint in (value.re.lo, value.re.hi, value.im.lo, value.im.hi)
    )
    frozen_parameter = (
        final_box[2].re.lo <= epsilon.re.lo
        and epsilon.re.hi <= final_box[2].re.hi
        and final_box[2].im.lo <= epsilon.im.lo
        and epsilon.im.hi <= final_box[2].im.hi
    )
    enclosed = finite and frozen_parameter
    sup_upper = max(value.mag for value in final_box)
    honesty = build_honesty(
        fixed_time_complex_flow_enclosed=enclosed,
        complex_first_hit_holomorphic=False,
        actual_return_ln_membership_proved=False,
        g3_passed=False,
        full_hilbert16_solved=False,
    )
    return ComplexNormalFlowReport(
        epsilon_cell=epsilon,
        final_box=final_box,
        fixed_time_complex_flow_enclosed=enclosed,
        flow_sup_upper=sup_upper,
        honesty=honesty,
    )
