# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
"""Uniform comparison first-hit for every eps in (0, 1/16] and every r1 in [0, 1].

On lambda1 = -2 the matching-chart height field is

    dx/dσ = eps * (y + (x - r1) * (x - r2)),    dy/dσ = x * y,

with r2 = 2 - r1. The excess (x-r1)(x-r2) - x(x-2) = 2*r1 - r1^2 lies
in [0, 1] for every r1 in [0, 1]. From (x, y) = (1/4, 1),

    dy/dx = x y / (eps * (y + x(x-2) + e)).

For eps <= 1/16 and e <= 1 this is at least the phase-wise lower bound
integrated below. That bound keeps y + x(x-2) > 1/2 on a grid of width
1/40 from x = 1/4 to x = 2, and y(2) > 16. For x >= 2 the product
x(x-2) is nonnegative and y is still larger, so

    dx/dσ >= eps / 2 > 0

on the whole half-line x >= 1/4. Every such orbit therefore hits its
matching outgoing section x = (1/4)/eps, and the hitting time is at most
(1 - eps) / (2 eps^2). At eps = 1/16 that majorant equals 120. Holding
y at 1 stalls on the phase through x = 1.

This is a comparison first-hit for every eps in (0, 1/16] and every r1
in [0, 1] from one fixed start. It is not a Lohner tube, not eps > 1/16,
not the shrinking interface x = r1(1+theta), not complete first-hit on
chart O, C2, dx_e off the kill line, G1, or Hilbert XVI. The slab
[1/32, 1/16] out to x = 8 is stage_c_compare.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from fractions import Fraction

from omnibias.core.collapse.verdict import adjudicate_residual
from omnibias.core.verified.interval import Interval
from omnibias.dynamics.honesty_inventory import build_honesty

__all__ = [
    "StageCUniformReport",
    "enclose_stage_c_uniform",
    "identity_verdicts",
    "report",
    "residual_unif_neck",
    "residual_unif_phases",
    "residual_unif_time",
]

_EPS_HI = Fraction(1, 16)
_EMAX = Fraction(1)
_X0 = Fraction(1, 4)
_X_NECK = Fraction(2)
_WIDTH = Fraction(7, 4)
_STEP = Fraction(1, 40)
_N_PHASES = 70
_X_SEC = Fraction(4)
_TIME_AT_HI = Fraction(120)
_GAP_FLOOR = 0.5
_Y_NECK_FLOOR = 16.0


def _honesty(*, stage_c_uniform: bool) -> dict[str, object]:
    return build_honesty(
        g1_passed=False,
        g4_passed=False,
        full_graphic_cyclicity_proved=False,
        full_hilbert16_solved=False,
        physical_return_membership_proved=False,
        stage_c_compare=False,
        stage_c_uniform=stage_c_uniform,
        outgoing_first_hit=False,
        hk_theorem_24_used=False,
    )


def residual_unif_neck() -> Fraction:
    """``1/4 + 7/4 = 2``: the neck grid closes at ``x = 2``."""
    return _X0 + _WIDTH - _X_NECK


def residual_unif_phases() -> Fraction:
    """``70 * (1/40) = 7/4``: the neck grid step count."""
    return Fraction(_N_PHASES) * _STEP - (_X_NECK - _X0)


def residual_unif_time() -> Fraction:
    """``2 * (4 - 1/4) / (1/16) = 120``: majorant at the large eps."""
    return Fraction(2) * (_X_SEC - _X0) / _EPS_HI - _TIME_AT_HI


def _verdict(residual: Fraction) -> str:
    box = Interval.point(float(residual)) if residual == 0 else Interval.from_rational(residual)
    return adjudicate_residual(box, existential=False).status


def identity_verdicts() -> dict[str, str]:
    return {
        "unif_neck": _verdict(residual_unif_neck()),
        "unif_phases": _verdict(residual_unif_phases()),
        "unif_time": _verdict(residual_unif_time()),
    }


def _quad_min(left: Fraction, right: Fraction) -> Fraction:
    def quad(z: Fraction) -> Fraction:
        return z * (z - 2)

    values = [quad(left), quad(right)]
    if left <= 1 <= right:
        values.append(quad(Fraction(1)))
    return min(values)


def _quad_max(left: Fraction, right: Fraction) -> Fraction:
    def quad(z: Fraction) -> Fraction:
        return z * (z - 2)

    return max(quad(left), quad(right))


def enclose_stage_c_uniform(*, grow: bool = True) -> dict[str, float | bool | int]:
    """Neck clearance on ``[1/4, 2]`` for every ``eps in (0, 1/16]``.

    ``grow=False`` holds ``y`` at 1. That run stalls, so the ``y`` update
    is what keeps the speed positive.
    """
    x = _X0
    y = Interval.from_rational(1)
    min_gap = 1.0
    phases = 0
    while x < _X_NECK:
        x1 = min(_X_NECK, x + _STEP)
        prod_lo = _quad_min(x, x1)
        prod_hi = _quad_max(x, x1)
        gap = y + Interval.from_rational(prod_lo)
        if gap.lo < min_gap:
            min_gap = gap.lo
        if gap.lo <= 0.0:
            return {
                "reached": False,
                "phases": phases,
                "y_neck": float(y.lo),
                "gap_lo": float(gap.lo),
                "stall_x": float(x),
            }
        if grow:
            excess = prod_hi + _EMAX
            span = Interval.from_rational((x1 * x1 - x * x) / 2)
            eps = Interval.from_rational(_EPS_HI)
            if excess < 0:
                margin = y + Interval.from_rational(excess)
                if margin.lo <= 0.0:
                    return {
                        "reached": False,
                        "phases": phases,
                        "y_neck": float(y.lo),
                        "gap_lo": float(gap.lo),
                        "stall_x": float(x),
                    }
                dy = span / eps
            else:
                dy = (y / (y + Interval.from_rational(excess))) * span / eps
            y = y + dy
        phases += 1
        x = x1
    return {
        "reached": True,
        "phases": phases,
        "y_neck": float(y.lo),
        "gap_lo": float(min_gap),
        "stall_x": float(_X_NECK),
    }


@dataclass(frozen=True)
class StageCUniformReport:
    """Comparison first-hit for every small eps and every r1 in [0, 1]. Not G1."""

    identities: Mapping[str, str]
    enclosure: Mapping[str, float | bool | int]
    stall: Mapping[str, float | bool | int]
    stage_c_uniform: bool
    outgoing_first_hit: bool
    honesty: Mapping[str, object]

    def to_payload(self) -> dict[str, object]:
        return {
            "schema": "hilbert16-stage-c-uniform-v1",
            "identities": dict(self.identities),
            "enclosure": dict(self.enclosure),
            "stall": dict(self.stall),
            "stage_c_uniform": self.stage_c_uniform,
            "outgoing_first_hit": self.outgoing_first_hit,
            "honesty": dict(self.honesty),
            "scope": (
                "Comparison first-hit of x=(1/4)/eps from (x,y)=(1/4,1) "
                "for every r1 in [0,1] and every eps in (0, 1/16], with "
                "dx/dsigma >= eps/2 and time at most (1-eps)/(2 eps^2). "
                "Holding y at 1 stalls. Not a Lohner tube, not eps>1/16, "
                "not the shrinking interface, not complete first-hit on "
                "chart O, C2, G1, or Hilbert XVI."
            ),
        }


def report() -> StageCUniformReport:
    """Replay the neck identities and the uniform speed bound."""
    identities = identity_verdicts()
    enclosure = enclose_stage_c_uniform()
    stall = enclose_stage_c_uniform(grow=False)
    sealed = (
        all(status == "PROVED" for status in identities.values())
        and bool(enclosure["reached"])
        and int(enclosure["phases"]) == _N_PHASES
        and float(enclosure["gap_lo"]) > _GAP_FLOOR
        and float(enclosure["y_neck"]) > _Y_NECK_FLOOR
        and not bool(stall["reached"])
    )
    return StageCUniformReport(
        identities=identities,
        enclosure=enclosure,
        stall=stall,
        stage_c_uniform=sealed,
        outgoing_first_hit=False,
        honesty=_honesty(stage_c_uniform=sealed),
    )
