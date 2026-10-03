# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
"""Comparison first-hit from every matching-chart start in (0, 1/2].

On lambda1 = -2, with root excess in [0, 1], the height field from
(x, y) = (x0, 1) satisfies y + x(x-2) >= 1/4 on (0, 1/2]: the most
negative product there is (1/2)(1/2-2) = -3/4. The orbit therefore
reaches x = 1/2 with y >= 1. From that worst entrance, the same
dy/dx bound used for the fixed start x = 1/4 keeps the speed gap
above 1/5 out to x = 2, with y(2) > 16. Past x = 2 the product
x(x-2) is nonnegative, so

    dx/dσ >= eps / 5

for every eps in (0, 1/16] and every r1 in [0, 1]. Every such orbit
hits x = (1/4)/eps. The longest majorant, from x = 0 at eps = 1/16,
is 320. Holding y at 1 stalls through x = 1.

Any shrinking interface x = r1(1+theta) that lands in (0, 1/2] is one
of these starts. Chart-O sequences r1 -> 0 do land there. This is not
a Lohner tube, not the height-section flag on L = 1/n, not eps > 1/16,
not an interface that sits past x = 1/2, not complete first-hit on
chart O, C2, G1, or Hilbert XVI. The fixed start x = 1/4 is
stage_c_uniform.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from fractions import Fraction

from omnibias.core.collapse.verdict import adjudicate_residual
from omnibias.core.verified.interval import Interval
from omnibias.dynamics.honesty_inventory import build_honesty

__all__ = [
    "StageCInterfaceReport",
    "enclose_stage_c_interface",
    "identity_verdicts",
    "report",
    "residual_iface_edge",
    "residual_iface_phases",
    "residual_iface_time",
]

_X_ENTER = Fraction(1, 2)
_ENTRANCE_GAP = Fraction(1, 4)
_X_NECK = Fraction(2)
_WIDTH = Fraction(3, 2)
_STEP = Fraction(1, 40)
_N_PHASES = 60
_EPS_HI = Fraction(1, 16)
_EMAX = Fraction(1)
_TIME_CAP = Fraction(320)
_GAP_FLOOR = 0.2
_Y_NECK_FLOOR = 16.0


def _honesty(*, stage_c_interface: bool) -> dict[str, object]:
    return build_honesty(
        g1_passed=False,
        g4_passed=False,
        full_graphic_cyclicity_proved=False,
        full_hilbert16_solved=False,
        physical_return_membership_proved=False,
        stage_c_uniform=False,
        stage_c_interface=stage_c_interface,
        outgoing_first_hit=False,
        hk_theorem_24_used=False,
    )


def residual_iface_edge() -> Fraction:
    """``1 + (1/2)*(1/2-2) = 1/4``: entrance gap on ``(0, 1/2]``."""
    return Fraction(1) + _X_ENTER * (_X_ENTER - 2) - _ENTRANCE_GAP


def residual_iface_phases() -> Fraction:
    """``60 * (1/40) = 3/2``: neck grid from ``1/2`` to ``2``."""
    return Fraction(_N_PHASES) * _STEP - _WIDTH


def residual_iface_time() -> Fraction:
    """``5 * (1/4) / (1/16)^2 = 320``: longest majorant at ``eps = 1/16``."""
    return Fraction(5) * Fraction(1, 4) / (_EPS_HI * _EPS_HI) - _TIME_CAP


def _verdict(residual: Fraction) -> str:
    box = Interval.point(float(residual)) if residual == 0 else Interval.from_rational(residual)
    return adjudicate_residual(box, existential=False).status


def identity_verdicts() -> dict[str, str]:
    return {
        "iface_edge": _verdict(residual_iface_edge()),
        "iface_phases": _verdict(residual_iface_phases()),
        "iface_time": _verdict(residual_iface_time()),
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


def enclose_stage_c_interface(*, grow: bool = True) -> dict[str, float | bool | int]:
    """Neck clearance from the worst entrance ``x = 1/2``, ``y = 1``.

    Starts in ``(0, 1/2)`` reach this entrance with ``y >= 1`` because
    the entrance gap is ``1/4``. ``grow=False`` holds ``y`` at 1 and stalls.
    """
    x = _X_ENTER
    y = Interval.from_rational(1)
    min_gap = 1.0
    phases = 0
    while x < _X_NECK:
        x1 = min(_X_NECK, x + _STEP)
        gap = y + Interval.from_rational(_quad_min(x, x1))
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
            excess = _quad_max(x, x1) + _EMAX
            span = Interval.from_rational((x1 * x1 - x * x) / 2)
            eps = Interval.from_rational(_EPS_HI)
            if excess < 0:
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
class StageCInterfaceReport:
    """Comparison first-hit from every start in (0, 1/2]. Not the height flag or G1."""

    identities: Mapping[str, str]
    enclosure: Mapping[str, float | bool | int]
    stall: Mapping[str, float | bool | int]
    stage_c_interface: bool
    outgoing_first_hit: bool
    honesty: Mapping[str, object]

    def to_payload(self) -> dict[str, object]:
        return {
            "schema": "hilbert16-stage-c-interface-v1",
            "identities": dict(self.identities),
            "enclosure": dict(self.enclosure),
            "stall": dict(self.stall),
            "stage_c_interface": self.stage_c_interface,
            "outgoing_first_hit": self.outgoing_first_hit,
            "honesty": dict(self.honesty),
            "scope": (
                "Comparison first-hit of x=(1/4)/eps from every start "
                "(x0,1) with x0 in (0, 1/2], for every r1 in [0,1] and "
                "every eps in (0, 1/16]. An interface r1(1+theta) in that "
                "interval is included. Holding y at 1 stalls. Not a Lohner "
                "tube, not the height-section flag, not eps>1/16, not "
                "complete first-hit on chart O, C2, G1, or Hilbert XVI."
            ),
        }


def report() -> StageCInterfaceReport:
    """Replay entrance identities and the worst-start neck bound."""
    identities = identity_verdicts()
    enclosure = enclose_stage_c_interface()
    stall = enclose_stage_c_interface(grow=False)
    sealed = (
        all(status == "PROVED" for status in identities.values())
        and _ENTRANCE_GAP > Fraction(1, 5)
        and bool(enclosure["reached"])
        and int(enclosure["phases"]) == _N_PHASES
        and float(enclosure["gap_lo"]) > _GAP_FLOOR
        and float(enclosure["y_neck"]) > _Y_NECK_FLOOR
        and not bool(stall["reached"])
    )
    return StageCInterfaceReport(
        identities=identities,
        enclosure=enclosure,
        stall=stall,
        stage_c_interface=sealed,
        outgoing_first_hit=False,
        honesty=_honesty(stage_c_interface=sealed),
    )
