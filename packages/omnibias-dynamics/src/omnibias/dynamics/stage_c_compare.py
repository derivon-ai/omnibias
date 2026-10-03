# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
"""Uniform-in-r1 comparison first-hit on an eps slab of the matching chart.

On lambda1 = -2 the matching-chart height field is

    dx/dσ = eps * (y + (x-r1)(x-r2)),    dy/dσ = x y,

with r2 = 2-r1. The root excess

    (x-r1)(x-r2) - x(x-2) = 2 r1 - r1^2

lies in [0, 1] for every r1 in [0, 1]. From (x, y) = (1/4, 1), every
such field with eps in [1/32, 1/16] is therefore at least as fast in x
as eps * (y + x(x-2)). Phase-wise Interval bounds, using exp of the
integral of x for y, keep that lower speed positive on a grid of width
1/40 from x = 1/4 out to x = 8. The far section (1/4)/(1/32) = 8 is the
matching outgoing section at the small end of the eps slab, so every
larger eps has already crossed its own section x = (1/4)/eps. Freezing
y at 1 stalls at the x = 1 neck.

This is a comparison first-hit on one eps slab and one start, not a
Lohner tube, not every eps, not the shrinking interface
x = r1(1+theta), not complete first-hit on chart O, C2, dx_e off the
kill line, G1, or Hilbert XVI. The x = 1/32 Lohner cover is
stage_c_origin_x32.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from fractions import Fraction

from omnibias.core.collapse.verdict import adjudicate_residual
from omnibias.core.verified.interval import Interval
from omnibias.core.verified.transcend import exp_iv
from omnibias.dynamics.honesty_inventory import build_honesty

__all__ = [
    "StageCCompareReport",
    "enclose_stage_c_compare",
    "identity_verdicts",
    "report",
    "residual_cmp_emax",
    "residual_cmp_phases",
    "residual_cmp_xfar",
]

_EPS_LO = Fraction(1, 32)
_EPS_HI = Fraction(1, 16)
_EMAX = Fraction(1)
_X0 = Fraction(1, 4)
_X_FAR = Fraction(8)
_STEP = Fraction(1, 40)
_N_PHASES = 310
_FREEZE = 3.0
_GAP_FLOOR = 0.125
_TIME_CAP = 250.0


def _honesty(*, stage_c_compare: bool) -> dict[str, object]:
    return build_honesty(
        g1_passed=False,
        g4_passed=False,
        full_graphic_cyclicity_proved=False,
        full_hilbert16_solved=False,
        physical_return_membership_proved=False,
        stage_c_origin_x32=False,
        stage_c_compare=stage_c_compare,
        outgoing_first_hit=False,
        hk_theorem_24_used=False,
    )


def residual_cmp_emax() -> Fraction:
    """``2*1 - 1^2 = 1``: root excess at ``r1 = 1``."""
    return Fraction(2) - Fraction(1) - Fraction(1)


def residual_cmp_xfar() -> Fraction:
    """``(1/4)/(1/32) = 8``: matching section at the small eps."""
    return Fraction(1, 4) / _EPS_LO - _X_FAR


def residual_cmp_phases() -> Fraction:
    """``310 * (1/40) = 31/4``: the phase grid fills ``[1/4, 8]``."""
    return Fraction(_N_PHASES) * _STEP - (_X_FAR - _X0)


def _verdict(residual: Fraction) -> str:
    box = Interval.point(float(residual)) if residual == 0 else Interval.from_rational(residual)
    return adjudicate_residual(box, existential=False).status


def identity_verdicts() -> dict[str, str]:
    return {
        "cmp_emax": _verdict(residual_cmp_emax()),
        "cmp_xfar": _verdict(residual_cmp_xfar()),
        "cmp_phases": _verdict(residual_cmp_phases()),
    }


def _quad_min(left: Fraction, right: Fraction) -> Fraction:
    """Minimum of ``z(z-2)`` on ``[left, right]`` (vertex at ``z=1``)."""

    def quad(z: Fraction) -> Fraction:
        return z * (z - 2)

    values = [quad(left), quad(right)]
    if left <= 1 <= right:
        values.append(quad(Fraction(1)))
    return min(values)


def _quad_max(left: Fraction, right: Fraction) -> Fraction:
    """Maximum of ``z(z-2)`` on ``[left, right]`` (attained at an endpoint)."""

    def quad(z: Fraction) -> Fraction:
        return z * (z - 2)

    return max(quad(left), quad(right))


def enclose_stage_c_compare(*, freeze_log: float = _FREEZE) -> dict[str, float | bool | int]:
    """Comparison transit of ``[1/4, 8]`` for every ``r1 in [0, 1]``.

    ``freeze_log`` caps the integral of ``x`` at which ``y`` is still
    refreshed. The sealed run uses ``3``. A cap of ``0`` never refreshes
    ``y`` and stalls at the neck ``x = 1``.
    """
    x = _X0
    log_lo = Interval.from_rational(0)
    log_hi = Interval.from_rational(0)
    y_lo = Interval.from_rational(1)
    elapsed = Interval.from_rational(0)
    phases = 0
    min_gap = 1.0
    while x < _X_FAR:
        x1 = min(_X_FAR, x + _STEP)
        prod_lo = _quad_min(x, x1)
        prod_hi = _quad_max(x, x1)
        gap = y_lo + Interval.from_rational(prod_lo)
        if gap.lo < min_gap:
            min_gap = gap.lo
        if gap.lo <= 0.0:
            return {
                "reached": False,
                "phases": phases,
                "time_hi": float(elapsed.hi),
                "y_lo": float(y_lo.lo),
                "gap_lo": float(gap.lo),
                "stall_x": float(x),
            }
        vmin = Interval.from_rational(_EPS_LO) * gap
        dx = Interval.from_rational(x1 - x)
        dT = dx / vmin
        elapsed = elapsed + dT
        log_hi = log_hi + Interval.from_rational(x1) * dT
        if log_hi.hi < freeze_log:
            y_hi = exp_iv(Interval(0.0, log_hi.hi))
            parenthesis = y_hi + Interval.from_rational(prod_hi + _EMAX)
            vmax = Interval.from_rational(_EPS_HI) * parenthesis
            dT_min = dx / vmax
            log_lo = log_lo + Interval.from_rational(x) * dT_min
            y_new = exp_iv(Interval(log_lo.lo, log_lo.hi))
            y_lo = Interval(y_new.lo, y_new.hi)
        phases += 1
        x = x1
    return {
        "reached": True,
        "phases": phases,
        "time_hi": float(elapsed.hi),
        "y_lo": float(y_lo.lo),
        "gap_lo": float(min_gap),
        "stall_x": float(_X_FAR),
    }


@dataclass(frozen=True)
class StageCCompareReport:
    """Uniform-in-r1 comparison first-hit on one eps slab. Not every eps or G1."""

    identities: Mapping[str, str]
    enclosure: Mapping[str, float | bool | int]
    stall: Mapping[str, float | bool | int]
    stage_c_compare: bool
    outgoing_first_hit: bool
    honesty: Mapping[str, object]

    def to_payload(self) -> dict[str, object]:
        return {
            "schema": "hilbert16-stage-c-compare-v1",
            "identities": dict(self.identities),
            "enclosure": dict(self.enclosure),
            "stall": dict(self.stall),
            "stage_c_compare": self.stage_c_compare,
            "outgoing_first_hit": self.outgoing_first_hit,
            "honesty": dict(self.honesty),
            "scope": (
                "Comparison first-hit of the matching outgoing section "
                "from (x,y)=(1/4,1) for every r1 in [0,1] and every eps "
                "in [1/32, 1/16], out to x=8. Freezing y at 1 stalls. "
                "Not a Lohner tube, not every eps, not the shrinking "
                "interface, not complete first-hit on chart O, C2, G1, "
                "or Hilbert XVI."
            ),
        }


def report() -> StageCCompareReport:
    """Replay the excess identities and the phase-wise comparison transit."""
    identities = identity_verdicts()
    enclosure = enclose_stage_c_compare()
    stall = enclose_stage_c_compare(freeze_log=0.0)
    sealed = (
        all(status == "PROVED" for status in identities.values())
        and bool(enclosure["reached"])
        and int(enclosure["phases"]) == _N_PHASES
        and float(enclosure["gap_lo"]) > _GAP_FLOOR
        and float(enclosure["time_hi"]) < _TIME_CAP
        and not bool(stall["reached"])
    )
    return StageCCompareReport(
        identities=identities,
        enclosure=enclosure,
        stall=stall,
        stage_c_compare=sealed,
        outgoing_first_hit=False,
        honesty=_honesty(stage_c_compare=sealed),
    )
