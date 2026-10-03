# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
"""Stage-A shrinking-rectangle wall on the kill line.

On lambda1 = -2 the slow-line midpoint is rstar = 1 and
r1 = 1 - sep/2. With theta = 1/8 the Stage-A walls are
a = r1 - theta sep and bnd = r1 + theta sep. The limiting
quadratic satisfies

    B_-(a) = theta (1 + theta) sep^2,
    B_-'(bnd) = -sep (1 - 2 theta).

For sep in [0, 1] the left wall stays in [3/8, 1], so a declared
floor a >= 1/4 holds after Interval wrapping. The leading Psi_pre
factor B_-(a)/(B_-(0) sep^2) is below 1/4. This is the geometric
core of the chi-rectangle, not dx_e/dkappa, not a chi threshold,
not Stage C, first-hit, G1, or Hilbert XVI. Kill-line Stage B is
stage_b.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from fractions import Fraction

from omnibias.core.collapse.verdict import adjudicate_residual
from omnibias.core.verified.interval import Interval
from omnibias.dynamics.honesty_inventory import build_honesty
from omnibias.dynamics.stage_b import r1_kill

__all__ = [
    "StageAReport",
    "B_minus",
    "enclose_stage_a",
    "identity_verdicts",
    "r2_kill",
    "report",
    "residual_a_min",
    "residual_B_slope",
    "residual_B_wall",
    "sample_stage_a",
    "wall_left",
    "wall_right",
]

_SAMPLE_SEP = Fraction(3, 5)
_THETA = Fraction(1, 8)
_RSTAR = Fraction(1)
_A_MIN_EXACT = Fraction(3, 8)
_DECLARED_A_MIN = Fraction(1, 4)
_SLOPE_FACTOR = Fraction(3, 4)
_WALL_FACTOR = Fraction(9, 64)


def _honesty(*, stage_a_wall: bool) -> dict[str, object]:
    return build_honesty(
        g1_passed=False,
        g4_passed=False,
        full_graphic_cyclicity_proved=False,
        full_hilbert16_solved=False,
        physical_return_membership_proved=False,
        orbit_continuation_on_kill=False,
        stage_a_wall=stage_a_wall,
        hk_theorem_24_used=False,
    )


def r2_kill(sep: Fraction) -> Fraction:
    """Outgoing root on ``lambda1 = -2``: ``r2 = 1 + sep/2``."""
    return _RSTAR + sep / 2


def wall_left(sep: Fraction, theta: Fraction = _THETA) -> Fraction:
    """Left Stage-A wall ``a = r1 - theta sep``."""
    return r1_kill(sep) - theta * sep


def wall_right(sep: Fraction, theta: Fraction = _THETA) -> Fraction:
    """Right Stage-A wall ``bnd = r1 + theta sep``."""
    return r1_kill(sep) + theta * sep


def B_minus(x: Fraction, sep: Fraction) -> Fraction:
    """Kill-line slow-line quadratic ``B_-(x) = (x - r1)(x - r2)``."""
    return (x - r1_kill(sep)) * (x - r2_kill(sep))


def residual_B_wall(sep: Fraction) -> Fraction:
    """``B_-(r1 - theta sep) - theta (1 + theta) sep^2``."""
    return B_minus(wall_left(sep), sep) - _THETA * (1 + _THETA) * sep * sep


def residual_B_slope(sep: Fraction) -> Fraction:
    """Kill-line ``B_-' = 2(x - 1)`` at the right wall plus ``sep (1 - 2 theta)``."""
    return 2 * (wall_right(sep) - 1) + sep * (1 - 2 * _THETA)


def residual_a_min() -> Fraction:
    """Worst-case left wall at ``sep = 1``: ``1 - (1/2 + 1/8) = 3/8``."""
    return Fraction(1) - (Fraction(1, 2) + _THETA) - _A_MIN_EXACT


def _verdict(residual: Fraction) -> str:
    box = Interval.point(float(residual)) if residual == 0 else Interval.from_rational(residual)
    return adjudicate_residual(box, existential=False).status


def identity_verdicts() -> dict[str, str]:
    return {
        "B_wall": _verdict(residual_B_wall(_SAMPLE_SEP)),
        "B_slope": _verdict(residual_B_slope(_SAMPLE_SEP)),
        "a_min_kill": _verdict(residual_a_min()),
    }


def _contains(outer: Interval, inner: Interval) -> bool:
    return outer.lo <= inner.lo and inner.hi <= outer.hi


def enclose_stage_a() -> dict[str, float | bool]:
    """Interval image of the kill-line Stage-A left wall on ``sep in [0, 1]``."""
    sep = Interval.hull(Fraction(0), Fraction(1))
    one = Interval.from_rational(Fraction(1))
    half = Interval.from_rational(Fraction(1, 2))
    four = Interval.from_rational(Fraction(4))
    theta = Interval.from_rational(_THETA)
    r1 = one - sep / Interval.from_rational(Fraction(2))
    a = r1 - theta * sep
    wall = Interval.from_rational(_WALL_FACTOR)
    slope = Interval.from_rational(_SLOPE_FACTOR)
    b0 = one - (sep * sep) / four
    psi_factor = wall / b0
    a_lo = float(a.lo)
    a_hi = float(a.hi)
    declared = float(_DECLARED_A_MIN)
    psi_hi = float(psi_factor.hi)
    return {
        "a_lo": a_lo,
        "a_hi": a_hi,
        "wall_factor_lo": wall.lo,
        "wall_factor_hi": wall.hi,
        "slope_factor_lo": slope.lo,
        "slope_factor_hi": slope.hi,
        "psi_factor_lo": psi_factor.lo,
        "psi_factor_hi": psi_hi,
        "b0_lo": b0.lo,
        "b0_hi": b0.hi,
        "r1_lo": r1.lo,
        "r1_hi": r1.hi,
        "half_lo": half.lo,
        "below_declared": a_lo > declared,
        "psi_below_declared": psi_hi < declared,
        "excludes_zero": a_lo > 0.0,
        "slope_ok": slope.lo > 0.5,
        "finite": a_lo > declared and a_lo > 0.0 and psi_hi < declared,
        "theta": float(_THETA),
    }


def sample_stage_a() -> Interval:
    """Sound left wall at ``sep = 3/5``."""
    return Interval.from_rational(wall_left(_SAMPLE_SEP))


@dataclass(frozen=True)
class StageAReport:
    """Kill-line Stage-A wall enclosure. Not dx_e/dkappa, chi, Stage C, or G1."""

    identities: Mapping[str, str]
    sample_a: Mapping[str, float]
    enclosure: Mapping[str, float | bool]
    sample_inside: bool
    stage_a_wall: bool
    honesty: Mapping[str, object]

    def to_payload(self) -> dict[str, object]:
        return {
            "schema": "hilbert16-stage-a-v1",
            "identities": dict(self.identities),
            "sample_a": dict(self.sample_a),
            "enclosure": dict(self.enclosure),
            "sample_inside": self.sample_inside,
            "stage_a_wall": self.stage_a_wall,
            "honesty": dict(self.honesty),
            "scope": (
                "Exact kill-line Stage-A wall identities B_-(a) = "
                "theta(1+theta) sep^2 and B_-'(bnd) = -sep(1-2 theta) "
                "at theta=1/8, plus Interval a >= 1/4 and leading "
                "Psi_pre factor B_-(a)/(B_-(0) sep^2) < 1/4 on "
                "sep in [0, 1]. Not dx_e/dkappa, not a chi threshold, "
                "not Stage C, first-hit, G1, or Hilbert XVI."
            ),
        }


def report() -> StageAReport:
    """Replay Stage-A wall identities and enclose a >= 1/4 on the kill line."""
    identities = identity_verdicts()
    sample = sample_stage_a()
    try:
        enclosure = enclose_stage_a()
        inside = bool(enclosure["finite"]) and _contains(
            Interval(float(enclosure["a_lo"]), float(enclosure["a_hi"])),
            sample,
        )
        sealed = (
            all(status == "PROVED" for status in identities.values())
            and bool(enclosure["below_declared"])
            and bool(enclosure["psi_below_declared"])
            and bool(enclosure["excludes_zero"])
            and bool(enclosure["slope_ok"])
            and inside
        )
    except (ValueError, ZeroDivisionError, ArithmeticError):
        enclosure = {
            "a_lo": float("nan"),
            "a_hi": float("nan"),
            "below_declared": False,
            "psi_below_declared": False,
            "excludes_zero": False,
            "slope_ok": False,
            "finite": False,
        }
        inside = False
        sealed = False
    return StageAReport(
        identities=identities,
        sample_a={"lo": sample.lo, "hi": sample.hi},
        enclosure=enclosure,
        sample_inside=inside,
        stage_a_wall=sealed,
        honesty=_honesty(stage_a_wall=sealed),
    )
