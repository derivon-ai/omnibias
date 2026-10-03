# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
"""Kill-line Stage-C continuation rectangle past T=h.

On lambda1 = -2 the sealed T-h < 1 bound gives T < h+1, so
|V| = sqrt(2T) < sqrt(2h+2). At hmax = 1 that left wall is
sqrt(4) = 2. The matching-chart floor |V| = eps x with x >= 1/4
gives the right wall eps/4 = 1/64 at the compact edge. Interval
wrapping still keeps the left wall < 3 and the right wall > 0.
The comparison orbit therefore stays in the rectangle
V in [-2, -1/64], h in [h_1, 1] on this compact. This is a sealed
continuation rectangle, not first-hit of the selected large
section, C2, dx_e off the kill line, G1, or Hilbert XVI. The T-h
margin is stage_c_boot. The a_min floor is stage_c.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from fractions import Fraction

from omnibias.core.collapse.verdict import adjudicate_residual
from omnibias.core.verified.interval import Interval
from omnibias.dynamics.honesty_inventory import build_honesty
from omnibias.dynamics.stage_c_boot import enclose_stage_c_boot

__all__ = [
    "StageCRectReport",
    "enclose_stage_c_rect",
    "identity_verdicts",
    "report",
    "residual_amin_eps",
    "residual_hmax_two",
    "residual_twice_two",
    "sample_stage_c_rect",
]

_HMAX = Fraction(1)
_MARGIN = Fraction(1)
_T_WALL = Fraction(2)
_V2 = Fraction(4)
_V_LEFT = Fraction(2)
_EPS_HI = Fraction(1, 16)
_AMIN = Fraction(1, 4)
_V_RIGHT = Fraction(1, 64)
_DECLARED_LEFT = 3.0
_DECLARED_RIGHT = 0.03125


def _honesty(*, stage_c_rect: bool) -> dict[str, object]:
    return build_honesty(
        g1_passed=False,
        g4_passed=False,
        full_graphic_cyclicity_proved=False,
        full_hilbert16_solved=False,
        physical_return_membership_proved=False,
        orbit_continuation_on_kill=False,
        stage_a_wall=False,
        chi_b_bound=False,
        dx_e_leading=False,
        dx_e_unif=False,
        stage_c_amin=False,
        stage_c_exit=False,
        stage_c_th=False,
        stage_c_gap=False,
        stage_c_env=False,
        stage_c_if=False,
        stage_c_int=False,
        stage_c_lo=False,
        stage_c_k=False,
        stage_c_boot=False,
        stage_c_rect=stage_c_rect,
        hk_theorem_24_used=False,
    )


def residual_hmax_two() -> Fraction:
    """``1 + 1 = 2``: T-wall at hmax after the sealed T-h < 1 margin."""
    return _HMAX + _MARGIN - _T_WALL


def residual_twice_two() -> Fraction:
    """``2 * 2 = 4``: V^2 = 2T at the left wall."""
    return 2 * _T_WALL - _V2


def residual_amin_eps() -> Fraction:
    """``(1/16) * (1/4) = 1/64``: matching-chart |V| floor at the compact edge."""
    return _EPS_HI * _AMIN - _V_RIGHT


def _verdict(residual: Fraction) -> str:
    box = Interval.point(float(residual)) if residual == 0 else Interval.from_rational(residual)
    return adjudicate_residual(box, existential=False).status


def identity_verdicts() -> dict[str, str]:
    return {
        "hmax_two": _verdict(residual_hmax_two()),
        "twice_two": _verdict(residual_twice_two()),
        "amin_eps": _verdict(residual_amin_eps()),
    }


def _contains(outer: Interval, inner: Interval) -> bool:
    return outer.lo <= inner.lo and inner.hi <= outer.hi


def enclose_stage_c_rect() -> dict[str, float | bool]:
    """Stage-C continuation rectangle: left wall < 3, right wall > 0."""
    boot = enclose_stage_c_boot()
    two_t = Interval.from_rational(2) * Interval.from_rational(_T_WALL)
    v_left = two_t.sqrt()
    v_right = Interval.from_rational(_EPS_HI) * Interval.from_rational(_AMIN)
    v_left_lo = float(v_left.lo)
    v_left_hi = float(v_left.hi)
    v_right_lo = float(v_right.lo)
    v_right_hi = float(v_right.hi)
    return {
        "v_left_lo": v_left_lo,
        "v_left_hi": v_left_hi,
        "v_right_lo": v_right_lo,
        "v_right_hi": v_right_hi,
        "total_hi": float(boot["total_hi"]),
        "below_declared": v_left_hi < _DECLARED_LEFT and v_right_hi < _DECLARED_RIGHT,
        "boot_below": float(boot["total_hi"]) < 1.0,
        "excludes_zero": v_left_lo > 0.0 and v_right_lo > 0.0,
        "finite": (
            v_left_hi < _DECLARED_LEFT
            and v_right_hi < _DECLARED_RIGHT
            and float(boot["total_hi"]) < 1.0
        ),
    }


def sample_stage_c_rect() -> Interval:
    """Sound left-wall ``|V| = 2`` at ``hmax = 1``."""
    return Interval.from_rational(_V_LEFT)


@dataclass(frozen=True)
class StageCRectReport:
    """Kill-line Stage-C continuation rectangle. Not first-hit or G1."""

    identities: Mapping[str, str]
    sample_left: Mapping[str, float]
    enclosure: Mapping[str, float | bool]
    sample_inside: bool
    stage_c_rect: bool
    honesty: Mapping[str, object]

    def to_payload(self) -> dict[str, object]:
        return {
            "schema": "hilbert16-stage-c-rect-v1",
            "identities": dict(self.identities),
            "sample_left": dict(self.sample_left),
            "enclosure": dict(self.enclosure),
            "sample_inside": self.sample_inside,
            "stage_c_rect": self.stage_c_rect,
            "honesty": dict(self.honesty),
            "scope": (
                "Exact kill-line Stage-C continuation-rectangle identities: "
                "1+1=2, 2*2=4, and (1/16)*(1/4)=1/64, plus Interval left wall "
                "< 3 and right wall > 0. Not first-hit, C2, G1, or Hilbert XVI."
            ),
        }


def report() -> StageCRectReport:
    """Replay Stage-C rectangle identities and enclose the walls."""
    identities = identity_verdicts()
    sample = sample_stage_c_rect()
    try:
        enclosure = enclose_stage_c_rect()
        inside = bool(enclosure["finite"]) and _contains(
            Interval(float(enclosure["v_left_lo"]), float(enclosure["v_left_hi"])),
            sample,
        )
        sealed = (
            all(status == "PROVED" for status in identities.values())
            and bool(enclosure["below_declared"])
            and bool(enclosure["boot_below"])
            and bool(enclosure["excludes_zero"])
            and inside
        )
    except (ValueError, ZeroDivisionError, ArithmeticError):
        enclosure = {
            "v_left_lo": float("nan"),
            "v_left_hi": float("nan"),
            "v_right_lo": float("nan"),
            "v_right_hi": float("nan"),
            "below_declared": False,
            "boot_below": False,
            "excludes_zero": False,
            "finite": False,
        }
        inside = False
        sealed = False
    return StageCRectReport(
        identities=identities,
        sample_left={"lo": sample.lo, "hi": sample.hi},
        enclosure=enclosure,
        sample_inside=inside,
        stage_c_rect=sealed,
        honesty=_honesty(stage_c_rect=sealed),
    )
