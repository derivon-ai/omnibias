# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
"""Kill-line Stage-C leading T_h floor at y_1 = 1.

On lambda1 = -2 the leading height derivative at matching-chart
exit is T_h = 1 + (x-r1)(x-r2)/y_1. Stage C starts at the
sep-independent height y_1 = 1 after Stage B. At the midpoint
x = 1 the product is -sep^2/4, so the worst value on sep in [0, 1]
is 3/4. Interval wrapping on the Stage-B end box still yields
T_h > 1/2.

This is a sealed Stage-C leading T_h floor on the kill line, not an
outgoing orbit, first-hit, C2, dx_e off the kill line, G1, or
Hilbert XVI. Exit energy is stage_c_exit. The a_min floor is stage_c.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from fractions import Fraction

from omnibias.core.collapse.verdict import adjudicate_residual
from omnibias.core.verified.interval import Interval
from omnibias.dynamics.honesty_inventory import build_honesty
from omnibias.dynamics.stage_a import B_minus
from omnibias.dynamics.stage_c import enclose_stage_c

__all__ = [
    "StageCThReport",
    "enclose_stage_c_th",
    "identity_verdicts",
    "report",
    "residual_mid_product",
    "residual_th_half_room",
    "residual_th_three_four",
    "sample_stage_c_th",
]

_SAMPLE_SEP = Fraction(3, 5)
_Y1 = Fraction(1)
_MID = Fraction(1)
_WORST_PROD = Fraction(1, 4)
_TH_EXACT = Fraction(3, 4)
_TH_FLOOR = Fraction(1, 2)
_ROOM = Fraction(1, 4)
_DECLARED_TH = 0.5
_DECLARED_TH_HI = 2.0


def _honesty(*, stage_c_th: bool) -> dict[str, object]:
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
        stage_c_th=stage_c_th,
        hk_theorem_24_used=False,
    )


def residual_mid_product(sep: Fraction = _SAMPLE_SEP) -> Fraction:
    """``(1-r1)(1-r2) + sep^2/4 = 0`` on ``lambda1 = -2``."""
    return B_minus(_MID, sep) + (sep**2) / 4


def residual_th_three_four() -> Fraction:
    """Worst leading ``T_h = 1 - 1/4 = 3/4`` at ``x = 1``, ``sep = 1``, ``y_1 = 1``."""
    return _Y1 - _WORST_PROD - _TH_EXACT


def residual_th_half_room() -> Fraction:
    """Declared floor ``3/4 - 1/2 = 1/4``."""
    return _TH_EXACT - _TH_FLOOR - _ROOM


def _verdict(residual: Fraction) -> str:
    box = Interval.point(float(residual)) if residual == 0 else Interval.from_rational(residual)
    return adjudicate_residual(box, existential=False).status


def identity_verdicts() -> dict[str, str]:
    return {
        "mid_product": _verdict(residual_mid_product()),
        "th_three_four": _verdict(residual_th_three_four()),
        "th_half_room": _verdict(residual_th_half_room()),
    }


def _contains(outer: Interval, inner: Interval) -> bool:
    return outer.lo <= inner.lo and inner.hi <= outer.hi


def enclose_stage_c_th() -> dict[str, float | bool]:
    """Leading T_h = 1 + (x-r1)(x-r2) on the Stage-B end box at y_1 = 1."""
    walls = enclose_stage_c()
    end = Interval(float(walls["end_lo"]), float(walls["end_hi"]))
    sep = Interval.hull(Fraction(0), Fraction(1))
    one = Interval.from_rational(Fraction(1))
    two = Interval.from_rational(Fraction(2))
    delta = end - one
    prod = delta * delta - (sep / two) * (sep / two)
    th = one + prod / Interval.from_rational(_Y1)
    th_lo = float(th.lo)
    th_hi = float(th.hi)
    return {
        "th_lo": th_lo,
        "th_hi": th_hi,
        "prod_lo": prod.lo,
        "prod_hi": prod.hi,
        "end_lo": end.lo,
        "end_hi": end.hi,
        "y1": float(_Y1),
        "below_declared": th_hi < _DECLARED_TH_HI,
        "th_above_floor": th_lo > _DECLARED_TH,
        "excludes_zero": th_lo > 0.0,
        "finite": th_hi < _DECLARED_TH_HI and th_lo > _DECLARED_TH,
    }


def sample_stage_c_th() -> Interval:
    """Sound leading ``T_h`` at ``x = 1``, ``sep = 3/5``, ``y_1 = 1``."""
    return Interval.from_rational(_Y1 + B_minus(_MID, _SAMPLE_SEP))


@dataclass(frozen=True)
class StageCThReport:
    """Kill-line Stage-C leading T_h. Not an outgoing orbit, first-hit, or G1."""

    identities: Mapping[str, str]
    sample_th: Mapping[str, float]
    enclosure: Mapping[str, float | bool]
    sample_inside: bool
    stage_c_th: bool
    honesty: Mapping[str, object]

    def to_payload(self) -> dict[str, object]:
        return {
            "schema": "hilbert16-stage-c-th-v1",
            "identities": dict(self.identities),
            "sample_th": dict(self.sample_th),
            "enclosure": dict(self.enclosure),
            "sample_inside": self.sample_inside,
            "stage_c_th": self.stage_c_th,
            "honesty": dict(self.honesty),
            "scope": (
                "Exact kill-line Stage-C T_h identities: midpoint "
                "product -sep^2/4, worst leading T_h = 3/4, and room "
                "1/4 above 1/2, plus Interval T_h > 1/2 on the Stage-B "
                "end box at y_1 = 1. Not an outgoing orbit, first-hit, "
                "C2, G1, or Hilbert XVI."
            ),
        }


def report() -> StageCThReport:
    """Replay Stage-C T_h identities and enclose the leading floor."""
    identities = identity_verdicts()
    sample = sample_stage_c_th()
    try:
        enclosure = enclose_stage_c_th()
        inside = bool(enclosure["finite"]) and _contains(
            Interval(float(enclosure["th_lo"]), float(enclosure["th_hi"])),
            sample,
        )
        sealed = (
            all(status == "PROVED" for status in identities.values())
            and bool(enclosure["below_declared"])
            and bool(enclosure["th_above_floor"])
            and bool(enclosure["excludes_zero"])
            and inside
        )
    except (ValueError, ZeroDivisionError, ArithmeticError):
        enclosure = {
            "th_lo": float("nan"),
            "th_hi": float("nan"),
            "below_declared": False,
            "th_above_floor": False,
            "excludes_zero": False,
            "finite": False,
        }
        inside = False
        sealed = False
    return StageCThReport(
        identities=identities,
        sample_th={"lo": sample.lo, "hi": sample.hi},
        enclosure=enclosure,
        sample_inside=inside,
        stage_c_th=sealed,
        honesty=_honesty(stage_c_th=sealed),
    )
