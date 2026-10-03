# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
"""Kill-line Stage-C comparison first-hit of matching-chart E_out.

On lambda1 = -2 the Stage-C start box has V in [-4 eps/3, -eps/4],
so the leading gap rho - 4 eps/3 is 1/4 - 1/12 = 1/6 > 0. Height
corrections still leave E_out > 0 there. Along the sealed T_h > 1/2
continuation, |V_h| = T_h/|V| >= (1/2)/2 = 1/4 on the rectangle
|V| <= 2, while the E_out height-correction derivative is at most
1/64 + 1/256 = 5/256. Interval wrapping keeps dE_out/dh < 0 and the
height to T = rho^2/2 below 1/8 < 1. This is a unique comparison
first-hit of the selected large outgoing physical section E_out from
Stage-C start, not a Lohner orbit, not chart O, C2, dx_e off the kill
line, G1, or Hilbert XVI. The height-section hit of h=1 is
stage_c_hit. Matching-chart Lohner E_out from the GRAZING start is
e_out_section.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from fractions import Fraction

from omnibias.core.collapse.verdict import adjudicate_residual
from omnibias.core.verified.interval import Interval
from omnibias.dynamics.e_out_section import e_out
from omnibias.dynamics.honesty_inventory import build_honesty
from omnibias.dynamics.stage_c_exit import enclose_stage_c_exit
from omnibias.dynamics.stage_c_th import enclose_stage_c_th

__all__ = [
    "StageCSecReport",
    "enclose_stage_c_sec",
    "identity_verdicts",
    "report",
    "residual_corr_sum",
    "residual_rho_start",
    "residual_vh_floor",
    "sample_stage_c_sec",
]

_RHO = Fraction(1, 4)
_EPS_HI = Fraction(1, 16)
_X_HI = Fraction(4, 3)
_V_START = -_EPS_HI * _X_HI
_START_GAP = Fraction(1, 6)
_TH_FLOOR = Fraction(1, 2)
_V_LEFT = Fraction(2)
_VH_FLOOR = Fraction(1, 4)
_NU = _EPS_HI
_C = Fraction(2)
_CORR_LIN = Fraction(1, 64)
_CORR_QUAD = Fraction(1, 256)
_CORR_SUM = Fraction(5, 256)
_TSEC = Fraction(1, 32)
_H1 = Fraction(1, 4096)
_EPS2 = Fraction(1, 256)
_HMAX = Fraction(1)
_DECLARED_HHIT = 0.125
_DECLARED_ROOM = 0.0


def _honesty(*, stage_c_sec: bool) -> dict[str, object]:
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
        stage_c_rect=False,
        stage_c_hit=False,
        stage_c_sec=stage_c_sec,
        outgoing_first_hit=False,
        hk_theorem_24_used=False,
    )


def residual_rho_start() -> Fraction:
    """``1/4 - 1/12 = 1/6``: leading E_out gap at the most-negative start."""
    return _RHO - (-_V_START) - _START_GAP


def residual_vh_floor() -> Fraction:
    """``(1/2) / 2 = 1/4``: min |V_h| on |V| <= 2 with T_h >= 1/2."""
    return _TH_FLOOR / _V_LEFT - _VH_FLOOR


def residual_corr_sum() -> Fraction:
    """``1/64 + 1/256 = 5/256``: E_out height-correction derivative at h=1."""
    return _CORR_LIN + _CORR_QUAD - _CORR_SUM


def _verdict(residual: Fraction) -> str:
    box = Interval.point(float(residual)) if residual == 0 else Interval.from_rational(residual)
    return adjudicate_residual(box, existential=False).status


def identity_verdicts() -> dict[str, str]:
    return {
        "rho_start": _verdict(residual_rho_start()),
        "vh_floor": _verdict(residual_vh_floor()),
        "corr_sum": _verdict(residual_corr_sum()),
    }


def _contains(outer: Interval, inner: Interval) -> bool:
    return outer.lo <= inner.lo and inner.hi <= outer.hi


def enclose_stage_c_sec() -> dict[str, float | bool]:
    """Stage-C comparison first-hit of E_out: start gap > 0, dE/dh < 0, h_hit < 1/8."""
    v_hi = -_EPS_HI * Fraction(1, 4)
    vbox = Interval(float(_V_START), float(v_hi))
    h0 = Interval.from_rational(_H1)
    corr = (
        Interval.from_rational(1)
        + Interval.from_rational(_NU) * h0
        + Interval.from_rational(_C * _NU * _NU) * h0 * h0
    )
    e_start = vbox + Interval.from_rational(_RHO) * corr
    vh = Interval.from_rational(_TH_FLOOR) / Interval.from_rational(_V_LEFT)
    corr_d = Interval.from_rational(_CORR_LIN) + Interval.from_rational(_CORR_QUAD)
    room = vh - corr_d
    exit_box = enclose_stage_c_exit()
    th_box = enclose_stage_c_th()
    te = Interval(float(exit_box["te_lo"]), float(exit_box["te_hi"])) * Interval.from_rational(
        _EPS2
    )
    tsec = Interval.from_rational(_TSEC)
    th_lo = Interval.point(float(th_box["th_lo"]))
    dh = (tsec - Interval.point(float(te.lo))) / th_lo
    hhit = Interval.from_rational(_H1) + dh
    e_lo = float(e_start.lo)
    e_hi = float(e_start.hi)
    room_lo = float(room.lo)
    room_hi = float(room.hi)
    hhit_hi = float(hhit.hi)
    return {
        "e_lo": e_lo,
        "e_hi": e_hi,
        "room_lo": room_lo,
        "room_hi": room_hi,
        "hhit_lo": float(hhit.lo),
        "hhit_hi": hhit_hi,
        "te_lo": float(te.lo),
        "th_lo": float(th_box["th_lo"]),
        "below_declared": (
            e_lo > _DECLARED_ROOM
            and room_lo > _DECLARED_ROOM
            and hhit_hi < _DECLARED_HHIT
        ),
        "start_positive": e_lo > 0.0,
        "transverse": room_lo > 0.0,
        "before_hmax": hhit_hi < float(_HMAX),
        "excludes_zero": e_lo > 0.0 and room_lo > 0.0,
        "finite": (
            e_lo > 0.0
            and room_lo > 0.0
            and hhit_hi < _DECLARED_HHIT
            and hhit_hi < float(_HMAX)
        ),
    }


def sample_stage_c_sec() -> Interval:
    """Sound E_out at the most-negative Stage-C start."""
    return Interval.from_rational(
        e_out(_V_START, _H1, _RHO, _NU, _C)
    )


@dataclass(frozen=True)
class StageCSecReport:
    """Kill-line Stage-C comparison first-hit of E_out. Not Lohner or G1."""

    identities: Mapping[str, str]
    sample_e: Mapping[str, float]
    enclosure: Mapping[str, float | bool]
    sample_inside: bool
    stage_c_sec: bool
    honesty: Mapping[str, object]

    def to_payload(self) -> dict[str, object]:
        return {
            "schema": "hilbert16-stage-c-sec-v1",
            "identities": dict(self.identities),
            "sample_e": dict(self.sample_e),
            "enclosure": dict(self.enclosure),
            "sample_inside": self.sample_inside,
            "stage_c_sec": self.stage_c_sec,
            "honesty": dict(self.honesty),
            "scope": (
                "Exact kill-line Stage-C signed-section identities: "
                "1/4-1/12=1/6, (1/2)/2=1/4, and 1/64+1/256=5/256, plus "
                "Interval E_out > 0 at start, dE_out/dh < 0, and h_hit < 1/8. "
                "Not Lohner, chart O, C2, G1, or Hilbert XVI."
            ),
        }


def report() -> StageCSecReport:
    """Replay Stage-C E_out identities and enclose the comparison first-hit."""
    identities = identity_verdicts()
    sample = sample_stage_c_sec()
    try:
        enclosure = enclose_stage_c_sec()
        inside = bool(enclosure["finite"]) and _contains(
            Interval(float(enclosure["e_lo"]), float(enclosure["e_hi"])),
            sample,
        )
        sealed = (
            all(status == "PROVED" for status in identities.values())
            and bool(enclosure["below_declared"])
            and bool(enclosure["start_positive"])
            and bool(enclosure["transverse"])
            and bool(enclosure["before_hmax"])
            and inside
        )
    except (ValueError, ZeroDivisionError, ArithmeticError):
        enclosure = {
            "e_lo": float("nan"),
            "e_hi": float("nan"),
            "room_lo": float("nan"),
            "room_hi": float("nan"),
            "hhit_hi": float("nan"),
            "below_declared": False,
            "start_positive": False,
            "transverse": False,
            "before_hmax": False,
            "excludes_zero": False,
            "finite": False,
        }
        inside = False
        sealed = False
    return StageCSecReport(
        identities=identities,
        sample_e={"lo": sample.lo, "hi": sample.hi},
        enclosure=enclosure,
        sample_inside=inside,
        stage_c_sec=sealed,
        honesty=_honesty(stage_c_sec=sealed),
    )
