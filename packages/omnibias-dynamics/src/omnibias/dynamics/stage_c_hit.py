# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
"""Kill-line Stage-C comparison first-hit of the height section h=1.

On lambda1 = -2 the sealed rectangle has |V| >= 1/64, so
hdot = -V h >= (1/64) h. Height is therefore strictly increasing.
From h_1 = eps^3 = 1/4096 < 1 the time to hmax = 1 is at most
64 * 3 * log(1/eps) = 192 ln(16) at the compact edge, and Interval
wrapping stays < 1024. This is a unique comparison first-hit of
{h=1}, not a Lohner orbit, not the physical signed-label section,
not chart O, C2, dx_e off the kill line, G1, or Hilbert XVI. The
continuation rectangle is stage_c_rect.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from fractions import Fraction

from omnibias.core.collapse.verdict import adjudicate_residual
from omnibias.core.verified.interval import Interval
from omnibias.core.verified.transcend import ln_iv
from omnibias.dynamics.honesty_inventory import build_honesty
from omnibias.dynamics.stage_c_rect import enclose_stage_c_rect

__all__ = [
    "StageCHitReport",
    "enclose_stage_c_hit",
    "identity_verdicts",
    "report",
    "residual_hmax_gap",
    "residual_inv_floor",
    "residual_time_pre",
    "sample_stage_c_hit",
]

_FLOOR = Fraction(1, 64)
_INV = Fraction(64)
_CUBE = Fraction(3)
_PRE = Fraction(192)
_HMAX = Fraction(1)
_H1 = Fraction(1, 4096)
_GAP = Fraction(4095, 4096)
_DECLARED_TIME = 1024.0
_DECLARED_PRE = 256.0


def _honesty(*, stage_c_hit: bool) -> dict[str, object]:
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
        stage_c_hit=stage_c_hit,
        outgoing_first_hit=False,
        hk_theorem_24_used=False,
    )


def residual_inv_floor() -> Fraction:
    """``1 / (1/64) = 64``: reciprocal of the ḣ/h floor."""
    return Fraction(1) / _FLOOR - _INV


def residual_time_pre() -> Fraction:
    """``64 * 3 = 192``: time prefactor times log(1/h_1) = 3 log(1/eps)."""
    return _INV * _CUBE - _PRE


def residual_hmax_gap() -> Fraction:
    """``1 - 1/4096 = 4095/4096``: hmax sits strictly above h_1."""
    return _HMAX - _H1 - _GAP


def _verdict(residual: Fraction) -> str:
    box = Interval.point(float(residual)) if residual == 0 else Interval.from_rational(residual)
    return adjudicate_residual(box, existential=False).status


def identity_verdicts() -> dict[str, str]:
    return {
        "inv_floor": _verdict(residual_inv_floor()),
        "time_pre": _verdict(residual_time_pre()),
        "hmax_gap": _verdict(residual_hmax_gap()),
    }


def _contains(outer: Interval, inner: Interval) -> bool:
    return outer.lo <= inner.lo and inner.hi <= outer.hi


def enclose_stage_c_hit() -> dict[str, float | bool]:
    """Stage-C comparison first-hit of h=1: time < 1024, h_1 < 1."""
    rect = enclose_stage_c_rect()
    floor = Interval(float(rect["v_right_lo"]), float(rect["v_right_hi"]))
    inv = Interval.from_rational(1) / floor
    pre = inv * Interval.from_rational(_CUBE)
    ln = ln_iv(Interval.from_rational(Fraction(16)))
    time = pre * ln
    pre_lo = float(pre.lo)
    pre_hi = float(pre.hi)
    time_lo = float(time.lo)
    time_hi = float(time.hi)
    return {
        "pre_lo": pre_lo,
        "pre_hi": pre_hi,
        "time_lo": time_lo,
        "time_hi": time_hi,
        "floor_lo": float(floor.lo),
        "h1": float(_H1),
        "below_declared": time_hi < _DECLARED_TIME and pre_hi < _DECLARED_PRE,
        "h1_below": float(_H1) < float(_HMAX),
        "excludes_zero": time_lo > 0.0 and float(floor.lo) > 0.0,
        "finite": (
            time_hi < _DECLARED_TIME
            and pre_hi < _DECLARED_PRE
            and float(_H1) < float(_HMAX)
        ),
    }


def sample_stage_c_hit() -> Interval:
    """Sound time prefactor ``192`` at ``|V| >= 1/64``."""
    return Interval.from_rational(_PRE)


@dataclass(frozen=True)
class StageCHitReport:
    """Kill-line Stage-C comparison first-hit of h=1. Not Lohner or G1."""

    identities: Mapping[str, str]
    sample_pre: Mapping[str, float]
    enclosure: Mapping[str, float | bool]
    sample_inside: bool
    stage_c_hit: bool
    honesty: Mapping[str, object]

    def to_payload(self) -> dict[str, object]:
        return {
            "schema": "hilbert16-stage-c-hit-v1",
            "identities": dict(self.identities),
            "sample_pre": dict(self.sample_pre),
            "enclosure": dict(self.enclosure),
            "sample_inside": self.sample_inside,
            "stage_c_hit": self.stage_c_hit,
            "honesty": dict(self.honesty),
            "scope": (
                "Exact kill-line Stage-C height-section identities: "
                "1/(1/64)=64, 64*3=192, and 1-1/4096=4095/4096, plus "
                "Interval time < 1024 to h=1. Not Lohner, signed-label "
                "section, chart O, C2, G1, or Hilbert XVI."
            ),
        }


def report() -> StageCHitReport:
    """Replay Stage-C height-section identities and enclose the hitting time."""
    identities = identity_verdicts()
    sample = sample_stage_c_hit()
    try:
        enclosure = enclose_stage_c_hit()
        inside = bool(enclosure["finite"]) and _contains(
            Interval(float(enclosure["pre_lo"]), float(enclosure["pre_hi"])),
            sample,
        )
        sealed = (
            all(status == "PROVED" for status in identities.values())
            and bool(enclosure["below_declared"])
            and bool(enclosure["h1_below"])
            and bool(enclosure["excludes_zero"])
            and inside
        )
    except (ValueError, ZeroDivisionError, ArithmeticError):
        enclosure = {
            "pre_lo": float("nan"),
            "pre_hi": float("nan"),
            "time_lo": float("nan"),
            "time_hi": float("nan"),
            "below_declared": False,
            "h1_below": False,
            "excludes_zero": False,
            "finite": False,
        }
        inside = False
        sealed = False
    return StageCHitReport(
        identities=identities,
        sample_pre={"lo": sample.lo, "hi": sample.hi},
        enclosure=enclosure,
        sample_inside=inside,
        stage_c_hit=sealed,
        honesty=_honesty(stage_c_hit=sealed),
    )
