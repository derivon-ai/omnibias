# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
"""Kill-line Stage-C C=2 lower T(h) envelope from T_h >= 1/2.

On lambda1 = -2 the sealed leading floor is T_h > 1/2 at y_1 = 1,
and the worst midpoint product -sep^2/4 only gets weaker as y
grows, so T_h >= 1/2 persists on the Stage-C rectangle. Integrating
the comparison slope 1/2 from h_1 = eps^3 gives

    T(h) >= T_e + (1/2) (h - h_1).

The start remainder T_e/eps^2 - (1/32)(1+eps) is 47/512 at the
written wall and Interval wrapping stays > 1/16. The h-coefficient
1/2 - 1/32 = 15/32 is positive, so the worst height is h = h_1.
Therefore

    T(h) >= (1/32) (eps^2 + h)

on the C=2 comparison, jointly with the sealed upper majorant
T(h) <= 64 (eps^2+h). This is a sealed two-sided Stage-C energy
sandwich, not first-hit, C2, dx_e off the kill line, G1, or
Hilbert XVI. The upper bound is stage_c_int. Leading T_h is
stage_c_th. Exit energy is stage_c_exit.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from fractions import Fraction

from omnibias.core.collapse.verdict import adjudicate_residual
from omnibias.core.verified.interval import Interval
from omnibias.dynamics.honesty_inventory import build_honesty
from omnibias.dynamics.stage_b import r1_kill
from omnibias.dynamics.stage_c_exit import enclose_stage_c_exit
from omnibias.dynamics.stage_c_th import enclose_stage_c_th

__all__ = [
    "StageCLoReport",
    "enclose_stage_c_lo",
    "identity_verdicts",
    "report",
    "residual_half_minus",
    "residual_one_eps",
    "residual_wrap_room",
    "sample_stage_c_lo",
]

_SAMPLE_SEP = Fraction(3, 5)
_HALF = Fraction(1, 2)
_C_LO = Fraction(1, 32)
_HALF_C = Fraction(15, 32)
_EPS_HI = Fraction(1, 16)
_ONE_EPS = Fraction(17, 16)
_TE_LO = Fraction(1, 8)
_WRAP = Fraction(1, 16)
_EDGE = Fraction(17, 512)
_WRAP_ROOM = Fraction(15, 512)
_DECLARED_START = 0.0625
_DECLARED_HALF = 0.25
_DECLARED_START_HI = 2.0


def _honesty(*, stage_c_lo: bool) -> dict[str, object]:
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
        stage_c_lo=stage_c_lo,
        hk_theorem_24_used=False,
    )


def residual_half_minus() -> Fraction:
    """``1/2 - 1/32 = 15/32``: h-coefficient of the lower remainder."""
    return _HALF - _C_LO - _HALF_C


def residual_one_eps() -> Fraction:
    """``1 + 1/16 = 17/16``: compact-edge ``1 + eps`` at ``eps = 1/16``."""
    return 1 + _EPS_HI - _ONE_EPS


def residual_wrap_room() -> Fraction:
    """``1/16 - 17/512 = 15/512``: wrapping ``T_e/eps^2`` start remainder."""
    return _WRAP - _EDGE - _WRAP_ROOM


def _verdict(residual: Fraction) -> str:
    box = Interval.point(float(residual)) if residual == 0 else Interval.from_rational(residual)
    return adjudicate_residual(box, existential=False).status


def identity_verdicts() -> dict[str, str]:
    return {
        "half_minus": _verdict(residual_half_minus()),
        "one_eps": _verdict(residual_one_eps()),
        "wrap_room": _verdict(residual_wrap_room()),
    }


def _contains(outer: Interval, inner: Interval) -> bool:
    return outer.lo <= inner.lo and inner.hi <= outer.hi


def enclose_stage_c_lo() -> dict[str, float | bool]:
    """Stage-C C=2 start remainder T_e/eps^2 - (1/32)(1+eps)."""
    energy = enclose_stage_c_exit()
    floor = enclose_stage_c_th()
    te = Interval(float(energy["te_lo"]), float(energy["te_hi"]))
    c_box = Interval.from_rational(_C_LO)
    one_eps = Interval.from_rational(_ONE_EPS)
    start = te - c_box * one_eps
    half_c = Interval.from_rational(_HALF) - c_box
    start_lo = float(start.lo)
    start_hi = float(start.hi)
    half_lo = float(half_c.lo)
    th_lo = float(floor["th_lo"])
    return {
        "start_lo": start_lo,
        "start_hi": start_hi,
        "half_lo": half_lo,
        "th_lo": th_lo,
        "te_lo": float(energy["te_lo"]),
        "te_hi": float(energy["te_hi"]),
        "below_declared": start_hi < _DECLARED_START_HI,
        "start_above": start_lo > _DECLARED_START,
        "half_above": half_lo > _DECLARED_HALF,
        "th_above": th_lo > float(_HALF),
        "excludes_zero": start_lo > 0.0 and half_lo > 0.0,
        "finite": (
            start_hi < _DECLARED_START_HI
            and start_lo > _DECLARED_START
            and half_lo > _DECLARED_HALF
            and th_lo > float(_HALF)
        ),
    }


def sample_stage_c_lo() -> Interval:
    """Sound start remainder at ``sep = 3/5``, ``eps = 1/16``."""
    kinetic = (r1_kill(_SAMPLE_SEP) ** 2) / 2
    return Interval.from_rational(kinetic - _C_LO * _ONE_EPS)


@dataclass(frozen=True)
class StageCLoReport:
    """Kill-line Stage-C C=2 lower T(h) envelope. Not first-hit or G1."""

    identities: Mapping[str, str]
    sample_start: Mapping[str, float]
    enclosure: Mapping[str, float | bool]
    sample_inside: bool
    stage_c_lo: bool
    honesty: Mapping[str, object]

    def to_payload(self) -> dict[str, object]:
        return {
            "schema": "hilbert16-stage-c-lo-v1",
            "identities": dict(self.identities),
            "sample_start": dict(self.sample_start),
            "enclosure": dict(self.enclosure),
            "sample_inside": self.sample_inside,
            "stage_c_lo": self.stage_c_lo,
            "honesty": dict(self.honesty),
            "scope": (
                "Exact kill-line Stage-C C=2 lower-envelope identities: "
                "1/2 - 1/32 = 15/32, 1 + 1/16 = 17/16, and wrapping "
                "start remainder 15/512, plus Interval "
                "T(h) >= (1/32)(eps^2+h). Not first-hit, C2, G1, or "
                "Hilbert XVI."
            ),
        }


def report() -> StageCLoReport:
    """Replay Stage-C C=2 lower-envelope identities and enclose the floor."""
    identities = identity_verdicts()
    sample = sample_stage_c_lo()
    try:
        enclosure = enclose_stage_c_lo()
        inside = bool(enclosure["finite"]) and _contains(
            Interval(float(enclosure["start_lo"]), float(enclosure["start_hi"])),
            sample,
        )
        sealed = (
            all(status == "PROVED" for status in identities.values())
            and bool(enclosure["below_declared"])
            and bool(enclosure["start_above"])
            and bool(enclosure["half_above"])
            and bool(enclosure["th_above"])
            and bool(enclosure["excludes_zero"])
            and inside
        )
    except (ValueError, ZeroDivisionError, ArithmeticError):
        enclosure = {
            "start_lo": float("nan"),
            "start_hi": float("nan"),
            "half_lo": float("nan"),
            "th_lo": float("nan"),
            "below_declared": False,
            "start_above": False,
            "half_above": False,
            "th_above": False,
            "excludes_zero": False,
            "finite": False,
        }
        inside = False
        sealed = False
    return StageCLoReport(
        identities=identities,
        sample_start={"lo": sample.lo, "hi": sample.hi},
        enclosure=enclosure,
        sample_inside=inside,
        stage_c_lo=sealed,
        honesty=_honesty(stage_c_lo=sealed),
    )
