# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
"""Kill-line Stage-C exit energy T_e = Theta(eps^2).

On lambda1 = -2 matching-chart exit is V = -eps x, so

    T_e = eps^2 x_e^2 / 2,    h_e = eps^3 y0,

with y0 <= mu = 1/16. After Stage B the end box has x >= 1/4, and
x^2/2 at the written wall x = 1/2 is 1/8. Interval wrapping still
gives T_e / eps^2 > 1/16. The product eps y0 is at most 1/256, so
the exit gap T_e - h_e stays positive: the orbit starts above
T = h.

This is a sealed Stage-C exit-energy enclosure on the kill line, not
an outgoing orbit, first-hit, C2, dx_e off the kill line, G1, or
Hilbert XVI. The a_min floor is stage_c.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from fractions import Fraction

from omnibias.core.collapse.verdict import adjudicate_residual
from omnibias.core.verified.interval import Interval
from omnibias.dynamics.honesty_inventory import build_honesty
from omnibias.dynamics.stage_b import r1_kill
from omnibias.dynamics.stage_c import enclose_stage_c

__all__ = [
    "StageCExitReport",
    "enclose_stage_c_exit",
    "identity_verdicts",
    "report",
    "residual_eps_y0",
    "residual_gap_room",
    "residual_te_half",
    "sample_stage_c_exit",
]

_SAMPLE_SEP = Fraction(3, 5)
_X_LO = Fraction(1, 2)
_TE_LO = Fraction(1, 8)
_MU = Fraction(1, 16)
_EPS_HI = Fraction(1, 16)
_EPS_Y0 = Fraction(1, 256)
_GAP_EXACT = Fraction(31, 256)
_DECLARED_TE_LO = 0.0625
_DECLARED_TE_HI = 1.0


def _honesty(*, stage_c_exit: bool) -> dict[str, object]:
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
        stage_c_exit=stage_c_exit,
        hk_theorem_24_used=False,
    )


def residual_te_half() -> Fraction:
    """``(1/2)^2 / 2 = 1/8``: exact ``T_e / eps^2`` at the written wall."""
    return (_X_LO**2) / 2 - _TE_LO


def residual_eps_y0() -> Fraction:
    """``eps y0 <= (1/16)*(1/16) = 1/256`` on the Stage-B compact."""
    return _EPS_HI * _MU - _EPS_Y0


def residual_gap_room() -> Fraction:
    """``1/8 - 1/256 = 31/256``: room of ``T_e/eps^2`` over ``eps y0``."""
    return _TE_LO - _EPS_Y0 - _GAP_EXACT


def _verdict(residual: Fraction) -> str:
    box = Interval.point(float(residual)) if residual == 0 else Interval.from_rational(residual)
    return adjudicate_residual(box, existential=False).status


def identity_verdicts() -> dict[str, str]:
    return {
        "te_half": _verdict(residual_te_half()),
        "eps_y0": _verdict(residual_eps_y0()),
        "gap_room": _verdict(residual_gap_room()),
    }


def _contains(outer: Interval, inner: Interval) -> bool:
    return outer.lo <= inner.lo and inner.hi <= outer.hi


def enclose_stage_c_exit() -> dict[str, float | bool]:
    """Kill-line Stage-C T_e / eps^2 box and T_e > h_e gap."""
    walls = enclose_stage_c()
    end = Interval(float(walls["end_lo"]), float(walls["end_hi"]))
    two = Interval.from_rational(Fraction(2))
    te = (end * end) / two
    eps_y0 = Interval.from_rational(_EPS_Y0)
    gap = te - eps_y0
    te_lo = float(te.lo)
    te_hi = float(te.hi)
    gap_lo = float(gap.lo)
    return {
        "te_lo": te_lo,
        "te_hi": te_hi,
        "gap_lo": gap_lo,
        "gap_hi": gap.hi,
        "end_lo": end.lo,
        "end_hi": end.hi,
        "eps_y0": float(_EPS_Y0),
        "below_declared": te_hi < _DECLARED_TE_HI,
        "te_above_floor": te_lo > _DECLARED_TE_LO,
        "gap_positive": gap_lo > 0.0,
        "excludes_zero": te_lo > 0.0 and gap_lo > 0.0,
        "finite": te_hi < _DECLARED_TE_HI and te_lo > _DECLARED_TE_LO and gap_lo > 0.0,
    }


def sample_stage_c_exit() -> Interval:
    """Sound ``T_e / eps^2 = r1^2 / 2`` at ``sep = 3/5``."""
    return Interval.from_rational((r1_kill(_SAMPLE_SEP) ** 2) / 2)


@dataclass(frozen=True)
class StageCExitReport:
    """Kill-line Stage-C exit energy. Not an outgoing orbit, first-hit, or G1."""

    identities: Mapping[str, str]
    sample_te: Mapping[str, float]
    enclosure: Mapping[str, float | bool]
    sample_inside: bool
    stage_c_exit: bool
    honesty: Mapping[str, object]

    def to_payload(self) -> dict[str, object]:
        return {
            "schema": "hilbert16-stage-c-exit-v1",
            "identities": dict(self.identities),
            "sample_te": dict(self.sample_te),
            "enclosure": dict(self.enclosure),
            "sample_inside": self.sample_inside,
            "stage_c_exit": self.stage_c_exit,
            "honesty": dict(self.honesty),
            "scope": (
                "Exact kill-line Stage-C exit identities: T_e/eps^2 = 1/8 "
                "at x = 1/2, eps y0 = 1/256, and gap room 31/256, plus "
                "Interval T_e/eps^2 in (1/16, 1) and T_e > h_e on the "
                "Stage-B end box. Not an outgoing orbit, first-hit, C2, "
                "G1, or Hilbert XVI."
            ),
        }


def report() -> StageCExitReport:
    """Replay Stage-C exit identities and enclose T_e / eps^2."""
    identities = identity_verdicts()
    sample = sample_stage_c_exit()
    try:
        enclosure = enclose_stage_c_exit()
        inside = bool(enclosure["finite"]) and _contains(
            Interval(float(enclosure["te_lo"]), float(enclosure["te_hi"])),
            sample,
        )
        sealed = (
            all(status == "PROVED" for status in identities.values())
            and bool(enclosure["below_declared"])
            and bool(enclosure["te_above_floor"])
            and bool(enclosure["gap_positive"])
            and bool(enclosure["excludes_zero"])
            and inside
        )
    except (ValueError, ZeroDivisionError, ArithmeticError):
        enclosure = {
            "te_lo": float("nan"),
            "te_hi": float("nan"),
            "gap_lo": float("nan"),
            "below_declared": False,
            "te_above_floor": False,
            "gap_positive": False,
            "excludes_zero": False,
            "finite": False,
        }
        inside = False
        sealed = False
    return StageCExitReport(
        identities=identities,
        sample_te={"lo": sample.lo, "hi": sample.hi},
        enclosure=enclosure,
        sample_inside=inside,
        stage_c_exit=sealed,
        honesty=_honesty(stage_c_exit=sealed),
    )
