# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
"""Kill-line Stage-C start gap T_e > h_1 at y_1 = 1.

On lambda1 = -2 matching-chart Stage C starts at the sep-independent
height y_1 = 1 after Stage B, so h_1 = eps^3. Then

    (T_e - h_1) / eps^2 = x_e^2 / 2 - eps.

At the written wall x = 1/2 this is 1/8 - eps, and on eps in [0, 1/16]
the exact edge value is 1/4096. Interval wrapping on the Stage-B end
box still yields a positive gap: the orbit starts above T = h at the
actual Stage C height, not the Stage A y0 used by stage_c_exit.

This is a sealed Stage-C start-gap enclosure on the kill line, not an
outgoing orbit, first-hit, C2, dx_e off the kill line, G1, or
Hilbert XVI. Leading T_h is stage_c_th. Exit energy is stage_c_exit.
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

__all__ = [
    "StageCGapReport",
    "enclose_stage_c_gap",
    "identity_verdicts",
    "report",
    "residual_h1_cube",
    "residual_start_gap",
    "residual_wall_gap",
    "sample_stage_c_gap",
]

_SAMPLE_SEP = Fraction(3, 5)
_TE_LO = Fraction(1, 8)
_EPS_HI = Fraction(1, 16)
_ROOM = Fraction(1, 16)
_H1_CUBE = Fraction(1, 4096)
_DECLARED_GAP = 0.03125
_DECLARED_GAP_HI = 1.0


def _honesty(*, stage_c_gap: bool) -> dict[str, object]:
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
        stage_c_gap=stage_c_gap,
        hk_theorem_24_used=False,
    )


def residual_start_gap() -> Fraction:
    """``1/8 - 1/16 = 1/16``: wall ``T_e/eps^2`` minus ``eps`` at the compact edge."""
    return _TE_LO - _EPS_HI - _ROOM


def residual_h1_cube() -> Fraction:
    """``h_1 / eps^2 = eps`` at ``eps = 1/16`` is ``(1/16)^3`` in raw ``T-h``."""
    return _EPS_HI**3 - _H1_CUBE


def residual_wall_gap() -> Fraction:
    """Exact wall gap ``(1/8) eps^2 - eps^3 = 1/4096`` at ``eps = 1/16``."""
    return _TE_LO * (_EPS_HI**2) - _EPS_HI**3 - _H1_CUBE


def _verdict(residual: Fraction) -> str:
    box = Interval.point(float(residual)) if residual == 0 else Interval.from_rational(residual)
    return adjudicate_residual(box, existential=False).status


def identity_verdicts() -> dict[str, str]:
    return {
        "start_gap": _verdict(residual_start_gap()),
        "h1_cube": _verdict(residual_h1_cube()),
        "wall_gap": _verdict(residual_wall_gap()),
    }


def _contains(outer: Interval, inner: Interval) -> bool:
    return outer.lo <= inner.lo and inner.hi <= outer.hi


def enclose_stage_c_gap() -> dict[str, float | bool]:
    """Stage-C start gap (T_e - h_1)/eps^2 at y_1 = 1."""
    energy = enclose_stage_c_exit()
    te = Interval(float(energy["te_lo"]), float(energy["te_hi"]))
    eps = Interval.hull(Fraction(0), _EPS_HI)
    gap = te - eps
    gap_lo = float(gap.lo)
    gap_hi = float(gap.hi)
    return {
        "gap_lo": gap_lo,
        "gap_hi": gap_hi,
        "te_lo": te.lo,
        "te_hi": te.hi,
        "eps_hi": float(_EPS_HI),
        "below_declared": gap_hi < _DECLARED_GAP_HI,
        "gap_above_floor": gap_lo > _DECLARED_GAP,
        "excludes_zero": gap_lo > 0.0,
        "finite": gap_hi < _DECLARED_GAP_HI and gap_lo > _DECLARED_GAP,
    }


def sample_stage_c_gap() -> Interval:
    """Sound ``r1^2/2 - 1/16`` at ``sep = 3/5``, ``eps = 1/16``."""
    return Interval.from_rational((r1_kill(_SAMPLE_SEP) ** 2) / 2 - _EPS_HI)


@dataclass(frozen=True)
class StageCGapReport:
    """Kill-line Stage-C start gap. Not an outgoing orbit, first-hit, or G1."""

    identities: Mapping[str, str]
    sample_gap: Mapping[str, float]
    enclosure: Mapping[str, float | bool]
    sample_inside: bool
    stage_c_gap: bool
    honesty: Mapping[str, object]

    def to_payload(self) -> dict[str, object]:
        return {
            "schema": "hilbert16-stage-c-gap-v1",
            "identities": dict(self.identities),
            "sample_gap": dict(self.sample_gap),
            "enclosure": dict(self.enclosure),
            "sample_inside": self.sample_inside,
            "stage_c_gap": self.stage_c_gap,
            "honesty": dict(self.honesty),
            "scope": (
                "Exact kill-line Stage-C start-gap identities: wall "
                "1/8 - 1/16 = 1/16, h_1 cube 1/4096, and exact wall "
                "T-h = 1/4096 at eps = 1/16, plus Interval "
                "(T_e - h_1)/eps^2 > 1/32 at y_1 = 1. Not an outgoing "
                "orbit, first-hit, C2, G1, or Hilbert XVI."
            ),
        }


def report() -> StageCGapReport:
    """Replay Stage-C start-gap identities and enclose (T_e - h_1)/eps^2."""
    identities = identity_verdicts()
    sample = sample_stage_c_gap()
    try:
        enclosure = enclose_stage_c_gap()
        inside = bool(enclosure["finite"]) and _contains(
            Interval(float(enclosure["gap_lo"]), float(enclosure["gap_hi"])),
            sample,
        )
        sealed = (
            all(status == "PROVED" for status in identities.values())
            and bool(enclosure["below_declared"])
            and bool(enclosure["gap_above_floor"])
            and bool(enclosure["excludes_zero"])
            and inside
        )
    except (ValueError, ZeroDivisionError, ArithmeticError):
        enclosure = {
            "gap_lo": float("nan"),
            "gap_hi": float("nan"),
            "below_declared": False,
            "gap_above_floor": False,
            "excludes_zero": False,
            "finite": False,
        }
        inside = False
        sealed = False
    return StageCGapReport(
        identities=identities,
        sample_gap={"lo": sample.lo, "hi": sample.hi},
        enclosure=enclosure,
        sample_inside=inside,
        stage_c_gap=sealed,
        honesty=_honesty(stage_c_gap=sealed),
    )
