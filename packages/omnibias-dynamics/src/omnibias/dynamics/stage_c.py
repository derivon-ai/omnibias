# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
"""Kill-line Stage-C a_min floor after Stage B.

On lambda1 = -2 the written Stage-C rectangle uses a_min ~ rstar/2 = 1/2.
After Stage-B Picard the end box stays inside the guess [1/4, 4/3], so a
declared floor a_min = 1/4 holds after Interval wrapping. The integrating
factor 1/x is then at most 4 on the guess. Matching-chart |V| = eps x
therefore stays at least eps/4 once x has left Stage B.

This is a sealed Stage-C geometric floor on the kill line, not an
outgoing orbit, first-hit, C2, dx_e off the kill line, G1, or
Hilbert XVI. Height inflation is stage_b.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from fractions import Fraction

from omnibias.core.collapse.verdict import adjudicate_residual
from omnibias.core.verified.interval import Interval
from omnibias.dynamics.honesty_inventory import build_honesty
from omnibias.dynamics.stage_b import enclose_stage_b, r1_kill

__all__ = [
    "StageCReport",
    "enclose_stage_c",
    "identity_verdicts",
    "report",
    "residual_amin_floor",
    "residual_amin_written",
    "residual_inv_guess",
    "sample_stage_c",
]

_SAMPLE_SEP = Fraction(3, 5)
_RSTAR = Fraction(1)
_AMIN_WRITTEN = Fraction(1, 2)
_AMIN_FLOOR = Fraction(1, 4)
_INV_GUESS = Fraction(4)
_DECLARED_AMIN = 0.25
_DECLARED_INV = 8.0


def _honesty(*, stage_c_amin: bool) -> dict[str, object]:
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
        stage_c_amin=stage_c_amin,
        hk_theorem_24_used=False,
    )


def residual_amin_written() -> Fraction:
    """Written Stage-C floor ``rstar/2 = 1/2`` on ``lambda1 = -2``."""
    return _RSTAR / 2 - _AMIN_WRITTEN


def residual_inv_guess() -> Fraction:
    """Integrating factor ``1/(1/4) = 4`` on the Stage-B Picard guess."""
    return 1 / _AMIN_FLOOR - _INV_GUESS


def residual_amin_floor() -> Fraction:
    """Declared floor ``(1/2)/2 = 1/4`` after Interval wrapping."""
    return _AMIN_WRITTEN / 2 - _AMIN_FLOOR


def _verdict(residual: Fraction) -> str:
    box = Interval.point(float(residual)) if residual == 0 else Interval.from_rational(residual)
    return adjudicate_residual(box, existential=False).status


def identity_verdicts() -> dict[str, str]:
    return {
        "amin_written": _verdict(residual_amin_written()),
        "inv_guess": _verdict(residual_inv_guess()),
        "amin_floor": _verdict(residual_amin_floor()),
    }


def _contains(outer: Interval, inner: Interval) -> bool:
    return outer.lo <= inner.lo and inner.hi <= outer.hi


def enclose_stage_c() -> dict[str, float | bool]:
    """Stage-C a_min floor and integrating-factor bound on the Stage-B end box."""
    stage_b = enclose_stage_b()
    end = Interval(float(stage_b["end_lo"]), float(stage_b["end_hi"]))
    if end.lo <= 0.0:
        raise ValueError("enclose_stage_c requires a positive Stage-B end box")
    inv = Interval.from_rational(Fraction(1)) / end
    end_lo = float(end.lo)
    inv_hi = float(inv.hi)
    return {
        "end_lo": end_lo,
        "end_hi": end.hi,
        "inv_lo": inv.lo,
        "inv_hi": inv_hi,
        "start_lo": float(stage_b["start_lo"]),
        "guess_lo": float(stage_b["guess_lo"]),
        "below_declared": inv_hi < _DECLARED_INV,
        "amin_above_floor": end_lo > _DECLARED_AMIN,
        "excludes_zero": end_lo > 0.0 and inv.lo > 0.0,
        "finite": inv_hi < _DECLARED_INV and end_lo > _DECLARED_AMIN,
        "picard_included": bool(stage_b["picard_included"]),
    }


def sample_stage_c() -> Interval:
    """Sound integrating factor ``1/r1`` at ``sep = 3/5``."""
    return Interval.from_rational(1 / r1_kill(_SAMPLE_SEP))


@dataclass(frozen=True)
class StageCReport:
    """Kill-line Stage-C a_min. Not an outgoing orbit, first-hit, or G1."""

    identities: Mapping[str, str]
    sample_inv: Mapping[str, float]
    enclosure: Mapping[str, float | bool]
    sample_inside: bool
    stage_c_amin: bool
    honesty: Mapping[str, object]

    def to_payload(self) -> dict[str, object]:
        return {
            "schema": "hilbert16-stage-c-v1",
            "identities": dict(self.identities),
            "sample_inv": dict(self.sample_inv),
            "enclosure": dict(self.enclosure),
            "sample_inside": self.sample_inside,
            "stage_c_amin": self.stage_c_amin,
            "honesty": dict(self.honesty),
            "scope": (
                "Exact kill-line Stage-C identities: written a_min = 1/2, "
                "integrating factor 1/(1/4) = 4, and declared floor 1/4, "
                "plus Interval end_lo > 1/4 and 1/x < 8 on the Stage-B "
                "end box. Not an outgoing orbit, first-hit, C2, G1, or "
                "Hilbert XVI."
            ),
        }


def report() -> StageCReport:
    """Replay Stage-C identities and enclose the a_min floor."""
    identities = identity_verdicts()
    sample = sample_stage_c()
    try:
        enclosure = enclose_stage_c()
        inside = bool(enclosure["finite"]) and _contains(
            Interval(float(enclosure["inv_lo"]), float(enclosure["inv_hi"])),
            sample,
        )
        sealed = (
            all(status == "PROVED" for status in identities.values())
            and bool(enclosure["below_declared"])
            and bool(enclosure["amin_above_floor"])
            and bool(enclosure["excludes_zero"])
            and bool(enclosure["picard_included"])
            and inside
        )
    except (ValueError, ZeroDivisionError, ArithmeticError):
        enclosure = {
            "end_lo": float("nan"),
            "inv_hi": float("nan"),
            "below_declared": False,
            "amin_above_floor": False,
            "excludes_zero": False,
            "finite": False,
            "picard_included": False,
        }
        inside = False
        sealed = False
    return StageCReport(
        identities=identities,
        sample_inv={"lo": sample.lo, "hi": sample.hi},
        enclosure=enclosure,
        sample_inside=inside,
        stage_c_amin=sealed,
        honesty=_honesty(stage_c_amin=sealed),
    )
