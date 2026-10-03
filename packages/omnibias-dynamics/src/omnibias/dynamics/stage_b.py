# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
"""Stage-B height-inflation Picard enclosure on the kill line.

On lambda1 = -2 the slow-line midpoint is rstar = 1 and
r1 = 1 - sep/2. For sep in (0, 1] the incoming wall lies in
[1/2, 1]. After Stage A, height inflation has dx/dy = eps k / x
with k = 1, independently of height. Interval Picard on

    eps in [0, 1/16],  x-guess [1/4, 4/3]

includes the image of the start box [1/2, 1] and encloses
|Delta x| < 1/3. The tracked exponent 2 - 2 alpha with alpha = 2 eps
stays positive on this compact, so the sep-power of the tracked
product is at most 1 for sep in (0, 1].

This is a sealed Stage-B displacement on the kill line, not Stage A/C,
not a C2 remainder, not first-hit, G1, or Hilbert XVI. The algebraic
height-inflation identity is entry_exit_leading.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from fractions import Fraction

from omnibias.core.collapse.verdict import adjudicate_residual
from omnibias.core.verified.interval import Interval
from omnibias.dynamics.honesty_inventory import build_honesty

__all__ = [
    "StageBReport",
    "enclose_stage_b",
    "identity_verdicts",
    "r1_kill",
    "report",
    "residual_alpha_pos",
    "residual_dx_declared",
    "residual_r1_half",
    "sample_stage_b",
]

_SAMPLE_SEP = Fraction(3, 5)
_EPS_HI = Fraction(1, 16)
_RSTAR = Fraction(1)
_DECLARED_DX = Fraction(1, 3)
_GUESS_LO = Fraction(1, 4)
_GUESS_HI = Fraction(4, 3)
_START_LO = Fraction(1, 2)
_START_HI = Fraction(1)
_ALPHA_EDGE = Fraction(7, 4)
_DECLARED_MAG = 1.0 / 3.0


def _honesty(*, orbit_continuation_on_kill: bool) -> dict[str, object]:
    return build_honesty(
        g1_passed=False,
        g4_passed=False,
        full_graphic_cyclicity_proved=False,
        full_hilbert16_solved=False,
        physical_return_membership_proved=False,
        fold_leading_c2_remainder=False,
        z_x_bound=False,
        orbit_continuation_on_kill=orbit_continuation_on_kill,
        hk_theorem_24_used=False,
    )


def r1_kill(sep: Fraction) -> Fraction:
    """Incoming root on ``lambda1 = -2``: ``r1 = 1 - sep/2``."""
    return _RSTAR - sep / 2


def residual_r1_half(sep: Fraction) -> Fraction:
    """``2 r1 + sep - 2``; the kill-line sum ``r1 + r2 = 2`` with ``sep = r2 - r1``."""
    return 2 * r1_kill(sep) + sep - 2


def residual_dx_declared() -> Fraction:
    """Naive majorant ``(1/16) / (1/4) = 1/4`` before the Picard pad."""
    return _EPS_HI / _GUESS_LO - Fraction(1, 4)


def residual_alpha_pos() -> Fraction:
    """``2(1 - 2 eps) = 7/4`` at ``eps = 1/16``; tracked exponent stays positive."""
    return 2 * (1 - 2 * _EPS_HI) - _ALPHA_EDGE


def _verdict(residual: Fraction) -> str:
    box = Interval.point(float(residual)) if residual == 0 else Interval.from_rational(residual)
    return adjudicate_residual(box, existential=False).status


def identity_verdicts() -> dict[str, str]:
    return {
        "r1_half": _verdict(residual_r1_half(_SAMPLE_SEP)),
        "dx_declared": _verdict(residual_dx_declared()),
        "alpha_pos": _verdict(residual_alpha_pos()),
    }


def _contains(outer: Interval, inner: Interval) -> bool:
    return outer.lo <= inner.lo and inner.hi <= outer.hi


def enclose_stage_b() -> dict[str, float | bool]:
    """Picard inclusion of kill-line Stage-B height inflation."""
    eps = Interval.hull(Fraction(0), _EPS_HI)
    guess = Interval.hull(_GUESS_LO, _GUESS_HI)
    start = Interval.hull(_START_LO, _START_HI)
    dx = eps / guess
    end = start + dx
    included = _contains(guess, start) and _contains(guess, end)
    mag = max(abs(dx.lo), abs(dx.hi)) if included else float("inf")
    alpha = Interval.from_rational(Fraction(2)) * eps
    pow_sep = Interval.from_rational(Fraction(2)) - Interval.from_rational(Fraction(2)) * alpha
    return {
        "picard_included": included,
        "dx_lo": dx.lo,
        "dx_hi": dx.hi,
        "dx_mag": mag,
        "end_lo": end.lo,
        "end_hi": end.hi,
        "guess_lo": guess.lo,
        "guess_hi": guess.hi,
        "start_lo": start.lo,
        "start_hi": start.hi,
        "pow_sep_lo": pow_sep.lo,
        "pow_sep_hi": pow_sep.hi,
        "alpha_positive": pow_sep.lo > 0.0,
        "below_declared": bool(included) and mag < _DECLARED_MAG,
        "finite": bool(included) and mag < float("inf"),
        "eps_hi": float(_EPS_HI),
    }


def sample_stage_b() -> Interval:
    """Sound ``Delta x`` at ``sep = 3/5``, ``eps = 1/16``, ``x = r1``."""
    sep = Interval.from_rational(_SAMPLE_SEP)
    r1 = Interval.from_rational(_RSTAR) - sep / Interval.from_rational(Fraction(2))
    eps = Interval.from_rational(_EPS_HI)
    dy = Interval.from_rational(Fraction(1)) - sep * sep
    return eps / r1 * dy


@dataclass(frozen=True)
class StageBReport:
    """Kill-line Stage-B Picard enclosure. Not Stage A/C, C2, first-hit, or G1."""

    identities: Mapping[str, str]
    sample_dx: Mapping[str, float]
    enclosure: Mapping[str, float | bool]
    sample_inside: bool
    stage_b_dx: bool
    orbit_continuation_on_kill: bool
    honesty: Mapping[str, object]

    def to_payload(self) -> dict[str, object]:
        return {
            "schema": "hilbert16-stage-b-v1",
            "identities": dict(self.identities),
            "sample_dx": dict(self.sample_dx),
            "enclosure": dict(self.enclosure),
            "sample_inside": self.sample_inside,
            "stage_b_dx": self.stage_b_dx,
            "orbit_continuation_on_kill": self.orbit_continuation_on_kill,
            "honesty": dict(self.honesty),
            "scope": (
                "Exact kill-line r1 = 1 - sep/2 identities, Picard "
                "inclusion of Stage-B height inflation with |Delta x|<1/3 "
                "on lambda1=-2, sep in (0, 1], eps in [0, 1/16], and a "
                "positive tracked exponent 2-2 alpha. Not Stage A/C, not "
                "C2, not first-hit, G1, or Hilbert XVI."
            ),
        }


def report() -> StageBReport:
    """Replay Stage-B identities and enclose |Delta x|<1/3 on the kill line."""
    identities = identity_verdicts()
    sample = sample_stage_b()
    try:
        enclosure = enclose_stage_b()
        inside = bool(enclosure["finite"]) and _contains(
            Interval(float(enclosure["dx_lo"]), float(enclosure["dx_hi"])),
            sample,
        )
        sealed = (
            all(status == "PROVED" for status in identities.values())
            and bool(enclosure["picard_included"])
            and bool(enclosure["below_declared"])
            and bool(enclosure["alpha_positive"])
            and inside
        )
    except (ValueError, ZeroDivisionError):
        enclosure = {
            "picard_included": False,
            "dx_lo": float("nan"),
            "dx_hi": float("nan"),
            "dx_mag": float("inf"),
            "alpha_positive": False,
            "below_declared": False,
            "finite": False,
        }
        inside = False
        sealed = False
    return StageBReport(
        identities=identities,
        sample_dx={"lo": sample.lo, "hi": sample.hi},
        enclosure=enclosure,
        sample_inside=inside,
        stage_b_dx=sealed,
        orbit_continuation_on_kill=sealed,
        honesty=_honesty(orbit_continuation_on_kill=sealed),
    )
