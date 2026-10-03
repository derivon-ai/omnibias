# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
"""Slow-line I-map from the shrinking-root interface to a compact x-section.

The inner rescaling ``x = r1 xi`` attracts to ``xi = 1``. Outgoing continuation
to a physical section at fixed ``x_* in (0, r2)`` is a different chart: the
cleared antiderivative

    J(x) = -r1 log|x-r1| + r2 log|x-r2|

has derivative ``(r2-r1) x / ((x-r1)(x-r2))``. From the matching interface
``x = r1 (1+theta)`` the potentially exploding term is ``r1 log r1``, which
is majorized by ``2 sqrt(r1) - 2 r1 -> 0`` on ``(0, 1]``. The slow time to
a compact ``x_*`` therefore stays bounded as ``r1 -> 0``.

This is not first-hit of the large height section, a uniform ``a_min``,
G1, or Hilbert XVI. The first-root wall ``a = r1 - d`` is negative once
``r1 < d``.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from fractions import Fraction

from omnibias.core.collapse.verdict import adjudicate_residual
from omnibias.core.verified.interval import Interval
from omnibias.dynamics.honesty_inventory import build_honesty

__all__ = [
    "OutgoingCorridorReport",
    "a_wall_gap",
    "identity_verdicts",
    "r1_log_majorant",
    "report",
    "residual_J_numerator",
    "residual_interface_embed",
    "residual_root_sum",
]

_SAMPLE_R1 = Fraction(1, 5)
_SAMPLE_R2 = Fraction(4)
_SAMPLE_LAM1 = -(_SAMPLE_R1 + _SAMPLE_R2)
_SAMPLE_THETA = Fraction(1, 2)
_SAMPLE_X = Fraction(1)
_SAMPLE_EPS = Fraction(1, 7)
_SAMPLE_D = Fraction(1, 4)


def _honesty(*, outgoing_x_corridor_bounded: bool) -> dict[str, object]:
    return build_honesty(
        g1_passed=False,
        g4_passed=False,
        full_graphic_cyclicity_proved=False,
        full_hilbert16_solved=False,
        physical_return_membership_proved=False,
        fold_leading_c2_remainder=False,
        outgoing_x_corridor_bounded=outgoing_x_corridor_bounded,
        outgoing_first_hit=False,
        orbit_continuation_on_kill=False,
        hk_theorem_24_used=False,
    )


def residual_J_numerator(x: Fraction, r1: Fraction, r2: Fraction) -> Fraction:
    """Cleared derivative of ``J``: ``-r1(x-r2)+r2(x-r1)-(r2-r1)x``."""
    return -r1 * (x - r2) + r2 * (x - r1) - (r2 - r1) * x


def residual_root_sum(r1: Fraction, r2: Fraction, lam1: Fraction) -> Fraction:
    """``r1 + r2 + lambda1``; the sum of slow-line roots is ``-lambda1``."""
    return r1 + r2 + lam1


def residual_interface_embed(
    v_coord: Fraction, eps: Fraction, r1: Fraction, theta: Fraction, x: Fraction
) -> Fraction:
    """``(V + eps x) + (x - r1(1+theta))`` at the matching interface."""
    return (v_coord + eps * x) + (x - r1 * (1 + theta))


def a_wall_gap(r1: Fraction, d: Fraction) -> Fraction:
    """``r1 - d``; negative iff the first-root wall ``a = r1 - d`` has failed."""
    return r1 - d


def r1_log_majorant(r1: Interval) -> Interval:
    """Sound bound ``0 <= -r1 log r1 <= 2 sqrt(r1) - 2 r1`` on ``(0, 1]``."""
    if r1.lo <= 0.0 or r1.hi > 1.0:
        raise ValueError("r1_log_majorant requires r1 in (0, 1]")
    two = Interval.point(2.0)
    return two * r1.sqrt() - two * r1


def _verdict(residual: Fraction) -> str:
    box = Interval.point(float(residual)) if residual == 0 else Interval.from_rational(residual)
    return adjudicate_residual(box, existential=False).status


def identity_verdicts() -> dict[str, str]:
    r1, r2, lam1 = _SAMPLE_R1, _SAMPLE_R2, _SAMPLE_LAM1
    theta, x, eps = _SAMPLE_THETA, _SAMPLE_X, _SAMPLE_EPS
    x_lo = r1 * (1 + theta)
    v_lo = -eps * x_lo
    return {
        "J_numerator": _verdict(residual_J_numerator(x, r1, r2)),
        "root_sum": _verdict(residual_root_sum(r1, r2, lam1)),
        "interface_embed": _verdict(
            residual_interface_embed(v_lo, eps, r1, theta, x_lo)
        ),
    }


def _sequence_majorant() -> dict[str, float | bool]:
    """``2 sqrt(r1)-2 r1`` on ``r1 = 1/n^2``, a vanishing sample of ``r1 log r1``."""
    samples = []
    for n in (4, 10, 25, 100):
        r1 = Interval.from_rational(Fraction(1, n * n))
        bound = r1_log_majorant(r1)
        samples.append(bound.hi)
    decreasing = all(samples[i] > samples[i + 1] for i in range(len(samples) - 1))
    small = samples[-1] < 0.3
    return {
        "majorant_hi": samples[-1],
        "decreasing": decreasing,
        "vanishes": decreasing and small,
    }


@dataclass(frozen=True)
class OutgoingCorridorReport:
    """Bounded x-corridor I-map. Not height-section first-hit or G1."""

    identities: Mapping[str, str]
    a_wall_failed: bool
    majorant: Mapping[str, float | bool]
    outgoing_x_corridor_bounded: bool
    outgoing_first_hit: bool
    honesty: Mapping[str, object]

    def to_payload(self) -> dict[str, object]:
        return {
            "schema": "hilbert16-outgoing-corridor-v1",
            "identities": dict(self.identities),
            "a_wall_failed": self.a_wall_failed,
            "majorant": dict(self.majorant),
            "outgoing_x_corridor_bounded": self.outgoing_x_corridor_bounded,
            "outgoing_first_hit": self.outgoing_first_hit,
            "honesty": dict(self.honesty),
            "scope": (
                "Cleared two-root I-map from r1(1+theta) to a compact x_*; "
                "r1 log r1 majorized by 2 sqrt(r1)-2 r1. Not height-section "
                "first-hit, uniform a_min, G1, or Hilbert XVI."
            ),
        }


def report() -> OutgoingCorridorReport:
    """Replay corridor identities and a vanishing r1 log r1 majorant."""
    identities = identity_verdicts()
    a_wall_failed = a_wall_gap(_SAMPLE_R1, _SAMPLE_D) < 0
    try:
        majorant = _sequence_majorant()
        sealed = all(status == "PROVED" for status in identities.values()) and bool(
            majorant["vanishes"]
        )
    except (ValueError, ZeroDivisionError):
        majorant = {"majorant_hi": float("inf"), "decreasing": False, "vanishes": False}
        sealed = False
    return OutgoingCorridorReport(
        identities=identities,
        a_wall_failed=a_wall_failed,
        majorant=majorant,
        outgoing_x_corridor_bounded=sealed,
        outgoing_first_hit=False,
        honesty=_honesty(outgoing_x_corridor_bounded=sealed),
    )
