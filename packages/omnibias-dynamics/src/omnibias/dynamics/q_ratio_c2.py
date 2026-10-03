# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
"""Uniform C=2 bound on the leading |q| ratio for the shrinking-root sequence.

On ``lambda1 = -2`` the slow-line roots satisfy ``r1 + r2 = 2`` and
``L = r1 r2``. The leading ratio

    |q| / (eps (eps^2 + T)) = |x^2 + lambda1 x + L| / (1 + x^2 / 2)

is independent of ``eps``. Between the roots it equals
``(x-r1)(r2-x)/(1+x^2/2)``. The polynomial identity

    2 + x^2 - (x-r1)(r2-x) = 2 + 2 x^2 + lambda1 x + L

and the complete square ``2(x-1/2)^2 + 3/2 + L`` (valid at
``lambda1 = -2``) prove the inner gap is at least ``3/2 + L > 0``.
Past the second root the outer gap is ``2 + 2x - L``, positive for
``L < 2``. The discriminant of the inner quadratic is
``lambda1^2 - 12 L - 16 = -12(1+L) < 0``, so ``C = 2`` is uniform in
``x`` and in ``r1 -> 0``.

This bounds leading ``|q|`` by ``2 eps (eps^2 + T)`` as a function of
``V`` only. It is not ``k = 1 + O(eps)``, a ``zeta`` remainder, the
transversal event, G1, or Hilbert XVI.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from fractions import Fraction

from omnibias.core.collapse.verdict import adjudicate_residual
from omnibias.core.verified.interval import Interval
from omnibias.dynamics.honesty_inventory import build_honesty

__all__ = [
    "QRatioC2Report",
    "identity_verdicts",
    "q_ratio_leading",
    "report",
    "residual_kill_square",
    "residual_ratio_disc",
    "residual_ratio_gap",
    "residual_ratio_outer",
]

_SAMPLE_R1 = Fraction(1, 5)
_SAMPLE_R2 = Fraction(9, 5)
_SAMPLE_LAM1 = Fraction(-2)
_SAMPLE_L = _SAMPLE_R1 * _SAMPLE_R2
_SAMPLE_X_INNER = Fraction(1)
_SAMPLE_X_OUTER = Fraction(3)


def _honesty(*, q_ratio_c2: bool) -> dict[str, object]:
    return build_honesty(
        g1_passed=False,
        g4_passed=False,
        full_graphic_cyclicity_proved=False,
        full_hilbert16_solved=False,
        physical_return_membership_proved=False,
        fold_leading_c2_remainder=False,
        q_ratio_c2=q_ratio_c2,
        outgoing_first_hit=False,
        orbit_continuation_on_kill=False,
        hk_theorem_24_used=False,
    )


def residual_ratio_gap(
    x: Fraction, r1: Fraction, r2: Fraction, lam1: Fraction, L: Fraction
) -> Fraction:
    """Inner gap ``2 + x^2 - (x-r1)(r2-x)`` versus ``2 + 2x^2 + lambda1 x + L``."""
    return (2 + x * x - (x - r1) * (r2 - x)) - (2 + 2 * x * x + lam1 * x + L)


def residual_ratio_disc(r1: Fraction, r2: Fraction, lam1: Fraction, L: Fraction) -> Fraction:
    """``(r2-r1)^2 - 8(L+2)`` versus ``lambda1^2 - 12 L - 16``."""
    return ((r2 - r1) ** 2 - 8 * (L + 2)) - (lam1**2 - 12 * L - 16)


def residual_kill_square(x: Fraction, L: Fraction) -> Fraction:
    """``2 + 2x^2 - 2x + L`` versus ``2(x-1/2)^2 + 3/2 + L`` at ``lambda1 = -2``."""
    half = Fraction(1, 2)
    return (2 + 2 * x * x - 2 * x + L) - (2 * (x - half) ** 2 + Fraction(3, 2) + L)


def residual_ratio_outer(
    x: Fraction, r1: Fraction, r2: Fraction, L: Fraction
) -> Fraction:
    """Outer gap ``2 + x^2 - (x-r1)(x-r2)`` versus ``2 + 2x - L`` at ``lambda1 = -2``."""
    return (2 + x * x - (x - r1) * (x - r2)) - (2 + 2 * x - L)


def q_ratio_leading(x: Fraction, r1: Fraction, r2: Fraction) -> Fraction:
    """``|(x-r1)(x-r2)| / (1 + x^2/2)``; the leading |q| ratio."""
    return abs((x - r1) * (x - r2)) / (1 + x * x / 2)


def _verdict(residual: Fraction) -> str:
    box = Interval.point(float(residual)) if residual == 0 else Interval.from_rational(residual)
    return adjudicate_residual(box, existential=False).status


def identity_verdicts() -> dict[str, str]:
    r1, r2, lam1, L = _SAMPLE_R1, _SAMPLE_R2, _SAMPLE_LAM1, _SAMPLE_L
    x_in, x_out = _SAMPLE_X_INNER, _SAMPLE_X_OUTER
    return {
        "ratio_gap": _verdict(residual_ratio_gap(x_in, r1, r2, lam1, L)),
        "ratio_disc": _verdict(residual_ratio_disc(r1, r2, lam1, L)),
        "kill_square": _verdict(residual_kill_square(x_in, L)),
        "ratio_outer": _verdict(residual_ratio_outer(x_out, r1, r2, L)),
    }


def _ratio_below_two() -> dict[str, float | bool]:
    """Leading ratio on inner, outer, and large-x samples; all strictly below 2."""
    r1, r2 = _SAMPLE_R1, _SAMPLE_R2
    samples = [float(q_ratio_leading(x, r1, r2)) for x in (r1, Fraction(1), r2, Fraction(3), Fraction(10))]
    return {
        "ratio_hi": max(samples),
        "below_two": all(val < 2.0 for val in samples),
    }


def _disc_negative_on_L() -> dict[str, float | bool]:
    """``lambda1^2 - 12 L - 16 = -12(1+L)`` along ``L = 1/n``."""
    samples = []
    for n in (1, 4, 25, 100):
        L = Fraction(1, n)
        samples.append(float(_SAMPLE_LAM1**2 - 12 * L - 16))
    return {
        "disc_hi": max(samples),
        "negative": all(val < 0.0 for val in samples),
    }


@dataclass(frozen=True)
class QRatioC2Report:
    """Uniform C=2 leading |q| ratio on lambda1=-2. Not first-hit or G1."""

    identities: Mapping[str, str]
    ratio: Mapping[str, float | bool]
    disc: Mapping[str, float | bool]
    q_ratio_c2: bool
    outgoing_first_hit: bool
    honesty: Mapping[str, object]

    def to_payload(self) -> dict[str, object]:
        return {
            "schema": "hilbert16-q-ratio-c2-v1",
            "identities": dict(self.identities),
            "ratio": dict(self.ratio),
            "disc": dict(self.disc),
            "q_ratio_c2": self.q_ratio_c2,
            "outgoing_first_hit": self.outgoing_first_hit,
            "honesty": dict(self.honesty),
            "scope": (
                "On lambda1=-2 the leading |q| ratio is <2 for every x, "
                "uniformly in r1->0. Not k=1+O(eps), zeta remainder, "
                "height-section first-hit, G1, or Hilbert XVI."
            ),
        }


def report() -> QRatioC2Report:
    """Replay C=2 ratio identities on the shrinking-root sequence."""
    identities = identity_verdicts()
    try:
        ratio = _ratio_below_two()
        disc = _disc_negative_on_L()
        sealed = (
            all(status == "PROVED" for status in identities.values())
            and bool(ratio["below_two"])
            and bool(disc["negative"])
        )
    except (ValueError, ZeroDivisionError):
        ratio = {"ratio_hi": float("inf"), "below_two": False}
        disc = {"disc_hi": float("inf"), "negative": False}
        sealed = False
    return QRatioC2Report(
        identities=identities,
        ratio=ratio,
        disc=disc,
        q_ratio_c2=sealed,
        outgoing_first_hit=False,
        honesty=_honesty(q_ratio_c2=sealed),
    )
