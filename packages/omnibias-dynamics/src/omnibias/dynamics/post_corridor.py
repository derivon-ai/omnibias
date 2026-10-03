# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
"""Restored first-root (V,h) hypotheses after the shrinking-root x-corridor.

The saddle wall ``a = r1 - d`` fails once ``r1 < d``. After the x-corridor
the matching section sits at a compact ``x_* in (r1, r2)``:

    V_* = -eps x_*,     T_* = eps^2 x_*^2 / 2,     h_e = eps^3 y0.

Then ``|V_*|/eps = x_*`` and ``T_* = Theta(eps^2)`` are independent of
``r1``. The leading slow-line cubic at ``V = -eps x`` is

    q / eps^3 = L + lambda1 x + x^2 = (x-r1)(x-r2),

so the height derivative ``T_h = q/h + k`` has leading value
``1 + (x-r1)(x-r2)/y0`` at ``h_e``. On ``(r1, r2)`` the product is
negative; a fixed ``y0 > -2 (x_*-r1)(x_*-r2)`` yields ``T_h > 1/2``.

The continuation factor ``(h/h_e)^{C eps}`` has logarithm
``C eps log(h/y0) + 3 C eps log(1/eps)``. The second term is majorized
by ``6 (sqrt(eps) - eps)`` via ``log u <= 2 (sqrt(u) - 1)`` on ``u >= 1``.
Along ``eps = 1/n`` that majorant tends to 0, independently of ``r1``
and of ``a_min``.

This restores the *hypotheses* of the written first-root height
continuation. It is not first-hit of the large height section, a sealed
``T - h = O(eps)`` bootstrap, G1, or Hilbert XVI. The saddle wall
``a = r1 - d`` still fails.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from fractions import Fraction

from omnibias.core.collapse.verdict import adjudicate_residual
from omnibias.core.verified.interval import Interval
from omnibias.dynamics.honesty_inventory import build_honesty
from omnibias.dynamics.outgoing_corridor import a_wall_gap

__all__ = [
    "PostCorridorReport",
    "identity_verdicts",
    "integrating_factor_exponent_majorant",
    "report",
    "residual_T_kinetic",
    "residual_V_embed",
    "residual_h_exit",
    "residual_q_leading",
    "th_leading_value",
]

_SAMPLE_R1 = Fraction(1, 5)
_SAMPLE_R2 = Fraction(4)
_SAMPLE_LAM1 = -(_SAMPLE_R1 + _SAMPLE_R2)
_SAMPLE_L = _SAMPLE_R1 * _SAMPLE_R2
_SAMPLE_XSTAR = Fraction(1)
_SAMPLE_EPS = Fraction(1, 7)
_SAMPLE_Y0 = Fraction(8)
_SAMPLE_D = Fraction(1, 4)


def _honesty(*, post_corridor_margin: bool) -> dict[str, object]:
    return build_honesty(
        g1_passed=False,
        g4_passed=False,
        full_graphic_cyclicity_proved=False,
        full_hilbert16_solved=False,
        physical_return_membership_proved=False,
        fold_leading_c2_remainder=False,
        post_corridor_margin=post_corridor_margin,
        outgoing_first_hit=False,
        orbit_continuation_on_kill=False,
        hk_theorem_24_used=False,
    )


def residual_V_embed(v_coord: Fraction, eps: Fraction, xstar: Fraction) -> Fraction:
    """``V + eps x_*`` at the compact matching section."""
    return v_coord + eps * xstar


def residual_T_kinetic(kinetic: Fraction, eps: Fraction, xstar: Fraction) -> Fraction:
    """``2 T - eps^2 x_*^2``; ``T = V^2 / 2`` on ``V = -eps x_*``."""
    return 2 * kinetic - eps**2 * xstar**2


def residual_h_exit(height: Fraction, eps: Fraction, y0: Fraction) -> Fraction:
    """``h - eps^3 y0`` at the fast-fiber exit used by first-root continuation."""
    return height - eps**3 * y0


def residual_q_leading(
    L: Fraction, lam1: Fraction, x: Fraction, r1: Fraction, r2: Fraction
) -> Fraction:
    """``L + lambda1 x + x^2 - (x-r1)(x-r2)``; leading ``q / eps^3``."""
    return L + lam1 * x + x * x - (x - r1) * (x - r2)


def th_leading_value(y0: Fraction, x: Fraction, r1: Fraction, r2: Fraction) -> Fraction:
    """``y0 * T_h`` at leading order: ``y0 + (x-r1)(x-r2)``."""
    return y0 + (x - r1) * (x - r2)


def integrating_factor_exponent_majorant(eps: Interval) -> Interval:
    """Upper bound ``6 (sqrt(eps) - eps)`` for ``3 eps log(1/eps)`` on ``(0, 1]``.

    Uses ``log u <= 2 (sqrt(u) - 1)`` at ``u = 1/eps``. When ``h <= y0`` the
    leftover ``eps log(h/y0)`` is non-positive and may be dropped.
    """
    if eps.lo <= 0.0 or eps.hi > 1.0:
        raise ValueError("integrating_factor_exponent_majorant requires eps in (0, 1]")
    six = Interval.point(6.0)
    return six * (eps.sqrt() - eps)


def _verdict(residual: Fraction) -> str:
    box = Interval.point(float(residual)) if residual == 0 else Interval.from_rational(residual)
    return adjudicate_residual(box, existential=False).status


def identity_verdicts() -> dict[str, str]:
    r1, r2, lam1, L = _SAMPLE_R1, _SAMPLE_R2, _SAMPLE_LAM1, _SAMPLE_L
    xstar, eps, y0 = _SAMPLE_XSTAR, _SAMPLE_EPS, _SAMPLE_Y0
    v_star = -eps * xstar
    t_star = (eps * xstar) ** 2 / 2
    h_e = eps**3 * y0
    return {
        "V_embed": _verdict(residual_V_embed(v_star, eps, xstar)),
        "T_kinetic": _verdict(residual_T_kinetic(t_star, eps, xstar)),
        "h_exit": _verdict(residual_h_exit(h_e, eps, y0)),
        "q_leading": _verdict(residual_q_leading(L, lam1, xstar, r1, r2)),
    }


def _sequence_majorant() -> dict[str, float | bool]:
    """``6 (sqrt(eps) - eps)`` on ``eps = 1/n``, vanishing as ``n -> inf``."""
    samples = []
    for n in (4, 10, 25, 100):
        eps = Interval.from_rational(Fraction(1, n))
        bound = integrating_factor_exponent_majorant(eps)
        samples.append(bound.hi)
    decreasing = all(samples[i] > samples[i + 1] for i in range(len(samples) - 1))
    small = samples[-1] < 0.6
    return {
        "majorant_hi": samples[-1],
        "decreasing": decreasing,
        "vanishes": decreasing and small,
    }


@dataclass(frozen=True)
class PostCorridorReport:
    """Restored (V,h) hypotheses after the x-corridor. Not height first-hit."""

    identities: Mapping[str, str]
    a_wall_failed: bool
    restored_x_margin: bool
    th_leading_half: bool
    majorant: Mapping[str, float | bool]
    post_corridor_margin: bool
    outgoing_first_hit: bool
    honesty: Mapping[str, object]

    def to_payload(self) -> dict[str, object]:
        return {
            "schema": "hilbert16-post-corridor-v1",
            "identities": dict(self.identities),
            "a_wall_failed": self.a_wall_failed,
            "restored_x_margin": self.restored_x_margin,
            "th_leading_half": self.th_leading_half,
            "majorant": dict(self.majorant),
            "post_corridor_margin": self.post_corridor_margin,
            "outgoing_first_hit": self.outgoing_first_hit,
            "honesty": dict(self.honesty),
            "scope": (
                "After the x-corridor, T_*=Theta(eps^2) and (h/h_e)^(C eps)->1 "
                "independently of r1. Not height-section first-hit, uniform "
                "a_min, sealed T-h=O(eps), G1, or Hilbert XVI."
            ),
        }


def report() -> PostCorridorReport:
    """Replay post-corridor identities and the integrating-factor majorant."""
    identities = identity_verdicts()
    r1, r2, xstar, y0 = _SAMPLE_R1, _SAMPLE_R2, _SAMPLE_XSTAR, _SAMPLE_Y0
    a_wall_failed = a_wall_gap(r1, _SAMPLE_D) < 0
    restored_x_margin = xstar > r1
    th_val = th_leading_value(y0, xstar, r1, r2)
    th_leading_half = th_val > y0 / 2
    try:
        majorant = _sequence_majorant()
        sealed = (
            all(status == "PROVED" for status in identities.values())
            and a_wall_failed
            and restored_x_margin
            and th_leading_half
            and bool(majorant["vanishes"])
        )
    except (ValueError, ZeroDivisionError):
        majorant = {"majorant_hi": float("inf"), "decreasing": False, "vanishes": False}
        sealed = False
    return PostCorridorReport(
        identities=identities,
        a_wall_failed=a_wall_failed,
        restored_x_margin=restored_x_margin,
        th_leading_half=th_leading_half,
        majorant=majorant,
        post_corridor_margin=sealed,
        outgoing_first_hit=False,
        honesty=_honesty(post_corridor_margin=sealed),
    )
