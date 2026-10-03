# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
"""Comparison-bootstrap integral of T-h on the first-root height chart.

Along an actual outgoing branch, ``(T-h)_h = q/h + k - 1``. On a compact
where ``T <= K (eps^2 + h)``, ``|q| <= C eps (eps^2 + T)``, and
``|k-1| <= 3 nu``, the slope splits exactly as

    C eps (eps^2 + K (eps^2 + h)) / h + 3 nu
        = C K eps + 3 nu + C (1+K) eps^3 / h.

The linear piece integrates to ``O(eps)``. The ``eps^3 / h`` piece
integrates to ``O(eps^3 log(hmax/h_e))``, majorized by the rational
sqrt bound ``log u <= 2 (sqrt(u) - 1)``. On the declared compact
``eps = 1/n`` for square ``n >= 16``, ``y0 = 4``, ``hmax = 1``,
``x_* = 1``, ``C = K = 2``, ``nu = eps``, the resulting majorant of
``T-h`` is ``< 9 eps``.

This is the comparison bootstrap, not a Lohner-validated actual
``(V,h)`` orbit, not height-section first-hit, G1, or Hilbert XVI.
The pointwise ``T_h`` gap is orbit_th.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from fractions import Fraction

from omnibias.core.collapse.verdict import adjudicate_residual
from omnibias.core.verified.interval import Interval
from omnibias.dynamics.honesty_inventory import build_honesty

__all__ = [
    "ThIntegralReport",
    "enclose_th_majorant",
    "identity_verdicts",
    "report",
    "residual_ftc_linear",
    "residual_log_sqrt_prefactor",
    "residual_majorant_split",
    "residual_sqrt_ratio_sq",
    "residual_th_dh",
]

_SAMPLE_Q = Fraction(3, 16)
_SAMPLE_H = Fraction(1, 4)
_SAMPLE_KAY = Fraction(17, 16)
_SAMPLE_C = Fraction(2)
_SAMPLE_KBOUND = Fraction(2)
_SAMPLE_EPS = Fraction(1, 16)
_SAMPLE_TE = Fraction(1, 8)
_SAMPLE_HE = Fraction(1, 16)
_SAMPLE_ALPHA = Fraction(1, 4)
_SAMPLE_Y0 = Fraction(4)
_SAMPLE_HMAX = Fraction(1)
_SAMPLE_S = Fraction(32)
_MARGIN = 9.0


def _honesty(*, th_integral_majorant: bool) -> dict[str, object]:
    return build_honesty(
        g1_passed=False,
        g4_passed=False,
        full_graphic_cyclicity_proved=False,
        full_hilbert16_solved=False,
        physical_return_membership_proved=False,
        fold_leading_c2_remainder=False,
        orbit_th_c0=False,
        th_integral_majorant=th_integral_majorant,
        outgoing_first_hit=False,
        orbit_continuation_on_kill=False,
        hk_theorem_24_used=False,
    )


def residual_th_dh(q: Fraction, height: Fraction, kay: Fraction) -> Fraction:
    """Cleared ``(T-h)_h = q/h + k - 1``: ``(q + k h - h) - (q + (k-1) h)``."""
    return (q + kay * height - height) - (q + (kay - 1) * height)


def residual_majorant_split(
    c: Fraction, kay: Fraction, eps: Fraction, height: Fraction
) -> Fraction:
    """``C eps (eps^2 + K (eps^2 + h)) - (C K eps h + C (1+K) eps^3)``."""
    left = c * eps * (eps * eps + kay * (eps * eps + height))
    right = c * kay * eps * height + c * (1 + kay) * eps**3
    return left - right


def residual_ftc_linear(
    t_exit: Fraction,
    h_exit: Fraction,
    alpha: Fraction,
    height: Fraction,
) -> Fraction:
    """Linear-slope orbit: ``(T-h) - (T_e-h_e) - alpha (h-h_e)`` at ``T_h = 1+alpha``."""
    kinetic = t_exit + (alpha + 1) * (height - h_exit)
    return (kinetic - height) - (t_exit - h_exit) - alpha * (height - h_exit)


def residual_sqrt_ratio_sq(
    eps: Fraction, s: Fraction, y0: Fraction, hmax: Fraction
) -> Fraction:
    """``s^2 = hmax / (eps^3 y0)`` cleared: ``eps^3 s^2 y0 - hmax``."""
    return eps**3 * s * s * y0 - hmax


def residual_log_sqrt_prefactor(
    eps: Fraction, s: Fraction, y0: Fraction, hmax: Fraction
) -> Fraction:
    """``eps^6 s^2 y0 - eps^3 hmax``; follows from ``sqrt_ratio_sq`` by ``eps^3``."""
    return eps**6 * s * s * y0 - eps**3 * hmax


def _verdict(residual: Fraction) -> str:
    box = Interval.point(float(residual)) if residual == 0 else Interval.from_rational(residual)
    return adjudicate_residual(box, existential=False).status


def identity_verdicts() -> dict[str, str]:
    q, h, kay = _SAMPLE_Q, _SAMPLE_H, _SAMPLE_KAY
    c, kbound, eps = _SAMPLE_C, _SAMPLE_KBOUND, _SAMPLE_EPS
    te, he, alpha = _SAMPLE_TE, _SAMPLE_HE, _SAMPLE_ALPHA
    s, y0, hmax = _SAMPLE_S, _SAMPLE_Y0, _SAMPLE_HMAX
    return {
        "th_dh": _verdict(residual_th_dh(q, h, kay)),
        "majorant_split": _verdict(residual_majorant_split(c, kbound, eps, h)),
        "ftc_linear": _verdict(residual_ftc_linear(te, he, alpha, h)),
        "sqrt_ratio_sq": _verdict(residual_sqrt_ratio_sq(eps, s, y0, hmax)),
        "log_sqrt_prefactor": _verdict(residual_log_sqrt_prefactor(eps, s, y0, hmax)),
    }


def enclose_th_majorant(
    *,
    y0: Fraction = _SAMPLE_Y0,
    hmax: Fraction = _SAMPLE_HMAX,
    xstar: Fraction = Fraction(1),
    c: Fraction = _SAMPLE_C,
    kbound: Fraction = _SAMPLE_KBOUND,
    ns: tuple[int, ...] = (16, 36, 64, 100),
    margin: float = _MARGIN,
) -> dict[str, float | bool]:
    """Sqrt-majorized ``T-h`` on ``eps=1/n``, ``nu=eps``. Not a Lohner orbit."""
    ratios: list[float] = []
    te_above = True
    within = True
    two = Interval.point(2.0)
    prefactor = Interval.from_rational(c * (1 + kbound))
    for n in ns:
        eps_f = Fraction(1, n)
        eps = Interval.from_rational(eps_f)
        t_exit = (eps_f * xstar) ** 2 / 2
        h_exit = eps_f**3 * y0
        if t_exit <= h_exit:
            te_above = False
            within = False
        gap0 = Interval.from_rational(t_exit - h_exit)
        alpha = Interval.from_rational(c * kbound * eps_f + 3 * eps_f)
        linear = alpha * Interval.from_rational(hmax - h_exit)
        ratio_he = Interval.from_rational(hmax / h_exit)
        log_piece = two * prefactor * (eps**3) * (ratio_he.sqrt() - Interval.point(1.0))
        total = gap0 + linear + log_piece
        ratio = (total / eps).hi
        ratios.append(ratio)
        if ratio >= margin:
            within = False
    decreasing = len(ratios) >= 2 and all(
        ratios[i] > ratios[i + 1] for i in range(len(ratios) - 1)
    )
    return {
        "ratio_hi": ratios[0] if ratios else float("inf"),
        "ratio_lo_end": ratios[-1] if ratios else float("inf"),
        "decreasing": decreasing,
        "te_above_he": te_above,
        "within_margin": within and decreasing and te_above,
    }


@dataclass(frozen=True)
class ThIntegralReport:
    """Comparison-bootstrap T-h integral. Not a Lohner orbit, first-hit, or G1."""

    identities: Mapping[str, str]
    majorant: Mapping[str, float | bool]
    th_integral_majorant: bool
    outgoing_first_hit: bool
    honesty: Mapping[str, object]

    def to_payload(self) -> dict[str, object]:
        return {
            "schema": "hilbert16-th-integral-v1",
            "identities": dict(self.identities),
            "majorant": dict(self.majorant),
            "th_integral_majorant": self.th_integral_majorant,
            "outgoing_first_hit": self.outgoing_first_hit,
            "honesty": dict(self.honesty),
            "scope": (
                "Comparison-bootstrap integral of (T-h)_h after "
                "T<=K(eps^2+h); majorant < 9 eps on a declared compact. "
                "Not a Lohner-validated (V,h) orbit, first-hit, G1, or "
                "Hilbert XVI."
            ),
        }


def report() -> ThIntegralReport:
    """Replay the T-h integral identities and the sqrt majorant."""
    identities = identity_verdicts()
    try:
        majorant = enclose_th_majorant()
        wide = enclose_th_majorant(ns=(8,))
        sealed = (
            all(status == "PROVED" for status in identities.values())
            and bool(majorant["within_margin"])
            and not bool(wide["te_above_he"])
        )
    except (ValueError, ZeroDivisionError):
        majorant = {
            "ratio_hi": float("inf"),
            "ratio_lo_end": float("inf"),
            "decreasing": False,
            "te_above_he": False,
            "within_margin": False,
        }
        sealed = False
    return ThIntegralReport(
        identities=identities,
        majorant=majorant,
        th_integral_majorant=sealed,
        outgoing_first_hit=False,
        honesty=_honesty(th_integral_majorant=sealed),
    )
