# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
"""C!=0 height mixing: ell and V pick up C nu^2 h, so |g_h|=O(nu^2).

On the normal chart the exact polynomials are

    ell = 1 + 2 nu v + C nu^2 h,
    V   = 1 - v - nu v^2 - C nu^2 v h.

Differentiating gives V_v + ell = 0 and V_h + C nu^2 v = 0, hence the
inverse estimates v_V = -1/ell and v_h = -C nu^2 v / ell. The first-order
jet of GRAZING (2.4) has g = -1 + nu (V-1), independent of h. The C-term
in ell and V is O(nu^2), so |g_h| <= M nu^2 on a declared compact with
ell bounded away from zero.

This is not T-h along the actual (V,h) orbit, not height-section
first-hit, G1, or Hilbert XVI. The slow-line Z bound is cancelled_n.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from fractions import Fraction

from omnibias.core.collapse.verdict import adjudicate_residual
from omnibias.core.verified.interval import Interval
from omnibias.dynamics.honesty_inventory import build_honesty

__all__ = [
    "HeightMixReport",
    "ell_full",
    "enclose_g_h",
    "identity_verdicts",
    "report",
    "residual_V_h_c",
    "residual_V_mix",
    "residual_V_v_ell",
    "residual_ell_mix",
    "residual_g_lead",
    "v_coord_full",
]

_SAMPLE_NU = Fraction(1, 16)
_SAMPLE_VCHART = Fraction(1, 2)
_SAMPLE_C = Fraction(2)
_SAMPLE_H = Fraction(1, 4)
_SAMPLE_VCOORD = Fraction(1, 3)
_NU_HI = Fraction(1, 8)
_C_HI = Fraction(3)
_VCHART_HI = Fraction(2)
_H_HI = Fraction(2)


def _honesty(*, height_mix_gh: bool) -> dict[str, object]:
    return build_honesty(
        g1_passed=False,
        g4_passed=False,
        full_graphic_cyclicity_proved=False,
        full_hilbert16_solved=False,
        physical_return_membership_proved=False,
        fold_leading_c2_remainder=False,
        cancelled_n_usable_z=False,
        height_mix_gh=height_mix_gh,
        outgoing_first_hit=False,
        orbit_continuation_on_kill=False,
        hk_theorem_24_used=False,
    )


def ell_full(nu: Fraction, v: Fraction, c: Fraction, h: Fraction) -> Fraction:
    """``ell = 1 + 2 nu v + C nu^2 h``."""
    return 1 + 2 * nu * v + c * nu * nu * h


def v_coord_full(
    nu: Fraction, v: Fraction, c: Fraction, h: Fraction
) -> Fraction:
    """``V = 1 - v - nu v^2 - C nu^2 v h``."""
    return 1 - v - nu * v * v - c * nu * nu * v * h


def residual_ell_mix(nu: Fraction, v: Fraction, c: Fraction, h: Fraction) -> Fraction:
    """``ell`` versus the C=0 factor plus ``C nu^2 h``."""
    return ell_full(nu, v, c, h) - (1 + 2 * nu * v) - c * nu * nu * h


def residual_V_mix(nu: Fraction, v: Fraction, c: Fraction, h: Fraction) -> Fraction:
    """``V`` versus the slow-line factor minus ``C nu^2 v h``."""
    slow = 1 - v - nu * v * v
    return v_coord_full(nu, v, c, h) - slow + c * nu * nu * v * h


def residual_V_v_ell(nu: Fraction, v: Fraction, c: Fraction, h: Fraction) -> Fraction:
    """``V_v + ell``; the inverse slope is ``v_V = -1/ell``."""
    v_v = -1 - 2 * nu * v - c * nu * nu * h
    return v_v + ell_full(nu, v, c, h)


def residual_V_h_c(nu: Fraction, v: Fraction, c: Fraction) -> Fraction:
    """``V_h + C nu^2 v``; hence ``v_h = -C nu^2 v / ell``."""
    v_h = -c * nu * nu * v
    return v_h + c * nu * nu * v


def residual_g_lead(nu: Fraction, v_coord: Fraction, h: Fraction) -> Fraction:
    """GRAZING (2.4) jet: ``(Vdot - f)/h = -1 + nu (V-1)``."""
    cubic = v_coord**3 - 3 * v_coord * v_coord
    vdot = -h + nu * (cubic + (v_coord - 1) * h)
    f_jet = nu * cubic
    return (vdot - f_jet) - h * (-1 + nu * (v_coord - 1))


def _verdict(residual: Fraction) -> str:
    box = Interval.point(float(residual)) if residual == 0 else Interval.from_rational(residual)
    return adjudicate_residual(box, existential=False).status


def identity_verdicts() -> dict[str, str]:
    nu, v, c, h = _SAMPLE_NU, _SAMPLE_VCHART, _SAMPLE_C, _SAMPLE_H
    return {
        "ell_mix": _verdict(residual_ell_mix(nu, v, c, h)),
        "V_mix": _verdict(residual_V_mix(nu, v, c, h)),
        "V_v_ell": _verdict(residual_V_v_ell(nu, v, c, h)),
        "V_h_c": _verdict(residual_V_h_c(nu, v, c)),
        "g_lead": _verdict(residual_g_lead(nu, _SAMPLE_VCOORD, h)),
    }


def enclose_g_h(
    *,
    nu_hi: Fraction = _NU_HI,
    c_hi: Fraction = _C_HI,
    v_hi: Fraction = _VCHART_HI,
    h_hi: Fraction = _H_HI,
) -> dict[str, float | bool]:
    """Sound ``|C nu^2|`` and ``ell`` lower bound on a declared compact."""
    nu = Interval.from_rational(nu_hi)
    c_abs = Interval.from_rational(c_hi)
    v_abs = Interval.from_rational(v_hi)
    h_abs = Interval.from_rational(h_hi)
    mix = c_abs * nu * nu
    two = Interval.from_rational(Fraction(2))
    ell_lo = (
        Interval.from_rational(Fraction(1))
        - two * nu * v_abs
        - mix * h_abs
    )
    v_h_bound = mix * v_abs / ell_lo if ell_lo.lo > 0.0 else Interval(float("inf"), float("inf"))
    nu2 = float(nu_hi * nu_hi)
    mix_ok = mix.hi <= float(c_hi) * nu2 * 1.01
    return {
        "nu_hi": float(nu_hi),
        "c_hi": float(c_hi),
        "mix_hi": mix.hi,
        "mix_over_nu2": mix.hi / nu2 if nu2 else float("inf"),
        "ell_lo": ell_lo.lo,
        "v_h_hi": v_h_bound.hi,
        "ell_positive": ell_lo.lo > 0.0,
        "g_h_o_nu2": bool(ell_lo.lo > 0.0) and mix_ok and v_h_bound.hi < float("inf"),
    }


@dataclass(frozen=True)
class HeightMixReport:
    """C!=0 height mixing. Not T-h along the orbit, first-hit, or G1."""

    identities: Mapping[str, str]
    enclosure: Mapping[str, float | bool]
    height_mix_gh: bool
    outgoing_first_hit: bool
    honesty: Mapping[str, object]

    def to_payload(self) -> dict[str, object]:
        return {
            "schema": "hilbert16-height-mix-v1",
            "identities": dict(self.identities),
            "enclosure": dict(self.enclosure),
            "height_mix_gh": self.height_mix_gh,
            "outgoing_first_hit": self.outgoing_first_hit,
            "honesty": dict(self.honesty),
            "scope": (
                "C!=0 ell/V mixing and |g_h|=O(nu^2) on a declared compact. "
                "Not T-h along the orbit, first-hit, G1, or Hilbert XVI."
            ),
        }


def report() -> HeightMixReport:
    """Replay mixing identities and enclose |g_h| = O(nu^2)."""
    identities = identity_verdicts()
    enclosure = enclose_g_h()
    sealed = (
        all(status == "PROVED" for status in identities.values())
        and bool(enclosure["ell_positive"])
        and bool(enclosure["g_h_o_nu2"])
    )
    return HeightMixReport(
        identities=identities,
        enclosure=enclosure,
        height_mix_gh=sealed,
        outgoing_first_hit=False,
        honesty=_honesty(height_mix_gh=sealed),
    )
