# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
"""Kill-line Stage-A dx_e/dkappa leading factors.

On lambda1 = -2 the slow-line event derivative factors as

    dx_e / d kappa = (A / x_e) exp(Psi_pre)
                     * exp integral eps (B' + y k_x) d tau.

The algebraic prefactor is theta(1+theta) sep^2 / x_e. With
theta = 1/8 and x_e >= a > 1/4 this is below (1/2) sep^2.
The integrand is at most -sep (1-2 theta)/2 = -3 sep/8 and
X <= 2, so the tau-coefficient is 3 sep/16. At the chi_b
threshold kappa = 4/sep the net exponent
(3/16)(4 - sep S_pre) stays above 1/8 after the y0 = (1/16) sep^2
log remainder on sep in [1/2^16, 1], eps in [0, 1/16].

This is a sealed leading-factor enclosure, not the uniform-in-chi
bound C sep^2 exp(-c chi), not Stage C, first-hit, G1, or Hilbert XVI.
The wall is stage_a. The chi threshold is chi_b.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from fractions import Fraction

from omnibias.core.collapse.verdict import adjudicate_residual
from omnibias.core.verified.interval import Interval
from omnibias.core.verified.transcend import ln_iv
from omnibias.dynamics.chi_b import enclose_chi_b
from omnibias.dynamics.honesty_inventory import build_honesty
from omnibias.dynamics.stage_a import enclose_stage_a, wall_left

__all__ = [
    "DxELeadingReport",
    "enclose_dx_e_leading",
    "identity_verdicts",
    "report",
    "residual_net_floor",
    "residual_prefactor",
    "residual_slope_half",
    "sample_dx_e_leading",
]

_SAMPLE_SEP = Fraction(3, 5)
_THETA = Fraction(1, 8)
_WALL = Fraction(9, 64)
_PREF_EXACT = Fraction(3, 8)
_SLOPE_HALF = Fraction(3, 8)
_COEFF = Fraction(3, 16)
_KAPPA_NUM = Fraction(4)
_K = Fraction(3)
_MU = Fraction(1, 16)
_SEP_LO = Fraction(1, 2**16)
_SEP_MID = Fraction(1, 64)
_SEP_HI = Fraction(1)
_EPS_HI = Fraction(1, 16)
_DECLARED_PREF = 0.5
_DECLARED_NET = 0.125


def _honesty(*, dx_e_leading: bool) -> dict[str, object]:
    return build_honesty(
        g1_passed=False,
        g4_passed=False,
        full_graphic_cyclicity_proved=False,
        full_hilbert16_solved=False,
        physical_return_membership_proved=False,
        orbit_continuation_on_kill=False,
        stage_a_wall=False,
        chi_b_bound=False,
        dx_e_leading=dx_e_leading,
        hk_theorem_24_used=False,
    )


def residual_prefactor() -> Fraction:
    """``(9/64) / (3/8) = 3/8``: wall factor over the exact left wall at ``sep = 1``."""
    return _WALL * 8 / 3 - _PREF_EXACT


def residual_slope_half() -> Fraction:
    """``(1 - 2 theta)/2 = 3/8`` at ``theta = 1/8``."""
    return (1 - 2 * _THETA) / 2 - _SLOPE_HALF


def residual_net_floor() -> Fraction:
    """Threshold net ``(3/16)(4 - 3) = 3/16`` with declared ``K = 3``."""
    return _COEFF * (_KAPPA_NUM - _K) - _COEFF


def _verdict(residual: Fraction) -> str:
    box = Interval.point(float(residual)) if residual == 0 else Interval.from_rational(residual)
    return adjudicate_residual(box, existential=False).status


def identity_verdicts() -> dict[str, str]:
    return {
        "dx_prefactor": _verdict(residual_prefactor()),
        "slope_half": _verdict(residual_slope_half()),
        "net_floor": _verdict(residual_net_floor()),
    }


def _contains(outer: Interval, inner: Interval) -> bool:
    return outer.lo <= inner.lo and inner.hi <= outer.hi


def _log_remainder(sep: Interval) -> Interval:
    """``(3/16) eps sep (2 ln sep + ln mu)`` on ``eps in [0, 1/16]``."""
    if sep.lo <= 0.0:
        raise ValueError("_log_remainder requires sep.lo > 0")
    eps = Interval.hull(Fraction(0), _EPS_HI)
    ln_mu = ln_iv(Interval.from_rational(_MU))
    two = Interval.from_rational(Fraction(2))
    return Interval.from_rational(_COEFF) * eps * sep * (two * ln_iv(sep) + ln_mu)


def enclose_dx_e_leading() -> dict[str, float | bool]:
    """Prefactor and threshold net-exponent floor on the chi_b compact."""
    walls = enclose_stage_a()
    chi = enclose_chi_b()
    a = Interval(float(walls["a_lo"]), float(walls["a_hi"]))
    pref = Interval.from_rational(_WALL) / a
    four = Interval.from_rational(_KAPPA_NUM)
    s_hi = float(chi["s_hi"])
    gap = four - Interval(0.0, s_hi)
    net = Interval.from_rational(_COEFF) * gap
    rem_lo = _log_remainder(Interval.hull(_SEP_LO, _SEP_MID))
    rem_hi = _log_remainder(Interval.hull(_SEP_MID, _SEP_HI))
    rem_min = min(rem_lo.lo, rem_hi.lo)
    after = net.lo + rem_min
    pref_hi = float(pref.hi)
    return {
        "pref_lo": pref.lo,
        "pref_hi": pref_hi,
        "net_lo": net.lo,
        "net_hi": net.hi,
        "rem_lo": rem_min,
        "after_lo": after,
        "a_lo": a.lo,
        "a_hi": a.hi,
        "s_hi": s_hi,
        "below_declared": pref_hi < _DECLARED_PREF,
        "net_below_declared": after > _DECLARED_NET,
        "excludes_zero": pref.lo > 0.0 and after > 0.0,
        "finite": pref_hi < _DECLARED_PREF and after > _DECLARED_NET,
        "mu": float(_MU),
    }


def sample_dx_e_leading() -> Interval:
    """Sound algebraic prefactor ``(9/64)/a`` at ``sep = 3/5``."""
    a = Interval.from_rational(wall_left(_SAMPLE_SEP))
    return Interval.from_rational(_WALL) / a


@dataclass(frozen=True)
class DxELeadingReport:
    """Kill-line dx_e leading factors. Not uniform-in-chi, Stage C, or G1."""

    identities: Mapping[str, str]
    sample_pref: Mapping[str, float]
    enclosure: Mapping[str, float | bool]
    sample_inside: bool
    dx_e_leading: bool
    honesty: Mapping[str, object]

    def to_payload(self) -> dict[str, object]:
        return {
            "schema": "hilbert16-dx-e-leading-v1",
            "identities": dict(self.identities),
            "sample_pref": dict(self.sample_pref),
            "enclosure": dict(self.enclosure),
            "sample_inside": self.sample_inside,
            "dx_e_leading": self.dx_e_leading,
            "honesty": dict(self.honesty),
            "scope": (
                "Exact kill-line dx_e identities: algebraic prefactor "
                "3/8, slope half 3/8, and threshold net floor 3/16, "
                "plus Interval prefactor <1/2 and net exponent >1/8 "
                "after the y0 log remainder on the chi_b compact. "
                "Not the uniform-in-chi bound C sep^2 exp(-c chi), "
                "not Stage C, first-hit, G1, or Hilbert XVI."
            ),
        }


def report() -> DxELeadingReport:
    """Replay dx_e leading identities and enclose the threshold factors."""
    identities = identity_verdicts()
    sample = sample_dx_e_leading()
    try:
        enclosure = enclose_dx_e_leading()
        inside = bool(enclosure["finite"]) and _contains(
            Interval(float(enclosure["pref_lo"]), float(enclosure["pref_hi"])),
            sample,
        )
        sealed = (
            all(status == "PROVED" for status in identities.values())
            and bool(enclosure["below_declared"])
            and bool(enclosure["net_below_declared"])
            and bool(enclosure["excludes_zero"])
            and inside
        )
    except (ValueError, ZeroDivisionError, ArithmeticError):
        enclosure = {
            "pref_lo": float("nan"),
            "pref_hi": float("nan"),
            "after_lo": float("nan"),
            "below_declared": False,
            "net_below_declared": False,
            "excludes_zero": False,
            "finite": False,
        }
        inside = False
        sealed = False
    return DxELeadingReport(
        identities=identities,
        sample_pref={"lo": sample.lo, "hi": sample.hi},
        enclosure=enclosure,
        sample_inside=inside,
        dx_e_leading=sealed,
        honesty=_honesty(dx_e_leading=sealed),
    )
