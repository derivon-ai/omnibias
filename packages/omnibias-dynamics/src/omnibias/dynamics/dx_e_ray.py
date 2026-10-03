# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
"""dx_e leading factors for every ``lambda1 <= -2``.

The residence denominator satisfies ``X <= 2 rstar`` with
``rstar = -lambda1/2``. The integrand bound ``-3 sep/8`` then gives
the tau-coefficient ``3 sep / (16 rstar)``. On ``rstar >= 1`` and
``sep in (0, 1]`` the wall ``a >= 3/8`` and ``u = sep/r1 <= 2`` still
hold, so the sealed ratio ``h(u) < 11/5`` yields

    sep * S_pre < r1 * (11/5).

The threshold ``kappa * sep = r1*(11/5) + rstar`` leaves a main
exponent ``3/16``. The log remainder is ``(1/rstar)`` times the
``rstar = 1`` remainder, so its most negative value is the one
already enclosed at the kill edge. The net stays above ``1/8``,
``chi_b <= 21/5``, and ``C < 2``. Dropping the ``rstar`` surplus
stalls.

This covers every ``lambda1 < -4`` as well as the slab
``[-4, -2]``. It does not cover ``lambda1 in (-2, 0)``. It is not
Stage C, first-hit, G1, or Hilbert XVI. The bounded slab is
``dx_e_off``.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from fractions import Fraction

from omnibias.core.collapse.verdict import adjudicate_residual
from omnibias.core.verified.interval import Interval
from omnibias.core.verified.transcend import exp_iv
from omnibias.dynamics.dx_e_off import enclose_dx_e_off, sample_dx_e_off
from omnibias.dynamics.honesty_inventory import build_honesty

__all__ = [
    "DxERayReport",
    "enclose_dx_e_ray",
    "identity_verdicts",
    "report",
    "residual_ray_chi",
    "residual_ray_coeff",
    "residual_ray_x",
    "sample_dx_e_ray",
]

_RSTAR_EDGE = Fraction(1)
_X_FACTOR = Fraction(2)
_SLOPE = Fraction(3, 8)
_NET = Fraction(3, 16)
_H_CAP = Fraction(11, 5)
_CHI_CAP = Fraction(21, 5)
_PREF = Fraction(3, 8)
_EXTRA = Fraction(3, 32)
_DECLARED_NET = 0.125
_DECLARED_C = 2.0
_DECLARED_EXTRA = 0.0625


def _honesty(*, dx_e_ray: bool) -> dict[str, object]:
    return build_honesty(
        g1_passed=False,
        g4_passed=False,
        full_graphic_cyclicity_proved=False,
        full_hilbert16_solved=False,
        physical_return_membership_proved=False,
        orbit_continuation_on_kill=False,
        outgoing_first_hit=False,
        dx_e_leading=False,
        dx_e_off=False,
        dx_e_ray=dx_e_ray,
        hk_theorem_24_used=False,
    )


def residual_ray_x() -> Fraction:
    """``X = 2 rstar`` at the kill edge ``rstar = 1`` recovers ``X = 2``."""
    return _X_FACTOR * _RSTAR_EDGE - 2


def residual_ray_coeff() -> Fraction:
    """``(3/8) / (2 rstar) * rstar = 3/16`` at ``rstar = 1``."""
    return _SLOPE / (_X_FACTOR * _RSTAR_EDGE) * _RSTAR_EDGE - _NET


def residual_ray_chi() -> Fraction:
    """Worst chi ratio ``rstar / (rstar - 1/2) = 2`` at ``rstar = 1``, so ``11/5+2=21/5``."""
    return _H_CAP + (_RSTAR_EDGE / (_RSTAR_EDGE - Fraction(1, 2))) - _CHI_CAP


def _verdict(residual: Fraction) -> str:
    box = Interval.point(float(residual)) if residual == 0 else Interval.from_rational(residual)
    return adjudicate_residual(box, existential=False).status


def identity_verdicts() -> dict[str, str]:
    return {
        "ray_x": _verdict(residual_ray_x()),
        "ray_coeff": _verdict(residual_ray_coeff()),
        "ray_chi": _verdict(residual_ray_chi()),
    }


def _contains(outer: Interval, inner: Interval) -> bool:
    return outer.lo <= inner.lo and inner.hi <= outer.hi


def sample_dx_e_ray() -> Interval:
    """The sealed ratio ``h(1/2)``, reused on the whole ray."""
    return sample_dx_e_off()


def enclose_dx_e_ray(*, use_bound: bool = True) -> dict[str, float | bool]:
    """Net exponent and ``C`` for every ``rstar >= 1``.

    ``use_bound=False`` drops the ``rstar`` surplus and stalls.
    """
    prior = enclose_dx_e_off()
    h_hi = float(prior["h_hi"])
    rem = float(prior["rem_lo"])
    cap = Interval.from_rational(_H_CAP)
    below = bool(prior["below_cap"]) and h_hi < cap.lo and rem < 0.0
    main = _NET if use_bound else Fraction(0)
    after = float(main) + rem
    net_ok = after > _DECLARED_NET
    chi_hi = float(_CHI_CAP)
    pref = Interval.from_rational(_PREF)
    e_th = exp_iv(Interval.point(-after))
    lift = Interval.from_rational(_EXTRA) * Interval(0.0, chi_hi)
    c_hi = float((pref * e_th * exp_iv(lift)).hi)
    extra_lo = float(Interval.from_rational(_EXTRA).lo)
    finite = bool(
        below and net_ok and c_hi < _DECLARED_C and extra_lo > _DECLARED_EXTRA and use_bound
    )
    return {
        "h_hi": h_hi,
        "rem_lo": rem,
        "after_lo": after,
        "below_cap": below,
        "net_below_declared": net_ok,
        "chi_hi": chi_hi,
        "c_hi": c_hi,
        "extra_lo": extra_lo,
        "below_c": c_hi < _DECLARED_C,
        "extra_above_floor": extra_lo > _DECLARED_EXTRA,
        "excludes_zero": after > 0.0 and c_hi > 0.0,
        "finite": finite,
    }


@dataclass(frozen=True)
class DxERayReport:
    """dx_e factors for every ``lambda1 <= -2``. Not G1."""

    identities: Mapping[str, str]
    sample_h: Mapping[str, float]
    enclosure: Mapping[str, float | bool]
    stall: Mapping[str, float | bool]
    sample_inside: bool
    dx_e_ray: bool
    honesty: Mapping[str, object]

    def to_payload(self) -> dict[str, object]:
        return {
            "schema": "hilbert16-dx-e-ray-v1",
            "identities": dict(self.identities),
            "sample_h": dict(self.sample_h),
            "enclosure": dict(self.enclosure),
            "stall": dict(self.stall),
            "sample_inside": self.sample_inside,
            "dx_e_ray": self.dx_e_ray,
            "outgoing_first_hit": False,
            "honesty": dict(self.honesty),
            "scope": (
                "Exact identities: X=2*rstar equals 2 at rstar=1, "
                "(3/8)/(2 rstar)*rstar=3/16, and 11/5+2=21/5. "
                "For every rstar>=1 and every sep in (0, 1] the "
                "net exponent stays above 1/8, chi_b<=21/5, and "
                "C<2. Dropping the rstar surplus stalls. Not "
                "lambda1 in (-2, 0), not Stage C, first-hit, G1, "
                "or Hilbert XVI."
            ),
        }


def report() -> DxERayReport:
    """Replay the ray identities and enclose the dx_e factors."""
    identities = identity_verdicts()
    sample = sample_dx_e_ray()
    try:
        enclosure = enclose_dx_e_ray()
        stall = enclose_dx_e_ray(use_bound=False)
        prior = enclose_dx_e_off()
        inside = bool(enclosure["finite"]) and _contains(
            Interval(float(prior["h_lo"]), float(prior["h_hi"])),
            sample,
        )
        proved = all(status == "PROVED" for status in identities.values())
        sealed = (
            proved
            and bool(enclosure["finite"])
            and bool(enclosure["net_below_declared"])
            and bool(enclosure["below_c"])
            and bool(enclosure["extra_above_floor"])
            and bool(enclosure["excludes_zero"])
            and inside
            and not bool(stall["finite"])
            and float(stall["after_lo"]) < 0.0
        )
    except (ValueError, ZeroDivisionError, ArithmeticError):
        enclosure = {
            "after_lo": float("nan"),
            "finite": False,
            "net_below_declared": False,
        }
        stall = {"finite": True, "after_lo": 1.0}
        inside = False
        sealed = False
    return DxERayReport(
        identities=identities,
        sample_h={"lo": sample.lo, "hi": sample.hi},
        enclosure=enclosure,
        stall=stall,
        sample_inside=inside,
        dx_e_ray=sealed,
        honesty=_honesty(dx_e_ray=sealed),
    )
