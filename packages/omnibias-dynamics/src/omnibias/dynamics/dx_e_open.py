# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
"""dx_e leading factors for every ``lambda1`` in ``(-3/2, 0)``.

On ``rstar = -lambda1/2`` in ``(0, 3/4)`` the rectangle exists only for
``sep < (8/5) rstar``, where ``a > 0`` and ``u = sep/r1 < 8``. The
ratio ``h(u)`` stays below ``11/5`` on ``(0, 8]``. The residence bound
``X <= 2 rstar`` still leaves the main exponent ``3/16`` after the
threshold ``kappa * sep = r1*(11/5) + rstar``.

The log remainder is at most ``(3/10) eps (2 ln sep + ln mu)``. The
cap ``eps <= (1/6) / -(2 ln sep + ln mu)`` keeps that remainder above
``-1/20``, so the net stays above ``11/80 > 1/8``. As ``a -> 0`` the
chi cap approaches ``36/5`` and the extra coefficient approaches
``3/80``. The decay ``1/32`` is strictly weaker, and the factored
constant satisfies ``C < (1/5) / a``. Holding ``eps = 1/16`` at a
small ``sep`` stalls.

This covers every ``lambda1`` in ``(-3/2, 0)`` on the geometric
``sep`` range. It is not Stage C, first-hit, G1, or Hilbert XVI.
The slab ``[-3/2, -2)`` is ``dx_e_near``.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from fractions import Fraction

from omnibias.core.collapse.verdict import adjudicate_residual
from omnibias.core.verified.interval import Interval
from omnibias.core.verified.transcend import exp_iv, ln_iv
from omnibias.dynamics.dx_e_near import enclose_dx_e_near
from omnibias.dynamics.honesty_inventory import build_honesty

__all__ = [
    "DxEOpenReport",
    "enclose_dx_e_open",
    "identity_verdicts",
    "report",
    "residual_open_chi",
    "residual_open_gap",
    "residual_open_wall",
    "sample_dx_e_open",
]

_H_CAP = Fraction(11, 5)
_CHI_CAP = Fraction(36, 5)
_EXTRA = Fraction(3, 80)
_BETA = Fraction(1, 32)
_MAIN = Fraction(3, 16)
_FLOOR = Fraction(11, 80)
_GAP = Fraction(1, 20)
_EPS_NUM = Fraction(1, 6)
_MU = Fraction(1, 16)
_PREF_NUM = Fraction(9, 64)
_LIFT = Fraction(9, 40)
_U_HI = Fraction(8)
_DECLARED_NET = 0.125
_DECLARED_FACTOR = 0.2
_SAMPLE_SEP = Fraction(1, 2)
_SAMPLE_U = Fraction(6)
_STALL_SEP = Fraction(1, 4096)
_STALL_EPS = Fraction(1, 16)


def _honesty(*, dx_e_open: bool) -> dict[str, object]:
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
        dx_e_ray=False,
        dx_e_near=False,
        dx_e_open=dx_e_open,
        hk_theorem_24_used=False,
    )


def residual_open_wall() -> Fraction:
    """``1 - (5/8)*(8/5) = 0``: the wall ``a`` vanishes at ``sep = (8/5) rstar``."""
    return 1 - Fraction(5, 8) * Fraction(8, 5)


def residual_open_chi() -> Fraction:
    """``11/5 + 5 = 36/5``: chi cap as ``rstar/r1 -> 5``."""
    return _H_CAP + 5 - _CHI_CAP


def residual_open_gap() -> Fraction:
    """``3/16 - 1/20 = 11/80``: net floor under the ``eps`` cap."""
    return _MAIN - _GAP - _FLOOR


def _verdict(residual: Fraction) -> str:
    box = Interval.point(float(residual)) if residual == 0 else Interval.from_rational(residual)
    return adjudicate_residual(box, existential=False).status


def identity_verdicts() -> dict[str, str]:
    return {
        "open_wall": _verdict(residual_open_wall()),
        "open_chi": _verdict(residual_open_chi()),
        "open_gap": _verdict(residual_open_gap()),
    }


def _contains(outer: Interval, inner: Interval) -> bool:
    return outer.lo <= inner.lo and inner.hi <= outer.hi


def _h_box(u: Interval) -> Interval:
    """Enclosure of ``h(u) = sep * S_pre / r1``."""
    if u.lo <= 0.0:
        raise ValueError("_h_box requires u.lo > 0")
    one = Interval.from_rational(Fraction(1))
    one_u = one + u
    return (
        u * ln_iv(u)
        - one_u * ln_iv(one_u)
        + one_u * ln_iv(Interval.from_rational(Fraction(9, 8)))
        - ln_iv(Interval.from_rational(Fraction(1, 8)))
    )


def _h_above_four() -> tuple[float, float]:
    """Upper and lower bounds of ``h`` on ``(4, 8]``."""
    h_hi = float("-inf")
    h_lo = float("inf")
    i = 64
    while Fraction(i, 16) < _U_HI:
        slab = Interval.hull(Fraction(i, 16), Fraction(i + 1, 16))
        box = _h_box(slab)
        h_hi = max(h_hi, box.hi)
        h_lo = min(h_lo, box.lo)
        i += 1
    return h_lo, h_hi


def _log_gap(sep: Fraction) -> Interval:
    """``2 ln sep + ln(1/16)``."""
    if sep <= 0:
        raise ValueError("_log_gap requires sep > 0")
    ln_mu = ln_iv(Interval.from_rational(_MU))
    return Interval.from_rational(2) * ln_iv(Interval.from_rational(sep)) + ln_mu


def _majorant(sep: Fraction, eps: Interval) -> Interval:
    """Lower bound of ``(3/10) eps (2 ln sep + ln mu)``."""
    return Interval.from_rational(Fraction(3, 10)) * eps * _log_gap(sep)


def sample_dx_e_open() -> Interval:
    """Sound ``h(6)`` on the slab ``u in (4, 8]``."""
    return _h_box(Interval.from_rational(_SAMPLE_U))


def _factor() -> Interval:
    """Uniform factor in ``C < factor / a``."""
    return (
        Interval.from_rational(_PREF_NUM)
        * exp_iv(-Interval.from_rational(_FLOOR))
        * exp_iv(Interval.from_rational(_LIFT))
    )


def enclose_dx_e_open(*, use_bound: bool = True) -> dict[str, float | bool]:
    """Net exponent and ``1/a`` factor for every ``lambda1`` in ``(-3/2, 0)``.

    ``use_bound=False`` holds ``eps = 1/16`` at ``sep = 1/4096`` and stalls.
    """
    prior = enclose_dx_e_near()
    lo_hi, hi_hi = _h_above_four()
    h_hi = max(float(prior["h_hi"]), hi_hi)
    h_lo = min(float(prior["h_lo"]), lo_hi)
    cap = Interval.from_rational(_H_CAP)
    below = bool(prior["below_cap"]) and h_hi < cap.lo and hi_hi < cap.lo
    ln_mu = ln_iv(Interval.from_rational(_MU))
    monotone = ln_mu.hi < -2.0
    if use_bound:
        gap = _log_gap(_SAMPLE_SEP)
        if not (gap.hi < 0.0):
            raise ValueError("sample log gap must be negative")
        eps = Interval.from_rational(_EPS_NUM) / (-gap)
        rem = _majorant(_SAMPLE_SEP, eps)
    else:
        eps = Interval.from_rational(_STALL_EPS)
        rem = _majorant(_STALL_SEP, eps)
    after = float(_MAIN) + rem.lo
    net_ok = after > _DECLARED_NET
    factor_hi = float(_factor().hi)
    extra_lo = float(Interval.from_rational(_EXTRA).lo)
    beta_hi = float(Interval.from_rational(_BETA).hi)
    finite = bool(
        below
        and monotone
        and use_bound
        and net_ok
        and factor_hi < _DECLARED_FACTOR
        and extra_lo > beta_hi
    )
    return {
        "h_lo": h_lo,
        "h_hi": h_hi,
        "rem_lo": rem.lo,
        "after_lo": after,
        "eps_hi": float(eps.hi),
        "below_cap": below,
        "monotone": monotone,
        "net_below_declared": net_ok,
        "chi_hi": float(_CHI_CAP),
        "factor_hi": factor_hi,
        "extra_lo": extra_lo,
        "below_factor": factor_hi < _DECLARED_FACTOR,
        "extra_above_beta": extra_lo > beta_hi,
        "excludes_zero": after > 0.0 and factor_hi > 0.0,
        "finite": finite,
    }


@dataclass(frozen=True)
class DxEOpenReport:
    """dx_e factors for every ``lambda1`` in ``(-3/2, 0)``. Not G1."""

    identities: Mapping[str, str]
    sample_h: Mapping[str, float]
    enclosure: Mapping[str, float | bool]
    stall: Mapping[str, float | bool]
    sample_inside: bool
    dx_e_open: bool
    honesty: Mapping[str, object]

    def to_payload(self) -> dict[str, object]:
        return {
            "schema": "hilbert16-dx-e-open-v1",
            "identities": dict(self.identities),
            "sample_h": dict(self.sample_h),
            "enclosure": dict(self.enclosure),
            "stall": dict(self.stall),
            "sample_inside": self.sample_inside,
            "dx_e_open": self.dx_e_open,
            "outgoing_first_hit": False,
            "honesty": dict(self.honesty),
            "scope": (
                "Exact identities: 1-(5/8)*(8/5)=0, 11/5+5=36/5, and "
                "3/16-1/20=11/80. For every lambda1 in (-3/2, 0) and "
                "every sep in (0, min(1, (8/5) rstar)), the eps cap "
                "(1/6)/-(2 ln sep + ln(1/16)) keeps the net exponent "
                "above 1/8, with C < (1/5)/a. Holding eps=1/16 at "
                "sep=1/4096 stalls. Not Stage C, first-hit, G1, or "
                "Hilbert XVI."
            ),
        }


def report() -> DxEOpenReport:
    """Replay the open-side identities and enclose the dx_e factors."""
    identities = identity_verdicts()
    sample = sample_dx_e_open()
    try:
        enclosure = enclose_dx_e_open()
        stall = enclose_dx_e_open(use_bound=False)
        inside = bool(enclosure["finite"]) and _contains(
            Interval(float(enclosure["h_lo"]), float(enclosure["h_hi"])),
            sample,
        )
        proved = all(status == "PROVED" for status in identities.values())
        sealed = (
            proved
            and bool(enclosure["finite"])
            and bool(enclosure["net_below_declared"])
            and bool(enclosure["below_factor"])
            and bool(enclosure["extra_above_beta"])
            and bool(enclosure["excludes_zero"])
            and bool(enclosure["monotone"])
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
    return DxEOpenReport(
        identities=identities,
        sample_h={"lo": sample.lo, "hi": sample.hi},
        enclosure=enclosure,
        stall=stall,
        sample_inside=inside,
        dx_e_open=sealed,
        honesty=_honesty(dx_e_open=sealed),
    )
