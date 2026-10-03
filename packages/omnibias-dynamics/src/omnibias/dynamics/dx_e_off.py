# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
"""dx_e leading factors on ``lambda1`` in ``[-4, -2]``.

The ratio ``sep * S_pre / r1`` depends only on ``u = sep / r1``:

    h(u) = u ln u - (1+u) ln(1+u) + (1+u) ln(9/8) - ln(1/8).

Slabs on ``u in (0, 2]`` enclose ``h(u) < 11/5``. For every
``rstar = -lambda1/2`` in ``[1, 2]`` and every ``sep`` in ``(0, 1]``,
``r1 = rstar - sep/2`` stays at least ``1/2``, so ``u <= 2`` and

    sep * S_pre < r1 * (11/5),
    a = r1 - sep/8 >= 3/8,
    bnd = r1 + sep/8 <= rstar <= 2.

The kill-line slope bound ``-3 sep/8`` and ``X <= 2`` therefore still
give the tau-coefficient ``3 sep/16``. The threshold
``kappa * sep = r1*(11/5) + rstar/2`` leaves a net exponent above
``3/16`` before the ``y0`` log remainder, and above ``1/8`` after it.
Then ``chi_b <= 21/5`` and the factored majorant has ``C < 2``.

``lambda1 = -2`` is the edge ``rstar = 1``. Every ``lambda1`` in
``[-4, -2)`` is off that edge and is included. Feeding the cap ``1``
in place of ``11/5`` stalls. This does not cover ``lambda1 < -4`` or
``lambda1 in (-2, 0)``, and it is not Stage C, first-hit, G1, or
Hilbert XVI. The kill-line factors are ``dx_e_leading``.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from fractions import Fraction

from omnibias.core.collapse.verdict import adjudicate_residual
from omnibias.core.verified.interval import Interval
from omnibias.core.verified.transcend import exp_iv, ln_iv
from omnibias.dynamics.honesty_inventory import build_honesty

__all__ = [
    "DxEOffReport",
    "enclose_dx_e_off",
    "identity_verdicts",
    "report",
    "residual_off_chi",
    "residual_off_u",
    "residual_off_wall",
    "sample_dx_e_off",
]

_THETA = Fraction(1, 8)
_TAIL_POW = 48
_TAIL = Fraction(1, 2**_TAIL_POW)
_U_HI = Fraction(2)
_H_CAP = Fraction(11, 5)
_CHI_CAP = Fraction(21, 5)
_A_MIN = Fraction(3, 8)
_WALL = Fraction(9, 64)
_PREF = Fraction(3, 8)
_COEFF = Fraction(3, 16)
_EXTRA = Fraction(3, 32)
_NET = Fraction(3, 16)
_EPS_HI = Fraction(1, 16)
_MU = Fraction(1, 16)
_DECLARED_NET = 0.125
_DECLARED_C = 2.0
_DECLARED_EXTRA = 0.0625
_SAMPLE_U = Fraction(1, 2)
_STALL_CAP = Fraction(1)


def _honesty(*, dx_e_off: bool) -> dict[str, object]:
    return build_honesty(
        g1_passed=False,
        g4_passed=False,
        full_graphic_cyclicity_proved=False,
        full_hilbert16_solved=False,
        physical_return_membership_proved=False,
        orbit_continuation_on_kill=False,
        outgoing_first_hit=False,
        dx_e_leading=False,
        dx_e_unif=False,
        dx_e_off=dx_e_off,
        hk_theorem_24_used=False,
    )


def residual_off_wall() -> Fraction:
    """``1 - 5/8 = 3/8``: left wall at ``rstar = 1``, ``sep = 1``."""
    return 1 - (Fraction(1, 2) + _THETA) - _A_MIN


def residual_off_u() -> Fraction:
    """``1 / (1/2) = 2``: largest ``sep/r1`` on the rectangle."""
    return Fraction(1) / Fraction(1, 2) - _U_HI


def residual_off_chi() -> Fraction:
    """``11/5 + 2 = 21/5``: chi cap at ``r1 = 1/2``."""
    return _H_CAP + 2 - _CHI_CAP


def _verdict(residual: Fraction) -> str:
    box = Interval.point(float(residual)) if residual == 0 else Interval.from_rational(residual)
    return adjudicate_residual(box, existential=False).status


def identity_verdicts() -> dict[str, str]:
    return {
        "off_wall": _verdict(residual_off_wall()),
        "off_u": _verdict(residual_off_u()),
        "off_chi": _verdict(residual_off_chi()),
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
        - ln_iv(Interval.from_rational(_THETA))
    )


def _tail_h() -> Interval:
    """Upper bound of ``h`` on ``(0, 2^{-48}]``.

    ``u ln u <= 0`` and ``-(1+u) ln(1+u) <= 0``, so ``h`` is at most
    ``(1+u) ln(9/8) - ln(1/8)``.
    """
    one_u = Interval.hull(Fraction(1), 1 + _TAIL)
    return one_u * ln_iv(Interval.from_rational(Fraction(9, 8))) - ln_iv(
        Interval.from_rational(_THETA)
    )


def _log_remainder(sep: Interval) -> Interval:
    """``(3/16) eps sep (2 ln sep + ln mu)`` on ``eps in [0, 1/16]``."""
    if sep.lo <= 0.0:
        raise ValueError("_log_remainder requires sep.lo > 0")
    eps = Interval.hull(Fraction(0), _EPS_HI)
    ln_mu = ln_iv(Interval.from_rational(_MU))
    two = Interval.from_rational(Fraction(2))
    return Interval.from_rational(_COEFF) * eps * sep * (two * ln_iv(sep) + ln_mu)


def _remainder_min() -> float:
    rem_min = float("inf")
    for k in range(_TAIL_POW):
        if k == 0:
            slab = Interval.hull(Fraction(1, 2), Fraction(1))
        else:
            slab = Interval.hull(Fraction(1, 2 ** (k + 1)), Fraction(1, 2**k))
        rem_min = min(rem_min, _log_remainder(slab).lo)
    ln2 = ln_iv(Interval.from_rational(Fraction(2)))
    ln_mu = ln_iv(Interval.from_rational(_MU))
    mag = Interval.from_rational(_TAIL_POW) * Interval.from_rational(_TAIL) * ln2
    h_lo = (-Interval.from_rational(Fraction(2)) * mag).lo + (
        Interval.from_rational(_TAIL) * ln_mu
    ).lo
    tail = Interval.from_rational(_COEFF * _EPS_HI) * Interval(h_lo, 0.0)
    return min(rem_min, tail.lo)


def sample_dx_e_off() -> Interval:
    """Sound ``h(1/2)``."""
    return _h_box(Interval.from_rational(_SAMPLE_U))


def enclose_dx_e_off(*, use_bound: bool = True) -> dict[str, float | bool | int]:
    """Enclosure of ``h``, the net exponent, and the factored ``C``.

    ``use_bound=False`` compares ``h`` with ``1`` and stalls.
    """
    h_hi = float("-inf")
    h_lo = float("inf")
    n_slabs = 0
    for k in range(4, _TAIL_POW):
        slab = Interval.hull(Fraction(1, 2 ** (k + 1)), Fraction(1, 2**k))
        box = _h_box(slab)
        h_hi = max(h_hi, box.hi)
        h_lo = min(h_lo, box.lo)
        n_slabs += 1
    i = 1
    while Fraction(i, 16) < _U_HI:
        slab = Interval.hull(Fraction(i, 16), Fraction(i + 1, 16))
        box = _h_box(slab)
        h_hi = max(h_hi, box.hi)
        h_lo = min(h_lo, box.lo)
        n_slabs += 1
        i += 1
    tail = _tail_h()
    h_hi = max(h_hi, tail.hi)
    h_lo = min(h_lo, tail.lo)
    cap = _H_CAP if use_bound else _STALL_CAP
    cap_iv = Interval.from_rational(cap)
    below = h_hi < cap_iv.lo
    rem_min = _remainder_min()
    after = float(_NET) + rem_min
    net_ok = after > _DECLARED_NET
    chi_hi = float(_CHI_CAP)
    pref = Interval.from_rational(_PREF)
    e_th = exp_iv(Interval.point(-after))
    lift = Interval.from_rational(_EXTRA) * Interval(0.0, chi_hi)
    c_box = pref * e_th * exp_iv(lift)
    c_hi = float(c_box.hi)
    extra_lo = float(Interval.from_rational(_EXTRA).lo)
    finite = bool(
        below and net_ok and c_hi < _DECLARED_C and extra_lo > _DECLARED_EXTRA and use_bound
    )
    return {
        "h_lo": h_lo,
        "h_hi": h_hi,
        "h_cap": float(_H_CAP),
        "below_cap": below,
        "slabs": n_slabs,
        "rem_lo": rem_min,
        "after_lo": after,
        "net_below_declared": net_ok,
        "chi_hi": chi_hi,
        "pref_hi": float(pref.hi),
        "c_hi": c_hi,
        "extra_lo": extra_lo,
        "below_c": c_hi < _DECLARED_C,
        "extra_above_floor": extra_lo > _DECLARED_EXTRA,
        "a_min": float(_A_MIN),
        "excludes_zero": h_hi > 0.0 and after > 0.0 and c_hi > 0.0,
        "finite": finite,
    }


@dataclass(frozen=True)
class DxEOffReport:
    """dx_e factors on ``lambda1`` in ``[-4, -2]``. Not G1."""

    identities: Mapping[str, str]
    sample_h: Mapping[str, float]
    enclosure: Mapping[str, float | bool | int]
    stall: Mapping[str, float | bool | int]
    sample_inside: bool
    dx_e_off: bool
    honesty: Mapping[str, object]

    def to_payload(self) -> dict[str, object]:
        return {
            "schema": "hilbert16-dx-e-off-v1",
            "identities": dict(self.identities),
            "sample_h": dict(self.sample_h),
            "enclosure": dict(self.enclosure),
            "stall": dict(self.stall),
            "sample_inside": self.sample_inside,
            "dx_e_off": self.dx_e_off,
            "outgoing_first_hit": False,
            "honesty": dict(self.honesty),
            "scope": (
                "Exact identities: 1-5/8=3/8, 1/(1/2)=2, and "
                "11/5+2=21/5, plus an Interval enclosure h(u)<11/5 "
                "on u in (0, 2]. For lambda1 in [-4, -2] and sep in "
                "(0, 1], sep*S_pre < r1*(11/5), the net exponent "
                "stays above 1/8, chi_b <= 21/5, and C < 2. The cap "
                "1 stalls. Not lambda1 < -4, not lambda1 in (-2, 0), "
                "not Stage C, first-hit, G1, or Hilbert XVI."
            ),
        }


def report() -> DxEOffReport:
    """Replay the off-edge identities and enclose the dx_e factors."""
    identities = identity_verdicts()
    sample = sample_dx_e_off()
    try:
        enclosure = enclose_dx_e_off()
        stall = enclose_dx_e_off(use_bound=False)
        inside = bool(enclosure["finite"]) and _contains(
            Interval(float(enclosure["h_lo"]), float(enclosure["h_hi"])),
            sample,
        )
        proved = all(status == "PROVED" for status in identities.values())
        sealed = (
            proved
            and bool(enclosure["finite"])
            and bool(enclosure["below_cap"])
            and bool(enclosure["net_below_declared"])
            and bool(enclosure["below_c"])
            and bool(enclosure["extra_above_floor"])
            and bool(enclosure["excludes_zero"])
            and inside
            and not bool(stall["finite"])
            and _WALL / _A_MIN == _PREF
        )
    except (ValueError, ZeroDivisionError, ArithmeticError):
        enclosure = {
            "h_hi": float("nan"),
            "after_lo": float("nan"),
            "below_cap": False,
            "finite": False,
            "net_below_declared": False,
        }
        stall = {"finite": True}
        inside = False
        sealed = False
    return DxEOffReport(
        identities=identities,
        sample_h={"lo": sample.lo, "hi": sample.hi, "u": float(_SAMPLE_U)},
        enclosure=enclosure,
        stall=stall,
        sample_inside=inside,
        dx_e_off=sealed,
        honesty=_honesty(dx_e_off=sealed),
    )
