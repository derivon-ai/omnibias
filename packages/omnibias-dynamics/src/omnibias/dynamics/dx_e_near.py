# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
"""dx_e leading factors for every ``lambda1`` in ``[-3/2, -2)``.

On ``rstar = -lambda1/2`` in ``[3/4, 1)`` and ``sep`` in ``(0, 1]``,

    a = rstar - (5/8) sep >= 1/8,
    r1 = rstar - sep/2 >= 1/4,
    u = sep/r1 <= 4.

The ratio ``h(u)`` stays below ``11/5`` on ``(0, 4]``. The residence
bound ``X <= 2 rstar`` and the slope ``-3 sep/8`` give the
tau-coefficient ``3 sep/(16 rstar)``. The threshold
``kappa * sep = r1*(11/5) + rstar`` leaves the main exponent ``3/16``.
The log remainder is most negative at the corner ``rstar = 3/4``,
``sep = 1``, ``eps = 1/16``, and the net stays above ``1/8``. Then
``chi_b <= 26/5``, the extra coefficient is at least ``1/16``, the
prefactor is at most ``9/8``, and ``C < 2``. Dropping the ``rstar``
surplus stalls.

This is the slab of ``lambda1`` in ``[-3/2, -2)``. It does not cover
``lambda1`` in ``(-3/2, 0)``. It is not Stage C, first-hit, G1, or
Hilbert XVI. The ray ``lambda1 <= -2`` is ``dx_e_ray``.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from fractions import Fraction

from omnibias.core.collapse.verdict import adjudicate_residual
from omnibias.core.verified.interval import Interval
from omnibias.core.verified.transcend import exp_iv, ln_iv
from omnibias.dynamics.dx_e_off import enclose_dx_e_off
from omnibias.dynamics.honesty_inventory import build_honesty

__all__ = [
    "DxENearReport",
    "enclose_dx_e_near",
    "identity_verdicts",
    "report",
    "residual_near_chi",
    "residual_near_u",
    "residual_near_wall",
    "sample_dx_e_near",
]

_RSTAR_EDGE = Fraction(3, 4)
_A_MIN = Fraction(1, 8)
_R1_MIN = Fraction(1, 4)
_U_HI = Fraction(4)
_H_CAP = Fraction(11, 5)
_CHI_CAP = Fraction(26, 5)
_PREF = Fraction(9, 8)
_EXTRA = Fraction(1, 16)
_NET = Fraction(3, 16)
_MU = Fraction(1, 16)
_DECLARED_NET = 0.125
_DECLARED_C = 2.0
_DECLARED_EXTRA = 0.05
_SAMPLE_U = Fraction(3)


def _honesty(*, dx_e_near: bool) -> dict[str, object]:
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
        dx_e_near=dx_e_near,
        hk_theorem_24_used=False,
    )


def residual_near_wall() -> Fraction:
    """``3/4 - 5/8 = 1/8``: left wall at ``rstar = 3/4``, ``sep = 1``."""
    return _RSTAR_EDGE - Fraction(5, 8) - _A_MIN


def residual_near_u() -> Fraction:
    """``1 / (1/4) = 4``: largest ``sep/r1`` on the rectangle."""
    return Fraction(1) / _R1_MIN - _U_HI


def residual_near_chi() -> Fraction:
    """``11/5 + 3 = 26/5``: chi cap at ``rstar/r1 = 3``."""
    return _H_CAP + (_RSTAR_EDGE / _R1_MIN) - _CHI_CAP


def _verdict(residual: Fraction) -> str:
    box = Interval.point(float(residual)) if residual == 0 else Interval.from_rational(residual)
    return adjudicate_residual(box, existential=False).status


def identity_verdicts() -> dict[str, str]:
    return {
        "near_wall": _verdict(residual_near_wall()),
        "near_u": _verdict(residual_near_u()),
        "near_chi": _verdict(residual_near_chi()),
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


def _h_above_two() -> tuple[float, float]:
    """Upper and lower bounds of ``h`` on ``(2, 4]``."""
    h_hi = float("-inf")
    h_lo = float("inf")
    i = 32
    while Fraction(i, 16) < _U_HI:
        slab = Interval.hull(Fraction(i, 16), Fraction(i + 1, 16))
        box = _h_box(slab)
        h_hi = max(h_hi, box.hi)
        h_lo = min(h_lo, box.lo)
        i += 1
    return h_lo, h_hi


def _corner_remainder() -> Interval:
    """Most negative log remainder, at ``rstar = 3/4`` and ``sep = 1``.

    ``ln(1/16) < -2`` forces ``sep (2 ln sep + ln mu)`` to decrease on
    ``(0, 1]``, and ``1/rstar <= 4/3``, so the corner coefficient
    ``(3/16)*(4/3)*(1/16) = 1/64`` is the worst.
    """
    ln_mu = ln_iv(Interval.from_rational(_MU))
    if not (ln_mu.hi < -2.0):
        raise ValueError("log remainder is not monotone on (0, 1]")
    return Interval.from_rational(Fraction(1, 64)) * ln_mu


def sample_dx_e_near() -> Interval:
    """Sound ``h(3)`` on the slab ``u in (2, 4]``."""
    return _h_box(Interval.from_rational(_SAMPLE_U))


def enclose_dx_e_near(*, use_bound: bool = True) -> dict[str, float | bool]:
    """Net exponent and ``C`` for every ``lambda1`` in ``[-3/2, -2)``.

    ``use_bound=False`` drops the ``rstar`` surplus and stalls.
    """
    prior = enclose_dx_e_off()
    lo_hi, hi_hi = _h_above_two()
    h_hi = max(float(prior["h_hi"]), hi_hi)
    h_lo = min(float(prior["h_lo"]), lo_hi)
    cap = Interval.from_rational(_H_CAP)
    below = bool(prior["below_cap"]) and h_hi < cap.lo and hi_hi < cap.lo
    rem = _corner_remainder()
    main = _NET if use_bound else Fraction(0)
    after = float(main) + rem.lo
    net_ok = after > _DECLARED_NET
    pref = Interval.from_rational(_PREF)
    lift = Interval.from_rational(_EXTRA) * Interval.from_rational(_CHI_CAP)
    c_hi = float((pref * exp_iv(Interval.point(-after)) * exp_iv(lift)).hi)
    extra_lo = float(Interval.from_rational(_EXTRA).lo)
    finite = bool(
        below and net_ok and c_hi < _DECLARED_C and extra_lo > _DECLARED_EXTRA and use_bound
    )
    return {
        "h_lo": h_lo,
        "h_hi": h_hi,
        "rem_lo": rem.lo,
        "after_lo": after,
        "below_cap": below,
        "net_below_declared": net_ok,
        "chi_hi": float(_CHI_CAP),
        "c_hi": c_hi,
        "extra_lo": extra_lo,
        "below_c": c_hi < _DECLARED_C,
        "extra_above_floor": extra_lo > _DECLARED_EXTRA,
        "excludes_zero": after > 0.0 and c_hi > 0.0,
        "finite": finite,
    }


@dataclass(frozen=True)
class DxENearReport:
    """dx_e factors for every ``lambda1`` in ``[-3/2, -2)``. Not G1."""

    identities: Mapping[str, str]
    sample_h: Mapping[str, float]
    enclosure: Mapping[str, float | bool]
    stall: Mapping[str, float | bool]
    sample_inside: bool
    dx_e_near: bool
    honesty: Mapping[str, object]

    def to_payload(self) -> dict[str, object]:
        return {
            "schema": "hilbert16-dx-e-near-v1",
            "identities": dict(self.identities),
            "sample_h": dict(self.sample_h),
            "enclosure": dict(self.enclosure),
            "stall": dict(self.stall),
            "sample_inside": self.sample_inside,
            "dx_e_near": self.dx_e_near,
            "outgoing_first_hit": False,
            "honesty": dict(self.honesty),
            "scope": (
                "Exact identities: 3/4-5/8=1/8, 1/(1/4)=4, and "
                "11/5+3=26/5. For every lambda1 in [-3/2, -2) and "
                "every sep in (0, 1] the net exponent stays above "
                "1/8, chi_b<=26/5, and C<2. Dropping the rstar "
                "surplus stalls. Not lambda1 in (-3/2, 0), not "
                "Stage C, first-hit, G1, or Hilbert XVI."
            ),
        }


def report() -> DxENearReport:
    """Replay the near-side identities and enclose the dx_e factors."""
    identities = identity_verdicts()
    sample = sample_dx_e_near()
    try:
        enclosure = enclose_dx_e_near()
        stall = enclose_dx_e_near(use_bound=False)
        inside = bool(enclosure["finite"]) and _contains(
            Interval(float(enclosure["h_lo"]), float(enclosure["h_hi"])),
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
    return DxENearReport(
        identities=identities,
        sample_h={"lo": sample.lo, "hi": sample.hi},
        enclosure=enclosure,
        stall=stall,
        sample_inside=inside,
        dx_e_near=sealed,
        honesty=_honesty(dx_e_near=sealed),
    )
