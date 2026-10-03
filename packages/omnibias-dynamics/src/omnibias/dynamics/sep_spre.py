# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
"""Kill-line ``sep * S_pre`` on every ``sep`` in ``(0, 1]``.

``sep * ln sep -> 0`` as ``sep -> 0``, and the smooth part of
``sep * S_pre`` tends to ``ln 9 < 11/5``. Dyadic slabs from
``2^{-48}`` to ``1``, plus a tail on ``(0, 2^{-48}]`` that drops the
non-positive ``sep * ln sep``, enclose ``sep * S_pre < 11/5 < 3``.
On the same interval ``r1 >= 1/2``, so ``chi_b = 4/r1 <= 8 < 9``.

The ``dx_e`` log remainder ``(3/16) eps sep (2 ln sep + ln(1/16))``
on ``eps in [0, 1/16]`` is enclosed on the same slabs. Its lower
bound stays above the value needed for a net exponent ``> 1/8``
once ``sep * S_pre < 11/5``. Replacing that cap by ``4`` stalls.
``ln(1/16) < -2`` is the witness that the remainder decreases in
``sep`` on ``(0, 1]``.

This removes the ``sep in (0, 1/2^{16})`` hole from the kill-line
chi threshold and from the kill-line leading net exponent. It is
not ``dx_e`` off the kill line, not a uniform-in-chi bound, not
Stage C, first-hit, G1, or Hilbert XVI. The compact chi threshold
is ``chi_b``. The compact leading factors are ``dx_e_leading``.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from fractions import Fraction

from omnibias.core.collapse.verdict import adjudicate_residual
from omnibias.core.verified.interval import Interval
from omnibias.core.verified.transcend import ln_iv
from omnibias.dynamics.chi_b import sample_chi_b, sep_times_spre
from omnibias.dynamics.honesty_inventory import build_honesty
from omnibias.dynamics.stage_a import enclose_stage_a

__all__ = [
    "SepSpreReport",
    "enclose_sep_spre",
    "identity_verdicts",
    "report",
    "residual_spre_chi",
    "residual_spre_net",
    "residual_spre_tail",
    "sample_sep_spre",
]

_TAIL_POW = 48
_TAIL = Fraction(1, 2**_TAIL_POW)
_THETA = Fraction(1, 8)
_ONE_THETA = Fraction(9, 8)
_S_CAP = Fraction(11, 5)
_CHI_WALL = 8
_CHI_DECLARED = 9
_DECLARED_NET = 0.125
_DECLARED_PREF = 0.5
_WALL = Fraction(9, 64)
_COEFF = Fraction(3, 16)
_EPS_HI = Fraction(1, 16)
_MU = Fraction(1, 16)
_KAPPA_NUM = Fraction(4)
_STALL_S = 4.0
_SAMPLE_SEP = Fraction(3, 5)


def _honesty(*, sep_spre: bool) -> dict[str, object]:
    return build_honesty(
        g1_passed=False,
        g4_passed=False,
        full_graphic_cyclicity_proved=False,
        full_hilbert16_solved=False,
        physical_return_membership_proved=False,
        orbit_continuation_on_kill=False,
        outgoing_first_hit=False,
        chi_b_bound=False,
        dx_e_leading=False,
        sep_spre=sep_spre,
        hk_theorem_24_used=False,
    )


def residual_spre_tail() -> Fraction:
    """``2^{48} * 2^{-48} = 1``: the tail starts at the last dyadic node."""
    return Fraction(2**_TAIL_POW) * _TAIL - 1


def residual_spre_chi() -> Fraction:
    """``4 / (1/2) = 8``: ``chi_b`` at ``sep = 1``, where ``r1`` is smallest."""
    return Fraction(4) / Fraction(1, 2) - _CHI_WALL


def residual_spre_net() -> Fraction:
    """``(3/16)(4 - 11/5) = 27/80``: net floor under the ``sep * S_pre`` cap."""
    return _COEFF * (_KAPPA_NUM - _S_CAP) - Fraction(27, 80)


def _verdict(residual: Fraction) -> str:
    box = Interval.point(float(residual)) if residual == 0 else Interval.from_rational(residual)
    return adjudicate_residual(box, existential=False).status


def identity_verdicts() -> dict[str, str]:
    return {
        "spre_tail": _verdict(residual_spre_tail()),
        "spre_chi": _verdict(residual_spre_chi()),
        "spre_net": _verdict(residual_spre_net()),
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


def _tail_smooth() -> Interval:
    """Smooth part of ``sep * S_pre`` on ``sep in (0, 2^{-48}]``.

    ``sep * ln sep <= 0`` on this tail, so the sum is at most this image.
    """
    r1 = Interval.hull(1 - _TAIL / 2, Fraction(1))
    r2 = Interval.hull(Fraction(1), 1 + _TAIL / 2)
    theta = Interval.from_rational(_THETA)
    one_theta = Interval.from_rational(_ONE_THETA)
    return r1 * ln_iv(r1) - r2 * ln_iv(r2) + r2 * ln_iv(one_theta) - r1 * ln_iv(theta)


def _tail_remainder() -> Interval:
    """Lower bound of the log remainder on ``sep in (0, 2^{-48}]``.

    On ``(0, 1/e]``, ``-s ln s`` increases with ``s``, so
    ``|s ln s| <= 48 * 2^{-48} * ln 2``. ``ln(1/16) < 0`` makes
    ``s ln(1/16)`` most negative at the right endpoint.
    """
    ln2 = ln_iv(Interval.from_rational(Fraction(2)))
    ln_mu = ln_iv(Interval.from_rational(_MU))
    mag = Interval.from_rational(_TAIL_POW) * Interval.from_rational(_TAIL) * ln2
    two = Interval.from_rational(Fraction(2))
    s_ln = -two * mag
    s_mu = Interval.from_rational(_TAIL) * ln_mu
    h_lo = s_ln.lo + s_mu.lo
    return Interval.from_rational(_COEFF * _EPS_HI) * Interval(h_lo, 0.0)


def enclose_sep_spre(*, use_bound: bool = True) -> dict[str, float | bool | int]:
    """Dyadic-plus-tail enclosure of ``sep * S_pre`` and the net exponent.

    ``use_bound=False`` feeds ``S = 4`` into the net floor and stalls.
    """
    s_hi = float("-inf")
    s_lo = float("inf")
    rem_min = float("inf")
    for k in range(_TAIL_POW):
        if k == 0:
            slab = Interval.hull(Fraction(1, 2), Fraction(1))
        else:
            slab = Interval.hull(Fraction(1, 2 ** (k + 1)), Fraction(1, 2**k))
        box = sep_times_spre(slab)
        rem = _log_remainder(slab)
        s_hi = max(s_hi, box.hi)
        s_lo = min(s_lo, box.lo)
        rem_min = min(rem_min, rem.lo)
    smooth = _tail_smooth()
    s_hi = max(s_hi, smooth.hi)
    s_lo = min(s_lo, smooth.lo)
    rem_min = min(rem_min, _tail_remainder().lo)
    cap = Interval.from_rational(_S_CAP)
    ln_mu = ln_iv(Interval.from_rational(_MU))
    s_for_net = s_hi if use_bound else _STALL_S
    gap = Interval.from_rational(_KAPPA_NUM) - Interval(0.0, s_for_net)
    net = Interval.from_rational(_COEFF) * gap
    after = net.lo + rem_min
    walls = enclose_stage_a()
    a = Interval(float(walls["a_lo"]), float(walls["a_hi"]))
    pref = Interval.from_rational(_WALL) / a
    s_below = s_hi < cap.lo
    chi_below = _CHI_WALL < _CHI_DECLARED
    decreasing = ln_mu.hi < -2.0
    net_ok = after > _DECLARED_NET
    pref_ok = float(pref.hi) < _DECLARED_PREF
    finite = bool(s_below and chi_below and decreasing and net_ok and pref_ok and use_bound)
    return {
        "s_lo": s_lo,
        "s_hi": s_hi,
        "s_cap": float(_S_CAP),
        "below_cap": s_below,
        "chi_wall": _CHI_WALL,
        "chi_below_declared": chi_below,
        "decreasing": decreasing,
        "ln_mu_hi": ln_mu.hi,
        "rem_lo": rem_min,
        "net_lo": net.lo,
        "after_lo": after,
        "net_below_declared": net_ok,
        "pref_hi": float(pref.hi),
        "pref_below_declared": pref_ok,
        "slabs": _TAIL_POW,
        "tail_pow": _TAIL_POW,
        "excludes_zero": s_hi > 0.0 and after > 0.0 and pref.lo > 0.0,
        "finite": finite,
    }


def sample_sep_spre() -> Interval:
    """Sound ``sep * S_pre`` at ``sep = 3/5``, inside the full-interval cap."""
    return sample_chi_b()


@dataclass(frozen=True)
class SepSpreReport:
    """Kill-line ``sep * S_pre`` on ``(0, 1]``. Not off-kill ``dx_e`` or G1."""

    identities: Mapping[str, str]
    sample_s: Mapping[str, float]
    enclosure: Mapping[str, float | bool | int]
    stall: Mapping[str, float | bool | int]
    sample_inside: bool
    sep_spre: bool
    honesty: Mapping[str, object]

    def to_payload(self) -> dict[str, object]:
        return {
            "schema": "hilbert16-sep-spre-v1",
            "identities": dict(self.identities),
            "sample_s": dict(self.sample_s),
            "enclosure": dict(self.enclosure),
            "stall": dict(self.stall),
            "sample_inside": self.sample_inside,
            "sep_spre": self.sep_spre,
            "outgoing_first_hit": False,
            "honesty": dict(self.honesty),
            "scope": (
                "Exact kill-line identities: 2^48 * 2^{-48} = 1, "
                "4/(1/2) = 8, and (3/16)(4 - 11/5) = 27/80, plus a "
                "dyadic-plus-tail enclosure sep*S_pre < 11/5 and "
                "chi_b <= 8 on every sep in (0, 1]. The dx_e log "
                "remainder on that interval keeps the net exponent "
                "above 1/8. Feeding S = 4 into the net floor stalls. "
                "Not dx_e off the kill line, not uniform-in-chi, "
                "not Stage C, first-hit, G1, or Hilbert XVI."
            ),
        }


def report() -> SepSpreReport:
    """Replay the full-interval identities and enclose ``sep * S_pre``."""
    identities = identity_verdicts()
    sample = sample_sep_spre()
    try:
        enclosure = enclose_sep_spre()
        stall = enclose_sep_spre(use_bound=False)
        inside = bool(enclosure["finite"]) and _contains(
            Interval(float(enclosure["s_lo"]), float(enclosure["s_hi"])),
            sample,
        )
        proved = all(status == "PROVED" for status in identities.values())
        sealed = (
            proved
            and bool(enclosure["finite"])
            and bool(enclosure["below_cap"])
            and bool(enclosure["chi_below_declared"])
            and bool(enclosure["decreasing"])
            and bool(enclosure["net_below_declared"])
            and bool(enclosure["pref_below_declared"])
            and bool(enclosure["excludes_zero"])
            and inside
            and not bool(stall["finite"])
        )
    except (ValueError, ZeroDivisionError, ArithmeticError):
        enclosure = {
            "s_hi": float("nan"),
            "after_lo": float("nan"),
            "below_cap": False,
            "finite": False,
            "net_below_declared": False,
        }
        stall = {"finite": True, "net_below_declared": True}
        inside = False
        sealed = False
    return SepSpreReport(
        identities=identities,
        sample_s={"lo": sample.lo, "hi": sample.hi, "sep": float(_SAMPLE_SEP)},
        enclosure=enclosure,
        stall=stall,
        sample_inside=inside,
        sep_spre=sealed,
        honesty=_honesty(sep_spre=sealed),
    )
