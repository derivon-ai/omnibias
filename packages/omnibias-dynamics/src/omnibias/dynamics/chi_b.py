# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
"""Kill-line chi_b threshold from the Stage-A pre-rectangle time.

On lambda1 = -2 the slow-line integrand x/B_-(x) has antiderivative
whose log-sep pieces leave

    sep * S_pre = sep ln sep + r1 ln r1 - r2 ln r2
                  + r2 ln(1+theta) - r1 ln theta

with theta = 1/8. Interval arithmetic on two slabs covering
sep in [1/2^16, 1] encloses sep * S_pre < 3, so S_pre <= 3/sep.
Then kappa_b = 4/sep and chi_b = (sep/r1) kappa_b = 4/r1 < 9
since r1 >= 1/2. The decay constant of the chi-rectangle is
c = 1/16. The sep -> 0 limit of sep * S_pre is ln 9.

This is a sealed chi threshold on a declared compact, not
dx_e/dkappa, not Stage C, first-hit, G1, or Hilbert XVI.
The Stage-A wall is stage_a.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from fractions import Fraction

from omnibias.core.collapse.verdict import adjudicate_residual
from omnibias.core.verified.interval import Interval
from omnibias.core.verified.transcend import ln_iv
from omnibias.dynamics.honesty_inventory import build_honesty

__all__ = [
    "ChiBReport",
    "enclose_chi_b",
    "identity_verdicts",
    "report",
    "residual_c_decay",
    "residual_chi_declared",
    "residual_limit_nine",
    "sample_chi_b",
    "sep_times_spre",
]

_SAMPLE_SEP = Fraction(3, 5)
_THETA = Fraction(1, 8)
_K = Fraction(3)
_SEP_LO = Fraction(1, 2**16)
_SEP_MID = Fraction(1, 64)
_SEP_HI = Fraction(1)
_DECLARED_S = 3.0
_DECLARED_CHI = 9.0
_C_DECAY = Fraction(1, 16)


def _honesty(*, chi_b_bound: bool) -> dict[str, object]:
    return build_honesty(
        g1_passed=False,
        g4_passed=False,
        full_graphic_cyclicity_proved=False,
        full_hilbert16_solved=False,
        physical_return_membership_proved=False,
        orbit_continuation_on_kill=False,
        stage_a_wall=False,
        chi_b_bound=chi_b_bound,
        hk_theorem_24_used=False,
    )


def residual_limit_nine() -> Fraction:
    """``(1 + theta)/theta = 9`` at ``theta = 1/8``; the sep -> 0 limit of ``e^{sep S_pre}``."""
    return (1 + _THETA) / _THETA - 9


def residual_c_decay() -> Fraction:
    """Kill-line ``c = (lmin/8)(1-2 theta)/(2 sqrt(Lmax)+1) = 1/16``."""
    return (Fraction(1, 4) * Fraction(3, 4) / Fraction(3)) - _C_DECAY


def residual_chi_declared() -> Fraction:
    """Worst-case ``(K+1)/r1 = 8`` at ``K = 3``, ``r1 = 1/2``."""
    return 2 * (_K + 1) - 8


def _verdict(residual: Fraction) -> str:
    box = Interval.point(float(residual)) if residual == 0 else Interval.from_rational(residual)
    return adjudicate_residual(box, existential=False).status


def identity_verdicts() -> dict[str, str]:
    return {
        "limit_nine": _verdict(residual_limit_nine()),
        "c_decay": _verdict(residual_c_decay()),
        "chi_declared": _verdict(residual_chi_declared()),
    }


def _contains(outer: Interval, inner: Interval) -> bool:
    return outer.lo <= inner.lo and inner.hi <= outer.hi


def sep_times_spre(sep: Interval) -> Interval:
    """Sound enclosure of ``sep * S_pre`` on a positive ``sep`` box."""
    if sep.lo <= 0.0:
        raise ValueError("sep_times_spre requires sep.lo > 0")
    one = Interval.from_rational(Fraction(1))
    two = Interval.from_rational(Fraction(2))
    r1 = one - sep / two
    r2 = one + sep / two
    theta = Interval.from_rational(_THETA)
    one_theta = Interval.from_rational(1 + _THETA)
    return sep * ln_iv(sep) + r1 * ln_iv(r1) - r2 * ln_iv(r2) + r2 * ln_iv(one_theta) - r1 * ln_iv(theta)


def enclose_chi_b() -> dict[str, float | bool]:
    """Two-slab enclosure of ``sep * S_pre`` and ``chi_b = 4/r1``."""
    lo_box = sep_times_spre(Interval.hull(_SEP_LO, _SEP_MID))
    hi_box = sep_times_spre(Interval.hull(_SEP_MID, _SEP_HI))
    mag = max(lo_box.hi, hi_box.hi)
    sep = Interval.hull(_SEP_LO, _SEP_HI)
    one = Interval.from_rational(Fraction(1))
    r1 = one - sep / Interval.from_rational(Fraction(2))
    chi = Interval.from_rational(_K + 1) / r1
    c_box = Interval.from_rational(_C_DECAY)
    return {
        "s_lo": min(lo_box.lo, hi_box.lo),
        "s_hi": mag,
        "lo_slab_lo": lo_box.lo,
        "lo_slab_hi": lo_box.hi,
        "hi_slab_lo": hi_box.lo,
        "hi_slab_hi": hi_box.hi,
        "chi_lo": chi.lo,
        "chi_hi": chi.hi,
        "c_lo": c_box.lo,
        "c_hi": c_box.hi,
        "r1_lo": r1.lo,
        "r1_hi": r1.hi,
        "below_declared": mag < _DECLARED_S,
        "chi_below_declared": chi.hi < _DECLARED_CHI,
        "excludes_zero": mag > 0.0 and chi.lo > 0.0,
        "finite": mag < _DECLARED_S and chi.hi < _DECLARED_CHI,
        "sep_lo": float(_SEP_LO),
        "sep_hi": float(_SEP_HI),
    }


def sample_chi_b() -> Interval:
    """Sound ``sep * S_pre`` at ``sep = 3/5``."""
    return sep_times_spre(Interval.from_rational(_SAMPLE_SEP))


@dataclass(frozen=True)
class ChiBReport:
    """Kill-line chi_b threshold. Not dx_e/dkappa, Stage C, first-hit, or G1."""

    identities: Mapping[str, str]
    sample_s: Mapping[str, float]
    enclosure: Mapping[str, float | bool]
    sample_inside: bool
    chi_b_bound: bool
    honesty: Mapping[str, object]

    def to_payload(self) -> dict[str, object]:
        return {
            "schema": "hilbert16-chi-b-v1",
            "identities": dict(self.identities),
            "sample_s": dict(self.sample_s),
            "enclosure": dict(self.enclosure),
            "sample_inside": self.sample_inside,
            "chi_b_bound": self.chi_b_bound,
            "honesty": dict(self.honesty),
            "scope": (
                "Exact kill-line chi_b identities: (1+theta)/theta = 9, "
                "decay c = 1/16, and worst-case (K+1)/r1 = 8, plus a "
                "two-slab Interval enclosure sep*S_pre < 3 and "
                "chi_b < 9 on sep in [1/2^16, 1]. Not dx_e/dkappa, "
                "not Stage C, first-hit, G1, or Hilbert XVI."
            ),
        }


def report() -> ChiBReport:
    """Replay chi_b identities and enclose the kill-line threshold."""
    identities = identity_verdicts()
    sample = sample_chi_b()
    try:
        enclosure = enclose_chi_b()
        inside = bool(enclosure["finite"]) and _contains(
            Interval(float(enclosure["hi_slab_lo"]), float(enclosure["hi_slab_hi"])),
            sample,
        )
        sealed = (
            all(status == "PROVED" for status in identities.values())
            and bool(enclosure["below_declared"])
            and bool(enclosure["chi_below_declared"])
            and bool(enclosure["excludes_zero"])
            and inside
        )
    except (ValueError, ZeroDivisionError, ArithmeticError):
        enclosure = {
            "s_lo": float("nan"),
            "s_hi": float("nan"),
            "below_declared": False,
            "chi_below_declared": False,
            "excludes_zero": False,
            "finite": False,
        }
        inside = False
        sealed = False
    return ChiBReport(
        identities=identities,
        sample_s={"lo": sample.lo, "hi": sample.hi},
        enclosure=enclosure,
        sample_inside=inside,
        chi_b_bound=sealed,
        honesty=_honesty(chi_b_bound=sealed),
    )
