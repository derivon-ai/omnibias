# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
"""Kill-line Stage-C C=2 tight T(h) ratio from the edge factor.

On lambda1 = -2 the integrating-factor exponent at hmax = 1 is
6 eps log(1/eps). The map eps |-> eps log(1/eps) increases on
(0, 1/16], so the maximum is the compact edge. Interval

    (3/8) ln(16)

encloses that edge exponent and exp of it is < 3. Times the sealed
T_e/eps^2 < 1 this is a prefactor < 6. The slope 16/7 sits below 3.
Therefore

    T(h) <= 6 (eps^2 + h)

uniformly on the Stage-C compact, sharpening the coefficient-64
majorant. This is a sealed ratio bound, not T-h = O(eps), first-hit,
C2, dx_e off the kill line, G1, or Hilbert XVI. The coefficient-64
bound is stage_c_int. The lower envelope is stage_c_lo. The edge
logarithm is ln_iv(16).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from fractions import Fraction

from omnibias.core.collapse.verdict import adjudicate_residual
from omnibias.core.verified.interval import Interval
from omnibias.core.verified.transcend import exp_iv, ln_iv
from omnibias.dynamics.honesty_inventory import build_honesty
from omnibias.dynamics.stage_c_exit import enclose_stage_c_exit
from omnibias.dynamics.stage_c_int import enclose_stage_c_int

__all__ = [
    "StageCKReport",
    "enclose_stage_c_k",
    "identity_verdicts",
    "report",
    "residual_k_slope",
    "residual_six_eps",
    "residual_twice_three",
    "sample_stage_c_k",
]

_C = Fraction(2)
_CUBE = Fraction(3)
_SIX = Fraction(6)
_EPS_HI = Fraction(1, 16)
_PRE = Fraction(3, 8)
_THREE = Fraction(3)
_K = Fraction(6)
_SLOPE = Fraction(16, 7)
_K_SLOPE = Fraction(26, 7)
_DECLARED_F = 3.0
_DECLARED_K = 6.0


def _honesty(*, stage_c_k: bool) -> dict[str, object]:
    return build_honesty(
        g1_passed=False,
        g4_passed=False,
        full_graphic_cyclicity_proved=False,
        full_hilbert16_solved=False,
        physical_return_membership_proved=False,
        orbit_continuation_on_kill=False,
        stage_a_wall=False,
        chi_b_bound=False,
        dx_e_leading=False,
        dx_e_unif=False,
        stage_c_amin=False,
        stage_c_exit=False,
        stage_c_th=False,
        stage_c_gap=False,
        stage_c_env=False,
        stage_c_if=False,
        stage_c_int=False,
        stage_c_lo=False,
        stage_c_k=stage_c_k,
        hk_theorem_24_used=False,
    )


def residual_six_eps() -> Fraction:
    """``6 * (1/16) = 3/8``: edge prefactor of ``ln(1/eps)``."""
    return _C * _CUBE * _EPS_HI - _PRE


def residual_twice_three() -> Fraction:
    """``3 * 2 = 6``: declared factor times ``1 + T_e/eps^2`` cap."""
    return _THREE * 2 - _K


def residual_k_slope() -> Fraction:
    """``6 - 16/7 = 26/7``: declared K sits above the compact-edge slope."""
    return _K - _SLOPE - _K_SLOPE


def _verdict(residual: Fraction) -> str:
    box = Interval.point(float(residual)) if residual == 0 else Interval.from_rational(residual)
    return adjudicate_residual(box, existential=False).status


def identity_verdicts() -> dict[str, str]:
    return {
        "six_eps": _verdict(residual_six_eps()),
        "twice_three": _verdict(residual_twice_three()),
        "k_slope": _verdict(residual_k_slope()),
    }


def _contains(outer: Interval, inner: Interval) -> bool:
    return outer.lo <= inner.lo and inner.hi <= outer.hi


def enclose_stage_c_k() -> dict[str, float | bool]:
    """Stage-C C=2 tight T(h) <= 6 (eps^2+h) from the edge factor."""
    energy = enclose_stage_c_exit()
    slope = enclose_stage_c_int()
    ln = ln_iv(Interval.from_rational(Fraction(16)))
    expo = Interval.from_rational(_PRE) * ln
    factor = exp_iv(expo)
    te = Interval(0.0, float(energy["te_hi"]))
    pref = factor * (Interval.from_rational(1) + te)
    pref_hi = float(pref.hi)
    factor_hi = float(factor.hi)
    slope_hi = float(slope["slope_hi"])
    k_hi = max(pref_hi, slope_hi)
    return {
        "expo_lo": float(expo.lo),
        "expo_hi": float(expo.hi),
        "factor_lo": float(factor.lo),
        "factor_hi": factor_hi,
        "pref_hi": pref_hi,
        "slope_hi": slope_hi,
        "k_hi": k_hi,
        "te_hi": float(energy["te_hi"]),
        "below_declared": k_hi < _DECLARED_K and factor_hi < _DECLARED_F,
        "pref_below": pref_hi < _DECLARED_K,
        "excludes_zero": factor_hi > 1.0 and k_hi > 0.0,
        "finite": k_hi < _DECLARED_K and factor_hi < _DECLARED_F and pref_hi < _DECLARED_K,
    }


def sample_stage_c_k() -> Interval:
    """Sound compact-edge slope ``16/7`` inside the tight K box."""
    return Interval.from_rational(_SLOPE)


@dataclass(frozen=True)
class StageCKReport:
    """Kill-line Stage-C C=2 tight T(h) ratio. Not T-h=O(eps) or G1."""

    identities: Mapping[str, str]
    sample_slope: Mapping[str, float]
    enclosure: Mapping[str, float | bool]
    sample_inside: bool
    stage_c_k: bool
    honesty: Mapping[str, object]

    def to_payload(self) -> dict[str, object]:
        return {
            "schema": "hilbert16-stage-c-k-v1",
            "identities": dict(self.identities),
            "sample_slope": dict(self.sample_slope),
            "enclosure": dict(self.enclosure),
            "sample_inside": self.sample_inside,
            "stage_c_k": self.stage_c_k,
            "honesty": dict(self.honesty),
            "scope": (
                "Exact kill-line Stage-C C=2 tight-ratio identities: "
                "6*(1/16)=3/8, 3*2=6, and 6-16/7=26/7, plus Interval "
                "T(h) <= 6 (eps^2+h) from the edge factor. Not T-h=O(eps), "
                "first-hit, C2, G1, or Hilbert XVI."
            ),
        }


def report() -> StageCKReport:
    """Replay Stage-C C=2 tight-ratio identities and enclose K=6."""
    identities = identity_verdicts()
    sample = sample_stage_c_k()
    try:
        enclosure = enclose_stage_c_k()
        inside = bool(enclosure["finite"]) and _contains(
            Interval(2.0, float(enclosure["k_hi"])),
            sample,
        )
        sealed = (
            all(status == "PROVED" for status in identities.values())
            and bool(enclosure["below_declared"])
            and bool(enclosure["pref_below"])
            and bool(enclosure["excludes_zero"])
            and inside
        )
    except (ValueError, ZeroDivisionError, ArithmeticError):
        enclosure = {
            "expo_hi": float("nan"),
            "factor_hi": float("nan"),
            "pref_hi": float("nan"),
            "k_hi": float("nan"),
            "below_declared": False,
            "pref_below": False,
            "excludes_zero": False,
            "finite": False,
        }
        inside = False
        sealed = False
    return StageCKReport(
        identities=identities,
        sample_slope={"lo": sample.lo, "hi": sample.hi},
        enclosure=enclosure,
        sample_inside=inside,
        stage_c_k=sealed,
        honesty=_honesty(stage_c_k=sealed),
    )
