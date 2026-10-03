# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
"""Kill-line Stage-C C=2 T(h) integral majorant.

On lambda1 = -2 the comparison T_h <= C + C eps T/h + C eps^3/h with
C = 2 and integrating factor h^(-C eps) integrates to

    T(h) <= (h/h_1)^{C eps} T_e + (C/(1-alpha)) h + (h/h_1)^{C eps} eps^2

where alpha = C eps <= 1/8 on eps in (0, 1/16] and h_1 = eps^3,
hmax = 1. The sealed factor is < 32 and T_e/eps^2 < 1, so the
eps^2 coefficient is < 64. The slope C/(1-alpha) equals 16/7 at
the compact edge and Interval wrapping stays < 3. Therefore

    T(h) <= 64 (eps^2 + h)

on the C=2 comparison. This is a sealed Stage-C T(h) majorant, not
first-hit, C2, dx_e off the kill line, G1, or Hilbert XVI. The
integrating factor is stage_c_if. Exit energy is stage_c_exit.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from fractions import Fraction

from omnibias.core.collapse.verdict import adjudicate_residual
from omnibias.core.verified.interval import Interval
from omnibias.dynamics.honesty_inventory import build_honesty
from omnibias.dynamics.stage_c_exit import enclose_stage_c_exit
from omnibias.dynamics.stage_c_if import enclose_stage_c_if

__all__ = [
    "StageCIntReport",
    "enclose_stage_c_int",
    "identity_verdicts",
    "report",
    "residual_ceps_eight",
    "residual_inv_seven",
    "residual_one_minus",
    "sample_stage_c_int",
]

_C = Fraction(2)
_EPS_HI = Fraction(1, 16)
_ALPHA = Fraction(1, 8)
_ONE_M = Fraction(7, 8)
_SLOPE = Fraction(16, 7)
_DECLARED_C = 64.0
_DECLARED_SLOPE = 3.0


def _honesty(*, stage_c_int: bool) -> dict[str, object]:
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
        stage_c_int=stage_c_int,
        hk_theorem_24_used=False,
    )


def residual_ceps_eight() -> Fraction:
    """``2 * (1/16) = 1/8``: compact-edge ``alpha = C eps``."""
    return _C * _EPS_HI - _ALPHA


def residual_one_minus() -> Fraction:
    """``1 - 1/8 = 7/8``: compact-edge ``1 - alpha``."""
    return 1 - _ALPHA - _ONE_M


def residual_inv_seven() -> Fraction:
    """``2 / (7/8) = 16/7``: compact-edge slope ``C / (1-alpha)``."""
    return _C / _ONE_M - _SLOPE


def _verdict(residual: Fraction) -> str:
    box = Interval.point(float(residual)) if residual == 0 else Interval.from_rational(residual)
    return adjudicate_residual(box, existential=False).status


def identity_verdicts() -> dict[str, str]:
    return {
        "ceps_eight": _verdict(residual_ceps_eight()),
        "one_minus": _verdict(residual_one_minus()),
        "inv_seven": _verdict(residual_inv_seven()),
    }


def _contains(outer: Interval, inner: Interval) -> bool:
    return outer.lo <= inner.lo and inner.hi <= outer.hi


def enclose_stage_c_int() -> dict[str, float | bool]:
    """Stage-C C=2 T(h) <= 64 (eps^2+h) coefficient enclosure."""
    factor = enclose_stage_c_if()
    energy = enclose_stage_c_exit()
    f_box = Interval(1.0, float(factor["factor_hi"]))
    te = Interval(0.0, float(energy["te_hi"]))
    pref = f_box * (Interval.from_rational(1) + te)
    alpha = Interval.hull(Fraction(0), _ALPHA)
    one_m = Interval.from_rational(1) - alpha
    slope = Interval.from_rational(_C) / one_m
    pref_hi = float(pref.hi)
    slope_lo = float(slope.lo)
    slope_hi = float(slope.hi)
    c_env = max(pref_hi, slope_hi)
    return {
        "pref_hi": pref_hi,
        "slope_lo": slope_lo,
        "slope_hi": slope_hi,
        "c_env": c_env,
        "factor_hi": float(factor["factor_hi"]),
        "te_hi": float(energy["te_hi"]),
        "below_declared": c_env < _DECLARED_C and slope_hi < _DECLARED_SLOPE,
        "pref_below": pref_hi < _DECLARED_C,
        "excludes_zero": pref_hi > 0.0 and slope_lo > 0.0,
        "finite": c_env < _DECLARED_C and slope_hi < _DECLARED_SLOPE and pref_hi < _DECLARED_C,
    }


def sample_stage_c_int() -> Interval:
    """Sound compact-edge slope ``16/7`` at ``C = 2``, ``eps = 1/16``."""
    return Interval.from_rational(_SLOPE)


@dataclass(frozen=True)
class StageCIntReport:
    """Kill-line Stage-C C=2 T(h) majorant. Not first-hit or G1."""

    identities: Mapping[str, str]
    sample_slope: Mapping[str, float]
    enclosure: Mapping[str, float | bool]
    sample_inside: bool
    stage_c_int: bool
    honesty: Mapping[str, object]

    def to_payload(self) -> dict[str, object]:
        return {
            "schema": "hilbert16-stage-c-int-v1",
            "identities": dict(self.identities),
            "sample_slope": dict(self.sample_slope),
            "enclosure": dict(self.enclosure),
            "sample_inside": self.sample_inside,
            "stage_c_int": self.stage_c_int,
            "honesty": dict(self.honesty),
            "scope": (
                "Exact kill-line Stage-C C=2 T(h)-integral identities: "
                "C eps = 1/8, 1-alpha = 7/8, and slope 16/7, plus Interval "
                "T(h) <= 64 (eps^2+h). Not first-hit, C2, G1, or Hilbert XVI."
            ),
        }


def report() -> StageCIntReport:
    """Replay Stage-C C=2 T(h)-integral identities and enclose the majorant."""
    identities = identity_verdicts()
    sample = sample_stage_c_int()
    try:
        enclosure = enclose_stage_c_int()
        inside = bool(enclosure["finite"]) and _contains(
            Interval(float(enclosure["slope_lo"]), float(enclosure["slope_hi"])),
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
            "pref_hi": float("nan"),
            "slope_lo": float("nan"),
            "slope_hi": float("nan"),
            "c_env": float("nan"),
            "below_declared": False,
            "pref_below": False,
            "excludes_zero": False,
            "finite": False,
        }
        inside = False
        sealed = False
    return StageCIntReport(
        identities=identities,
        sample_slope={"lo": sample.lo, "hi": sample.hi},
        enclosure=enclosure,
        sample_inside=inside,
        stage_c_int=sealed,
        honesty=_honesty(stage_c_int=sealed),
    )
