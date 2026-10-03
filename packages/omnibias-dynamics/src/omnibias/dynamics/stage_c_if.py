# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
"""Kill-line Stage-C C=2 integrating factor (h/h_1)^{C eps}.

On lambda1 = -2 Stage C starts at y_1 = 1, so h_1 = eps^3. The
written comparison uses T_h <= C + C eps T/h + C eps^3/h with
integrating factor h^(-C eps). At C = 2 and hmax = 1 the exponent is

    C eps log(h/h_1) = 6 eps log(1/eps)

once log h <= 0. The public 3 eps log(1/eps) majorant 6 (sqrt(eps)-eps)
therefore lifts by C = 2 to 12 (sqrt(eps)-eps). On eps in (0, 1/16]
one has sqrt(eps) <= 1/4, so the exponent is at most 3 and
(h/h_1)^{C eps} < 32.

This is a sealed Stage-C C=2 integrating-factor bound on the kill
line, not T(h) <= C (eps^2+h) after the remaining integral, not
first-hit, C2, dx_e off the kill line, G1, or Hilbert XVI. The C=0
envelope is stage_c_env. The 3 eps log majorant is post_corridor.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from fractions import Fraction

from omnibias.core.collapse.verdict import adjudicate_residual
from omnibias.core.verified.interval import Interval
from omnibias.core.verified.transcend import exp_iv
from omnibias.dynamics.honesty_inventory import build_honesty
from omnibias.dynamics.post_corridor import integrating_factor_exponent_majorant

__all__ = [
    "StageCIfReport",
    "enclose_stage_c_if",
    "identity_verdicts",
    "report",
    "residual_c_triple",
    "residual_sqrt_edge",
    "residual_twice_six",
    "sample_stage_c_if",
]

_C = Fraction(2)
_CUBE = Fraction(3)
_BASE_SIX = Fraction(6)
_TWELVE = Fraction(12)
_EPS_HI = Fraction(1, 16)
_SQRT_CAP = Fraction(1, 4)
_EXPO_CAP = Fraction(3)
_EDGE = Fraction(9, 4)
_DECLARED_FACTOR = 32.0
_DECLARED_EXPO = 4.0


def _honesty(*, stage_c_if: bool) -> dict[str, object]:
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
        stage_c_if=stage_c_if,
        hk_theorem_24_used=False,
    )


def residual_c_triple() -> Fraction:
    """``2 * 3 = 6``: C times the three from ``h_1 = eps^3``."""
    return _C * _CUBE - _BASE_SIX


def residual_twice_six() -> Fraction:
    """``2 * 6 = 12``: C times the public ``6 (sqrt(eps)-eps)`` prefactor."""
    return _C * _BASE_SIX - _TWELVE


def residual_sqrt_edge() -> Fraction:
    """Edge exponent ``12 (1/4 - 1/16) = 9/4`` at ``eps = 1/16``."""
    return _TWELVE * (_SQRT_CAP - _EPS_HI) - _EDGE


def _verdict(residual: Fraction) -> str:
    box = Interval.point(float(residual)) if residual == 0 else Interval.from_rational(residual)
    return adjudicate_residual(box, existential=False).status


def identity_verdicts() -> dict[str, str]:
    return {
        "c_triple": _verdict(residual_c_triple()),
        "twice_six": _verdict(residual_twice_six()),
        "sqrt_edge": _verdict(residual_sqrt_edge()),
    }


def _contains(outer: Interval, inner: Interval) -> bool:
    return outer.lo <= inner.lo and inner.hi <= outer.hi


def enclose_stage_c_if() -> dict[str, float | bool]:
    """Stage-C C=2 integrating factor: exponent <= 3 and factor < 32."""
    expo = Interval.from_rational(_TWELVE) * Interval.from_rational(_SQRT_CAP)
    factor = exp_iv(Interval.hull(Fraction(0), _EXPO_CAP))
    edge = Interval.from_rational(2) * integrating_factor_exponent_majorant(
        Interval.from_rational(_EPS_HI)
    )
    factor_hi = float(factor.hi)
    expo_hi = float(expo.hi)
    return {
        "expo_hi": expo_hi,
        "factor_lo": factor.lo,
        "factor_hi": factor_hi,
        "edge_lo": edge.lo,
        "edge_hi": edge.hi,
        "sqrt_cap": float(_SQRT_CAP),
        "below_declared": factor_hi < _DECLARED_FACTOR and expo_hi <= _DECLARED_EXPO,
        "edge_below_cap": edge.hi <= expo_hi,
        "excludes_zero": factor_hi > 1.0 and expo_hi > 0.0,
        "finite": factor_hi < _DECLARED_FACTOR and expo_hi <= _DECLARED_EXPO,
    }


def sample_stage_c_if() -> Interval:
    """Sound edge exponent ``9/4`` at ``C = 2``, ``eps = 1/16``, ``h = 1``."""
    return Interval.from_rational(_EDGE)


@dataclass(frozen=True)
class StageCIfReport:
    """Kill-line Stage-C C=2 integrating factor. Not T(h) orbit, first-hit, or G1."""

    identities: Mapping[str, str]
    sample_expo: Mapping[str, float]
    enclosure: Mapping[str, float | bool]
    sample_inside: bool
    stage_c_if: bool
    honesty: Mapping[str, object]

    def to_payload(self) -> dict[str, object]:
        return {
            "schema": "hilbert16-stage-c-if-v1",
            "identities": dict(self.identities),
            "sample_expo": dict(self.sample_expo),
            "enclosure": dict(self.enclosure),
            "sample_inside": self.sample_inside,
            "stage_c_if": self.stage_c_if,
            "honesty": dict(self.honesty),
            "scope": (
                "Exact kill-line Stage-C C=2 integrating-factor identities: "
                "2*3=6, 2*6=12, and edge 12(1/4-1/16)=9/4, plus Interval "
                "exponent <= 3 and (h/h_1)^{C eps} < 32 on eps in (0, 1/16]. "
                "Not T(h)<=C(eps^2+h) after the remaining integral, first-hit, "
                "C2, G1, or Hilbert XVI."
            ),
        }


def report() -> StageCIfReport:
    """Replay Stage-C C=2 integrating-factor identities and enclose the factor."""
    identities = identity_verdicts()
    sample = sample_stage_c_if()
    try:
        enclosure = enclose_stage_c_if()
        inside = bool(enclosure["finite"]) and _contains(
            Interval(0.0, float(enclosure["expo_hi"])),
            sample,
        )
        sealed = (
            all(status == "PROVED" for status in identities.values())
            and bool(enclosure["below_declared"])
            and bool(enclosure["edge_below_cap"])
            and bool(enclosure["excludes_zero"])
            and inside
        )
    except (ValueError, ZeroDivisionError, ArithmeticError):
        enclosure = {
            "expo_hi": float("nan"),
            "factor_hi": float("nan"),
            "below_declared": False,
            "edge_below_cap": False,
            "excludes_zero": False,
            "finite": False,
        }
        inside = False
        sealed = False
    return StageCIfReport(
        identities=identities,
        sample_expo={"lo": sample.lo, "hi": sample.hi},
        enclosure=enclosure,
        sample_inside=inside,
        stage_c_if=sealed,
        honesty=_honesty(stage_c_if=sealed),
    )
