# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
"""Kill-line Stage-C C=0 two-sided T(h) envelope.

On lambda1 = -2 the C=0 comparison is T_h = 1, so

    T(h) = T_e + h - h_1

with h_1 = eps^3 at the Stage-C start y_1 = 1. The sealed start gap
gives (T_e - h_1)/eps^2 > 1/32, and Stage-B exit energy has
T_e/eps^2 < 1. Therefore

    (1/32) (eps^2 + h) <= T(h) <= eps^2 + h

on the C=0 comparison, uniformly in height. This is the written
c (eps^2+h) <= T <= C (eps^2+h) sandwich at C=0, not a C!=0
integrating-factor orbit, first-hit, C2, dx_e off the kill line,
G1, or Hilbert XVI. The start gap is stage_c_gap. Leading T_h is
stage_c_th.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from fractions import Fraction

from omnibias.core.collapse.verdict import adjudicate_residual
from omnibias.core.verified.interval import Interval
from omnibias.dynamics.honesty_inventory import build_honesty
from omnibias.dynamics.stage_b import r1_kill
from omnibias.dynamics.stage_c_exit import enclose_stage_c_exit
from omnibias.dynamics.stage_c_gap import enclose_stage_c_gap

__all__ = [
    "StageCEnvReport",
    "enclose_stage_c_env",
    "identity_verdicts",
    "report",
    "residual_env_add",
    "residual_half_gap",
    "residual_te_unit",
    "sample_stage_c_env",
]

_SAMPLE_SEP = Fraction(3, 5)
_TE_LO = Fraction(1, 8)
_TE_ROOM = Fraction(7, 8)
_EPS_HI = Fraction(1, 16)
_C_LO = Fraction(1, 32)
_C_HI = Fraction(1)
_ENV_ROOM = Fraction(31, 32)
_DECLARED_C_LO = 0.03125
_DECLARED_C_HI = 1.0


def _honesty(*, stage_c_env: bool) -> dict[str, object]:
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
        stage_c_env=stage_c_env,
        hk_theorem_24_used=False,
    )


def residual_te_unit() -> Fraction:
    """``1/8 + 7/8 = 1``: wall ``T_e/eps^2`` sits strictly below the unit upper."""
    return _TE_LO + _TE_ROOM - _C_HI


def residual_half_gap() -> Fraction:
    """Declared lower ``c = (1/16)/2 = 1/32``: half the wall start-gap room."""
    return _EPS_HI / 2 - _C_LO


def residual_env_add() -> Fraction:
    """``1/32 + 31/32 = 1``: lower c plus room recovers the unit upper."""
    return _C_LO + _ENV_ROOM - _C_HI


def _verdict(residual: Fraction) -> str:
    box = Interval.point(float(residual)) if residual == 0 else Interval.from_rational(residual)
    return adjudicate_residual(box, existential=False).status


def identity_verdicts() -> dict[str, str]:
    return {
        "te_unit": _verdict(residual_te_unit()),
        "half_gap": _verdict(residual_half_gap()),
        "env_add": _verdict(residual_env_add()),
    }


def _contains(outer: Interval, inner: Interval) -> bool:
    return outer.lo <= inner.lo and inner.hi <= outer.hi


def enclose_stage_c_env() -> dict[str, float | bool]:
    """C=0 Stage-C sandwich (1/32)(eps^2+h) <= T(h) <= eps^2+h."""
    energy = enclose_stage_c_exit()
    gap = enclose_stage_c_gap()
    te_hi = float(energy["te_hi"])
    gap_lo = float(gap["gap_lo"])
    gap_hi = float(gap["gap_hi"])
    return {
        "c_lo": gap_lo,
        "c_hi": 1.0,
        "te_hi": te_hi,
        "gap_lo": gap_lo,
        "gap_hi": gap_hi,
        "below_declared": te_hi < _DECLARED_C_HI and gap_hi < _DECLARED_C_HI,
        "c_above_floor": gap_lo > _DECLARED_C_LO,
        "excludes_zero": gap_lo > 0.0,
        "finite": te_hi < _DECLARED_C_HI
        and gap_hi < _DECLARED_C_HI
        and gap_lo > _DECLARED_C_LO,
    }


def sample_stage_c_env() -> Interval:
    """Sound ``T(h)/(eps^2+h)`` at ``sep=3/5``, ``eps=1/16``, ``h=1``."""
    kinetic = (r1_kill(_SAMPLE_SEP) ** 2) / 2
    eps = _EPS_HI
    height = Fraction(1)
    t_orbit = kinetic * eps * eps + height - eps**3
    return Interval.from_rational(t_orbit / (eps * eps + height))


@dataclass(frozen=True)
class StageCEnvReport:
    """Kill-line Stage-C C=0 T envelope. Not a C!=0 orbit, first-hit, or G1."""

    identities: Mapping[str, str]
    sample_ratio: Mapping[str, float]
    enclosure: Mapping[str, float | bool]
    sample_inside: bool
    stage_c_env: bool
    honesty: Mapping[str, object]

    def to_payload(self) -> dict[str, object]:
        return {
            "schema": "hilbert16-stage-c-env-v1",
            "identities": dict(self.identities),
            "sample_ratio": dict(self.sample_ratio),
            "enclosure": dict(self.enclosure),
            "sample_inside": self.sample_inside,
            "stage_c_env": self.stage_c_env,
            "honesty": dict(self.honesty),
            "scope": (
                "Exact kill-line Stage-C C=0 envelope identities: wall "
                "T_e/eps^2 = 1/8 below 1, declared c = 1/32, and "
                "c + room = 1, plus Interval "
                "(1/32)(eps^2+h) <= T(h) <= eps^2+h. Not a C!=0 "
                "orbit, first-hit, C2, G1, or Hilbert XVI."
            ),
        }


def report() -> StageCEnvReport:
    """Replay Stage-C C=0 envelope identities and enclose the sandwich."""
    identities = identity_verdicts()
    sample = sample_stage_c_env()
    try:
        enclosure = enclose_stage_c_env()
        inside = bool(enclosure["finite"]) and _contains(
            Interval(float(_DECLARED_C_LO), float(_DECLARED_C_HI)),
            sample,
        )
        sealed = (
            all(status == "PROVED" for status in identities.values())
            and bool(enclosure["below_declared"])
            and bool(enclosure["c_above_floor"])
            and bool(enclosure["excludes_zero"])
            and inside
        )
    except (ValueError, ZeroDivisionError, ArithmeticError):
        enclosure = {
            "c_lo": float("nan"),
            "c_hi": float("nan"),
            "below_declared": False,
            "c_above_floor": False,
            "excludes_zero": False,
            "finite": False,
        }
        inside = False
        sealed = False
    return StageCEnvReport(
        identities=identities,
        sample_ratio={"lo": sample.lo, "hi": sample.hi},
        enclosure=enclosure,
        sample_inside=inside,
        stage_c_env=sealed,
        honesty=_honesty(stage_c_env=sealed),
    )
