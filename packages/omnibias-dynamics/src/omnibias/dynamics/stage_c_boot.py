# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
"""Kill-line Stage-C C=2 T-h comparison bootstrap at K=6.

On lambda1 = -2 the sealed sandwich is T <= 6 (eps^2+h). Along the
comparison, |(T-h)_h| <= C K eps + 3 nu + C(1+K) eps^3/h with C=2
and nu=eps, so the linear coefficient is 15 eps and the log
coefficient is 42 eps^3 log(1/eps) after integrating from
h_1=eps^3 to hmax=1. Adding the sealed start gap, Interval wrapping
at the compact edge still yields T-h < 1. This is a fixed-margin
bound, not O(eps), not first-hit, C2, dx_e off the kill line, G1,
or Hilbert XVI. The tight ratio is stage_c_k. The start gap is
stage_c_gap.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from fractions import Fraction

from omnibias.core.collapse.verdict import adjudicate_residual
from omnibias.core.verified.interval import Interval
from omnibias.core.verified.transcend import ln_iv
from omnibias.dynamics.honesty_inventory import build_honesty
from omnibias.dynamics.stage_c_gap import enclose_stage_c_gap
from omnibias.dynamics.stage_c_k import enclose_stage_c_k

__all__ = [
    "StageCBootReport",
    "enclose_stage_c_boot",
    "identity_verdicts",
    "report",
    "residual_c_log",
    "residual_lin_fifteen",
    "residual_one_k",
    "sample_stage_c_boot",
]

_C = Fraction(2)
_K = Fraction(6)
_ONE_K = Fraction(7)
_THREE = Fraction(3)
_LIN = Fraction(15)
_C_LOG = Fraction(42)
_EPS_HI = Fraction(1, 16)
_SAMPLE_LIN = Fraction(15, 16)
_DECLARED_TH = 1.0
_DECLARED_LIN = 1.0


def _honesty(*, stage_c_boot: bool) -> dict[str, object]:
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
        stage_c_k=False,
        stage_c_boot=stage_c_boot,
        hk_theorem_24_used=False,
    )


def residual_one_k() -> Fraction:
    """``1 + 6 = 7``: compact ``1+K`` at the sealed ratio K=6."""
    return 1 + _K - _ONE_K


def residual_lin_fifteen() -> Fraction:
    """``2*6 + 3 = 15``: linear coefficient C K + 3 at nu=eps."""
    return _C * _K + _THREE - _LIN


def residual_c_log() -> Fraction:
    """``2*7*3 = 42``: log coefficient C(1+K) times 3 from h_1=eps^3."""
    return _C * _ONE_K * _THREE - _C_LOG


def _verdict(residual: Fraction) -> str:
    box = Interval.point(float(residual)) if residual == 0 else Interval.from_rational(residual)
    return adjudicate_residual(box, existential=False).status


def identity_verdicts() -> dict[str, str]:
    return {
        "one_k": _verdict(residual_one_k()),
        "lin_fifteen": _verdict(residual_lin_fifteen()),
        "c_log": _verdict(residual_c_log()),
    }


def _contains(outer: Interval, inner: Interval) -> bool:
    return outer.lo <= inner.lo and inner.hi <= outer.hi


def enclose_stage_c_boot() -> dict[str, float | bool]:
    """Stage-C C=2 T-h < 1 at the compact edge with K=6."""
    ratio = enclose_stage_c_k()
    gap = enclose_stage_c_gap()
    eps = Interval.from_rational(_EPS_HI)
    ln = ln_iv(Interval.from_rational(Fraction(16)))
    linear = Interval.from_rational(_LIN) * eps
    logp = Interval.from_rational(_C_LOG) * (eps**3) * ln
    g = Interval(float(gap["gap_lo"]), float(gap["gap_hi"]))
    gap0 = g * (eps * eps)
    total = gap0 + linear + logp
    lin_lo = float(linear.lo)
    lin_hi = float(linear.hi)
    total_lo = float(total.lo)
    total_hi = float(total.hi)
    return {
        "lin_lo": lin_lo,
        "lin_hi": lin_hi,
        "log_lo": float(logp.lo),
        "log_hi": float(logp.hi),
        "gap0_lo": float(gap0.lo),
        "gap0_hi": float(gap0.hi),
        "total_lo": total_lo,
        "total_hi": total_hi,
        "k_hi": float(ratio["k_hi"]),
        "below_declared": total_hi < _DECLARED_TH and lin_hi < _DECLARED_LIN,
        "k_below": float(ratio["k_hi"]) < 6.0,
        "excludes_zero": total_lo > 0.0 and lin_lo > 0.0,
        "finite": (
            total_hi < _DECLARED_TH
            and lin_hi < _DECLARED_LIN
            and float(ratio["k_hi"]) < 6.0
        ),
    }


def sample_stage_c_boot() -> Interval:
    """Sound linear piece ``15/16`` at ``C=2``, ``K=6``, ``eps=1/16``."""
    return Interval.from_rational(_SAMPLE_LIN)


@dataclass(frozen=True)
class StageCBootReport:
    """Kill-line Stage-C C=2 T-h bootstrap. Not first-hit or G1."""

    identities: Mapping[str, str]
    sample_lin: Mapping[str, float]
    enclosure: Mapping[str, float | bool]
    sample_inside: bool
    stage_c_boot: bool
    honesty: Mapping[str, object]

    def to_payload(self) -> dict[str, object]:
        return {
            "schema": "hilbert16-stage-c-boot-v1",
            "identities": dict(self.identities),
            "sample_lin": dict(self.sample_lin),
            "enclosure": dict(self.enclosure),
            "sample_inside": self.sample_inside,
            "stage_c_boot": self.stage_c_boot,
            "honesty": dict(self.honesty),
            "scope": (
                "Exact kill-line Stage-C C=2 T-h bootstrap identities: "
                "1+6=7, 2*6+3=15, and 2*7*3=42, plus Interval T-h < 1 "
                "at the compact edge. Not O(eps), first-hit, C2, G1, or "
                "Hilbert XVI."
            ),
        }


def report() -> StageCBootReport:
    """Replay Stage-C C=2 T-h bootstrap identities and enclose the margin."""
    identities = identity_verdicts()
    sample = sample_stage_c_boot()
    try:
        enclosure = enclose_stage_c_boot()
        inside = bool(enclosure["finite"]) and _contains(
            Interval(float(enclosure["lin_lo"]), float(enclosure["lin_hi"])),
            sample,
        )
        sealed = (
            all(status == "PROVED" for status in identities.values())
            and bool(enclosure["below_declared"])
            and bool(enclosure["k_below"])
            and bool(enclosure["excludes_zero"])
            and inside
        )
    except (ValueError, ZeroDivisionError, ArithmeticError):
        enclosure = {
            "lin_lo": float("nan"),
            "lin_hi": float("nan"),
            "total_lo": float("nan"),
            "total_hi": float("nan"),
            "below_declared": False,
            "k_below": False,
            "excludes_zero": False,
            "finite": False,
        }
        inside = False
        sealed = False
    return StageCBootReport(
        identities=identities,
        sample_lin={"lo": sample.lo, "hi": sample.hi},
        enclosure=enclosure,
        sample_inside=inside,
        stage_c_boot=sealed,
        honesty=_honesty(stage_c_boot=sealed),
    )
