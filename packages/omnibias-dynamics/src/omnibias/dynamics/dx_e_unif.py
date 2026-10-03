# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
"""Kill-line uniform-in-chi dx_e/dkappa majorant.

On lambda1 = -2 the Stage-A event derivative satisfies

    dx_e / d kappa <= (1/2) sep^2 exp(-after(chi))

on the chi_b compact once the leading factors are sealed. For every
chi >= chi_b the extra exponent is (3/16) r1 (chi - chi_b), and
(3/16) r1 >= 3/32. Absorbing the threshold floor then yields

    dx_e / d kappa <= C sep^2 exp(- (3/32) chi)

with Interval C < 2. The written decay c = 1/16 is strictly weaker
by 1/32. This is a sealed uniform-in-chi majorant on the declared
compact, not Stage C, first-hit, dx_e off the kill line, G1, or
Hilbert XVI. Leading factors are dx_e_leading. The chi threshold
is chi_b.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from fractions import Fraction

from omnibias.core.collapse.verdict import adjudicate_residual
from omnibias.core.verified.interval import Interval
from omnibias.core.verified.transcend import exp_iv
from omnibias.dynamics.chi_b import enclose_chi_b
from omnibias.dynamics.dx_e_leading import enclose_dx_e_leading
from omnibias.dynamics.honesty_inventory import build_honesty
from omnibias.dynamics.stage_b import r1_kill

__all__ = [
    "DxEUnifReport",
    "enclose_dx_e_unif",
    "identity_verdicts",
    "report",
    "residual_c_weaker",
    "residual_extra_half",
    "residual_lift_nine",
    "sample_dx_e_unif",
]

_SAMPLE_SEP = Fraction(3, 5)
_COEFF = Fraction(3, 16)
_R1_MIN = Fraction(1, 2)
_EXTRA = Fraction(3, 32)
_C_WRITTEN = Fraction(1, 16)
_GAP = Fraction(1, 32)
_CHI_DECL = Fraction(9)
_LIFT = Fraction(27, 32)
_DECLARED_C = 2.0
_DECLARED_EXTRA = 0.0625


def _honesty(*, dx_e_unif: bool) -> dict[str, object]:
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
        dx_e_unif=dx_e_unif,
        hk_theorem_24_used=False,
    )


def residual_extra_half() -> Fraction:
    """``(3/16)*(1/2) = 3/32``: extra chi-coefficient at ``r1 = 1/2``."""
    return _COEFF * _R1_MIN - _EXTRA


def residual_c_weaker() -> Fraction:
    """Written ``c = 1/16`` is ``1/32`` weaker than the extra coefficient."""
    return _EXTRA - _C_WRITTEN - _GAP


def residual_lift_nine() -> Fraction:
    """Threshold lift ``(3/32)*9 = 27/32`` at declared ``chi_b < 9``."""
    return _EXTRA * _CHI_DECL - _LIFT


def _verdict(residual: Fraction) -> str:
    box = Interval.point(float(residual)) if residual == 0 else Interval.from_rational(residual)
    return adjudicate_residual(box, existential=False).status


def identity_verdicts() -> dict[str, str]:
    return {
        "extra_half": _verdict(residual_extra_half()),
        "c_weaker": _verdict(residual_c_weaker()),
        "lift_nine": _verdict(residual_lift_nine()),
    }


def _contains(outer: Interval, inner: Interval) -> bool:
    return outer.lo <= inner.lo and inner.hi <= outer.hi


def enclose_dx_e_unif() -> dict[str, float | bool]:
    """Uniform-in-chi C majorant on the chi_b compact."""
    lead = enclose_dx_e_leading()
    chi = enclose_chi_b()
    r1 = Interval(float(chi["r1_lo"]), float(chi["r1_hi"]))
    extra = Interval.from_rational(_COEFF) * r1
    pref = Interval(0.0, float(lead["pref_hi"]))
    after_lo = float(lead["after_lo"])
    e_th = exp_iv(Interval.point(-after_lo))
    lift = Interval.from_rational(_EXTRA) * Interval(0.0, float(chi["chi_hi"]))
    e_lift = exp_iv(lift)
    c_box = pref * e_th * e_lift
    extra_lo = float(extra.lo)
    c_hi = float(c_box.hi)
    return {
        "extra_lo": extra_lo,
        "extra_hi": extra.hi,
        "c_lo": c_box.lo,
        "c_hi": c_hi,
        "pref_hi": float(lead["pref_hi"]),
        "after_lo": after_lo,
        "chi_hi": float(chi["chi_hi"]),
        "r1_lo": r1.lo,
        "r1_hi": r1.hi,
        "below_declared": c_hi < _DECLARED_C,
        "extra_above_floor": extra_lo > _DECLARED_EXTRA,
        "excludes_zero": extra_lo > 0.0 and c_hi > 0.0,
        "finite": c_hi < _DECLARED_C and extra_lo > _DECLARED_EXTRA,
        "c_decay": float(_EXTRA),
    }


def sample_dx_e_unif() -> Interval:
    """Sound extra coefficient ``(3/16) r1`` at ``sep = 3/5``."""
    return Interval.from_rational(_COEFF * r1_kill(_SAMPLE_SEP))


@dataclass(frozen=True)
class DxEUnifReport:
    """Kill-line uniform-in-chi dx_e. Not Stage C, first-hit, or G1."""

    identities: Mapping[str, str]
    sample_extra: Mapping[str, float]
    enclosure: Mapping[str, float | bool]
    sample_inside: bool
    dx_e_unif: bool
    honesty: Mapping[str, object]

    def to_payload(self) -> dict[str, object]:
        return {
            "schema": "hilbert16-dx-e-unif-v1",
            "identities": dict(self.identities),
            "sample_extra": dict(self.sample_extra),
            "enclosure": dict(self.enclosure),
            "sample_inside": self.sample_inside,
            "dx_e_unif": self.dx_e_unif,
            "honesty": dict(self.honesty),
            "scope": (
                "Exact kill-line uniform-in-chi dx_e identities: extra "
                "coefficient 3/32 at r1 = 1/2, written c = 1/16 weaker "
                "by 1/32, and threshold lift 27/32 at chi_b < 9, plus "
                "Interval C < 2 and extra > 1/16 on the chi_b compact. "
                "Not Stage C, first-hit, dx_e off the kill line, G1, "
                "or Hilbert XVI."
            ),
        }


def report() -> DxEUnifReport:
    """Replay uniform-in-chi identities and enclose the C majorant."""
    identities = identity_verdicts()
    sample = sample_dx_e_unif()
    try:
        enclosure = enclose_dx_e_unif()
        inside = bool(enclosure["finite"]) and _contains(
            Interval(float(enclosure["extra_lo"]), float(enclosure["extra_hi"])),
            sample,
        )
        sealed = (
            all(status == "PROVED" for status in identities.values())
            and bool(enclosure["below_declared"])
            and bool(enclosure["extra_above_floor"])
            and bool(enclosure["excludes_zero"])
            and inside
        )
    except (ValueError, ZeroDivisionError, ArithmeticError):
        enclosure = {
            "extra_lo": float("nan"),
            "extra_hi": float("nan"),
            "c_hi": float("nan"),
            "below_declared": False,
            "extra_above_floor": False,
            "excludes_zero": False,
            "finite": False,
        }
        inside = False
        sealed = False
    return DxEUnifReport(
        identities=identities,
        sample_extra={"lo": sample.lo, "hi": sample.hi},
        enclosure=enclosure,
        sample_inside=inside,
        dx_e_unif=sealed,
        honesty=_honesty(dx_e_unif=sealed),
    )
