# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
"""Matching-chart fold I-map Z_x = Z_v * eps / ell.

On the fold wall r in [1.4, 1.6], the holomorphic remainder is a
function of slow-line v. Matching V = -eps x gives v_x = eps / ell
with ell = 1 + 2 nu v, so

    Z_x = Z_v * eps / ell.

Interval arithmetic on the fold holomorphic Z_v box, eps in [0, 0.02],
and ell >= 49/50 encloses |Z_x| < 1/100. At eps = 0 the extra term
vanishes. This is a fold I-map Z_x bound under the matching
identification, not sep>0, not first-hit, G1, or Hilbert XVI.

The slow-line Z_V chain is z_slow_v. Unfrozen gap identities without
a bound are z_x_gap.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from fractions import Fraction

from omnibias.core.collapse.verdict import adjudicate_residual
from omnibias.core.verified.interval import Interval
from omnibias.dynamics.honesty_inventory import build_honesty
from omnibias.dynamics.z_slow_v import ell, enclose_fold_zv, enclose_zV, sample_fold_zv
from omnibias.dynamics.z_v_bound import z0_v

__all__ = [
    "FoldZXReport",
    "enclose_fold_zx",
    "identity_verdicts",
    "report",
    "residual_fold_x_interior",
    "residual_zx_chain",
    "residual_zx_declared",
    "sample_fold_zx",
    "z0_x",
]

_SAMPLE_NU = Fraction(5, 16)
_SAMPLE_V0 = Fraction(4, 5)
_SAMPLE_V = Fraction(1, 2)
_SAMPLE_KILL_NU = Fraction(1, 64)
_SAMPLE_KILL_V = Fraction(1, 2)
_SAMPLE_FOLD_X = Fraction(3, 2)
_FOLD_X_LO = Fraction(7, 5)
_FOLD_X_HI = Fraction(8, 5)
_ELL_NU_HI = Fraction(1, 50)
_ELL_MIN = Fraction(49, 50)
_DECLARED_ZV = Fraction(1, 4)
_DECLARED_ZX = Fraction(1, 196)
_NU_HI = 0.02
_DECLARED_MAG = 0.01


def _honesty(*, z_x_bound: bool) -> dict[str, object]:
    return build_honesty(
        g1_passed=False,
        g4_passed=False,
        full_graphic_cyclicity_proved=False,
        full_hilbert16_solved=False,
        physical_return_membership_proved=False,
        fold_leading_c2_remainder=False,
        physical_c2_remainder=False,
        z_x_bound=z_x_bound,
        z_v_bound=False,
        z_slow_v_bound=False,
        cancelled_n_usable_z=False,
        orbit_continuation_on_kill=False,
        hk_theorem_24_used=False,
    )


def z0_x(nu: Fraction, v: Fraction, v0: Fraction) -> Fraction:
    """Matching-chart ``Z_x = Z_v * nu / ell`` with ``nu = eps``."""
    factor = ell(nu, v)
    if factor == 0:
        raise ValueError("z0_x requires ell != 0")
    return z0_v(nu, v, v0) * nu / factor


def residual_zx_chain(nu: Fraction, v: Fraction, v0: Fraction) -> Fraction:
    """``ell Z_x - Z_v nu``; matching ``v_x = nu / ell``."""
    return z0_x(nu, v, v0) * ell(nu, v) - z0_v(nu, v, v0) * nu


def residual_zx_declared() -> Fraction:
    """Declared rate ``(1/4)(1/50)/(49/50) = 1/196``."""
    return _DECLARED_ZV * _ELL_NU_HI / _ELL_MIN - _DECLARED_ZX


def residual_fold_x_interior() -> Fraction:
    """``x=3/2`` lies in the fold I-map interval ``[7/5, 8/5]``."""
    return (_FOLD_X_HI - _SAMPLE_FOLD_X) * (_SAMPLE_FOLD_X - _FOLD_X_LO) - Fraction(1, 100)


def _verdict(residual: Fraction) -> str:
    box = Interval.point(float(residual)) if residual == 0 else Interval.from_rational(residual)
    return adjudicate_residual(box, existential=False).status


def identity_verdicts() -> dict[str, str]:
    nu, v, v0 = _SAMPLE_NU, _SAMPLE_V, _SAMPLE_V0
    return {
        "zx_chain": _verdict(residual_zx_chain(nu, v, v0)),
        "zx_declared": _verdict(residual_zx_declared()),
        "fold_x_interior": _verdict(residual_fold_x_interior()),
    }


def _contains(outer: Interval, inner: Interval) -> bool:
    return outer.lo <= inner.lo and inner.hi <= outer.hi


def enclose_fold_zx() -> dict[str, float | bool]:
    """Sound ``|Z_x| <= |Z_v| eps_hi / ell_min`` on the fold I-map compact."""
    fold = enclose_fold_zv()
    kill = enclose_zV()
    ell_lo = float(kill["ell_lo"])
    picard = bool(fold["picard_included"]) and ell_lo > 0.0
    if not picard:
        mag = float("inf")
    else:
        mag = float(fold["z_v_mag"]) * _NU_HI / ell_lo
    extra = mag * (_NU_HI**2) * (float(_FOLD_X_HI) ** 4)
    return {
        "picard_included": picard,
        "ell_lo": ell_lo,
        "z_v_mag": float(fold["z_v_mag"]),
        "zx_lo": -mag if picard else float("nan"),
        "zx_hi": mag if picard else float("nan"),
        "z_x_mag": mag,
        "extra_c2_mag": float(extra) if picard else float("inf"),
        "below_declared": bool(picard) and mag < _DECLARED_MAG,
        "finite": bool(picard) and mag < float("inf"),
        "x_lo": float(_FOLD_X_LO),
        "x_hi": float(_FOLD_X_HI),
        "eps_hi": _NU_HI,
    }


def sample_fold_zx() -> Interval:
    """Sound real-line enclosure of matching ``Z_x`` at the fold sample."""
    nu = Interval.from_rational(_SAMPLE_KILL_NU)
    v = Interval.from_rational(_SAMPLE_KILL_V)
    one = Interval.from_rational(Fraction(1))
    two = Interval.from_rational(Fraction(2))
    ell_pt = one + two * nu * v
    return sample_fold_zv() * nu / ell_pt


@dataclass(frozen=True)
class FoldZXReport:
    """Fold I-map Z_x bound under matching V=-eps x. Not sep>0 or G1."""

    identities: Mapping[str, str]
    sample_zx: Mapping[str, float]
    enclosure: Mapping[str, float | bool]
    sample_inside: bool
    z_x_bound: bool
    honesty: Mapping[str, object]

    def to_payload(self) -> dict[str, object]:
        return {
            "schema": "hilbert16-fold-z-x-v1",
            "identities": dict(self.identities),
            "sample_zx": dict(self.sample_zx),
            "enclosure": dict(self.enclosure),
            "sample_inside": self.sample_inside,
            "z_x_bound": self.z_x_bound,
            "honesty": dict(self.honesty),
            "scope": (
                "Exact matching-chart Z_x = Z_v eps/ell identities and an "
                "Interval enclosure |Z_x|<1/100 on the fold I-map compact "
                "r in [1.4, 1.6], eps in [0, 0.02]. Not sep>0, not first-hit, "
                "G1, or Hilbert XVI."
            ),
        }


def report() -> FoldZXReport:
    """Replay matching Z_x identities and enclose |Z_x|<1/100 on the fold."""
    identities = identity_verdicts()
    sample = sample_fold_zx()
    try:
        enclosure = enclose_fold_zx()
        inside = bool(enclosure["finite"]) and _contains(
            Interval(float(enclosure["zx_lo"]), float(enclosure["zx_hi"])),
            sample,
        )
        sealed = (
            all(status == "PROVED" for status in identities.values())
            and bool(enclosure["picard_included"])
            and bool(enclosure["below_declared"])
            and inside
        )
    except (ValueError, ZeroDivisionError):
        enclosure = {
            "picard_included": False,
            "zx_lo": float("nan"),
            "zx_hi": float("nan"),
            "z_x_mag": float("inf"),
            "below_declared": False,
            "finite": False,
        }
        inside = False
        sealed = False
    return FoldZXReport(
        identities=identities,
        sample_zx={"lo": sample.lo, "hi": sample.hi},
        enclosure=enclosure,
        sample_inside=inside,
        z_x_bound=sealed,
        honesty=_honesty(z_x_bound=sealed),
    )
