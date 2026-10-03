# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
"""Slow-line Z_V chain rule and a fold holomorphic Z_v enclosure.

On the slow line, V = 1 - v - nu v^2 so V_v = -ell with ell = 1 + 2 nu v.
The holomorphic remainder in the v-chart therefore satisfies

    Z_V = Z_v / V_v = - Z_v / ell.

Interval arithmetic on the cancelled-N kill compact encloses |Z_V| < 1/4
and excludes 0. The same holomorphic Z_v formula on the fold wall

    r in [1.4, 1.6],  L = r^2,  lambda1 = -2 r

encloses |Z_v| < 1/4 (and then |Z_V| < 1/4) with Picard-included k.

This is a slow-line V-chart bound plus a holomorphic fold Z_v bound, not
a fold I-map Z_x bound, not sep>0, not first-hit, G1, or Hilbert XVI.
The kill-compact Z_v enclosure is z_v_bound.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from fractions import Fraction

from omnibias.core.collapse.verdict import adjudicate_residual
from omnibias.core.verified.interval import Interval
from omnibias.dynamics.honesty_inventory import build_honesty
from omnibias.dynamics.z_v_bound import enclose_zv, sample_zv, z0_v

__all__ = [
    "ZSlowVReport",
    "V_v",
    "ell",
    "enclose_fold_zv",
    "enclose_zV",
    "identity_verdicts",
    "report",
    "residual_ZV_chain",
    "residual_V_v_slow",
    "residual_ell_min",
    "sample_fold_zv",
    "sample_zV",
    "z0_V",
]

_SAMPLE_NU = Fraction(5, 16)
_SAMPLE_V0 = Fraction(4, 5)
_SAMPLE_V = Fraction(1, 2)
_SAMPLE_KILL_NU = Fraction(1, 64)
_SAMPLE_KILL_V = Fraction(1, 2)
_SAMPLE_FOLD_R = Fraction(3, 2)
_NU_LO = 0.0
_NU_HI = 0.02
_VCHART_LO = -0.5
_VCHART_HI = 1.5
_FOLD_R_LO = 1.4
_FOLD_R_HI = 1.6
_K_PAD = 0.08
_SAMPLE_K_PAD = 0.05
_DECLARED_MAG = 0.25
_ELL_NU_HI = Fraction(1, 50)
_ELL_V_LO = Fraction(-1, 2)
_ELL_MIN = Fraction(49, 50)


def _honesty(*, z_slow_v_bound: bool) -> dict[str, object]:
    return build_honesty(
        g1_passed=False,
        g4_passed=False,
        full_graphic_cyclicity_proved=False,
        full_hilbert16_solved=False,
        physical_return_membership_proved=False,
        fold_leading_c2_remainder=False,
        physical_c2_remainder=False,
        z_x_bound=False,
        z_v_bound=False,
        z_slow_v_bound=z_slow_v_bound,
        cancelled_n_usable_z=False,
        orbit_continuation_on_kill=False,
        hk_theorem_24_used=False,
    )


def ell(nu: Fraction, v: Fraction) -> Fraction:
    """Slow-line factor ``1 + 2 nu v``."""
    return 1 + 2 * nu * v


def V_v(nu: Fraction, v: Fraction) -> Fraction:
    """``dV/dv`` for ``V = 1 - v - nu v^2``."""
    return -1 - 2 * nu * v


def z0_V(nu: Fraction, v: Fraction, v0: Fraction) -> Fraction:
    """Chain-rule ``d Z0 / d V = Z_v / V_v``."""
    vv = V_v(nu, v)
    if vv == 0:
        raise ValueError("z0_V requires V_v != 0")
    return z0_v(nu, v, v0) / vv


def residual_V_v_slow(nu: Fraction, v: Fraction) -> Fraction:
    """``V_v + ell``; both writings of ``1 + 2 nu v``."""
    return V_v(nu, v) + ell(nu, v)


def residual_ZV_chain(nu: Fraction, v: Fraction, v0: Fraction) -> Fraction:
    """``ell Z_V + Z_v``; the chain rule on the slow line."""
    return z0_V(nu, v, v0) * ell(nu, v) + z0_v(nu, v, v0)


def residual_ell_min() -> Fraction:
    """Declared ``ell`` floor ``1 + 2 (1/50) (-1/2) = 49/50``."""
    return ell(_ELL_NU_HI, _ELL_V_LO) - _ELL_MIN


def _verdict(residual: Fraction) -> str:
    box = Interval.point(float(residual)) if residual == 0 else Interval.from_rational(residual)
    return adjudicate_residual(box, existential=False).status


def identity_verdicts() -> dict[str, str]:
    nu, v, v0 = _SAMPLE_NU, _SAMPLE_V, _SAMPLE_V0
    return {
        "V_v_slow": _verdict(residual_V_v_slow(nu, v)),
        "ZV_chain": _verdict(residual_ZV_chain(nu, v, v0)),
        "ell_min": _verdict(residual_ell_min()),
    }


def _contains(outer: Interval, inner: Interval) -> bool:
    return outer.lo <= inner.lo and inner.hi <= outer.hi


def _holomorphic_zv(
    nu: Interval,
    v: Interval,
    lam0: Interval,
    lam1: Interval,
    k_pad: float,
) -> tuple[Interval, bool]:
    one = Interval.point(1.0)
    two = Interval.point(2.0)
    three = Interval.point(3.0)
    four = Interval.point(4.0)
    six = Interval.point(6.0)
    nine = Interval.point(9.0)
    v0 = two / (one + (one + four * nu).sqrt())
    ell0 = one + two * nu * v0
    k0 = (three * v0) / ell0
    guess = k0 + Interval(-k_pad, k_pad)
    phi = k0 + (nu * nu * guess * guess * lam1) / (ell0 * ell0) + (
        four * nu * nu * nu * nu * guess * guess * guess * lam0
    ) / (ell0 * ell0 * ell0 * ell0)
    included = _contains(guess, phi)
    kay = phi
    wall = one + nu * (v0 + v)
    s = v0 + v
    q1 = (
        four * v0
        - two * v0 * v0
        - three * v0 * v
        + nu * (four * v0 * v0 - three * v0 * s * s)
        - nu * nu * v0 * s * s * s
        + (v0 - v) * (-two - nu * v0)
    )
    q1v = -three * v0 - six * nu * v0 * s - three * (nu * nu) * v0 * (s * s) + two + nu * v0
    z0v = ell0 * (q1v * wall - three * nu * q1) / (nine * v0 * v0 * wall**4)
    d_ratio = nu * (-two * wall - three * ell0) / (wall**4)
    extra = (
        -(nu**2) * lam1 * d_ratio / (ell0 * ell0)
        - four * (nu**4) * kay * lam0 * d_ratio / (ell0**4)
    )
    return (k0 / kay) ** 2 * z0v + extra, included


def _ell_box(nu_lo: float, nu_hi: float, v_lo: float, v_hi: float) -> Interval:
    one = Interval.point(1.0)
    two = Interval.point(2.0)
    return one + two * Interval(nu_lo, nu_hi) * Interval(v_lo, v_hi)


def enclose_zV(
    *,
    nu_lo: float = _NU_LO,
    nu_hi: float = _NU_HI,
    v_lo: float = _VCHART_LO,
    v_hi: float = _VCHART_HI,
) -> dict[str, float | bool]:
    """Interval enclosure of slow-line ``Z_V = -Z_v / ell`` on the kill compact."""
    zv = enclose_zv(nu_lo=nu_lo, nu_hi=nu_hi, v_lo=v_lo, v_hi=v_hi)
    ell_box = _ell_box(nu_lo, nu_hi, v_lo, v_hi)
    picard = bool(zv["picard_included"])
    ell_away = ell_box.lo > 0.0 or ell_box.hi < 0.0
    if not picard or not ell_away:
        return {
            "nu_lo": nu_lo,
            "nu_hi": nu_hi,
            "v_lo": v_lo,
            "v_hi": v_hi,
            "picard_included": picard,
            "ell_lo": ell_box.lo,
            "ell_hi": ell_box.hi,
            "zV_lo": float("nan"),
            "zV_hi": float("nan"),
            "z_V_mag": float("inf"),
            "excludes_zero": False,
            "below_declared": False,
            "finite": False,
        }
    zV = -Interval(float(zv["zv_lo"]), float(zv["zv_hi"])) / ell_box
    mag = max(abs(zV.lo), abs(zV.hi))
    return {
        "nu_lo": nu_lo,
        "nu_hi": nu_hi,
        "v_lo": v_lo,
        "v_hi": v_hi,
        "picard_included": bool(zv["picard_included"]),
        "ell_lo": ell_box.lo,
        "ell_hi": ell_box.hi,
        "zV_lo": zV.lo,
        "zV_hi": zV.hi,
        "z_V_mag": mag,
        "excludes_zero": zV.hi < 0.0 or zV.lo > 0.0,
        "below_declared": mag < _DECLARED_MAG,
        "finite": mag < float("inf"),
    }


def enclose_fold_zv(
    *,
    r_lo: float = _FOLD_R_LO,
    r_hi: float = _FOLD_R_HI,
    nu_lo: float = _NU_LO,
    nu_hi: float = _NU_HI,
    v_lo: float = _VCHART_LO,
    v_hi: float = _VCHART_HI,
    k_pad: float = _K_PAD,
) -> dict[str, float | bool]:
    """Holomorphic ``Z_v`` on the fold wall ``L=r^2``, ``lambda1=-2 r``."""
    if r_hi < r_lo or r_lo <= 0.0 or nu_hi < nu_lo or v_hi < v_lo or k_pad <= 0.0:
        raise ValueError("enclose_fold_zv requires a declared fold compact")
    r = Interval(r_lo, r_hi)
    lam0 = r * r
    lam1 = Interval.point(-2.0) * r
    nu = Interval(nu_lo, nu_hi)
    v = Interval(v_lo, v_hi)
    zv, included = _holomorphic_zv(nu, v, lam0, lam1, k_pad)
    mag = max(abs(zv.lo), abs(zv.hi)) if included else float("inf")
    ell_box = _ell_box(nu_lo, nu_hi, v_lo, v_hi)
    if included and ell_box.lo > 0.0:
        zV = -zv / ell_box
        zV_mag = max(abs(zV.lo), abs(zV.hi))
        zV_lo, zV_hi = zV.lo, zV.hi
        zV_ex0 = zV.hi < 0.0 or zV.lo > 0.0
    else:
        zV_mag = float("inf")
        zV_lo, zV_hi = float("nan"), float("nan")
        zV_ex0 = False
    return {
        "r_lo": r_lo,
        "r_hi": r_hi,
        "nu_lo": nu_lo,
        "nu_hi": nu_hi,
        "v_lo": v_lo,
        "v_hi": v_hi,
        "picard_included": included,
        "zv_lo": zv.lo,
        "zv_hi": zv.hi,
        "z_v_mag": mag,
        "zV_lo": zV_lo,
        "zV_hi": zV_hi,
        "z_V_mag": zV_mag,
        "excludes_zero": bool(included) and (zv.hi < 0.0 or zv.lo > 0.0),
        "zV_excludes_zero": zV_ex0,
        "below_declared": bool(included) and mag < _DECLARED_MAG,
        "zV_below_declared": zV_mag < _DECLARED_MAG,
        "finite": bool(included) and mag < float("inf"),
        "L_lo": lam0.lo,
        "L_hi": lam0.hi,
        "lam1_lo": lam1.lo,
        "lam1_hi": lam1.hi,
    }


def sample_zV() -> Interval:
    """Sound real-line enclosure of ``Z_V`` at the kill sample of ``sample_zv``."""
    nu = Interval.from_rational(_SAMPLE_KILL_NU)
    v = Interval.from_rational(_SAMPLE_KILL_V)
    one = Interval.from_rational(Fraction(1))
    two = Interval.from_rational(Fraction(2))
    ell_pt = one + two * nu * v
    return -sample_zv() / ell_pt


def sample_fold_zv() -> Interval:
    """Sound real-line enclosure of holomorphic ``Z_v`` at a fold sample."""
    nu = Interval.from_rational(_SAMPLE_KILL_NU)
    v = Interval.from_rational(_SAMPLE_KILL_V)
    r = Interval.from_rational(_SAMPLE_FOLD_R)
    lam0 = r * r
    lam1 = Interval.from_rational(Fraction(-2)) * r
    zv, included = _holomorphic_zv(nu, v, lam0, lam1, _SAMPLE_K_PAD)
    if not included:
        raise ValueError("sample_fold_zv Picard box does not contain its image")
    return zv


@dataclass(frozen=True)
class ZSlowVReport:
    """Slow-line Z_V chain plus fold holomorphic Z_v. Not fold Z_x or G1."""

    identities: Mapping[str, str]
    sample_zV: Mapping[str, float]
    sample_fold_zv: Mapping[str, float]
    kill_enclosure: Mapping[str, float | bool]
    fold_enclosure: Mapping[str, float | bool]
    sample_inside: bool
    fold_sample_inside: bool
    z_slow_v_bound: bool
    z_x_bound: bool
    honesty: Mapping[str, object]

    def to_payload(self) -> dict[str, object]:
        return {
            "schema": "hilbert16-z-slow-v-v1",
            "identities": dict(self.identities),
            "sample_zV": dict(self.sample_zV),
            "sample_fold_zv": dict(self.sample_fold_zv),
            "kill_enclosure": dict(self.kill_enclosure),
            "fold_enclosure": dict(self.fold_enclosure),
            "sample_inside": self.sample_inside,
            "fold_sample_inside": self.fold_sample_inside,
            "z_slow_v_bound": self.z_slow_v_bound,
            "z_x_bound": self.z_x_bound,
            "honesty": dict(self.honesty),
            "scope": (
                "Exact slow-line Z_V = -Z_v/ell identities, an Interval "
                "enclosure |Z_V|<1/4 on the cancelled-N kill compact, and "
                "holomorphic |Z_v|<1/4 on the fold wall r in [1.4, 1.6]. "
                "Not fold I-map Z_x, not sep>0, not first-hit, G1, or "
                "Hilbert XVI."
            ),
        }


def report() -> ZSlowVReport:
    """Replay Z_V identities and enclose kill Z_V plus fold holomorphic Z_v."""
    identities = identity_verdicts()
    sample = sample_zV()
    fold_sample = sample_fold_zv()
    try:
        kill = enclose_zV()
        fold = enclose_fold_zv()
        inside = bool(kill["finite"]) and _contains(
            Interval(float(kill["zV_lo"]), float(kill["zV_hi"])),
            sample,
        )
        fold_inside = bool(fold["finite"]) and _contains(
            Interval(float(fold["zv_lo"]), float(fold["zv_hi"])),
            fold_sample,
        )
        sealed = (
            all(status == "PROVED" for status in identities.values())
            and bool(kill["picard_included"])
            and bool(kill["below_declared"])
            and bool(kill["excludes_zero"])
            and inside
            and bool(fold["picard_included"])
            and bool(fold["below_declared"])
            and bool(fold["excludes_zero"])
            and bool(fold["zV_below_declared"])
            and bool(fold["zV_excludes_zero"])
            and fold_inside
        )
    except (ValueError, ZeroDivisionError):
        kill = {
            "picard_included": False,
            "zV_lo": float("nan"),
            "zV_hi": float("nan"),
            "z_V_mag": float("inf"),
            "excludes_zero": False,
            "below_declared": False,
            "finite": False,
        }
        fold = {
            "picard_included": False,
            "zv_lo": float("nan"),
            "zv_hi": float("nan"),
            "z_v_mag": float("inf"),
            "z_V_mag": float("inf"),
            "excludes_zero": False,
            "zV_excludes_zero": False,
            "below_declared": False,
            "zV_below_declared": False,
            "finite": False,
        }
        inside = False
        fold_inside = False
        sealed = False
    return ZSlowVReport(
        identities=identities,
        sample_zV={"lo": sample.lo, "hi": sample.hi},
        sample_fold_zv={"lo": fold_sample.lo, "hi": fold_sample.hi},
        kill_enclosure=kill,
        fold_enclosure=fold,
        sample_inside=inside,
        fold_sample_inside=fold_inside,
        z_slow_v_bound=sealed,
        z_x_bound=False,
        honesty=_honesty(z_slow_v_bound=sealed),
    )
