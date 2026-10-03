# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
"""Holomorphic Z_v identities and a kill-compact Interval bound.

On the lambda=0 slice, Z0 = ell0 q1 / (9 v0^2 wall^3) is rational in
(nu, v, v0). Termwise differentiation of q1, the quotient rule for Z0,
and the product rule for (wall+ell0)/wall^3 are exact. Interval
arithmetic on the cancelled-N compact

    nu in [0, 0.02],  v in [-0.5, 1.5],  L in [0, 1],  lambda1 = -2

encloses |Z_v| < 1/4 with Picard-included k, and the box excludes 0.

This is a holomorphic-chart bound in the slow-line coordinate v, not a
fold Z_x bound, not sep>0, not first-hit, G1, or Hilbert XVI. The
unfrozen first-log-derivative identities are z_x_gap.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from fractions import Fraction

from omnibias.core.collapse.verdict import adjudicate_residual
from omnibias.core.verified.interval import Interval
from omnibias.dynamics.cancelled_n import q1_holomorphic
from omnibias.dynamics.honesty_inventory import build_honesty

__all__ = [
    "ZVBoundReport",
    "enclose_zv",
    "identity_verdicts",
    "q1_v",
    "report",
    "residual_q1_v",
    "residual_wall_ratio_v",
    "residual_zv_quotient",
    "sample_zv",
    "z0_v",
]

_SAMPLE_NU = Fraction(5, 16)
_SAMPLE_V0 = Fraction(4, 5)
_SAMPLE_V = Fraction(1, 2)
_SAMPLE_KILL_NU = Fraction(1, 64)
_SAMPLE_KILL_R1 = Fraction(1, 5)
_SAMPLE_KILL_V = Fraction(1, 2)
_NU_LO = 0.0
_NU_HI = 0.02
_VCHART_LO = -0.5
_VCHART_HI = 1.5
_L_LO = 0.0
_L_HI = 1.0
_K_PAD = 0.08
_KILL_LAM1 = -2.0
_DECLARED_MAG = 0.25


def _honesty(*, z_v_bound: bool) -> dict[str, object]:
    return build_honesty(
        g1_passed=False,
        g4_passed=False,
        full_graphic_cyclicity_proved=False,
        full_hilbert16_solved=False,
        physical_return_membership_proved=False,
        fold_leading_c2_remainder=False,
        physical_c2_remainder=False,
        z_x_bound=False,
        z_v_bound=z_v_bound,
        cancelled_n_usable_z=False,
        orbit_continuation_on_kill=False,
        hk_theorem_24_used=False,
    )


def q1_v(nu: Fraction, v: Fraction, v0: Fraction) -> Fraction:
    """Closed-form ``d q1 / d v`` on the holomorphic slice."""
    s = v0 + v
    return -3 * v0 - 6 * nu * v0 * s - 3 * (nu**2) * v0 * (s**2) + 2 + nu * v0


def _ell0(nu: Fraction, v0: Fraction) -> Fraction:
    return 1 + 2 * nu * v0


def z0_v(nu: Fraction, v: Fraction, v0: Fraction) -> Fraction:
    """Quotient-rule ``d Z0 / d v`` with ``Z0 = ell0 q1 / (9 v0^2 wall^3)``."""
    ell0 = _ell0(nu, v0)
    wall = 1 + nu * (v0 + v)
    if v0 == 0 or wall == 0:
        raise ValueError("z0_v requires nonzero v0 and wall")
    q1 = q1_holomorphic(nu, v, v0)
    return ell0 * (q1_v(nu, v, v0) * wall - 3 * nu * q1) / (9 * v0 * v0 * wall**4)


def residual_q1_v(nu: Fraction, v: Fraction, v0: Fraction) -> Fraction:
    """Termwise derivative of ``q1`` minus the closed form."""
    s = v0 + v
    termwise = (
        -3 * v0
        + nu * (-3 * v0 * 2 * s)
        - (nu**2) * v0 * (3 * s**2)
        + (2 + nu * v0)
    )
    return termwise - q1_v(nu, v, v0)


def residual_zv_quotient(nu: Fraction, v: Fraction, v0: Fraction) -> Fraction:
    """Quotient rule ``(f/g)' g^2 - (f' g - f g')`` for ``Z0``."""
    ell0 = _ell0(nu, v0)
    wall = 1 + nu * (v0 + v)
    if v0 == 0 or wall == 0:
        raise ValueError("residual_zv_quotient requires nonzero v0 and wall")
    q1 = q1_holomorphic(nu, v, v0)
    f = ell0 * q1
    g = 9 * v0 * v0 * wall**3
    f_v = ell0 * q1_v(nu, v, v0)
    g_v = 27 * v0 * v0 * wall**2 * nu
    return z0_v(nu, v, v0) * g * g - (f_v * g - f * g_v)


def residual_wall_ratio_v(nu: Fraction, v: Fraction, v0: Fraction) -> Fraction:
    """Product rule for ``d/dv[(wall+ell0)/wall^3]`` versus the closed form."""
    ell0 = _ell0(nu, v0)
    wall = 1 + nu * (v0 + v)
    if wall == 0:
        raise ValueError("residual_wall_ratio_v requires nonzero wall")
    product = nu / wall**3 - 3 * nu * (wall + ell0) / wall**4
    closed = nu * (-2 * wall - 3 * ell0) / wall**4
    return product - closed


def _verdict(residual: Fraction) -> str:
    box = Interval.point(float(residual)) if residual == 0 else Interval.from_rational(residual)
    return adjudicate_residual(box, existential=False).status


def identity_verdicts() -> dict[str, str]:
    nu, v, v0 = _SAMPLE_NU, _SAMPLE_V, _SAMPLE_V0
    return {
        "q1_v": _verdict(residual_q1_v(nu, v, v0)),
        "zv_quotient": _verdict(residual_zv_quotient(nu, v, v0)),
        "wall_ratio_v": _verdict(residual_wall_ratio_v(nu, v, v0)),
    }


def _contains(outer: Interval, inner: Interval) -> bool:
    return outer.lo <= inner.lo and inner.hi <= outer.hi


def enclose_zv(
    *,
    nu_lo: float = _NU_LO,
    nu_hi: float = _NU_HI,
    v_lo: float = _VCHART_LO,
    v_hi: float = _VCHART_HI,
    L_lo: float = _L_LO,
    L_hi: float = _L_HI,
    k_pad: float = _K_PAD,
) -> dict[str, float | bool]:
    """Interval enclosure of holomorphic ``Z_v`` on the cancelled-N compact."""
    if nu_hi < nu_lo or v_hi < v_lo or L_lo < 0.0 or L_hi < L_lo or k_pad <= 0.0:
        raise ValueError("enclose_zv requires a declared compact")
    one = Interval.point(1.0)
    two = Interval.point(2.0)
    three = Interval.point(3.0)
    four = Interval.point(4.0)
    six = Interval.point(6.0)
    nine = Interval.point(9.0)
    lam1 = Interval.point(_KILL_LAM1)
    nu = Interval(nu_lo, nu_hi)
    v = Interval(v_lo, v_hi)
    lam0 = Interval(L_lo, L_hi)
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
    zv = (k0 / kay) ** 2 * z0v + extra
    mag = max(abs(zv.lo), abs(zv.hi)) if included else float("inf")
    return {
        "nu_lo": nu_lo,
        "nu_hi": nu_hi,
        "v_lo": v_lo,
        "v_hi": v_hi,
        "L_lo": L_lo,
        "L_hi": L_hi,
        "picard_included": included,
        "zv_lo": zv.lo,
        "zv_hi": zv.hi,
        "z_v_mag": mag,
        "excludes_zero": bool(included) and (zv.hi < 0.0 or zv.lo > 0.0),
        "below_declared": bool(included) and mag < _DECLARED_MAG,
        "finite": bool(included) and mag < float("inf"),
    }


def sample_zv() -> Interval:
    """Sound real-line enclosure of holomorphic ``Z_v`` at a kill sample."""
    nu = Interval.from_rational(_SAMPLE_KILL_NU)
    r1 = Interval.from_rational(_SAMPLE_KILL_R1)
    v = Interval.from_rational(_SAMPLE_KILL_V)
    lam0 = r1 * (Interval.from_rational(Fraction(2)) - r1)
    lam1 = Interval.point(_KILL_LAM1)
    one = Interval.from_rational(Fraction(1))
    two = Interval.from_rational(Fraction(2))
    three = Interval.from_rational(Fraction(3))
    four = Interval.from_rational(Fraction(4))
    six = Interval.from_rational(Fraction(6))
    nine = Interval.from_rational(Fraction(9))
    v0 = two / (one + (one + four * nu).sqrt())
    ell0 = one + two * nu * v0
    k0 = (three * v0) / ell0
    pad = Interval(-0.05, 0.05)
    guess = k0 + pad
    phi = (
        k0
        + (nu * nu * guess * guess * lam1) / (ell0 * ell0)
        + (four * nu * nu * nu * nu * guess * guess * guess * lam0)
        / (ell0 * ell0 * ell0 * ell0)
    )
    if not _contains(guess, phi):
        raise ValueError("sample_zv Picard box does not contain its image")
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
    return (k0 / kay) ** 2 * z0v + extra


@dataclass(frozen=True)
class ZVBoundReport:
    """Holomorphic Z_v bound on the kill compact. Not fold Z_x or G1."""

    identities: Mapping[str, str]
    sample_zv: Mapping[str, float]
    enclosure: Mapping[str, float | bool]
    sample_inside: bool
    z_v_bound: bool
    z_x_bound: bool
    honesty: Mapping[str, object]

    def to_payload(self) -> dict[str, object]:
        return {
            "schema": "hilbert16-z-v-bound-v1",
            "identities": dict(self.identities),
            "sample_zv": dict(self.sample_zv),
            "enclosure": dict(self.enclosure),
            "sample_inside": self.sample_inside,
            "z_v_bound": self.z_v_bound,
            "z_x_bound": self.z_x_bound,
            "honesty": dict(self.honesty),
            "scope": (
                "Exact holomorphic Z_v identities and an Interval enclosure "
                "|Z_v|<1/4 on the cancelled-N kill compact, excluding 0. "
                "Holomorphic-chart v, not fold Z_x, not sep>0, not first-hit, "
                "G1, or Hilbert XVI."
            ),
        }


def report() -> ZVBoundReport:
    """Replay Z_v identities and enclose |Z_v|<1/4 on the kill compact."""
    identities = identity_verdicts()
    sample = sample_zv()
    try:
        enclosure = enclose_zv()
        inside = bool(enclosure["finite"]) and _contains(
            Interval(float(enclosure["zv_lo"]), float(enclosure["zv_hi"])),
            sample,
        )
        sealed = (
            all(status == "PROVED" for status in identities.values())
            and bool(enclosure["picard_included"])
            and bool(enclosure["below_declared"])
            and bool(enclosure["excludes_zero"])
            and inside
        )
    except (ValueError, ZeroDivisionError):
        enclosure = {
            "picard_included": False,
            "zv_lo": float("nan"),
            "zv_hi": float("nan"),
            "z_v_mag": float("inf"),
            "excludes_zero": False,
            "below_declared": False,
            "finite": False,
        }
        inside = False
        sealed = False
    return ZVBoundReport(
        identities=identities,
        sample_zv={"lo": sample.lo, "hi": sample.hi},
        enclosure=enclosure,
        sample_inside=inside,
        z_v_bound=sealed,
        z_x_bound=False,
        honesty=_honesty(z_v_bound=sealed),
    )

