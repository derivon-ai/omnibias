# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
"""Cauchy majorant for Z on the lambda1=-2 kill compact, including L=0.

The shrinking-root sequence is ``lambda1 = -2`` and ``L = r1 (2 - r1)``.
As ``r1 -> 0``, ``L -> 0``. That compact is ``L in [0, 1]``: every
two-root pair with sum 2 has product at most 1. On it the implicit ``k``
equation is a Picard contraction around ``k0 = 3 v0 / l`` for small
``nu``. Enclosing the holomorphic numerator

    N = Vdot/eps - eps^2 L - eps lambda1 V + V^2 - V^3 / 3

on a polydisc in ``(V, nu)`` supplies ``|Z| <= |N| / (R^3 E k_min)``.

The bound is finite and includes the kill limit ``L = 0``. It is a
rectangular majorant, not small enough for a ``C = 2 + delta``
comparison, not ``T-h`` along the orbit, not first-hit, G1, or Hilbert
XVI. The fold compact ``r in [1.4, 1.6]`` is ``fold_zeta``.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from fractions import Fraction

from omnibias.core.collapse.verdict import adjudicate_residual
from omnibias.core.verified.complex_interval import ComplexInterval
from omnibias.core.verified.interval import Interval
from omnibias.dynamics.canonical_zeta import enclose_v, enclose_v0, polydisc_box
from omnibias.dynamics.honesty_inventory import build_honesty

__all__ = [
    "KillZetaReport",
    "cauchy_majorant_kill",
    "identity_verdicts",
    "kill_L",
    "report",
    "residual_kill_disc_gap",
    "residual_kill_lambda",
    "residual_kill_product",
    "residual_kill_sum",
    "residual_kill_tworoot",
    "sample_z_kill",
]

_KILL_E = 0.02
_KILL_R = 0.08
_KILL_L_LO = 0.0
_KILL_L_HI = 1.0
_KILL_K_RAD = 0.4
_SAMPLE_NU = Fraction(1, 64)
_SAMPLE_R1 = Fraction(1, 5)
_SAMPLE_V = Fraction(1, 2)
_SAMPLE_LAM1 = Fraction(-2)


def _honesty(*, kill_z_compact_bound: bool) -> dict[str, object]:
    return build_honesty(
        g1_passed=False,
        g4_passed=False,
        full_graphic_cyclicity_proved=False,
        full_hilbert16_solved=False,
        physical_return_membership_proved=False,
        fold_leading_c2_remainder=False,
        inner_z_compact_bound=False,
        fold_z_compact_bound=False,
        kill_z_compact_bound=kill_z_compact_bound,
        outgoing_first_hit=False,
        orbit_continuation_on_kill=False,
        hk_theorem_24_used=False,
    )


def kill_L(r1: Fraction) -> Fraction:
    """``L = r1 (2 - r1)`` on ``lambda1 = -2``."""
    return r1 * (2 - r1)


def residual_kill_lambda(lam1: Fraction) -> Fraction:
    """``lambda1 + 2``."""
    return lam1 + 2


def residual_kill_product(L: Fraction, r1: Fraction) -> Fraction:
    """``L - r1 (2 - r1)``."""
    return L - kill_L(r1)


def residual_kill_sum(r1: Fraction, r2: Fraction) -> Fraction:
    """``r1 + r2 - 2``."""
    return r1 + r2 - 2


def residual_kill_disc_gap(lam1: Fraction, L: Fraction) -> Fraction:
    """``lambda1^2 - 4 L`` versus ``4(1 - L)`` at ``lambda1 = -2``."""
    return (lam1**2 - 4 * L) - 4 * (1 - L)


def residual_kill_tworoot(
    x: Fraction, r1: Fraction, r2: Fraction, L: Fraction
) -> Fraction:
    """``(x-r1)(x-r2)`` versus ``x^2 - 2 x + L`` on ``r1 + r2 = 2``."""
    return (x - r1) * (x - r2) - (x * x - 2 * x + L)


def _verdict(residual: Fraction) -> str:
    box = Interval.point(float(residual)) if residual == 0 else Interval.from_rational(residual)
    return adjudicate_residual(box, existential=False).status


def identity_verdicts() -> dict[str, str]:
    r1, r2, L = _SAMPLE_R1, 2 - _SAMPLE_R1, kill_L(_SAMPLE_R1)
    return {
        "kill_lambda": _verdict(residual_kill_lambda(_SAMPLE_LAM1)),
        "kill_product": _verdict(residual_kill_product(L, r1)),
        "kill_sum": _verdict(residual_kill_sum(r1, r2)),
        "kill_disc_gap": _verdict(residual_kill_disc_gap(_SAMPLE_LAM1, L)),
        "kill_tworoot": _verdict(residual_kill_tworoot(Fraction(1), r1, r2, L)),
    }


def _contains_box(outer: ComplexInterval, inner: ComplexInterval) -> bool:
    return (
        outer.re.lo <= inner.re.lo
        and inner.re.hi <= outer.re.hi
        and outer.im.lo <= inner.im.lo
        and inner.im.hi <= outer.im.hi
    )


def _enclose_k(
    nu: ComplexInterval,
    v0: ComplexInterval,
    lam0: ComplexInterval,
    lam1: ComplexInterval,
    k_rad: float,
) -> tuple[ComplexInterval, bool]:
    two = ComplexInterval.from_value(2)
    three = ComplexInterval.from_value(3)
    four = ComplexInterval.from_value(4)
    ell0 = ComplexInterval.one() + two * nu * v0
    k0 = (three * v0) / ell0
    pad = Interval(-k_rad, k_rad)
    guess = ComplexInterval(k0.re + pad, k0.im + pad)
    corr = (nu * nu * guess * guess * lam1) / (ell0 * ell0) + (
        four * nu * nu * nu * nu * guess * guess * guess * lam0
    ) / (ell0 * ell0 * ell0 * ell0)
    phi = k0 + corr
    return guess, _contains_box(guess, phi)


def _enclose_N(
    nu: ComplexInterval,
    v_coord: ComplexInterval,
    lam0: ComplexInterval,
    k_rad: float,
) -> tuple[ComplexInterval, ComplexInterval, bool]:
    lam1 = ComplexInterval.from_value(-2)
    v0 = enclose_v0(nu)
    v = enclose_v(nu, v_coord)
    kay, included = _enclose_k(nu, v0, lam0, lam1, k_rad)
    two = ComplexInterval.from_value(2)
    three = ComplexInterval.from_value(3)
    ell0 = ComplexInterval.one() + two * nu * v0
    ell = ComplexInterval.one() + two * nu * v
    f0 = (nu * nu * kay * kay * kay * lam0) / ell0
    f1 = -nu * kay * kay * lam1 - (two * nu * nu * nu * kay * kay * kay * lam0) / (
        ell0 * ell0
    )
    ptilde = three * v0 * v0 + f1
    mtilde = f0 - ptilde * v0 + v0 * v0 * v0
    slow = mtilde + ptilde * v - v * v * v
    vdot_over_eps = ell * slow / kay
    eps = nu * kay
    numer = (
        vdot_over_eps
        - eps * eps * lam0
        - eps * lam1 * v_coord
        + v_coord * v_coord
        - (v_coord * v_coord * v_coord) / three
    )
    return numer, kay, included


def cauchy_majorant_kill(
    *,
    radius_v: float = _KILL_R,
    radius_nu: float = _KILL_E,
    L_lo: float = _KILL_L_LO,
    L_hi: float = _KILL_L_HI,
    k_rad: float = _KILL_K_RAD,
) -> dict[str, float | bool]:
    """Cauchy majorant for ``Z`` on ``lambda1=-2``, ``L in [L_lo, L_hi]``."""
    if radius_v <= 0.0 or radius_nu <= 0.0 or L_lo < 0.0 or L_hi < L_lo:
        raise ValueError("cauchy_majorant_kill requires a nonnegative L compact")
    nu = polydisc_box(radius_nu)
    v_coord = polydisc_box(radius_v)
    lam0 = ComplexInterval.from_parts(Interval(L_lo, L_hi), Interval.point(0.0))
    numer, kay, included = _enclose_N(nu, v_coord, lam0, k_rad)
    k_min = kay.modulus().lo
    majorant = numer.mag
    bound = (
        majorant / (radius_v**3 * radius_nu * k_min)
        if k_min > 0.0 and included
        else float("inf")
    )
    return {
        "majorant": majorant,
        "k_min": k_min,
        "radius_v": radius_v,
        "radius_nu": radius_nu,
        "L_lo": L_lo,
        "L_hi": L_hi,
        "picard_included": included,
        "z_bound": bound,
        "finite": bool(included) and k_min > 0.0 and bound < float("inf"),
    }


def sample_z_kill() -> Interval:
    """Sound real-line enclosure of ``Z`` at ``nu=1/64``, ``r1=1/5``, ``v=1/2``."""
    nu = Interval.from_rational(_SAMPLE_NU)
    r1 = Interval.from_rational(_SAMPLE_R1)
    v = Interval.from_rational(_SAMPLE_V)
    lam0 = r1 * (Interval.from_rational(Fraction(2)) - r1)
    lam1 = Interval.from_rational(_SAMPLE_LAM1)
    disc = Interval.from_rational(Fraction(1)) + Interval.from_rational(Fraction(4)) * nu
    v0 = Interval.from_rational(Fraction(2)) / (
        Interval.from_rational(Fraction(1)) + disc.sqrt()
    )
    two = Interval.from_rational(Fraction(2))
    three = Interval.from_rational(Fraction(3))
    four = Interval.from_rational(Fraction(4))
    one = Interval.from_rational(Fraction(1))
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
    if not (guess.lo <= phi.lo and phi.hi <= guess.hi):
        raise ValueError("sample_z_kill Picard box does not contain its image")
    kay = phi
    ell = one + two * nu * v
    f0 = (nu * nu * kay * kay * kay * lam0) / ell0
    f1 = -nu * kay * kay * lam1 - (two * nu * nu * nu * kay * kay * kay * lam0) / (
        ell0 * ell0
    )
    ptilde = three * v0 * v0 + f1
    mtilde = f0 - ptilde * v0 + v0 * v0 * v0
    slow = mtilde + ptilde * v - v * v * v
    vdot_over_eps = ell * slow / kay
    eps = nu * kay
    v_coord = one - v - nu * v * v
    zeta = (vdot_over_eps - eps * eps * lam0 - eps * lam1 * v_coord) / (v_coord * v_coord)
    numer = zeta + one - v_coord / three
    return numer / (eps * v_coord)


@dataclass(frozen=True)
class KillZetaReport:
    """Kill-compact Cauchy majorant on lambda1=-2, L in [0, 1]. Not G1."""

    identities: Mapping[str, str]
    sample_abs_z: float
    cauchy: Mapping[str, float | bool]
    sample_below_bound: bool
    honesty: Mapping[str, object]

    def to_payload(self) -> dict[str, object]:
        return {
            "schema": "hilbert16-kill-zeta-v1",
            "identities": dict(self.identities),
            "sample_abs_z": self.sample_abs_z,
            "cauchy": dict(self.cauchy),
            "sample_below_bound": self.sample_below_bound,
            "kill_compact": "lambda1=-2, L in [0, 1], includes r1->0",
            "honesty": dict(self.honesty),
            "scope": (
                "Cauchy majorant for Z on the lambda1=-2 kill compact "
                "including L=0. Rectangular and not small enough for "
                "C=2+delta. Not T-h along the orbit, first-hit, G1, or "
                "Hilbert XVI."
            ),
        }


def report() -> KillZetaReport:
    """Replay kill identities and enclose a Cauchy majorant including L=0."""
    identities = identity_verdicts()
    sample = sample_z_kill()
    sample_abs = sample.abs().hi
    try:
        cauchy = cauchy_majorant_kill()
        sealed = bool(cauchy["finite"]) and all(
            status == "PROVED" for status in identities.values()
        )
        z_bound = float(cauchy["z_bound"])
        below = sealed and sample_abs <= z_bound
        flag = sealed and below
    except (ValueError, ZeroDivisionError):
        cauchy = {
            "majorant": float("inf"),
            "k_min": 0.0,
            "radius_v": _KILL_R,
            "radius_nu": _KILL_E,
            "L_lo": _KILL_L_LO,
            "L_hi": _KILL_L_HI,
            "picard_included": False,
            "z_bound": float("inf"),
            "finite": False,
        }
        below = False
        flag = False
    return KillZetaReport(
        identities=identities,
        sample_abs_z=sample_abs,
        cauchy=cauchy,
        sample_below_bound=below,
        honesty=_honesty(kill_z_compact_bound=flag),
    )
