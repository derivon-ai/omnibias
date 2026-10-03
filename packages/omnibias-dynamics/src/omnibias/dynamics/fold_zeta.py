# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
"""Cauchy majorant for Z on a declared fold compact of (L, lambda1).

The sep=0 fold is the discriminant wall ``lambda1^2 = 4 L`` with
``L = r^2`` and ``lambda1 = -2 r``. On that compact the implicit ``k``
equation is a contraction of a Picard box around ``k0 = 3 v0 / l`` for
small ``nu``. Enclosing the holomorphic numerator

    N = Vdot/eps - eps^2 L - eps lambda1 V + V^2 - V^3 / 3

on a polydisc in ``(V, nu)``, with ``r`` in a real interval, supplies a
Cauchy bound ``|Z| <= |N| / (R^3 E k_min)``.

This is not physical C2 of ``log D'``, first-hit, G1, or Hilbert XVI.
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
    "FoldZetaReport",
    "cauchy_majorant_fold",
    "fold_lambda",
    "identity_verdicts",
    "report",
    "residual_fold_bminus",
    "residual_fold_disc",
    "residual_fold_L",
    "residual_fold_lambda",
    "sample_z_fold",
]

_FOLD_E = 0.02
_FOLD_R = 0.08
_FOLD_RSTAR_LO = 1.4
_FOLD_RSTAR_HI = 1.6
_FOLD_K_RAD = 0.4
_SAMPLE_NU = Fraction(1, 64)
_SAMPLE_RSTAR = Fraction(3, 2)
_SAMPLE_V = Fraction(1)


def _honesty(*, fold_z_compact_bound: bool) -> dict[str, object]:
    return build_honesty(
        g1_passed=False,
        g4_passed=False,
        full_graphic_cyclicity_proved=False,
        full_hilbert16_solved=False,
        physical_return_membership_proved=False,
        fold_leading_c2_remainder=False,
        inner_z_compact_bound=False,
        fold_z_compact_bound=fold_z_compact_bound,
        orbit_continuation_on_kill=False,
        hk_theorem_24_used=False,
    )


def fold_lambda(rstar: Fraction) -> tuple[Fraction, Fraction]:
    """Fold coefficients ``(L, lambda1) = (r^2, -2 r)``."""
    return rstar**2, -2 * rstar


def residual_fold_lambda(rstar: Fraction, lam1: Fraction) -> Fraction:
    """``lambda1 + 2 r``."""
    return lam1 + 2 * rstar


def residual_fold_L(rstar: Fraction, lam0: Fraction) -> Fraction:
    """``L - r^2``."""
    return lam0 - rstar**2


def residual_fold_disc(lam0: Fraction, lam1: Fraction) -> Fraction:
    """``lambda1^2 - 4 L``; vanishes on the fold wall."""
    return lam1**2 - 4 * lam0


def residual_fold_bminus(x: Fraction, rstar: Fraction) -> Fraction:
    """``(x-r)^2 - (r^2 - 2 r x + x^2)``."""
    lam0, lam1 = fold_lambda(rstar)
    return (x - rstar) ** 2 - (lam0 + lam1 * x + x**2)


def _verdict(residual: Fraction) -> str:
    box = Interval.point(float(residual)) if residual == 0 else Interval.from_rational(residual)
    return adjudicate_residual(box, existential=False).status


def identity_verdicts() -> dict[str, str]:
    rstar, x = Fraction(3, 2), Fraction(2)
    lam0, lam1 = fold_lambda(rstar)
    return {
        "fold_lambda": _verdict(residual_fold_lambda(rstar, lam1)),
        "fold_L": _verdict(residual_fold_L(rstar, lam0)),
        "fold_disc": _verdict(residual_fold_disc(lam0, lam1)),
        "fold_bminus": _verdict(residual_fold_bminus(x, rstar)),
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
    rstar: ComplexInterval,
    k_rad: float,
) -> tuple[ComplexInterval, ComplexInterval, bool]:
    lam0 = rstar * rstar
    lam1 = ComplexInterval.from_value(-2) * rstar
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


def cauchy_majorant_fold(
    *,
    radius_v: float = _FOLD_R,
    radius_nu: float = _FOLD_E,
    rstar_lo: float = _FOLD_RSTAR_LO,
    rstar_hi: float = _FOLD_RSTAR_HI,
    k_rad: float = _FOLD_K_RAD,
) -> dict[str, float | bool]:
    """Cauchy majorant for ``Z`` on the fold compact ``r in [r_lo, r_hi]``."""
    if radius_v <= 0.0 or radius_nu <= 0.0 or rstar_lo <= 0.0 or rstar_hi <= rstar_lo:
        raise ValueError("cauchy_majorant_fold requires a positive fold chart")
    nu = polydisc_box(radius_nu)
    v_coord = polydisc_box(radius_v)
    rstar = ComplexInterval.from_parts(Interval(rstar_lo, rstar_hi), Interval.point(0.0))
    numer, kay, included = _enclose_N(nu, v_coord, rstar, k_rad)
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
        "rstar_lo": rstar_lo,
        "rstar_hi": rstar_hi,
        "picard_included": included,
        "z_bound": bound,
        "finite": bool(included) and k_min > 0.0 and bound < float("inf"),
    }


def sample_z_fold() -> Interval:
    """Sound real-line enclosure of ``Z`` at ``nu=1/64``, ``r=3/2``, ``v=1``."""
    nu = Interval.from_rational(_SAMPLE_NU)
    rstar = Interval.from_rational(_SAMPLE_RSTAR)
    v = Interval.from_rational(_SAMPLE_V)
    disc = Interval.from_rational(Fraction(1)) + Interval.from_rational(Fraction(4)) * nu
    v0 = Interval.from_rational(Fraction(2)) / (
        Interval.from_rational(Fraction(1)) + disc.sqrt()
    )
    lam0 = rstar * rstar
    lam1 = Interval.from_rational(Fraction(-2)) * rstar
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
        raise ValueError("sample_z_fold Picard box does not contain its image")
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
class FoldZetaReport:
    """Fold-compact Cauchy majorant. Not G1."""

    identities: Mapping[str, str]
    sample_abs_z: float
    cauchy: Mapping[str, float | bool]
    sample_below_bound: bool
    honesty: Mapping[str, object]

    def to_payload(self) -> dict[str, object]:
        return {
            "schema": "hilbert16-fold-zeta-v1",
            "identities": dict(self.identities),
            "sample_abs_z": self.sample_abs_z,
            "cauchy": dict(self.cauchy),
            "sample_below_bound": self.sample_below_bound,
            "fold_compact": "rstar in [1.4, 1.6], L=r^2, lambda1=-2 r",
            "honesty": dict(self.honesty),
            "scope": (
                "Cauchy majorant for Z on a declared fold compact of "
                "(L, lambda1). Not physical C2, first-hit, G1, or Hilbert XVI."
            ),
        }


def report() -> FoldZetaReport:
    """Replay fold identities and enclose a Cauchy majorant on the compact."""
    identities = identity_verdicts()
    sample = sample_z_fold()
    sample_abs = sample.abs().hi
    try:
        cauchy = cauchy_majorant_fold()
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
            "radius_v": _FOLD_R,
            "radius_nu": _FOLD_E,
            "rstar_lo": _FOLD_RSTAR_LO,
            "rstar_hi": _FOLD_RSTAR_HI,
            "picard_included": False,
            "z_bound": float("inf"),
            "finite": False,
        }
        below = False
        flag = False
    return FoldZetaReport(
        identities=identities,
        sample_abs_z=sample_abs,
        cauchy=cauchy,
        sample_below_bound=below,
        honesty=_honesty(fold_z_compact_bound=flag),
    )
