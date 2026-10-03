# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
"""Actual r=-1 slow-line zeta from the quadratic embedding, lambda=0 slice.

On the slow line h=0 of the canonical family, V = 1 - v - nu v^2 and
v0 solves nu v0^2 + v0 - 1 = 0. For lambda0 = lambda1 = 0 the height
coefficient is algebraic:

    zeta = - ell * l * (v + 2 v0) / (3 v0 D^2),

with ell = 1+2 nu v, l = 1+2 nu v0, D = 1+nu(v0+v). This yields
zeta(0, eps) = -1 and zeta(V, 0) = -1 + V/3 exactly, so the division

    zeta + 1 - V/3 = eps V Z

is holomorphic at (0,0). A rectangular ComplexInterval enclosure of the
numerator on a polydisc supplies a Cauchy majorant for Z on this slice.

The fold remainder uses a compact of (L, lambda1), sealed in
``omnibias.dynamics.fold_zeta``. The bound does not pass G1, physical C2,
first-hit, or Hilbert XVI.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from fractions import Fraction

from omnibias.core.collapse.verdict import adjudicate_residual
from omnibias.core.verified.complex_interval import ComplexInterval
from omnibias.core.verified.interval import Interval
from omnibias.dynamics.honesty_inventory import build_honesty

__all__ = [
    "CanonicalZetaReport",
    "SAMPLE_K",
    "SAMPLE_LAM0",
    "SAMPLE_LAM1",
    "SAMPLE_NU",
    "SAMPLE_V0",
    "cauchy_majorant",
    "complex_sqrt_right_halfplane",
    "enclose_v",
    "enclose_v0",
    "identity_verdicts",
    "k_lambda0",
    "polydisc_box",
    "report",
    "residual_V_factor",
    "residual_beta_linear_jet",
    "residual_c0_jet",
    "residual_field_matches_closed",
    "residual_k_implicit",
    "residual_k_lambda0",
    "residual_limiting_cubic",
    "residual_sqrt_inverse",
    "residual_v0_quadratic",
    "residual_zeta_at_zero",
    "residual_zeta_plus_one_canceled",
    "sample_z",
    "slow_line_V",
    "zeta_closed",
    "zeta_from_field",
]

SAMPLE_NU = Fraction(5, 16)
SAMPLE_V0 = Fraction(4, 5)
SAMPLE_K = Fraction(2)
SAMPLE_LAM0 = Fraction(0)
SAMPLE_LAM1 = Fraction(288, 125)
_POLYDISC_R = 0.125
_POLYDISC_E = 0.0625


def _honesty(*, inner_z_compact_bound: bool) -> dict[str, object]:
    return build_honesty(
        g1_passed=False,
        g4_passed=False,
        full_graphic_cyclicity_proved=False,
        full_hilbert16_solved=False,
        physical_return_membership_proved=False,
        fold_leading_c2_remainder=False,
        inner_z_compact_bound=inner_z_compact_bound,
        fold_z_compact_bound=False,
        orbit_continuation_on_kill=False,
        hk_theorem_24_used=False,
    )


def slow_line_V(nu: Fraction, v: Fraction) -> Fraction:
    """Exact slow-line coordinate ``V = 1 - v - nu v^2`` at ``r = -1``."""
    return 1 - v - nu * v**2


def k_lambda0(nu: Fraction, v0: Fraction) -> Fraction:
    """``k = 3 v0 / l`` on the ``lambda0 = lambda1 = 0`` slice."""
    ell0 = 1 + 2 * nu * v0
    if ell0 == 0 or v0 == 0:
        raise ValueError("k_lambda0 requires nonzero l and v0")
    return 3 * v0 / ell0


def zeta_closed(nu: Fraction, v: Fraction, v0: Fraction) -> Fraction:
    """Algebraic slow-line ``zeta`` at ``lambda0 = lambda1 = 0``."""
    if v0 == 0:
        raise ValueError("zeta_closed requires nonzero v0")
    ell = 1 + 2 * nu * v
    ell0 = 1 + 2 * nu * v0
    wall = 1 + nu * (v0 + v)
    if wall == 0:
        raise ValueError("zeta_closed requires nonzero D")
    return -(ell * ell0 * (v + 2 * v0)) / (3 * v0 * wall**2)


def residual_v0_quadratic(nu: Fraction, v0: Fraction) -> Fraction:
    """``nu v0^2 + v0 - 1``; vanishes on the slow-line root at ``r = -1``."""
    return nu * v0**2 + v0 - 1


def residual_sqrt_inverse(nu: Fraction, v: Fraction, v_coord: Fraction) -> Fraction:
    """``V - (1 - v - nu v^2)``."""
    return v_coord - slow_line_V(nu, v)


def residual_V_factor(nu: Fraction, v: Fraction, v0: Fraction, v_coord: Fraction) -> Fraction:
    """``V - (v0 - v)(1 + nu(v0 + v))``; vanishes once ``v0`` is a root."""
    return v_coord - (v0 - v) * (1 + nu * (v0 + v))


def residual_zeta_at_zero(nu: Fraction, v0: Fraction) -> Fraction:
    """``zeta(v0) + 1``; the V^2 jet is ``-eps`` so ``zeta(0, eps) = -1``."""
    return zeta_closed(nu, v0, v0) + 1


def residual_limiting_cubic(v_coord: Fraction) -> Fraction:
    """``zeta(V, 0) + 1 - V/3`` at ``nu = 0``, ``v = 1 - V``."""
    return zeta_closed(Fraction(0), 1 - v_coord, Fraction(1)) + 1 - v_coord / 3


def residual_k_lambda0(nu: Fraction, v0: Fraction) -> Fraction:
    """``k l - 3 v0`` on the ``lambda = 0`` slice."""
    ell0 = 1 + 2 * nu * v0
    return k_lambda0(nu, v0) * ell0 - 3 * v0


def residual_beta_linear_jet(nu: Fraction, v0: Fraction) -> Fraction:
    """Linear jet of zeta at V=0 equals ``1/(k l^2) = 1/(3 v0 l)``."""
    ell0 = 1 + 2 * nu * v0
    if v0 == 0 or ell0 == 0:
        raise ValueError("residual_beta_linear_jet requires nonzero v0 and l")
    kay = k_lambda0(nu, v0)
    return 1 / (3 * v0 * ell0) - 1 / (kay * ell0**2)


def residual_zeta_plus_one_canceled(
    nu: Fraction, v: Fraction, v0: Fraction
) -> Fraction:
    """Canceled numerator of ``zeta+1`` versus the closed form."""
    ell0 = 1 + 2 * nu * v0
    gap = v0 - v
    wall = 1 + nu * (v0 + v)
    if v0 == 0 or wall == 0:
        raise ValueError("residual_zeta_plus_one_canceled requires nonzero v0 and D")
    numer = gap * (ell0**2 + gap * (3 * v0 * nu**2 - 2 * ell0 * nu))
    denom = 3 * v0 * wall**2
    return (zeta_closed(nu, v, v0) + 1) * denom - numer


def k_implicit_rhs(
    nu: Fraction, v0: Fraction, kay: Fraction, lam0: Fraction, lam1: Fraction
) -> Fraction:
    """Right-hand side of the analytic ``k`` equation at ``r = -1``."""
    ell0 = 1 + 2 * nu * v0
    if ell0 == 0:
        raise ValueError("k_implicit_rhs requires nonzero l")
    return (
        3 * v0 / ell0
        + (nu**2 * kay**2 * lam1) / ell0**2
        + (4 * nu**4 * kay**3 * lam0) / ell0**4
    )


def residual_k_implicit(
    nu: Fraction, v0: Fraction, kay: Fraction, lam0: Fraction, lam1: Fraction
) -> Fraction:
    """``k`` minus the implicit unfolding right-hand side."""
    return kay - k_implicit_rhs(nu, v0, kay, lam0, lam1)


def zeta_from_field(
    nu: Fraction,
    v: Fraction,
    v0: Fraction,
    kay: Fraction,
    lam0: Fraction,
    lam1: Fraction,
) -> Fraction:
    """Slow-line ``zeta`` from the cubic embedding, including nonzero lambda."""
    if kay == 0:
        raise ValueError("zeta_from_field requires nonzero k")
    ell0 = 1 + 2 * nu * v0
    if ell0 == 0:
        raise ValueError("zeta_from_field requires nonzero l")
    ell = 1 + 2 * nu * v
    f0 = (nu**2 * kay**3 * lam0) / ell0
    f1 = -nu * kay**2 * lam1 - (2 * nu**3 * kay**3 * lam0) / ell0**2
    ptilde = 3 * v0**2 + f1
    mtilde = f0 - ptilde * v0 + v0**3
    slow = mtilde + ptilde * v - v**3
    vdot_over_eps = ell * slow / kay
    eps = nu * kay
    v_coord = slow_line_V(nu, v)
    if v_coord == 0:
        raise ValueError("zeta_from_field requires nonzero V")
    return (vdot_over_eps - eps**2 * lam0 - eps * lam1 * v_coord) / v_coord**2


def residual_field_matches_closed(nu: Fraction, v: Fraction, v0: Fraction) -> Fraction:
    """Field embedding at ``lambda=0`` recovers the closed-form ``zeta``."""
    kay = k_lambda0(nu, v0)
    return zeta_from_field(nu, v, v0, kay, Fraction(0), Fraction(0)) - zeta_closed(
        nu, v, v0
    )


def residual_c0_jet(
    nu: Fraction, v0: Fraction, kay: Fraction, lam0: Fraction, lam1: Fraction
) -> Fraction:
    """``Vdot`` at ``V=0`` equals ``eps^3 lambda0``."""
    ell0 = 1 + 2 * nu * v0
    if ell0 == 0:
        raise ValueError("residual_c0_jet requires nonzero l")
    f0 = (nu**2 * kay**3 * lam0) / ell0
    vdot = ell0 * nu * f0
    eps = nu * kay
    return vdot - eps**3 * lam0


def _verdict(residual: Fraction) -> str:
    box = Interval.point(float(residual)) if residual == 0 else Interval.from_rational(residual)
    return adjudicate_residual(box, existential=False).status


def identity_verdicts() -> dict[str, str]:
    nu, v0, v = SAMPLE_NU, SAMPLE_V0, Fraction(1, 2)
    v_coord = slow_line_V(nu, v)
    return {
        "v0_quadratic": _verdict(residual_v0_quadratic(nu, v0)),
        "sqrt_inverse": _verdict(residual_sqrt_inverse(nu, v, v_coord)),
        "V_factor": _verdict(residual_V_factor(nu, v, v0, v_coord)),
        "zeta_at_zero": _verdict(residual_zeta_at_zero(nu, v0)),
        "limiting_cubic": _verdict(residual_limiting_cubic(Fraction(2))),
        "k_lambda0": _verdict(residual_k_lambda0(nu, v0)),
        "beta_linear_jet": _verdict(residual_beta_linear_jet(nu, v0)),
        "zeta_plus_one_canceled": _verdict(residual_zeta_plus_one_canceled(nu, v, v0)),
        "k_implicit": _verdict(
            residual_k_implicit(nu, v0, SAMPLE_K, SAMPLE_LAM0, SAMPLE_LAM1)
        ),
        "field_matches_closed": _verdict(residual_field_matches_closed(nu, v, v0)),
        "c0_jet": _verdict(residual_c0_jet(nu, v0, SAMPLE_K, Fraction(1, 7), SAMPLE_LAM1)),
    }


def complex_sqrt_right_halfplane(z: ComplexInterval) -> ComplexInterval:
    """Principal square root on a rectangle with strictly positive real part."""
    if z.re.lo <= 0.0:
        raise ValueError("complex_sqrt_right_halfplane requires Re z > 0")
    mod = z.modulus()
    half = Interval.point(0.5)
    half_sum = (mod + z.re) * half
    if half_sum.lo < 0.0:
        half_sum = Interval(0.0, max(half_sum.hi, 0.0))
    sqrt_re = half_sum.sqrt()
    half_diff = (mod - z.re) * half
    half_diff = Interval(max(half_diff.lo, 0.0), max(half_diff.hi, 0.0))
    sqrt_im_abs = half_diff.sqrt()
    if z.im.lo >= 0.0:
        sqrt_im = sqrt_im_abs
    elif z.im.hi <= 0.0:
        sqrt_im = -sqrt_im_abs
    else:
        sqrt_im = Interval(-sqrt_im_abs.hi, sqrt_im_abs.hi)
    return ComplexInterval(sqrt_re, sqrt_im)


def polydisc_box(radius: float) -> ComplexInterval:
    return ComplexInterval.from_parts(Interval(-radius, radius), Interval(-radius, radius))


def enclose_v0(nu: ComplexInterval) -> ComplexInterval:
    four = ComplexInterval.from_value(4)
    root = complex_sqrt_right_halfplane(ComplexInterval.one() + four * nu)
    return ComplexInterval.from_value(2) / (ComplexInterval.one() + root)


def enclose_v(nu: ComplexInterval, v_coord: ComplexInterval) -> ComplexInterval:
    vm1 = v_coord - ComplexInterval.one()
    four = ComplexInterval.from_value(4)
    root = complex_sqrt_right_halfplane(ComplexInterval.one() - four * nu * vm1)
    return (ComplexInterval.from_value(-2) * vm1) / (ComplexInterval.one() + root)


def _enclose_zeta(
    nu: ComplexInterval, v: ComplexInterval, v0: ComplexInterval
) -> ComplexInterval:
    two = ComplexInterval.from_value(2)
    three = ComplexInterval.from_value(3)
    ell = ComplexInterval.one() + two * nu * v
    ell0 = ComplexInterval.one() + two * nu * v0
    wall = ComplexInterval.one() + nu * (v0 + v)
    return -(ell * ell0 * (v + two * v0)) / (three * v0 * (wall * wall))


def cauchy_majorant(
    *, radius_v: float = _POLYDISC_R, radius_nu: float = _POLYDISC_E
) -> dict[str, float | bool]:
    """Rectangular enclosure of ``zeta+1-V/3`` on a polydisc at ``lambda=0``.

    Schwarz on the axes ``V=0`` and ``nu=0`` then bounds ``Z = f/(eps V)``.
    The rectangle contains the polydisc, so the majorant is sound and possibly
    pessimistic. The fold ``(L, lambda1)`` compact is ``fold_zeta``.
    """
    if radius_v <= 0.0 or radius_nu <= 0.0:
        raise ValueError("cauchy_majorant requires positive radii")
    nu = polydisc_box(radius_nu)
    v_coord = polydisc_box(radius_v)
    v0 = enclose_v0(nu)
    v = enclose_v(nu, v_coord)
    zeta = _enclose_zeta(nu, v, v0)
    three = ComplexInterval.from_value(3)
    numerator = zeta + ComplexInterval.one() - v_coord / three
    ell0 = ComplexInterval.one() + ComplexInterval.from_value(2) * nu * v0
    kay = (ComplexInterval.from_value(3) * v0) / ell0
    k_min = kay.modulus().lo
    majorant = numerator.mag
    bound = (
        majorant / (radius_v * radius_nu * k_min) if k_min > 0.0 else float("inf")
    )
    return {
        "majorant": majorant,
        "k_min": k_min,
        "radius_v": radius_v,
        "radius_nu": radius_nu,
        "z_bound": bound,
        "finite": majorant < float("inf") and k_min > 0.0 and bound < float("inf"),
    }


def sample_z(nu: Fraction = SAMPLE_NU, v: Fraction = Fraction(1, 2)) -> Fraction:
    """Exact ``Z`` at a rational sample with ``eps V != 0``."""
    v0 = SAMPLE_V0
    v_coord = slow_line_V(nu, v)
    eps = nu * k_lambda0(nu, v0)
    if eps == 0 or v_coord == 0:
        raise ValueError("sample_z requires nonzero eps V")
    return (zeta_closed(nu, v, v0) + 1 - v_coord / 3) / (eps * v_coord)


@dataclass(frozen=True)
class CanonicalZetaReport:
    """Finite replay of the lambda=0 embedding. Not G1."""

    identities: Mapping[str, str]
    sample_abs_z: float
    cauchy: Mapping[str, float | bool]
    sample_below_bound: bool
    honesty: Mapping[str, object]

    def to_payload(self) -> dict[str, object]:
        return {
            "schema": "hilbert16-canonical-zeta-v1",
            "identities": dict(self.identities),
            "sample_abs_z": self.sample_abs_z,
            "cauchy": dict(self.cauchy),
            "sample_below_bound": self.sample_below_bound,
            "lambda_slice": "lambda0=lambda1=0",
            "honesty": dict(self.honesty),
            "scope": (
                "Algebraic r=-1 slow-line zeta and a Cauchy majorant on the "
                "lambda=0 slice. Fold compact is fold_zeta. Not G1 or "
                "Hilbert XVI."
            ),
        }


def report() -> CanonicalZetaReport:
    """Replay exact identities and enclose a Cauchy majorant on lambda=0."""
    identities = identity_verdicts()
    sample = abs(float(sample_z()))
    try:
        cauchy = cauchy_majorant()
        sealed = bool(cauchy["finite"]) and all(
            status == "PROVED" for status in identities.values()
        )
        z_bound = float(cauchy["z_bound"])
        below = sealed and sample <= z_bound
        inner = sealed and below
    except (ValueError, ZeroDivisionError):
        cauchy = {
            "majorant": float("inf"),
            "k_min": 0.0,
            "radius_v": _POLYDISC_R,
            "radius_nu": _POLYDISC_E,
            "z_bound": float("inf"),
            "finite": False,
        }
        below = False
        inner = False
    return CanonicalZetaReport(
        identities=identities,
        sample_abs_z=sample,
        cauchy=cauchy,
        sample_below_bound=below,
        honesty=_honesty(inner_z_compact_bound=inner),
    )
