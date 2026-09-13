# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
r"""Regular-parabola passage estimates for a quadratic Hilbert-16 family.

For ``A=1, mu=0`` in the Huzak--Kristiansen family, the exact coordinate
``w = y - x**2/2 + C/2`` gives ``xdot=q-w, wdot=2*x*w``, where
``q=((x+1)**2+C-1)/2``.  This module treats ``C>1`` only.  The normalized
section coordinate is ``z=w/q``; ``G_R`` denotes its transition from ``x=-R``
to ``x=R``.  It is not a Poincare return map.

The identity ``(w/(q+w)**2)' = -2*(w/(q+w)**2)/(q+w)`` (x derivative)
gives the analytic limit ``G=H^{-1}(c*H)``, ``H(z)=z/(1+z)**2``,
``c=exp(-4*pi/sqrt(C-1))``.  On the complex disk ``|z|<=1/16`` and ``R>=6``,

    sup |G_R-G| <= (12/25)*c*(exp(d_R)-1),
    d_R = 82*R/(25*q(-R)) + 8*R/(R**2-1).

Cauchy's estimate controls each fixed derivative on ``|z|<=1/32``.
See the accompanying HILBERT16.md for the proof and full-goal obligations.
The interval checks below can establish negative Schwarzian for this finite
regular passage.  They do not cover the singular passage, matching charts,
the ``C=1`` face, full quadratic cyclicity, or Hilbert's sixteenth problem.
No theorem-prover flag is asserted by this numerical module.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from math import factorial, isfinite

from omnibias.core.verified.interval import Interval, IntervalLike
from omnibias.core.verified.transcend import (
    PI_IV,
    atan_iv,
    exp_iv,
    ln_iv,
    require_rigorous_backend,
)

_HALF = Interval.from_rational(Fraction(1, 2))


def _finite(value: IntervalLike, name: str) -> Interval:
    out = Interval.from_value(value)
    if not isfinite(out.lo) or not isfinite(out.hi):
        raise ValueError(f"{name} must have finite endpoints")
    return out


def _parameter(c_parameter: IntervalLike) -> Interval:
    out = _finite(c_parameter, "C")
    if out.lo <= 1.0:
        raise ValueError("the regular-parabola estimate requires C > 1")
    return out


def _limit_multiplier(c_parameter: Interval) -> Interval:
    require_rigorous_backend()
    return exp_iv(-4 * PI_IV / (c_parameter - 1).sqrt())


def regular_outer_multiplier(
    c_parameter: IntervalLike, x_start: float, x_end: float
) -> Interval:
    r"""Enclose ``d w(x_end)/d w(x_start)`` at ``w=0`` in x-time.

    This is the derivative of a regular section-to-section passage in the
    unnormalized w coordinate.  It equals ``exp(integral 2*x/q(x) dx)``.
    """
    c_iv = _parameter(c_parameter)
    left, right = _finite(x_start, "x_start"), _finite(x_end, "x_end")
    if x_end < x_start:
        raise ValueError("x_end must be at least x_start")
    require_rigorous_backend()
    root = (c_iv - 1).sqrt()
    q_left = (left + 1).pow_int(2) + c_iv - 1
    q_right = (right + 1).pow_int(2) + c_iv - 1
    log_value = 2 * ln_iv(q_right / q_left) - (4 / root) * (
        atan_iv((right + 1) / root) - atan_iv((left + 1) / root)
    )
    return exp_iv(log_value)


def _h_derivatives(z: Interval) -> tuple[Interval, Interval, Interval]:
    plus = 1 + z
    return (
        (1 - z) / plus.pow_int(3),
        2 * (z - 2) / plus.pow_int(4),
        6 * (3 - z) / plus.pow_int(5),
    )


def normalized_outer_limit_jet(
    c_parameter: IntervalLike, z: IntervalLike
) -> tuple[Interval, Interval, Interval, Interval]:
    r"""Enclose ``(G, G', G'', G''')`` on a real box within ``[-1/32,1/32]``.

    These are derivatives, not factorial-normalized Taylor coefficients.
    Implicit differentiation of ``H(G)=c*H(z)`` avoids numerical differences.
    """
    c_iv, z_iv = _parameter(c_parameter), _finite(z, "z")
    if z_iv.lo < -1 / 32 or z_iv.hi > 1 / 32:
        raise ValueError("z must lie in [-1/32, 1/32]")
    c = _limit_multiplier(c_iv)
    target = c * z_iv / (1 + z_iv).pow_int(2)
    # Stable inverse of H on the branch G(0)=0; no subtraction of nearby roots.
    g = 2 * target / (1 - 2 * target + (1 - 4 * target).sqrt())
    h1, h2, h3 = _h_derivatives(z_iv)
    k1, k2, k3 = _h_derivatives(g)
    d1 = c * h1 / k1
    d2 = (c * h2 - k2 * d1.pow_int(2)) / k1
    d3 = (c * h3 - k3 * d1.pow_int(3) - 3 * k2 * d1 * d2) / k1
    return g, d1, d2, d3


@dataclass(frozen=True)
class OuterPassageError:
    """Uniform analytic comparison of G_R and G, not a return-map certificate."""

    c_parameter: Interval
    section_cutoff: float
    limit_multiplier: Interval
    exponent_bound: Interval
    uniform_value_error: float
    derivative_errors: tuple[float, float, float, float]


def normalized_outer_error(
    c_parameter: IntervalLike, section_cutoff: float
) -> OuterPassageError:
    """Bound derivatives 0..3 of G_R-G on the real interval [-1/32,1/32].

    The value bound holds on the larger complex disk of radius 1/16.
    A C interval is handled uniformly; no sampled-parameter inference is used.
    """
    c_iv = _parameter(c_parameter)
    radius = _finite(section_cutoff, "section_cutoff")
    if section_cutoff < 6:
        raise ValueError("section_cutoff must be at least 6")
    q_left = ((radius - 1).pow_int(2) + c_iv - 1) * _HALF
    d = 82 * radius / (25 * q_left) + 8 * radius / (radius.pow_int(2) - 1)
    c = _limit_multiplier(c_iv)
    error = Interval.from_rational(Fraction(12, 25)) * c * (exp_iv(d) - 1)
    bound = max(0.0, error.hi)
    error_list = [
        (Interval.from_rational(factorial(k) * 32**k) * Interval.point(bound)).hi
        for k in range(4)
    ]
    errors = (error_list[0], error_list[1], error_list[2], error_list[3])
    if not all(isfinite(item) for item in errors):
        raise ArithmeticError("the derivative error bounds overflow; refine the parameter range")
    return OuterPassageError(c_iv, section_cutoff, c, d, bound, errors)


@dataclass(frozen=True)
class OuterSchwarzianCheck:
    """A finite-R regular-passage check with visible unresolved section cells."""

    comparison: OuterPassageError
    cells: int
    schwarzian_bounds: tuple[Interval | None, ...]
    unresolved_cells: tuple[Interval, ...]

    @property
    def certified_negative(self) -> bool:
        """Every section cell has a strictly negative outward Schwarzian bound."""
        return not self.unresolved_cells

    @property
    def certified_for_larger_cutoffs(self) -> bool:
        """The same negative bound holds for every R >= the checked cutoff.

        R/q(-R) decreases when R**2 >= C, and R/(R**2-1) decreases for
        R>1.  Thus the comparison error only improves on this entire ray.
        """
        radius_squared = Interval.point(self.comparison.section_cutoff).pow_int(2)
        return self.certified_negative and self.comparison.c_parameter.hi <= radius_squared.lo


def certify_outer_schwarzian(
    c_parameter: IntervalLike,
    section_cutoff: float,
    *,
    cells: int = 32,
) -> OuterSchwarzianCheck:
    r"""Check ``S G_R < 0`` on [-1/32,1/32] for the base quadratic family.

    ``S f = f'''/f' - (3/2)*(f''/f')**2``.  This computation controls the
    actual finite passage through the proved complex comparison and Cauchy
    bounds.  Failure leaves unresolved cells; it does not infer a sign.
    """
    if isinstance(cells, bool) or not isinstance(cells, int) or cells < 1:
        raise ValueError("cells must be a positive integer")
    comparison = normalized_outer_error(c_parameter, section_cutoff)
    bounds: list[Interval | None] = []
    unresolved: list[Interval] = []
    for index in range(cells):
        left = Interval.from_rational(Fraction(-1, 32) + Fraction(index, 16 * cells))
        right = Interval.from_rational(Fraction(-1, 32) + Fraction(index + 1, 16 * cells))
        box = Interval(left.lo, right.hi)
        _, d1, d2, d3 = normalized_outer_limit_jet(comparison.c_parameter, box)
        d1 = d1 + Interval(-comparison.derivative_errors[1], comparison.derivative_errors[1])
        d2 = d2 + Interval(-comparison.derivative_errors[2], comparison.derivative_errors[2])
        d3 = d3 + Interval(-comparison.derivative_errors[3], comparison.derivative_errors[3])
        if d1.lo <= 0:
            bounds.append(None)
            unresolved.append(box)
            continue
        schwarzian = d3 / d1 - Interval.from_rational(Fraction(3, 2)) * (d2 / d1).pow_int(2)
        bounds.append(schwarzian)
        if schwarzian.hi >= 0:
            unresolved.append(box)
    return OuterSchwarzianCheck(comparison, cells, tuple(bounds), tuple(unresolved))


@dataclass(frozen=True)
class OuterSplittingAsymptotics:
    """First parameter variations at A=1, mu=0, z_in=0; not nonlinear bounds.

    The alpha, mu2, mu3 entries are limits of R**(-2) times the respective
    derivative of the normalized outer transition.  The mu1_log entry is
    the limit of that derivative divided by R**2 * log(R).
    """

    c_parameter: Interval
    alpha: Interval
    mu1_log: Interval
    mu2: Interval
    mu3: Interval


def outer_splitting_asymptotics(c_parameter: IntervalLike) -> OuterSplittingAsymptotics:
    r"""Closed-form leading sensitivity of the actual quadratic unfolding.

    With alpha=A-1 and c=exp(-4*pi/sqrt(C-1)), the ordinary leading
    combination is alpha+(C+6)*mu2+3*mu3.  The exceptional mu1 direction
    contributes 8*(C+3)*mu1*log(R) in the same normalization.
    This is a first-variation theorem; it does not bound nonlinear remainders
    uniformly for joint parameter/cutoff limits.
    """
    c_iv = _parameter(c_parameter)
    defect = 1 - _limit_multiplier(c_iv)
    alpha = -defect / (8 * (c_iv + 3))
    return OuterSplittingAsymptotics(c_iv, alpha, -defect, (c_iv + 6) * alpha, 3 * alpha)


@dataclass(frozen=True)
class OuterFirstVariationRay:
    """R**(-2) first-variation bounds for every cutoff R >= cutoff_min.

    All derivatives are evaluated at A=1, mu=0, with incoming normalized
    coordinate zero.  No finite-perturbation or return-map claim is made.
    The logarithmic mu1 direction is deliberately not bounded by these
    integrable-tail estimates.
    """

    c_parameter: Interval
    cutoff_min: float
    alpha: Interval
    mu2: Interval
    mu3: Interval


def outer_first_variation_ray(
    c_parameter: IntervalLike, cutoff_min: float
) -> OuterFirstVariationRay:
    r"""Enclose three parameter derivatives uniformly over an unbounded ray.

    At the base field, the exact first variation is
    ``q(R)*exp(-4*theta(R)/sqrt(C-1))*integral E*b/q**3``.
    Rescaling E by its right-end limit avoids exponentially large factors.
    Beyond |x|>=R>=sqrt(C), the omitted tails for alpha,mu2,mu3 are
    nonpositive and have magnitudes bounded respectively by
    ``64/(3*(R-1)**3), 128/(R-1), 64/(R-1)``.
    """
    c_iv = _parameter(c_parameter)
    radius = _finite(cutoff_min, "cutoff_min")
    if cutoff_min < 6 or c_iv.hi > radius.pow_int(2).lo:
        raise ValueError("need cutoff_min >= 6 and C <= cutoff_min**2")
    leading = outer_splitting_asymptotics(c_iv)
    p_upper = (
        _HALF + 1 / radius + c_iv / (2 * radius.pow_int(2))
    ) * exp_iv(4 / (radius + 1))
    prefactor = Interval(0.5, p_upper.hi)
    tails = (
        (64 / (3 * (radius - 1).pow_int(3))).hi,
        (128 / (radius - 1)).hi,
        (64 / (radius - 1)).hi,
    )
    # The full-line integral is twice its R**(-2) normalized limiting value.
    alpha = prefactor * (2 * leading.alpha + Interval(0.0, tails[0]))
    mu2 = prefactor * (2 * leading.mu2 + Interval(0.0, tails[1]))
    mu3 = prefactor * (2 * leading.mu3 + Interval(0.0, tails[2]))
    return OuterFirstVariationRay(c_iv, cutoff_min, alpha, mu2, mu3)


__all__ = [
    "OuterFirstVariationRay",
    "OuterPassageError",
    "OuterSchwarzianCheck",
    "OuterSplittingAsymptotics",
    "certify_outer_schwarzian",
    "normalized_outer_error",
    "normalized_outer_limit_jet",
    "outer_first_variation_ray",
    "outer_splitting_asymptotics",
    "regular_outer_multiplier",
]
