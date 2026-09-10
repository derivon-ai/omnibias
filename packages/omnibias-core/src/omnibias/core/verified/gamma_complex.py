# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Complex logarithm / Gamma enclosures for the functional-equation evaluator.

This module encloses ``log``, ``exp``, ``sin`` / ``cos``, and ``Gamma`` on
complex rectangles. It is the substrate for ``chi(s)``, ``xi(s)``, and the
left half-plane zeta evaluator. It does **not** prove analytic continuation
as a theorem, locate zeros, or infer the Riemann Hypothesis.

Stirling is applied only after a recurrence shift into the sector
``Re(w) >= shift_to`` and ``Re(w) >= |Im(w)|`` (so ``|arg w| <= pi/4`` over
the whole rectangle). On that sector the remainder after ``p`` Stirling
terms is bounded in magnitude by the first omitted term (Whittaker &
Watson, *A Course of Modern Analysis*, the classical Stirling remainder
on ``|arg z| <= pi/4``). The remainder is returned as an axis-aligned
square containing that disc.

Poles at the non-positive integers are refused. The principal logarithm
refuses the origin and the cut ``(-inf, 0]``.
"""

from __future__ import annotations

import math
from fractions import Fraction

from omnibias.core.verified.coeffs import bernoulli_number_exact
from omnibias.core.verified.complex_interval import ComplexInterval, ComplexLike
from omnibias.core.verified.interval import Interval
from omnibias.core.verified.transcend import (
    PI_IV,
    atan_iv,
    cos_iv,
    cosh_iv,
    exp_iv,
    ln_iv,
    sin_iv,
    sinh_iv,
)

_HALF = Interval.point(0.5)
_STIRLING_SHIFT_TO = 8.0
_STIRLING_TERMS = 6
_MAX_SHIFT = 512


def _contains_origin(z: ComplexInterval) -> bool:
    return z.re.contains_zero() and z.im.contains_zero()


def _crosses_branch_cut(z: ComplexInterval) -> bool:
    if _contains_origin(z):
        return True
    return z.re.hi <= 0.0 and z.im.contains_zero()


def _atan2_point(y: float, x: float) -> Interval:
    if x == 0.0 and y == 0.0:
        raise ValueError("atan2 is undefined at the origin")
    yi = Interval.point(y)
    xi = Interval.point(x)
    half = PI_IV / Interval.from_rational(2)
    if x > 0.0:
        return atan_iv(yi / xi)
    if x < 0.0 and y >= 0.0:
        return atan_iv(yi / xi) + PI_IV
    if x < 0.0 and y < 0.0:
        return atan_iv(yi / xi) - PI_IV
    if y > 0.0:
        return half
    return -half


def arg_ci(z: ComplexInterval) -> Interval:
    """Principal-argument enclosure, or raise if it is not well-defined."""
    if _contains_origin(z) or _crosses_branch_cut(z):
        raise ValueError("arg_ci: argument meets the origin or principal cut")
    corners = (
        (z.re.lo, z.im.lo),
        (z.re.lo, z.im.hi),
        (z.re.hi, z.im.lo),
        (z.re.hi, z.im.hi),
    )
    pieces = [_atan2_point(im, re) for re, im in corners]
    return Interval(min(p.lo for p in pieces), max(p.hi for p in pieces))


def log_ci(z: ComplexLike) -> ComplexInterval:
    r"""Principal logarithm ``ln|z| + i Arg(z)`` on a rectangle off the cut."""
    z_ci = ComplexInterval.from_value(z)
    arg = arg_ci(z_ci)
    mod = z_ci.modulus()
    if mod.lo <= 0.0:
        raise ValueError("log_ci: modulus is not strictly positive")
    return ComplexInterval(ln_iv(mod), arg)


def exp_ci(z: ComplexLike) -> ComplexInterval:
    r"""``exp(a+ib) = e^a (cos b + i sin b)``."""
    z_ci = ComplexInterval.from_value(z)
    ea = exp_iv(z_ci.re)
    return ComplexInterval(ea * cos_iv(z_ci.im), ea * sin_iv(z_ci.im))


def sin_ci(z: ComplexLike) -> ComplexInterval:
    r"""``sin(x+iy) = sin x cosh y + i cos x sinh y``."""
    z_ci = ComplexInterval.from_value(z)
    return ComplexInterval(
        sin_iv(z_ci.re) * cosh_iv(z_ci.im),
        cos_iv(z_ci.re) * sinh_iv(z_ci.im),
    )


def cos_ci(z: ComplexLike) -> ComplexInterval:
    r"""``cos(x+iy) = cos x cosh y - i sin x sinh y``."""
    z_ci = ComplexInterval.from_value(z)
    return ComplexInterval(
        cos_iv(z_ci.re) * cosh_iv(z_ci.im),
        -(sin_iv(z_ci.re) * sinh_iv(z_ci.im)),
    )


def _half_ln_two_pi() -> Interval:
    return _HALF * ln_iv(PI_IV * 2)


def _pow_neg_int(w: ComplexInterval, n: int) -> ComplexInterval:
    """Enclosure of ``w^{-n}`` for a positive integer ``n``."""
    if n < 1:
        raise ValueError(f"n must be >= 1, got {n}")
    return exp_ci(ComplexInterval.from_parts(Interval.point(float(-n))) * log_ci(w))


def _stirling_log_gamma_ci(w: ComplexInterval, terms: int) -> ComplexInterval:
    r"""Stirling series for ``ln Gamma(w)`` with a disc remainder.

    Requires the caller to have shifted ``w`` into ``Re(w) >= |Im(w)|`` and
    ``Re(w) >= shift_to`` so ``|arg w| <= pi/4`` and the first-omitted-term
    magnitude bounds the remainder.
    """
    ln_w = log_ci(w)
    main = (w - ComplexInterval.from_parts(_HALF)) * ln_w - w
    main = main + ComplexInterval.from_parts(_half_ln_two_pi())
    series = ComplexInterval.zero()
    for j in range(1, terms + 1):
        coeff = Interval.from_rational(
            bernoulli_number_exact(2 * j) / Fraction((2 * j) * (2 * j - 1))
        )
        series = series + ComplexInterval.from_parts(coeff) * _pow_neg_int(w, 2 * j - 1)
    p = terms + 1
    omitted = ComplexInterval.from_parts(
        Interval.from_rational(bernoulli_number_exact(2 * p) / Fraction((2 * p) * (2 * p - 1)))
    ) * _pow_neg_int(w, 2 * p - 1)
    radius = omitted.mag
    remainder = ComplexInterval(Interval(-radius, radius), Interval(-radius, radius))
    return main + series + remainder


def _contains_nonpositive_integer(z: ComplexInterval) -> bool:
    if not z.im.contains_zero():
        return False
    n0 = math.ceil(z.re.lo - 1e-15)
    n1 = math.floor(z.re.hi + 1e-15)
    for n in range(n0, n1 + 1):
        if n <= 0:
            return True
    return False


def _in_stirling_sector(w: ComplexInterval, shift_to: float) -> bool:
    return w.re.lo >= shift_to and w.re.lo >= w.im.mag


def _log_gamma_positive_half(z: ComplexInterval, *, terms: int, shift_to: float) -> ComplexInterval:
    """``ln Gamma`` after shifting into the Stirling sector; no reflection."""
    n = 0
    w = z
    while not _in_stirling_sector(w, shift_to):
        n += 1
        if n > _MAX_SHIFT:
            raise ValueError(
                "log_gamma_ci: could not shift into the Stirling sector "
                f"(Re.lo={z.re.lo!r}, Im.mag={z.im.mag!r})"
            )
        w = z + ComplexInterval.from_parts(Interval.point(float(n)))
    log_w = _stirling_log_gamma_ci(w, terms)
    shift = ComplexInterval.zero()
    for k in range(n):
        shift = shift + log_ci(z + ComplexInterval.from_parts(Interval.point(float(k))))
    return log_w - shift


def log_gamma_ci(
    z: ComplexLike,
    *,
    terms: int = _STIRLING_TERMS,
    shift_to: float = _STIRLING_SHIFT_TO,
) -> ComplexInterval:
    r"""Enclosure of ``ln Gamma(z)`` off the non-positive integers.

    Negative real parts use the multiplicative reflection
    ``Gamma(z) = pi / (sin(pi z) Gamma(1-z))`` and then take ``log`` of
    the enclosed value (principal value ``ln|g| + i pi`` on a negative-real
    image). This avoids a logarithm of ``sin(pi z)`` on the cut.
    """
    if terms < 1:
        raise ValueError(f"terms must be >= 1, got {terms}")
    if shift_to <= 0.0:
        raise ValueError(f"shift_to must be positive, got {shift_to}")
    z_ci = ComplexInterval.from_value(z)
    if _contains_nonpositive_integer(z_ci):
        raise ValueError("log_gamma_ci: rectangle contains a non-positive integer pole")
    if z_ci.re.hi < 0.5:
        pi_z = ComplexInterval.from_parts(PI_IV) * z_ci
        sine = sin_ci(pi_z)
        if _contains_origin(sine):
            raise ValueError("log_gamma_ci: sin(pi z) meets 0 (pole of Gamma)")
        g = gamma_ci(z_ci, terms=terms, shift_to=shift_to)
        try:
            return log_ci(g)
        except ValueError:
            if g.im.contains(0.0) and g.re.hi < 0.0 and not _contains_origin(g):
                return ComplexInterval(ln_iv(g.modulus()), PI_IV)
            raise
    return _log_gamma_positive_half(z_ci, terms=terms, shift_to=shift_to)


def gamma_ci(
    z: ComplexLike,
    *,
    terms: int = _STIRLING_TERMS,
    shift_to: float = _STIRLING_SHIFT_TO,
) -> ComplexInterval:
    r"""Enclosure of ``Gamma(z)``.

    On ``Re(z) < 1/2`` the multiplicative reflection
    ``Gamma(z) = pi / (sin(pi z) Gamma(1-z))`` is used so a negative-real
    sine does not require a logarithm on the principal cut.
    """
    if terms < 1:
        raise ValueError(f"terms must be >= 1, got {terms}")
    if shift_to <= 0.0:
        raise ValueError(f"shift_to must be positive, got {shift_to}")
    z_ci = ComplexInterval.from_value(z)
    if _contains_nonpositive_integer(z_ci):
        raise ValueError("log_gamma_ci: rectangle contains a non-positive integer pole")
    if z_ci.re.hi < 0.5:
        sine = sin_ci(ComplexInterval.from_parts(PI_IV) * z_ci)
        if _contains_origin(sine):
            raise ValueError("gamma_ci: sin(pi z) meets 0 (pole of Gamma)")
        return ComplexInterval.from_parts(PI_IV) / (
            sine * gamma_ci(ComplexInterval.one() - z_ci, terms=terms, shift_to=shift_to)
        )
    return exp_ci(_log_gamma_positive_half(z_ci, terms=terms, shift_to=shift_to))


__all__ = [
    "arg_ci",
    "cos_ci",
    "exp_ci",
    "gamma_ci",
    "log_ci",
    "log_gamma_ci",
    "sin_ci",
]
