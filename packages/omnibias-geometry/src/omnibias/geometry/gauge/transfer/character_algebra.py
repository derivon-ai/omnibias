# SPDX-License-Identifier: Apache-2.0
"""Exact SU(2) central character algebra and conditional analytic norm bounds.

Keys are doubled spins ell; chi_ell has dimension ell+1. The A_s norm is
sum |c_ell|*(ell+1)**(s+1). Bounds below propagate a supplied A_s enclosure;
they do not establish the origin of that enclosure or any quantum theory.
"""

from __future__ import annotations

from collections.abc import Mapping
from fractions import Fraction as Q

CharacterPolynomial = dict[int, Q]


def _q(value: int | Q) -> Q:
    if type(value) is not int and not isinstance(value, Q):
        raise TypeError("exact integer or Fraction required")
    return Q(value)


def _integer(value: int, *, minimum: int = 0) -> None:
    if type(value) is not int or value < minimum:
        raise ValueError(f"integer at least {minimum} required")


def _polynomial(values: Mapping[int, int | Q]) -> CharacterPolynomial:
    result = {}
    for spin, raw in values.items():
        _integer(spin)
        value = _q(raw)
        if value:
            result[spin] = value
    return result


def su2_character_product(
    left: Mapping[int, int | Q], right: Mapping[int, int | Q],
) -> CharacterPolynomial:
    """Pointwise product by the complete Clebsch--Gordan character rule."""
    result: CharacterPolynomial = {}
    for a, ca in _polynomial(left).items():
        for b, cb in _polynomial(right).items():
            for spin in range(abs(a - b), a + b + 1, 2):
                result[spin] = result.get(spin, Q(0)) + ca * cb
    return {spin: value for spin, value in result.items() if value}


def su2_character_convolution(
    left: Mapping[int, int | Q], right: Mapping[int, int | Q],
) -> CharacterPolynomial:
    """Normalized-Haar central convolution, c_ell=a_ell*b_ell/(ell+1)."""
    a, b = _polynomial(left), _polynomial(right)
    return {spin: ca * b[spin] / (spin + 1) for spin, ca in a.items() if spin in b}


def su2_character_norm(values: Mapping[int, int | Q], *, norm_order: int = 2) -> Q:
    """Weighted Wiener norm A_s; s=2 controls the Laplacian uniformly."""
    _integer(norm_order)
    return sum((abs(c) * (spin + 1)**(norm_order + 1)
                for spin, c in _polynomial(values).items()), Q(0))


def su2_character_round(
    values: Mapping[int, int | Q], *, bits: int = 128, norm_order: int = 2,
) -> tuple[CharacterPolynomial, Q]:
    """Round coefficients down to a dyadic grid and return the exact A_s error.

This compression never discards its error. Integral coefficients, including
the unit Haar mean of a probability density, remain exact.
"""
    _integer(bits, minimum=1)
    source = _polynomial(values)
    scale = 1 << bits
    rounded = {spin: Q((c.numerator * scale) // c.denominator, scale)
               for spin, c in source.items()}
    error = su2_character_norm({s: source[s] - c for s, c in rounded.items()}, norm_order=norm_order)
    return {s: c for s, c in rounded.items() if c}, error


def su2_character_even_tail(
    ratio: int | Q, *, first_omitted: int, circle_norm: int | Q,
    norm_order: int = 2,
) -> Q:
    """A_s tail for an even analytic series with support ell<=m at degree 2m.

Premises: the analytic circle L1 norm is at most circle_norm and ratio is
|g|/radius. Coefficients obey |c_ell|<=(ell+1)*circle_norm/radius**(2m).
Only s=0,1,2 is supported by these explicit rational generating functions.
"""
    _integer(first_omitted)
    _integer(norm_order)
    t, bound = _q(ratio), _q(circle_norm)
    if not 0 <= t < 1 or bound < 0 or norm_order > 2:
        raise ValueError("need 0<=ratio<1, nonnegative circle norm, and norm_order<=2")
    x = t * t
    numerators = ((1, 1), (1, 4, 1), (1, 11, 11, 1))
    total = sum((Q(c) * x**k for k, c in enumerate(numerators[norm_order])), Q(0))
    total /= (1 - x)**(norm_order + 4)
    initial = sum((Q(sum(d**(norm_order + 2) for d in range(1, m + 2))) * x**m
                   for m in range(first_omitted)), Q(0))
    return bound * (total - initial)


def su2_character_log_bound(
    density: Mapping[int, int | Q], *, error: int | Q = 0,
    order: int = 6, norm_order: int = 2,
) -> tuple[CharacterPolynomial, Q]:
    """Polynomial log and A_s error for a true density in a supplied A_s ball.

Requires ||density-1||_A_s+error<1. This earns a convergent logarithm
conditional on the input enclosure; a failed sufficient condition raises.
"""
    _integer(order, minimum=1)
    delta = _q(error)
    if delta < 0:
        raise ValueError("nonnegative enclosure error required")
    centered = _polynomial(density)
    centered[0] = centered.get(0, Q(0)) - 1
    centered = {spin: c for spin, c in centered.items() if c}
    size = su2_character_norm(centered, norm_order=norm_order)
    if size + delta >= 1:
        raise ValueError("the supplied density ball is outside the unit logarithm ball")
    result: CharacterPolynomial = {}
    power: CharacterPolynomial = {0: Q(1)}
    for degree in range(1, order + 1):
        power = su2_character_product(power, centered)
        sign = Q(1 if degree % 2 else -1, degree)
        for spin, c in power.items():
            result[spin] = result.get(spin, Q(0)) + sign * c
    remainder = delta / (1 - size - delta)
    remainder += size**(order + 1) / ((order + 1) * (1 - size))
    return {spin: c for spin, c in result.items() if c}, remainder


def su2_character_laplacian(values: Mapping[int, int | Q]) -> CharacterPolynomial:
    """Exact unit-Casimir Laplacian: Delta chi_ell=-ell*(ell+2)*chi_ell/4."""
    return {spin: -Q(spin * (spin + 2), 4) * c
            for spin, c in _polynomial(values).items() if spin}


def su2_character_potential_bound(
    log_vacuum: Mapping[int, int | Q], *, error: int | Q = 0,
    kinetic: int | Q = 1,
) -> tuple[CharacterPolynomial, Q]:
    """Enclose V=k*(Delta S+|grad S|^2) in sup norm from an A_2 log bound.

The Schrödinger operator -k*Delta+V has exp(S) as its zero-energy vacuum.
An additive log constant does not affect this potential. The returned
potential keeps its scalar coefficient, fixing that energy convention.
"""
    polynomial, delta, scale = _polynomial(log_vacuum), _q(error), _q(kinetic)
    if delta < 0 or scale <= 0:
        raise ValueError("nonnegative error and positive kinetic coefficient required")
    lap = su2_character_laplacian(polynomial)
    lap_square = su2_character_laplacian(su2_character_product(polynomial, polynomial))
    mixed = su2_character_product(polynomial, lap)
    result = dict(lap)
    for spin, value in lap_square.items():
        result[spin] = result.get(spin, Q(0)) + value / 2
    for spin, value in mixed.items():
        result[spin] = result.get(spin, Q(0)) - value
    # Removing the polynomial scalar tightens derivative estimates.
    size = su2_character_norm({s: c for s, c in polynomial.items() if s})
    remainder = scale * delta * (1 + 2 * size + delta) / 4
    return {spin: scale * value for spin, value in result.items() if value}, remainder
