# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Host arithmetic for centered collision coordinates and bounded moment atoms.

Input floats are interpreted as their stored dyadic values. Conversion errors
bound conversion to the returned floats; they cannot recover earlier rounding.
No finite-spread truncated moment expansion is called an exact coordinate change.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from math import factorial
from typing import Literal

from omnibias.core.verified.coeffs import sigmoid_poly_coeffs_exact, tanh_poly_coeffs_exact
from omnibias.core.verified.interval import Interval

Activation = Literal["sigmoid", "tanh"]
_ZERO = Interval.point(0.0)
_ONE = Interval.point(1.0)


@dataclass(frozen=True)
class MomentInitialization:
    """Taylor-normalized moments, exact source arithmetic, and rounding boxes."""

    moments: tuple[float, ...]
    exact_moments: tuple[Fraction, ...]
    conversion_errors: tuple[Interval, ...]
    center: Fraction
    offsets: tuple[Fraction, ...]
    coefficients: tuple[Fraction, ...]


def initialize_moments(
    offsets: tuple[float | Fraction, ...],
    coefficients: tuple[float | Fraction, ...],
    *,
    order: int,
    center: float | Fraction = 0.0,
) -> MomentInitialization:
    """Compute ``sum(a_i * (b_i-center)**k)/k!`` in exact rational arithmetic."""
    if order < 0 or not offsets or len(offsets) != len(coefficients):
        raise ValueError("nonempty matching arrays and nonnegative order are required")
    c = Fraction(center)
    bs = tuple(Fraction(b) - c for b in offsets)
    aa = tuple(Fraction(a) for a in coefficients)
    exact = tuple(
        sum((a * b**k for a, b in zip(aa, bs, strict=True)), Fraction()) / factorial(k)
        for k in range(order + 1)
    )
    rounded = tuple(float(q) for q in exact)
    errors = tuple(
        Interval.from_rational(q - Fraction(f)) for q, f in zip(exact, rounded, strict=True)
    )
    return MomentInitialization(rounded, exact, errors, c, bs, aa)


def pair_moments(a_plus: float, a_minus: float, h: float) -> MomentInitialization:
    """Exact stored-operand ``(m0,m1)`` for a pair with spread ``h``."""
    if h < 0:
        raise ValueError("h must be nonnegative")
    return initialize_moments((h, -h), (a_plus, a_minus), order=1)


def derivative_bound(activation: Activation, order: int) -> Interval:
    """Global bound on ``abs(sigma**(order))`` from the shared integer polynomial.

    This conservative coefficient bound uses sigmoid in [0,1] and tanh in
    [-1,1]. It avoids a numerical maximization being mistaken for a proof.
    """
    if order < 0:
        raise ValueError("order must be nonnegative")
    if activation not in ("sigmoid", "tanh"):
        raise ValueError("supported analytic bounds are sigmoid and tanh")
    coefficients = (
        sigmoid_poly_coeffs_exact(order)
        if activation == "sigmoid"
        else tanh_poly_coeffs_exact(order)
    )
    return Interval.from_rational(sum(abs(c) for c in coefficients))


def cluster_remainder(
    initialization: MomentInitialization,
    *,
    activation: Activation = "sigmoid",
    spatial_order: int = 0,
    weight: float = 1.0,
) -> Interval:
    """Uniform spatial-derivative error for a common-weight bias cluster.

    Includes moment float-conversion error. Spatial derivatives are along a
    scalar direction with common affine slope ``weight``.
    """
    if spatial_order < 0:
        raise ValueError("spatial_order must be nonnegative")
    n = len(initialization.moments)
    tail = sum(
        (
            abs(a) * abs(b) ** n
            for a, b in zip(initialization.coefficients, initialization.offsets, strict=True)
        ),
        Fraction(),
    ) / factorial(n)
    error = Interval.from_rational(tail) * derivative_bound(activation, n + spatial_order)
    for k, rounding in enumerate(initialization.conversion_errors):
        error += Interval.point(rounding.mag) * derivative_bound(activation, k + spatial_order)
    error *= Interval.point(abs(weight)) ** spatial_order
    return Interval(0.0, error.hi)


def pair_derivative_error(
    *,
    rho: Interval,
    eta: Interval,
    m0: Interval,
    m1: Interval,
    weight: Interval = _ONE,
    direction: Interval = _ZERO,
    spatial_order: int = 0,
    activation: Activation = "sigmoid",
) -> Interval:
    """Bound the pair-to-derivative limit error, including affine direction pairs.

    ``z=w*x+b``, ``eta=v*x+c``; weight encloses w, direction encloses v,
    eta encloses its values on the declared domain. Taylor's theorem in h
    is applied to ``sigma**(n)(z+h*eta)*(w+h*v)**n``. This also handles the
    product-rule terms in derivatives of ``eta*sigma'(z)``.
    """
    if rho.lo < 0 or spatial_order < 0:
        raise ValueError("rho and spatial_order must be nonnegative")
    if rho.hi == 0:
        return _ZERO
    from math import comb, inf, nextafter, sqrt

    h = Interval(0.0, nextafter(sqrt(rho.hi), inf))
    slope = Interval.point(weight.mag) + h * Interval.point(direction.mag)
    n = spatial_order

    def bound(k: int) -> Interval:
        total = Interval.point(0.0)
        for j in range(min(k, n) + 1):
            coefficient = comb(k, j) * factorial(n) // factorial(n - j)
            total += (
                coefficient
                * Interval.point(direction.mag) ** j
                * slope ** (n - j)
                * Interval.point(eta.mag) ** (k - j)
                * derivative_bound(activation, n + k - j)
            )
        return total

    error = rho * (Interval.point(m0.mag) * bound(2) / 2 + Interval.point(m1.mag) * bound(3) / 6)
    return Interval(0.0, error.hi)


def pair_series_error(
    *,
    rho: Interval,
    eta: Interval,
    terms: int,
    activation: Activation = "sigmoid",
    spatial_order: int = 0,
    weight: Interval = _ONE,
    direction: Interval = _ZERO,
) -> tuple[Interval, Interval]:
    """Declared truncation error of the even rho series in centered_pair.

    These bounds concern the analytic truncation, not backend floating-point
    evaluation. The rigorous register must also enclose arithmetic rounding.
    """
    if rho.lo < 0 or terms < 1 or spatial_order < 0:
        raise ValueError("nonnegative rho/order and positive series terms required")
    if rho.hi == 0:
        return _ZERO, _ZERO
    from math import comb, inf, nextafter, sqrt

    h = Interval(0.0, nextafter(sqrt(rho.hi), inf))
    slope = Interval.point(weight.mag) + h * Interval.point(direction.mag)
    n = spatial_order

    def bound(k: int) -> Interval:
        total = _ZERO
        for j in range(min(k, n) + 1):
            coefficient = comb(k, j) * factorial(n) // factorial(n - j)
            total += (
                coefficient
                * Interval.point(direction.mag) ** j
                * slope ** (n - j)
                * Interval.point(eta.mag) ** (k - j)
                * derivative_bound(activation, n + k - j)
            )
        return total

    k = 2 * terms
    a = rho**terms * bound(k) / factorial(k)
    b = rho**terms * bound(k + 1) / factorial(k + 1)
    return Interval(0, a.hi), Interval(0, b.hi)


__all__ = [
    "Activation",
    "MomentInitialization",
    "cluster_remainder",
    "derivative_bound",
    "initialize_moments",
    "pair_derivative_error",
    "pair_moments",
    "pair_series_error",
]
