# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Shared bounded interval actions for Gaussian log-weight derivatives.

Callers own aggregate storage reservations and the mathematical validity of
cached posterior, projector and series data. These algebraic helpers use the
caller's cumulative arithmetic meter and require no transcendental backend,
linear solve, certificate operation or backend-state mutation.
"""

from __future__ import annotations

from fractions import Fraction

from omnibias.core.gaussian_jet import _Arithmetic
from omnibias.core.verified._gaussian_moments import _add, _multiply, _zero
from omnibias.core.verified.gaussian_heat import _finite, _rational
from omnibias.core.verified.interval import Interval
from omnibias.core.verified.inverse_gaussian_heat import _sqrt_upper

Vector = tuple[Fraction, ...]
Matrix = tuple[Vector, ...]
Box = tuple[Interval, ...]
IntervalMatrix = tuple[Box, ...]
Pair = tuple[Fraction, Fraction]
_ZERO = Interval.point(0.0)


def _shape(probabilities: Box, centers: Matrix) -> tuple[int, int]:
    count = len(centers)
    if not count or not centers[0] or len(probabilities) != count:
        raise ValueError("posterior and projected-center shapes must match and be nonempty")
    dimension = len(centers[0])
    if any(len(row) != dimension for row in centers):
        raise ValueError("projected centers must be rectangular")
    if any(isinstance(value, bool) or not isinstance(value, (int, Fraction)) for row in centers for value in row):
        raise TypeError("projected centers must be exact int or Fraction")
    if any(centers[0]):
        raise ValueError("the first projected-center row must be exactly zero")
    for probability in probabilities:
        _finite(probability)
        if probability.lo < 0 or probability.hi > 1:
            raise ValueError("posterior intervals must lie in [0,1]")
    return count, dimension


def _bounds(value: Interval, arithmetic: _Arithmetic) -> Pair:
    _finite(value)
    return arithmetic.exact(Fraction(value.lo)), arithmetic.exact(Fraction(value.hi))


def _weighted_bounds(probabilities: tuple[Pair, ...], values: Vector, arithmetic: _Arithmetic) -> Pair:
    """Exact endpoint accumulation, clipped by the normalized-weight hull."""
    lower = upper = Fraction(0)
    for pair, value in zip(probabilities, values, strict=True):
        if value:
            products = tuple(arithmetic.mul(value, endpoint) for endpoint in pair)
            lower, upper = arithmetic.add(lower, min(products)), arithmetic.add(upper, max(products))
    return max(lower, min(values)), min(upper, max(values))


def _interval(pair: Pair) -> Interval:
    return Interval(_rational(pair[0]).lo, _rational(pair[1]).hi)


def _projected_mean(
    probabilities: Box, projected_centers: Matrix, arithmetic: _Arithmetic,
) -> Box:
    """Enclose E[u] in O(m*d), preserving exact zero projected coordinates.

    The supplied intervals must jointly enclose normalized nonnegative true
    probabilities. First-row-zero exact projected modes are checked; input
    intervals need not themselves sum to the point interval one. Coefficients
    and probability endpoints are multiplied exactly before binary conversion.
    """
    count, dimension = _shape(probabilities, projected_centers)
    arithmetic._work(64*(count+1)*(dimension+1))
    posterior = tuple(_bounds(value, arithmetic) for value in probabilities)
    result = []
    for axis in range(dimension):
        column = tuple(arithmetic.exact(row[axis]) for row in projected_centers)
        result.append(_interval(_weighted_bounds(posterior, column, arithmetic)) if any(column) else _ZERO)
    return tuple(result)


def _bias_covariance(
    probabilities: Box, projected_centers: Matrix, bias_direction: Vector, arithmetic: _Arithmetic,
) -> Box:
    """Enclose Cov_pi(u,beta) in O(m*d) without an absolute bias conversion.

    Both probability normalization and exact projected modes have the same
    contract as _projected_mean. The common beta[0] gauge is subtracted in
    exact arithmetic; a constant beta or zero projected coordinate produces
    exact zero. Products u_j*(beta_j-beta_0) are formed before conversion.
    """
    count, dimension = _shape(probabilities, projected_centers)
    if len(bias_direction) != count:
        raise ValueError("bias direction length must match the mode count")
    if any(isinstance(value, bool) or not isinstance(value, (int, Fraction)) for value in bias_direction):
        raise TypeError("bias direction entries must be exact int or Fraction")
    arithmetic._work(128*(count+1)*(dimension+1))
    reference = Fraction(bias_direction[0])
    relative = tuple(arithmetic.add(Fraction(value), -reference) for value in bias_direction)
    if not any(relative) or not any(value for row in projected_centers for value in row):
        return (_ZERO,)*dimension
    posterior = tuple(_bounds(value, arithmetic) for value in probabilities)
    beta_mean = _weighted_bounds(posterior, relative, arithmetic)
    result = []
    for axis in range(dimension):
        column = tuple(arithmetic.exact(row[axis]) for row in projected_centers)
        if not any(column):
            result.append(_ZERO)
            continue
        mean = _weighted_bounds(posterior, column, arithmetic)
        products = tuple(arithmetic.mul(value, beta) for value, beta in zip(column, relative, strict=True))
        mixed = _weighted_bounds(posterior, products, arithmetic)
        correction = tuple(arithmetic.mul(left, right) for left in mean for right in beta_mean)
        result.append(_interval((arithmetic.add(mixed[0], -max(correction)),
                                 arithmetic.add(mixed[1], -min(correction)))))
    return tuple(result)


def _interval_response(
    projector: Matrix, polynomial: IntervalMatrix, time: Fraction, tail: Fraction,
    tangent_forcing: Box, arithmetic: _Arithmetic,
) -> Box:
    """Enclose s*Q*f plus a tangent operator tail in O(d²).

    The caller supplies an exact orthogonal R, interval entries enclosing a
    true polynomial Q preserving its tangent space, and a true f in range(R)
    enclosed by tangent_forcing. Component boxes may include nontangent points.
    The remainder operator preserves range(R) and has norm at most ``tail``;
    this bound ALREADY includes any time scaling. Its i-th component has
    radius tail*sqrt(R_ii*upper(||f||²)). Zero inputs and coordinates with
    R_ii=0 return exact zero. No solve, sealing or transcendental is performed.
    """
    dimension = len(projector)
    if (not dimension or len(polynomial) != dimension or len(tangent_forcing) != dimension
            or any(len(row) != dimension for row in projector)
            or any(len(row) != dimension for row in polynomial)):
        raise ValueError("response matrices and forcing must have matching positive dimension")
    if isinstance(time, bool) or not isinstance(time, (int, Fraction)) or isinstance(tail, bool) or not isinstance(tail, (int, Fraction)):
        raise TypeError("response time and tail must be exact int or Fraction")
    if time <= 0 or tail < 0:
        raise ValueError("response time must be positive and tail nonnegative")
    arithmetic._work(128*(dimension+1)**2)
    scale, remainder = arithmetic.exact(time), arithmetic.exact(tail)
    for exact_row in projector:
        for exact_entry in exact_row:
            arithmetic.exact(exact_entry)
    for interval_row in polynomial:
        for interval_entry in interval_row:
            _finite(interval_entry)
    for component in tangent_forcing:
        _finite(component)
    if all(_zero(component) for component in tangent_forcing):
        return (_ZERO,)*dimension
    bounds = tuple(_bounds(value, arithmetic) for value in tangent_forcing)
    norm_squared = Fraction(0)
    scaled = []
    for lower, upper in bounds:
        magnitude = max(abs(lower), abs(upper))
        norm_squared = arithmetic.add(norm_squared, arithmetic.mul(magnitude, magnitude))
        scaled.append(_interval((arithmetic.mul(scale, lower), arithmetic.mul(scale, upper))))
    tail_squared = arithmetic.mul(remainder, remainder)
    result = []
    for axis, polynomial_row in enumerate(polynomial):
        diagonal = projector[axis][axis]
        if not 0 <= diagonal <= 1:
            raise ValueError("orthogonal projector diagonal must lie in [0,1]")
        if not diagonal:
            result.append(_ZERO)
            continue
        value = _ZERO
        for entry, coordinate in zip(polynomial_row, scaled, strict=True):
            value = _add(value, _multiply(entry, coordinate))
        squared_radius = arithmetic.mul(arithmetic.mul(tail_squared, diagonal), norm_squared)
        if squared_radius:
            arithmetic._work(64)
            radius = _sqrt_upper(squared_radius, arithmetic.limits)
            value = _add(value, Interval(-radius, radius))
        result.append(value)
    return tuple(result)
