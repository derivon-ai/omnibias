# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Global inverse certificates for sufficiently smoothed Gaussian heat fields.

The diameter covariance bound proves a Euclidean contraction on all of R^d.
Iteration only proposes a point; rigorous residuals and the exact contraction
constant certify its distance to the unique inverse. No affine-rank assumption,
local Newton convergence assumption, or bounded search domain is required.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from fractions import Fraction
from typing import TYPE_CHECKING, Literal

from omnibias.core.ridge_system import RidgeBudget
from omnibias.core.verified.gaussian_heat import GaussianHeatMap, _exact, _finite, _float, _rational
from omnibias.core.verified.interval import Interval
from omnibias.core.verified.transcend import backend_name

if TYPE_CHECKING:
    from omnibias.core.proof.certificate import Cert

Rational = int | Fraction
GaussianInverseStatus = Literal["certified", "not_certified", "outside_contraction_regime"]


@dataclass(frozen=True)
class GaussianInverseCertificate:
    """Unique global inverse enclosed in physical and scaled coordinates.

    ``scaled_point`` is a binary64 proposal y=x/s. ``scaled_error_bound`` is
    a Euclidean distance bound to its exact inverse; ``error_bound`` scales
    it by s. The boxes also intersect the global center-hull coordinate bounds.
    The sealed record is numerical evidence, not a Lean-kernel proof.
    """

    scaled_point: tuple[float, ...]
    scaled_enclosure: tuple[Interval, ...]
    enclosure: tuple[Interval, ...]
    scaled_error_bound: float
    error_bound: float
    contraction_bound: Fraction
    certificate: Cert


@dataclass(frozen=True)
class GaussianInverseResult:
    """Bounded iteration; a certificate can remain wider than the tolerance.

    ``outside_contraction_regime`` does not imply nonexistence or nonuniqueness.
    ``not_certified`` denotes numerical or resource failure. A ``certified``
    result proves a unique global inverse even if ``tolerance_met`` is false.
    """

    status: GaussianInverseStatus
    scaled_point: tuple[float, ...]
    iterations: int
    tolerance_met: bool
    certificate: GaussianInverseCertificate | None


def _inputs(
    model: GaussianHeatMap, target: Sequence[Rational], scaled_point: Sequence[float],
) -> tuple[tuple[Fraction, ...], tuple[float, ...]]:
    if not isinstance(model, GaussianHeatMap):
        raise TypeError("model must be a GaussianHeatMap")
    if not isinstance(target, Sequence) or not isinstance(scaled_point, Sequence):
        raise TypeError("target and scaled_point must be finite sequences")
    if len(target) != model.dimension or len(scaled_point) != model.dimension:
        raise ValueError("target and scaled_point dimensions must match the model")
    return (tuple(_exact(value, model.budget) for value in target),
            tuple(_float(value) for value in scaled_point))


def _sqrt_upper(value: Fraction, budget: RidgeBudget | None = None) -> float:
    """Outward binary64 square-root bound using exact integer comparisons."""
    if value < 0:
        raise ValueError("square-root bound must be nonnegative")
    if value == 0:
        return 0.0

    def checked(number: Fraction) -> Fraction:
        return number if budget is None else budget._coefficient(number)

    exponent = (value.numerator.bit_length() - value.denominator.bit_length()) // 2 - 52
    if budget is not None and abs(2 * exponent) + 1 > budget.max_coefficient_bits:
        raise ValueError("square-root scaling exceeds the exact coefficient budget")
    scaled = checked(value / checked(Fraction(2) ** (2 * exponent)))
    significand = math.isqrt(scaled.numerator // scaled.denominator)
    if checked(Fraction(significand * significand)) < scaled:
        significand += 1
    result = _rational(checked(significand * checked(Fraction(2) ** exponent))).hi
    # _rational rounds the exact upper rational outward, including subnormals.
    if checked(Fraction(result) ** 2) < value:
        raise ArithmeticError("square-root enclosure failed")
    return result


class _Work:
    def __init__(self, model: GaussianHeatMap) -> None:
        self.model = model
        self.used = 0

    def evaluation(self) -> None:
        self.used += self.model._evaluation_work() + 64 * self.model.dimension + 64
        self.model.budget._work(self.used)


def _pair(value: Fraction) -> list[int]:
    return [value.numerator, value.denominator]


def _certify(
    model: GaussianHeatMap, target: tuple[Fraction, ...], point: tuple[float, ...], work: _Work,
) -> GaussianInverseCertificate | None:
    q = model.contraction_bound
    if q >= 1:
        return None
    try:
        work.evaluation()
        image = model.scaled_fixed_point(tuple(Interval.point(value) for value in point), target)
        residual = tuple(_finite(Interval.point(value) - bound)
                         for value, bound in zip(point, image, strict=True))
        squared = Fraction(0)
        for value in residual:
            magnitude = max(abs(Fraction(value.lo)), abs(Fraction(value.hi)))
            square = model.budget._coefficient(magnitude * magnitude)
            squared = model.budget._coefficient(squared + square)
        margin = model.budget._coefficient(1 - q)
        margin_squared = model.budget._coefficient(margin * margin)
        radius_squared = model.budget._coefficient(squared / margin_squared)
        radius = _sqrt_upper(radius_squared, model.budget)
        physical_radius = _rational(model.budget._coefficient(Fraction(radius) * model.time)).hi
        scaled_box: list[Interval] = []
        box: list[Interval] = []
        for axis, center in enumerate(point):
            numerical = _finite(Interval.point(center) + Interval(-radius, radius))
            hull_lo = model.budget._coefficient(target[axis] + min(a[axis] for a in model.centers) / model.time)
            hull_hi = model.budget._coefficient(target[axis] + max(a[axis] for a in model.centers) / model.time)
            enclosure = Interval(max(numerical.lo, _rational(hull_lo).lo),
                                 min(numerical.hi, _rational(hull_hi).hi))
            scaled_box.append(enclosure)
            physical = _finite(enclosure * _rational(model.time))
            # Compute the physical hull directly, avoiding dependency widening
            # from the intermediate division by s followed by multiplication.
            lower = model.budget._coefficient(model.time * target[axis] + min(a[axis] for a in model.centers))
            upper = model.budget._coefficient(model.time * target[axis] + max(a[axis] for a in model.centers))
            box.append(Interval(max(physical.lo, _rational(lower).lo), min(physical.hi, _rational(upper).hi)))
        from omnibias.core.proof.certificate import encode_interval, make_certificate

        certificate = make_certificate(
            claim="Unique global inverse of the Gaussian heat Burgers velocity and minimizer of its target-tilted potential.",
            payload={
                "type": "gaussian_heat_inverse",
                "centers": [[_pair(value) for value in row] for row in model.centers],
                "biases": [_pair(value) for value in model.biases],
                "viscosity": _pair(model.viscosity), "time": _pair(model.time),
                "target": [_pair(value) for value in target],
                "diameter_squared": _pair(model.diameter_squared), "contraction_bound": _pair(q),
                "scaled_point": list(point), "fixed_point_image": [encode_interval(value) for value in image],
                "residual": [encode_interval(value) for value in residual],
                "residual_norm_squared_bound": _pair(squared),
                "error_squared_bound": _pair(radius_squared), "scaled_error_bound": radius,
                "error_bound": physical_radius,
                "scaled_enclosure": [encode_interval(value) for value in scaled_box],
                "enclosure": [encode_interval(value) for value in box],
                "global_argument": "Covariance <= diameter_squared/4 times identity implies a Euclidean contraction q<1 on all R^d; Banach gives existence, uniqueness, and residual/(1-q) error. The velocity potential is strongly convex.",
            },
            honesty={"unproven_claim": False, "unconditional_transcendentals": True, "continuum_pde_claim": False},
            meta={"transcend_backend": backend_name()},
        )
        return GaussianInverseCertificate(point, tuple(scaled_box), tuple(box), radius, physical_radius, q, certificate)
    except (ArithmeticError, ValueError):
        return None


def certify_gaussian_inverse(
    model: GaussianHeatMap, target: Sequence[Rational], scaled_point: Sequence[float],
) -> GaussianInverseCertificate | None:
    """Certify a proposal y=x/s, or return inconclusive ``None``.

    Any finite point may yield a coarse certificate if q<1. Exact target data
    must be integers or Fractions. Invalid inputs raise; arithmetic and resource
    failure return None. A missing rigorous transcendental backend raises.
    """
    frozen_target, point = _inputs(model, target, scaled_point)
    return _certify(model, frozen_target, point, _Work(model))


def inverse_gaussian_heat(
    model: GaussianHeatMap, target: Sequence[Rational], *, initial: Sequence[float] | None = None,
    max_iterations: int = 128, absolute_tolerance: float = 1e-10,
) -> GaussianInverseResult:
    """Propose a fixed-point inverse and certify its physical Euclidean error.

    ``initial`` is the scaled coordinate y=x/s. Default zero is always a valid
    mathematical starting point in the contraction regime. The solver shares
    one cumulative work budget and never treats floating-point stagnation as a
    proof. A valid last certificate survives later numerical/proposal failure.
    """
    if not isinstance(model, GaussianHeatMap):
        raise TypeError("model must be a GaussianHeatMap")
    if type(max_iterations) is not int:
        raise TypeError("max_iterations must be an integer")
    if max_iterations < 0:
        raise ValueError("max_iterations must be nonnegative")
    tolerance = _float(absolute_tolerance)
    if tolerance <= 0:
        raise ValueError("absolute_tolerance must be positive")
    frozen_target, point = _inputs(model, target, (0.0,) * model.dimension if initial is None else initial)
    if model.contraction_bound >= 1:
        return GaussianInverseResult("outside_contraction_regime", point, 0, False, None)
    work = _Work(model)
    best: GaussianInverseCertificate | None = None
    iteration = 0
    for iteration in range(max_iterations + 1):
        certificate = _certify(model, frozen_target, point, work)
        if certificate is not None and (best is None or certificate.error_bound < best.error_bound):
            best = certificate
        if best is not None and best.error_bound <= tolerance:
            return GaussianInverseResult("certified", best.scaled_point, iteration, True, best)
        if iteration == max_iterations:
            break
        try:
            work.evaluation()
            proposed = model.float_scaled_fixed_point(point, frozen_target)
            next_point = tuple(_float(value) for value in proposed)
        except (ArithmeticError, ValueError):
            break
        if next_point == point:
            break
        point = next_point
    return GaussianInverseResult("certified" if best is not None else "not_certified",
                                 best.scaled_point if best is not None else point, iteration, False, best)


__all__ = ["GaussianInverseCertificate", "GaussianInverseResult", "certify_gaussian_inverse", "inverse_gaussian_heat"]
