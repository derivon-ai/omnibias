# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Certified Chebyshev approximations of Gaussian inverse decision curves.

Only the scalar target parameter varies. The represented family is fixed;
the uniform value certificate is neither a derivative certificate nor a
roundoff guarantee for a separate tensor evaluator.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from copy import deepcopy
from dataclasses import dataclass
from fractions import Fraction
from typing import TYPE_CHECKING

from omnibias.core.gaussian_jet import _Arithmetic
from omnibias.core.verified.gaussian_heat import _exact, _finite, _float, _rational
from omnibias.core.verified.gaussian_tube import GaussianInverseTubeCertificate
from omnibias.core.verified.interval import Interval
from omnibias.core.verified.inverse_gaussian_heat import _pair, _sqrt_upper, inverse_gaussian_heat
from omnibias.core.verified.mean_map import _strict
from omnibias.core.verified.transcend import PI_IV, cos_iv

if TYPE_CHECKING:
    from omnibias.core.proof.certificate import Cert

Rational = int | Fraction
Vector = tuple[Fraction, ...]


def _midpoint(value: Interval) -> float:
    result = value.lo / 2 + value.hi / 2
    if not math.isfinite(result):
        raise ArithmeticError("nonfinite Chebyshev coefficient proposal")
    return result


def _cosine(index: int, degree: int) -> Interval:
    index %= 2 * degree
    index = min(index, 2 * degree - index)
    if index == 0:
        return Interval.point(1.0)
    if index == degree:
        return Interval.point(-1.0)
    if 2 * index == degree:
        return Interval.point(0.0)
    return _finite(cos_iv(_finite(PI_IV * _rational(Fraction(index, degree)))))


@dataclass(frozen=True, init=False)
class GaussianInverseApproximation:
    """A fixed polynomial and uniform coordinatewise inverse error bounds.

    For exact real |t|<=half_width, the polynomial is
    ``time*(base_target+t*direction)+reference_center+sum(c_k*T_k(t/half_width))``.
    Coefficients use ordinary Chebyshev normalization, without endpoint halves.
    ``evaluate`` returns its exact rational value; ``enclose`` uses outward
    Clenshaw evaluation and includes the uniform approximation error.
    """

    tube: GaussianInverseTubeCertificate
    base_target: Vector
    direction: Vector
    half_width: Fraction
    ellipse_parameter: Fraction | None
    coefficients: tuple[tuple[float, ...], ...]
    coefficient_enclosures: tuple[tuple[Interval, ...], ...]
    interpolation_error_bound: Fraction
    coefficient_error_bounds: Vector
    uniform_error_bounds: Vector
    certificate: Cert

    def __init__(self) -> None:
        raise TypeError("use approximate_gaussian_inverse")

    @property
    def degree(self) -> int:
        return len(self.coefficients) - 1

    def _parameter(self, parameter: Rational) -> Fraction:
        value = _exact(parameter, self.tube.model.budget)
        if abs(value) > self.half_width:
            raise ValueError("parameter lies outside the certified target segment")
        return value

    def evaluate(self, parameter: Rational) -> Vector:
        """Exact nominal polynomial, without its approximation error."""
        value = self._parameter(parameter)
        arithmetic = _Arithmetic(self.tube.model.budget)
        z = arithmetic._coefficient(value / self.half_width)
        two_z = arithmetic.mul(Fraction(2), z)
        result = []
        for axis in range(self.tube.model.dimension):
            b1 = b2 = Fraction(0)
            for row in reversed(self.coefficients[1:]):
                b0 = arithmetic.add(arithmetic.add(arithmetic.exact(Fraction(row[axis])),
                                                  arithmetic.mul(two_z, b1)), -b2)
                b2, b1 = b1, b0
            correction = arithmetic.add(arithmetic.add(arithmetic.exact(Fraction(self.coefficients[0][axis])),
                                                       arithmetic.mul(z, b1)), -b2)
            target = arithmetic.add(self.base_target[axis], arithmetic.mul(value, self.direction[axis]))
            affine = arithmetic.add(arithmetic.mul(self.tube.model.time, target), self.tube.reference_center[axis])
            result.append(arithmetic.add(affine, correction))
        return tuple(result)

    def enclose(self, parameter: Rational) -> tuple[Interval, ...]:
        """Outward enclosure of the true inverse, including evaluation rounding."""
        value = self._parameter(parameter)
        model = self.tube.model
        model.budget._entries(16 * model.dimension + 16)
        model.budget._work(64 * (self.degree + 1) * (model.dimension + 1))
        checked = model.budget._coefficient
        z = _rational(checked(value / self.half_width))
        two_z = _finite(Interval.point(2.0) * z)
        result = []
        for axis in range(model.dimension):
            b1 = b2 = Interval.point(0.0)
            for row in reversed(self.coefficients[1:]):
                b0 = _finite(_finite(Interval.point(row[axis]) + _finite(two_z * b1)) - b2)
                b2, b1 = b1, b0
            correction = _finite(_finite(Interval.point(self.coefficients[0][axis]) + _finite(z * b1)) - b2)
            target = checked(self.base_target[axis] + checked(value * self.direction[axis]))
            affine = checked(checked(model.time * target) + self.tube.reference_center[axis])
            error = _rational(self.uniform_error_bounds[axis]).hi
            result.append(_finite(_finite(_rational(affine) + correction) + Interval(-error, error)))
        return tuple(result)


def approximate_gaussian_inverse(
    tube: GaussianInverseTubeCertificate, base_target: Sequence[Rational], direction: Sequence[Rational], *,
    half_width: Rational = 1, degree: int = 32, ellipse_parameter: Rational | None = None,
    max_iterations: int = 128, absolute_tolerance: float = 1e-12,
) -> GaussianInverseApproximation | None:
    r"""Certify a Chebyshev approximation on v(t)=base_target+t*direction.

    The real segment is |t|<=half_width. A rational Bernstein ellipse rho>1
    must fit strictly inside the tube: ||direction||*half_width*(rho-1/rho)/2
    <target_radius. Automatic rho=1+target_radius/(half_width*||direction||_up)
    satisfies this. The nonlinear correction has Euclidean bound M on that
    ellipse, giving Lobatto interpolation error <=4*M*rho**(-degree)/(rho-1).

    Rigorous inverses at rounded nodes are transported to the ideal nodes
    using the correction's Lipschitz bound time*q/(1-q). Outward DCT-I
    arithmetic encloses the ideal interpolant coefficients. The error from
    replacing them by stored binary64 midpoints is included coordinatewise.

    The tolerance controls sample inverse solves, not the final uniform error;
    coarse certified samples remain valid. A single cumulative work allowance
    caps their iteration counts. Invalid inputs raise. Insufficient resources,
    an unproved ellipse, or unavailable finite enclosures return None. A missing
    rigorous transcendental backend raises, except for the exact affine case.
    """
    if not isinstance(tube, GaussianInverseTubeCertificate):
        raise TypeError("tube must be a GaussianInverseTubeCertificate")
    model = tube.model
    if not isinstance(base_target, Sequence) or not isinstance(direction, Sequence):
        raise TypeError("target and direction must be finite exact sequences")
    if len(base_target) != model.dimension or len(direction) != model.dimension:
        raise ValueError("target and direction dimensions must match the model")
    if type(degree) is not int or type(max_iterations) is not int:
        raise TypeError("degree and max_iterations must be integers, excluding bool")
    if degree < 1 or max_iterations < 0:
        raise ValueError("degree must be positive and max_iterations nonnegative")
    # Domain/type validation is distinct from later resource refusal.
    def exact_input(value: Rational) -> Fraction:
        if isinstance(value, bool) or not isinstance(value, (int, Fraction)):
            raise TypeError("approximation metadata must be exact int or Fraction")
        return Fraction(value)

    base = tuple(exact_input(value) for value in base_target)
    tangent = tuple(exact_input(value) for value in direction)
    width = exact_input(half_width)
    requested = None if ellipse_parameter is None else exact_input(ellipse_parameter)
    tolerance = _float(absolute_tolerance)
    if width <= 0 or tolerance <= 0 or (requested is not None and requested <= 1):
        raise ValueError("half_width and tolerance must be positive; ellipse_parameter must exceed one")
    try:
        arithmetic = _Arithmetic(model.budget)
        checked = arithmetic._coefficient
        if degree > model.budget.max_order:
            return None
        dimension, count = model.dimension, model.mode_count
        arithmetic._entries(32 * (degree + 1) * (count * dimension + count + dimension + 1)
                            + 32 * (degree + 1) * dimension + 128)
        arithmetic._work(64 * (degree + 1) ** 2 * (dimension + 1))
        from omnibias.core.proof.certificate import verify_certificate_digest

        if not verify_certificate_digest(tube.certificate):
            return None
        tube_certificate = deepcopy(tube.certificate)
        for value in (*base, *tangent, width):
            checked(value)
        if requested is not None:
            checked(requested)
        direction_squared = arithmetic.sum(tuple(arithmetic.mul(value, value) for value in tangent))
        affine = tube.correction_bound_squared == 0
        constant = direction_squared == 0
        rho: Fraction | None = None
        interpolation_error = Fraction(0)
        sampled_certificates: list[Cert] = []
        sampled_targets: list[Vector] = []
        nodes: list[Interval] = []
        sampled_enclosures: list[tuple[Interval, ...]] = []
        coefficient_intervals: tuple[tuple[Interval, ...], ...]
        if affine:
            coefficient_intervals = ((Interval.point(0.0),) * dimension,)
        else:
            direction_norm = checked(Fraction(_sqrt_upper(direction_squared, model.budget)))
            if not constant:
                assert tube.target_radius is not None
                scale = arithmetic.mul(width, direction_norm)
                rho = requested if requested is not None else arithmetic.add(Fraction(1), checked(tube.target_radius / scale))
                semiminor = checked(arithmetic.add(rho, -checked(1 / rho)) / 2)
                height_squared = arithmetic.mul(arithmetic.mul(direction_squared, arithmetic.mul(width, width)),
                                                arithmetic.mul(semiminor, semiminor))
                if height_squared >= arithmetic.mul(tube.target_radius, tube.target_radius):
                    return None
                magnitude = checked(Fraction(_sqrt_upper(tube.correction_bound_squared, model.budget)))
                power = Fraction(1)
                for _ in range(degree):
                    power = arithmetic.mul(power, rho)
                denominator = arithmetic.mul(power, arithmetic.add(rho, Fraction(-1)))
                interpolation_error = checked(arithmetic.mul(Fraction(4), magnitude) / denominator)
            sample_count = 1 if constant else degree + 1
            q = model.contraction_bound
            lipschitz = checked(arithmetic.mul(model.time, q) / arithmetic.add(Fraction(1), -q))
            evaluation_work = model._evaluation_work() + 64 * dimension + 64
            with _strict():
                for j in range(sample_count):
                    node = Interval.point(0.0) if constant else _cosine(j, degree)
                    midpoint = checked(Fraction(_midpoint(node)))
                    target = tuple(arithmetic.add(origin, arithmetic.mul(arithmetic.mul(width, rate), midpoint))
                                   for origin, rate in zip(base, tangent, strict=True))
                    # Reserve a whole next solve before calling it. Afterwards
                    # charge its actual iteration count conservatively.
                    remaining = model.budget.max_elimination_work - arithmetic.used
                    iterations = min(max_iterations, remaining // (2 * evaluation_work) - 1)
                    if iterations < 0:
                        return None
                    result = inverse_gaussian_heat(model, target, max_iterations=iterations, absolute_tolerance=tolerance)
                    arithmetic._work((2 * result.iterations + 2) * evaluation_work)
                    sample_certificate = result.certificate
                    if sample_certificate is None:
                        return None
                    delta = max(abs(arithmetic.add(Fraction(node.lo), -midpoint)),
                                abs(arithmetic.add(Fraction(node.hi), -midpoint)))
                    transport = arithmetic.mul(arithmetic.mul(arithmetic.mul(lipschitz, width), direction_norm), delta)
                    transport_upper = _rational(transport).hi
                    values = []
                    for axis, inverse_interval in enumerate(sample_certificate.enclosure):
                        trend = arithmetic.add(arithmetic.mul(model.time, target[axis]), tube.reference_center[axis])
                        values.append(_finite(_finite(inverse_interval - _rational(trend))
                                              + Interval(-transport_upper, transport_upper)))
                    nodes.append(node)
                    sampled_targets.append(target)
                    sampled_enclosures.append(tuple(values))
                    sampled_certificates.append(sample_certificate.certificate)
                if constant:
                    coefficient_intervals = (sampled_enclosures[0],)
                else:
                    rows = []
                    for k in range(degree + 1):
                        row = [Interval.point(0.0) for _ in range(dimension)]
                        for j, sample in enumerate(sampled_enclosures):
                            # All DCT phases belong to the already enclosed
                            # Lobatto cosine grid, modulo even symmetry.
                            phase = (k * j) % (2 * degree)
                            cosine = nodes[min(phase, 2 * degree - phase)]
                            sample_weight = Fraction(1, 2) if j in (0, degree) else Fraction(1)
                            factor = _finite(_rational(sample_weight) * cosine)
                            row = [_finite(total + _finite(factor * component))
                                   for total, component in zip(row, sample, strict=True)]
                        outer_weight = Fraction(1, degree) if k in (0, degree) else Fraction(2, degree)
                        rows.append(tuple(_finite(_rational(outer_weight) * component) for component in row))
                    coefficient_intervals = tuple(rows)
        coefficients = tuple(tuple(_midpoint(component) for component in row) for row in coefficient_intervals)
        coefficient_errors = []
        for axis in range(dimension):
            error = Fraction(0)
            for nominal, enclosure in zip(coefficients, coefficient_intervals, strict=True):
                value = checked(Fraction(nominal[axis]))
                delta = max(abs(arithmetic.add(value, -Fraction(enclosure[axis].lo))),
                            abs(arithmetic.add(Fraction(enclosure[axis].hi), -value)))
                error = arithmetic.add(error, delta)
            coefficient_errors.append(error)
        uniform_errors = tuple(arithmetic.add(interpolation_error, error) for error in coefficient_errors)
        from omnibias.core.proof.certificate import encode_interval, make_certificate

        certificate = make_certificate(
            claim="Uniform coordinatewise inverse approximation on an exact real Gaussian target segment.",
            payload={
                "type": "gaussian_inverse_chebyshev", "tube_certificate": tube_certificate,
                "base_target": [_pair(value) for value in base], "direction": [_pair(value) for value in tangent],
                "half_width": _pair(width), "ellipse_parameter": None if rho is None else _pair(rho),
                "degree": len(coefficients) - 1, "coefficients": [list(row) for row in coefficients],
                "coefficient_enclosures": [[encode_interval(value) for value in row] for row in coefficient_intervals],
                "interpolation_error_bound": _pair(interpolation_error),
                "coefficient_error_bounds": [_pair(value) for value in coefficient_errors],
                "uniform_error_bounds": [_pair(value) for value in uniform_errors],
                "sample_nodes": [encode_interval(value) for value in nodes],
                "sample_targets": [[_pair(value) for value in row] for row in sampled_targets],
                "sample_correction_enclosures": [[encode_interval(value) for value in row] for row in sampled_enclosures],
                "sample_inverse_certificates": sampled_certificates,
                "polynomial": "time*(base_target+t*direction)+reference_center+sum(coefficients[k]*T_k(t/half_width))",
                "error_scope": "coordinatewise uniform value error of the exact stored-coefficient polynomial, |t|<=half_width; no derivative or separate numerical evaluator claim",
                "argument": "Subtract the exact affine trend. The tube bounds the analytic correction by M on the strictly contained Bernstein ellipse. Vector Chebyshev coefficients and Lobatto aliasing give 4*M*rho^(-N)/(rho-1). Certified sample inverses at exact rounded-node targets are transported with correction Lipschitz time*q/(1-q). Directed DCT-I encloses the ideal interpolant coefficients. Since |T_k|<=1 on the real segment, summing coefficient uncertainties adds a coordinatewise uniform error. Affine and constant paths have no interpolation tail.",
            },
            honesty={"unproven_claim": False, "unconditional_transcendentals": not affine, "continuum_pde_claim": False},
            meta={"transcend_backend": "not_used" if affine else "mpmath"},
        )
        approximation = object.__new__(GaussianInverseApproximation)
        for name, stored in (("tube", tube), ("base_target", base), ("direction", tangent), ("half_width", width),
                             ("ellipse_parameter", rho), ("coefficients", coefficients),
                             ("coefficient_enclosures", coefficient_intervals),
                             ("interpolation_error_bound", interpolation_error),
                             ("coefficient_error_bounds", tuple(coefficient_errors)),
                             ("uniform_error_bounds", uniform_errors), ("certificate", certificate)):
            object.__setattr__(approximation, name, stored)
        return approximation
    except (ArithmeticError, ValueError):
        return None


__all__ = ["GaussianInverseApproximation", "approximate_gaussian_inverse"]
