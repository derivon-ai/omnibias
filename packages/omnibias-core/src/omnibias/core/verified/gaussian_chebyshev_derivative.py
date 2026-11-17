# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Uniform raw derivative bounds for fixed Gaussian inverse target curves.

The analytic interpolation tail and the uncertain sampled coefficients are
differentiated separately. A value approximation alone is insufficient for
this claim. Arithmetic here is rational; transcendental provenance is inherited
from the original approximation, whose certificate is retained as a snapshot.
"""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass
from fractions import Fraction
from typing import TYPE_CHECKING

from omnibias.core._chebyshev_calculus import (
    chebyshev_derivative_norm,
    chebyshev_derivative_tail,
    differentiated_chebyshev_coefficients,
)
from omnibias.core.gaussian_jet import _Arithmetic
from omnibias.core.verified.gaussian_chebyshev import GaussianInverseApproximation
from omnibias.core.verified.gaussian_heat import _finite, _rational
from omnibias.core.verified.interval import Interval
from omnibias.core.verified.inverse_gaussian_heat import _pair, _sqrt_upper

if TYPE_CHECKING:
    from omnibias.core.proof.certificate import Cert

Rational = int | Fraction
Vector = tuple[Fraction, ...]


@dataclass(frozen=True, init=False)
class GaussianInverseDerivative:
    """A polynomial and uniform bounds for ``d**order X(v0+t*h)/dt**order``.

    ``coefficients`` are exact ordinary Chebyshev coefficients of the raw
    derivative of the nonlinear correction, in the basis ``T_n(t/half_width)``.
    They exclude the affine trend, which evaluation restores. These are raw
    directional derivatives, not Taylor coefficients or a full target tensor.
    The coordinatewise error bounds concern exact polynomial evaluation;
    ``enclose`` additionally encloses its own floating-point rounding.
    """

    approximation: GaussianInverseApproximation
    order: int
    coefficients: tuple[Vector, ...]
    interpolation_error_bound: Fraction
    coefficient_error_bounds: Vector
    uniform_error_bounds: Vector
    certificate: Cert

    def __init__(self) -> None:
        raise TypeError("use certify_gaussian_inverse_derivative")

    @property
    def degree(self) -> int:
        return len(self.coefficients) - 1

    def _affine(self, value: Fraction, arithmetic: _Arithmetic) -> Vector:
        source = self.approximation
        if self.order == 0:
            return tuple(arithmetic.add(arithmetic.mul(source.tube.model.time,
                         arithmetic.add(origin, arithmetic.mul(value, rate))), center)
                         for origin, rate, center in zip(source.base_target, source.direction,
                                                        source.tube.reference_center, strict=True))
        if self.order == 1:
            return tuple(arithmetic.mul(source.tube.model.time, rate) for rate in source.direction)
        return (Fraction(0),) * source.tube.model.dimension

    def evaluate(self, parameter: Rational) -> Vector:
        """Evaluate the nominal raw derivative exactly, without its error bound."""
        value = self.approximation._parameter(parameter)
        arithmetic = _Arithmetic(self.approximation.tube.model.budget)
        z = arithmetic._coefficient(value / self.approximation.half_width)
        two_z = arithmetic.mul(Fraction(2), z)
        result = []
        for axis, affine in enumerate(self._affine(value, arithmetic)):
            b1 = b2 = Fraction(0)
            for row in reversed(self.coefficients[1:]):
                b0 = arithmetic.add(arithmetic.add(row[axis], arithmetic.mul(two_z, b1)), -b2)
                b2, b1 = b1, b0
            correction = arithmetic.add(arithmetic.add(self.coefficients[0][axis], arithmetic.mul(z, b1)), -b2)
            result.append(arithmetic.add(affine, correction))
        return tuple(result)

    def enclose(self, parameter: Rational) -> tuple[Interval, ...]:
        """Enclose the true raw derivative, including outward evaluation rounding."""
        value = self.approximation._parameter(parameter)
        arithmetic = _Arithmetic(self.approximation.tube.model.budget)
        dimension = self.approximation.tube.model.dimension
        arithmetic._entries(16 * dimension + 16)
        arithmetic._work(64 * (self.degree + 1) * (dimension + 1))
        z = _rational(arithmetic._coefficient(value / self.approximation.half_width))
        two_z = _finite(Interval.point(2.0) * z)
        result = []
        for axis, affine in enumerate(self._affine(value, arithmetic)):
            b1 = b2 = Interval.point(0.0)
            for row in reversed(self.coefficients[1:]):
                b0 = _finite(_finite(_rational(row[axis]) + _finite(two_z * b1)) - b2)
                b2, b1 = b1, b0
            correction = _finite(_finite(_rational(self.coefficients[0][axis]) + _finite(z * b1)) - b2)
            error = _rational(self.uniform_error_bounds[axis]).hi
            result.append(_finite(_finite(_rational(affine) + correction) + Interval(-error, error)))
        return tuple(result)


def certify_gaussian_inverse_derivative(
    approximation: GaussianInverseApproximation, order: int,
) -> GaussianInverseDerivative | None:
    r"""Bound a raw derivative of the true inverse uniformly on its target segment.

    For order k>=1, write C(n,k)=||T_n^(k)||_infinity, zero if n<k,
    r=1/rho and K=max(N+1,k). If the analytic correction is bounded by M
    on the original ellipse, its derivative interpolation error is at most
    2*M/L^k * (sum_{n>=K} C(n,k)*r^n + C(N,k)*r^K/(1-r)).
    Lobatto aliases have degree at most N; the second term bounds their
    derivatives and vanishes for k>N. The positive rational tail is summed
    exactly. Stored coefficient errors are weighted separately by C(n,k)/L^k.

    Order zero retains the original bounds. Affine/constant target paths have
    zero interpolation tail. Invalid types/orders raise; exhausted resources,
    a damaged dependency certificate, or unavailable finite bounds return None.
    Construction needs no new transcendental evaluation. The inherited backend
    stamp is retained; global strict certificate policy still applies.
    """
    if not isinstance(approximation, GaussianInverseApproximation):
        raise TypeError("approximation must be a GaussianInverseApproximation")
    if type(order) is not int:
        raise TypeError("order must be an integer, excluding bool")
    if order < 0:
        raise ValueError("order must be nonnegative")
    source = approximation
    model = source.tube.model
    try:
        if order > model.budget.max_order:
            return None
        arithmetic = _Arithmetic(model.budget)
        checked = arithmetic._coefficient
        dimension, degree = model.dimension, source.degree
        arithmetic._entries(16 * (degree + order + 2) * (dimension + 1) + 128)
        arithmetic._work(16 * (degree + order + 2) * (dimension + 1))
        from omnibias.core.proof.certificate import make_certificate, verify_certificate_digest

        try:
            intact = verify_certificate_digest(source.certificate)
        except (TypeError, ValueError, RecursionError):
            return None
        if not intact:
            return None
        dependency = deepcopy(source.certificate)
        metadata = dependency.get("meta")
        if not isinstance(metadata, Mapping):
            return None
        backend = metadata.get("transcend_backend")
        if backend not in ("not_used", "mpmath", "arb", "mpfr"):
            return None
        nominal = tuple(tuple(checked(Fraction(value)) for value in row) for row in source.coefficients)
        coefficients = differentiated_chebyshev_coefficients(nominal, order, source.half_width, arithmetic)
        if order == 0:
            interpolation_error = source.interpolation_error_bound
            coefficient_errors = source.coefficient_error_bounds
        else:
            inverse_width_power = Fraction(1)
            for _ in range(order):
                inverse_width_power = checked(inverse_width_power / source.half_width)
            interpolation_error = Fraction(0)
            if source.ellipse_parameter is not None:
                ratio = checked(1 / source.ellipse_parameter)
                start = max(degree + 1, order)
                tail = chebyshev_derivative_tail(ratio, start, order, arithmetic)
                alias_norm = chebyshev_derivative_norm(degree, order, arithmetic)
                alias_tail = Fraction(0)
                if alias_norm:
                    power = Fraction(1)
                    for _ in range(start):
                        power = arithmetic.mul(power, ratio)
                    alias_tail = checked(arithmetic.mul(alias_norm, power) / arithmetic.add(Fraction(1), -ratio))
                magnitude = checked(Fraction(_sqrt_upper(source.tube.correction_bound_squared, model.budget)))
                interpolation_error = arithmetic.mul(arithmetic.mul(Fraction(2), magnitude),
                    arithmetic.mul(inverse_width_power, arithmetic.add(tail, alias_tail)))
            errors = [Fraction(0)] * dimension
            for n in range(order, degree + 1):
                weight = arithmetic.mul(inverse_width_power, chebyshev_derivative_norm(n, order, arithmetic))
                for axis in range(dimension):
                    enclosure = source.coefficient_enclosures[n][axis]
                    delta = max(abs(arithmetic.add(nominal[n][axis], -Fraction(enclosure.lo))),
                                abs(arithmetic.add(Fraction(enclosure.hi), -nominal[n][axis])))
                    errors[axis] = arithmetic.add(errors[axis], arithmetic.mul(weight, delta))
            coefficient_errors = tuple(errors)
        uniform_errors = tuple(arithmetic.add(interpolation_error, error) for error in coefficient_errors)
        certificate = make_certificate(
            claim="Uniform coordinatewise raw derivative error for a fixed Gaussian inverse target curve.",
            payload={
                "type": "gaussian_inverse_chebyshev_derivative", "approximation_certificate": dependency,
                "order": order, "coefficients": [[_pair(value) for value in row] for row in coefficients],
                "interpolation_error_bound": _pair(interpolation_error),
                "coefficient_error_bounds": [_pair(value) for value in coefficient_errors],
                "uniform_error_bounds": [_pair(value) for value in uniform_errors],
                "derivative": "raw d^order/dt^order X(base_target+t*direction); |t|<=half_width",
                "polynomial": "raw derivative of affine trend + sum(coefficients[n]*T_n(t/half_width))",
                "error_scope": "exact stored-coefficient polynomial versus true inverse derivative; no separate tensor rounding or full target derivative tensor claim",
                "argument": "Chebyshev coefficients of the analytic correction have vector norm at most 2*M*rho^(-n). Lobatto interpolation aliases each n>N to a degree at most N. The endpoint bound C(n,k)=T_n^(k)(1) bounds each differentiated basis function on [-1,1]. Summing the positive exact tail C(n,k)*rho^(-n) and bounding alias derivatives by C(N,k) proves the analytic error. Individual stored-coefficient uncertainties gain weights C(n,k)/half_width^k. The affine trend is differentiated exactly. Order zero retains the source bound.",
            },
            honesty={"unproven_claim": False, "continuum_pde_claim": False},
            meta={"transcend_backend": backend, "construction_arithmetic": "exact_rational",
                  "transcendental_provenance": "inherited_from_approximation"},
        )
        result = object.__new__(GaussianInverseDerivative)
        for name, stored in (("approximation", source), ("order", order), ("coefficients", coefficients),
                             ("interpolation_error_bound", interpolation_error),
                             ("coefficient_error_bounds", coefficient_errors),
                             ("uniform_error_bounds", uniform_errors), ("certificate", certificate)):
            object.__setattr__(result, name, stored)
        return result
    except (ArithmeticError, ValueError):
        return None


__all__ = ["GaussianInverseDerivative", "certify_gaussian_inverse_derivative"]
