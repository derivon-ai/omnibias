# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Certified all-order inverse-age expansions for Gaussian heat decisions."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from fractions import Fraction
from typing import TYPE_CHECKING

from omnibias.core.gaussian_jet import _Arithmetic
from omnibias.core.verified._gaussian_moments import (
    _add,
    _multiply,
    _posterior_moments,
    _square,
)
from omnibias.core.verified.gaussian_heat import GaussianHeatMap, _finite, _rational
from omnibias.core.verified.interval import Interval
from omnibias.core.verified.inverse_gaussian_heat import _pair, _sqrt_upper
from omnibias.core.verified.mean_map import _probabilities, _strict

if TYPE_CHECKING:
    from omnibias.core.proof.certificate import Cert

Box = tuple[Interval, ...]
Matrix = tuple[Box, ...]
_ZERO = Interval.point(0.0)
_ONE = Interval.point(1.0)


def _sum(values: Sequence[Interval]) -> Interval:
    result = _ZERO
    for value in values:
        result = _add(result, value)
    return result


def _coefficients(
    posterior: Box, centered: Matrix, inverse_scale: Interval, order: int,
    radii: Sequence[Fraction], *, extra_costs: Box | None = None,
) -> Matrix:
    """Outward recurrence; aggregate work/storage are preflighted by caller."""
    dimension = len(centered[0])
    coefficients = [(_ZERO,) * dimension]
    logarithms: list[list[Interval]] = [[_ZERO] for _ in posterior]
    exponentials: list[list[Interval]] = [[_ONE] for _ in posterior]
    denominators = [_ONE]
    costs = tuple(_multiply(Interval.point(-0.5), _multiply(inverse_scale,
                  _add(_sum(tuple(_square(x) for x in row)), _ZERO if extra_costs is None else extra_costs[index])))
                  for index, row in enumerate(centered))
    for degree in range(1, order + 1):
        reciprocal = _rational(Fraction(1, degree))
        weights = []
        for index, (probability, row) in enumerate(zip(posterior, centered, strict=True)):
            log_coefficient = costs[index] if degree == 1 else _multiply(
                inverse_scale, _sum(tuple(_multiply(a, c) for a, c in zip(row, coefficients[-1], strict=True))),
            )
            logarithms[index].append(log_coefficient)
            exponential = _multiply(reciprocal, _sum(tuple(
                _multiply(_rational(Fraction(k)), _multiply(logarithms[index][k], exponentials[index][degree-k]))
                for k in range(1, degree+1)
            )))
            exponentials[index].append(exponential)
            weights.append(_multiply(probability, exponential))
        denominators.append(_sum(weights))
        radius = _rational(radii[degree]).hi
        bound = Interval(-radius, radius)
        row_result = []
        for axis in range(dimension):
            numerator = _sum(tuple(_multiply(weight, row[axis]) for weight, row in zip(weights, centered, strict=True)))
            correction = _sum(tuple(_multiply(denominators[k], coefficients[degree-k][axis]) for k in range(1, degree)))
            row_result.append(_add(numerator, -correction).intersect(bound))
        coefficients.append(tuple(row_result))
    return tuple(coefficients)


@dataclass(frozen=True, init=False)
class GaussianInverseAging:
    r"""Uniform enclosure from a truncated inverse-age expansion.

    ``correction_coefficients[n]`` encloses the ordinary Taylor coefficient
    c_n of X(v,s/t)-(s/t)*v-mu_infinity, with c_0=0. Thus the formal variable
    scales inverse age, not physical age. Only this centered correction is
    analytic at t=0; X itself has a pole there unless v=0. The predictor is
    s*v+mu_infinity+sum(c_n), evaluated at t=1.

    ``predictor_enclosure`` includes interval evaluation over ``target_box``.
    ``approximation_error_bound`` bounds the Euclidean distance between the
    exact truncated formula and the true inverse at the same target. The
    final ``enclosure`` adds that error and intersects the inverse center hull.
    These are value bounds, not bounds on derivatives or tensor roundoff.
    """

    model: GaussianHeatMap
    target_box: Box
    order: int
    relative_mean: Box
    correction_coefficients: Matrix
    predictor_enclosure: Box
    approximation_error_bound: Fraction
    enclosure: Box
    certificate: Cert

    def __init__(self) -> None:
        raise TypeError("use certify_gaussian_inverse_aging")


def certify_gaussian_inverse_aging(
    model: GaussianHeatMap, target_box: Sequence[Interval], *, order: int = 8,
) -> GaussianInverseAging | None:
    r"""Enclose the Gaussian inverse by all-order aging, without inverse solves.

    Require q=D²/(8*nu*s)<1/3. A complex contraction proof gives a uniform
    Euclidean truncation bound D/2*(3*q)**(order+1)/(1-3*q) for every real
    target. At orders zero and one, take the smaller of this bound and the
    established real bounds D*q/2 and D*q². Identical centers are exact affine.

    Coefficients are scaled by the supplied heat age during the recurrence;
    no unscaled inverse-age powers or covariance matrices are formed. Work
    is O((m+d)*N²+m*d*N) and working storage O(m*d+(m+d)*N), excluding the
    already constructed model. Outward intervals may widen with order; a
    smaller analytic tail does not guarantee a smaller returned enclosure.

    Invalid types/domains raise. Unsupported contraction, exhausted budgets
    and nonfinite enclosures return None. A missing rigorous transcendental
    backend raises except for the affine case. No convergence refusal is a
    nonexistence claim, and the certificate digest is not a formal proof.
    """
    if not isinstance(model, GaussianHeatMap):
        raise TypeError("model must be a GaussianHeatMap")
    if not isinstance(target_box, Sequence):
        raise TypeError("target_box must be a finite sequence of Interval objects")
    box = model._box(target_box)
    if type(order) is not int:
        raise TypeError("order must be an integer, excluding bool")
    if order < 0:
        raise ValueError("order must be nonnegative")
    q = model.contraction_bound
    if q >= Fraction(1, 3) or order > model.budget.max_order:
        return None
    try:
        arithmetic = _Arithmetic(model.budget)
        checked = arithmetic._coefficient
        count, dimension = model.mode_count, model.dimension
        arithmetic._entries(32*(count+1)*(dimension+1) + 16*(count+dimension+1)*(order+1) + 128)
        arithmetic._work(64*((count+dimension+1)*(order+1)**2 + (count+1)*(dimension+1)*(order+1)) + 512)
        posterior: Box = ()
        mean: Box = (_ZERO,) * dimension
        coefficients: Matrix = (mean,) * (order+1)
        error = Fraction(0)
        if q:
            relative = tuple(tuple(arithmetic.add(value, -reference) for value, reference in zip(row, model.centers[0], strict=True))
                             for row in model.centers)
            converted = tuple(tuple(_rational(value) for value in row) for row in relative)
            twice_nu = arithmetic.mul(Fraction(2), model.viscosity)
            logits = []
            for row, bias in zip(relative, model.biases, strict=True):
                logit = _rational(arithmetic.add(bias, -model.biases[0]))
                for value, target in zip(row, box, strict=True):
                    logit = _add(logit, _multiply(_rational(checked(value/twice_nu)), target))
                logits.append(logit)
            with _strict():
                posterior = _probabilities(tuple(logits))
                mean = _posterior_moments(posterior, converted, covariance=False).relative_mean
            centered = tuple(tuple(_ZERO if a.lo == a.hi == mu.lo == mu.hi else _finite(a-mu)
                                   for a, mu in zip(row, mean, strict=True)) for row in converted)
            diameter = checked(Fraction(_sqrt_upper(model.diameter_squared, model.budget)))
            ratio = arithmetic.mul(Fraction(3), q)
            radii = [checked(diameter/2)]
            for _ in range(order+1):
                radii.append(arithmetic.mul(radii[-1], ratio))
            error = checked(radii[-1]/arithmetic.add(Fraction(1), -ratio))
            if order == 0:
                error = min(error, checked(arithmetic.mul(diameter, q)/2))
            elif order == 1:
                error = min(error, arithmetic.mul(diameter, arithmetic.mul(q, q)))
            if order:
                inverse_scale = _rational(checked(1/arithmetic.mul(twice_nu, model.time)))
                coefficients = _coefficients(posterior, centered, inverse_scale, order, radii)
        error_upper = _rational(error).hi
        predictor = []
        enclosed = []
        for axis, (target, reference) in enumerate(zip(box, model.centers[0], strict=True)):
            low_affine = arithmetic.add(arithmetic.mul(model.time, Fraction(target.lo)), reference)
            high_affine = arithmetic.add(arithmetic.mul(model.time, Fraction(target.hi)), reference)
            affine = Interval(_rational(low_affine).lo, _rational(high_affine).hi)
            prediction = _add(_add(affine, mean[axis]), _sum(tuple(row[axis] for row in coefficients[1:])))
            predictor.append(prediction)
            expanded = _add(prediction, Interval(-error_upper, error_upper))
            low_hull = arithmetic.add(arithmetic.mul(model.time, Fraction(target.lo)), min(a[axis] for a in model.centers))
            high_hull = arithmetic.add(arithmetic.mul(model.time, Fraction(target.hi)), max(a[axis] for a in model.centers))
            enclosed.append(_finite(Interval(max(expanded.lo, _rational(low_hull).lo), min(expanded.hi, _rational(high_hull).hi))))

        from omnibias.core.proof.certificate import encode_interval, make_certificate

        certificate = make_certificate(
            claim="Uniform Gaussian inverse enclosure from a convergent inverse-age series.",
            payload={
                "type": "gaussian_inverse_aging", "centers": [[_pair(value) for value in row] for row in model.centers],
                "biases": [_pair(value) for value in model.biases], "viscosity": _pair(model.viscosity), "time": _pair(model.time),
                "target_box": [encode_interval(value) for value in box], "order": order,
                "diameter_squared": _pair(model.diameter_squared), "contraction_bound": _pair(q),
                "posterior": [encode_interval(value) for value in posterior], "relative_mean": [encode_interval(value) for value in mean],
                "correction_coefficients": [[encode_interval(value) for value in row] for row in coefficients],
                "predictor_enclosure": [encode_interval(value) for value in predictor],
                "approximation_error_bound": _pair(error), "enclosure": [encode_interval(value) for value in enclosed],
                "coefficient_convention": "Ordinary Taylor coefficients of X(v,s/t)-(s/t)*v-mu_infinity at t=0; c0=0; evaluated at t=1.",
                "argument": "For inverse age z, the centered inverse correction is the fixed point of the limiting-posterior mean tilted by z*(A.h-|A|^2/2)/(2*nu). On |z|<=8*nu/(3*D^2), ||h||<=D/2, the phase half-width is <=2/3, the h-Jacobian norm is <=3/5, and the forcing at h=0 is <=3*D/16. The map takes the closed spatial ball into its interior. Its holomorphic fixed point has norm <=D/2. Cauchy bounds give ||c_n||<=D/2*(3*q)^n and the stated geometric tail. Outward exponential/quotient recurrences enclose the coefficients, intersected with these Cauchy bounds. For orders 0 and 1 the established real moment-predictor error bounds may tighten the tail. Exact affine arithmetic and the independent inverse center hull complete the enclosure.",
                "error_scope": "Euclidean analytic value error at equal targets plus outward interval evaluation on target_box; no derivative or separate tensor-roundoff claim.",
            },
            honesty={"unproven_claim": False, "unconditional_transcendentals": q != 0, "continuum_pde_claim": False},
            meta={"transcend_backend": "not_used" if q == 0 else "mpmath"},
        )
        result = object.__new__(GaussianInverseAging)
        for name, stored in (("model", model), ("target_box", box), ("order", order), ("relative_mean", mean),
                             ("correction_coefficients", coefficients), ("predictor_enclosure", tuple(predictor)),
                             ("approximation_error_bound", error), ("enclosure", tuple(enclosed)), ("certificate", certificate)):
            object.__setattr__(result, name, stored)
        return result
    except (ArithmeticError, ValueError):
        return None


__all__ = ["GaussianInverseAging", "certify_gaussian_inverse_aging"]
