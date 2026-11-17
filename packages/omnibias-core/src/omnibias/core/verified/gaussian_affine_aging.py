# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Certified forward approximations of affine-constrained Gaussian inverses."""

from __future__ import annotations

from collections.abc import Sequence
from copy import deepcopy
from dataclasses import dataclass
from fractions import Fraction
from typing import TYPE_CHECKING, Literal

from omnibias.core.gaussian_jet import _Arithmetic
from omnibias.core.verified._gaussian_moments import _add, _multiply, _posterior_moments
from omnibias.core.verified.gaussian_affine import GaussianAffineRestriction
from omnibias.core.verified.gaussian_aging import _coefficients, _sum
from omnibias.core.verified.gaussian_heat import _finite, _rational
from omnibias.core.verified.interval import Interval
from omnibias.core.verified.inverse_gaussian_affine import _intact, _linear
from omnibias.core.verified.inverse_gaussian_heat import _pair, _sqrt_upper
from omnibias.core.verified.mean_map import _probabilities, _strict

if TYPE_CHECKING:
    from omnibias.core.proof.certificate import Cert

Box = tuple[Interval, ...]
Matrix = tuple[Box, ...]
Vector = tuple[Fraction, ...]
Mode = Literal["predictor", "series"]
_ZERO = Interval.point(0.0)


def _affine_box(
    restriction: GaussianAffineRestriction, box: Box, shift: Vector, scale: Fraction, arithmetic: _Arithmetic,
) -> Box:
    """Exact endpoint linear algebra before the binary interval boundary."""
    output = []
    for row, constant in zip(restriction.projector, shift, strict=True):
        lower = upper = constant
        for coefficient, value in zip(row, box, strict=True):
            weight = arithmetic.mul(scale, coefficient)
            lower = arithmetic.add(lower, arithmetic.mul(weight, Fraction(value.lo if weight >= 0 else value.hi)))
            upper = arithmetic.add(upper, arithmetic.mul(weight, Fraction(value.hi if weight >= 0 else value.lo)))
        output.append(Interval(_rational(lower).lo, _rational(upper).hi))
    return tuple(output)


@dataclass(frozen=True, init=False)
class GaussianAffineAging:
    r"""A constrained forward formula and rigorous optimizer/value bounds.

    ``mode`` distinguishes a real order-zero/one predictor from a convergent
    aging series. The limiting posterior uses original biases. Coefficients
    describe X(v,s/t)-(s/t)*P*v-x0-mu_infinity at t=0 with original model
    parameters and constraints fixed. They are ordinary Taylor coefficients.

    ``approximation_error_bound`` is the Euclidean error of the exact finite
    formula at each target; ``objective_gap_bound`` is its objective gap.
    ``predictor_enclosure`` also includes interval evaluation and variation
    over the target box. ``enclosure`` contains the exact constrained optimum.
    Its other points are not necessarily feasible.

    ``feasible_point`` is an exactly feasible rational projection of the
    predictor-box midpoint. Its separate error and objective-gap bounds include
    interval evaluation and target-box variation, and hold for every target
    in the box. A tensor evaluation needs its own rounding error assessment.
    """

    restriction: GaussianAffineRestriction
    target_box: Box
    projected_target_box: Box
    order: int
    mode: Mode
    aging_parameter: Fraction
    relative_mean: Box
    correction_coefficients: Matrix
    predictor_enclosure: Box
    approximation_error_bound: Fraction
    objective_gap_bound: Fraction
    enclosure: Box
    feasible_point: Vector
    feasible_point_error_bound: Fraction
    feasible_point_objective_gap_bound: Fraction
    certificate: Cert

    def __init__(self) -> None:
        raise TypeError("use certify_gaussian_affine_predictor or certify_gaussian_affine_aging")


def _build(
    restriction: GaussianAffineRestriction, target_box: Sequence[Interval], order: int, mode: Mode,
) -> GaussianAffineAging | None:
    if not isinstance(restriction, GaussianAffineRestriction):
        raise TypeError("restriction must be a GaussianAffineRestriction")
    if not isinstance(target_box, Sequence):
        raise TypeError("target_box must be a finite sequence of Interval objects")
    model, source = restriction.model, restriction.source_model
    box = model._box(target_box)
    if type(order) is not int:
        raise TypeError("order must be an integer, excluding bool")
    if order < 0 or (mode == "predictor" and order not in (0, 1)):
        raise ValueError("predictor order must be zero or one; series order must be nonnegative")
    if order > model.budget.max_order or model.contraction_bound >= 1 or not _intact(restriction.certificate):
        return None
    try:
        arithmetic = _Arithmetic(model.budget)
        checked = arithmetic._coefficient
        count, dimension = model.mode_count, model.dimension
        arithmetic._entries(64*(count+len(restriction.constraints)+1)*(dimension+1) + 32*(dimension+1)**2
                            + 32*(count+dimension+1)*(order+1) + 256)
        arithmetic._work(128*((count+dimension+1)*(order+1)**2 + (count+1)*(dimension+1)*(order+1))
                         + 256*(dimension+1)**2 + 64*(len(restriction.constraints)+1)*(dimension+1) + 1024)
        cost_range = arithmetic.add(max(restriction.normal_cost_differences), -min(restriction.normal_cost_differences))
        denominator = arithmetic.mul(arithmetic.mul(Fraction(8), source.viscosity), source.time)
        parameter = checked(arithmetic.add(model.diameter_squared, cost_range)/denominator)
        q = model.contraction_bound
        # A zero tangent diameter makes the inverse exactly affine even when
        # normal costs and their inverse-age parameter are arbitrarily large.
        if q and mode == "series" and parameter >= Fraction(1, 3):
            return None
        mean: Box = (_ZERO,)*dimension
        posterior: Box = ()
        coefficients: Matrix = (mean,)*(order+1)
        error = Fraction(0)
        if q:
            relative = tuple(tuple(arithmetic.add(value, -reference) for value, reference in zip(row, model.centers[0], strict=True))
                             for row in model.centers)
            converted = tuple(tuple(_rational(value) for value in row) for row in relative)
            twice_nu = arithmetic.mul(Fraction(2), source.viscosity)
            logits = []
            # Tangent center differences dot v equal their dot P*v. Use the
            # original box directly to avoid a needless lost correlation.
            for row, bias in zip(relative, source.biases, strict=True):
                logit = _rational(arithmetic.add(bias, -source.biases[0]))
                for value, target in zip(row, box, strict=True):
                    logit = _add(logit, _multiply(_rational(checked(value/twice_nu)), target))
                logits.append(logit)
            with _strict():
                posterior = _probabilities(tuple(logits))
                moments = _posterior_moments(posterior, converted, covariance=False)
            mean = moments.relative_mean
            centered = tuple(tuple(_ZERO if value.lo == value.hi == center.lo == center.hi else _finite(value-center)
                                   for value, center in zip(row, mean, strict=True)) for row in converted)
            diameter = checked(Fraction(_sqrt_upper(model.diameter_squared, model.budget)))
            costs = tuple(_rational(value) for value in restriction.normal_cost_differences)
            if mode == "predictor":
                error = checked(arithmetic.mul(diameter, parameter)/2)
                if order:
                    normal_covariance = tuple(_sum(tuple(_multiply(_multiply(p, cost), row[axis])
                                                         for p, cost, row in zip(posterior, costs, centered, strict=True)))
                                              for axis in range(dimension))
                    scale = _rational(checked(-1/arithmetic.mul(arithmetic.mul(Fraction(4), source.viscosity), source.time)))
                    coefficients = ((_ZERO,)*dimension,
                                    tuple(_multiply(scale, _add(trace, normal)) for trace, normal in
                                          zip(moments.third_trace, normal_covariance, strict=True)))
                    error = checked(arithmetic.mul(arithmetic.mul(diameter, parameter), arithmetic.add(parameter, q))/2)
            else:
                ratio = arithmetic.mul(Fraction(3), parameter)
                radii = [checked(diameter/2)]
                for _ in range(order+1):
                    radii.append(arithmetic.mul(radii[-1], ratio))
                error = checked(radii[-1]/arithmetic.add(Fraction(1), -ratio))
                if order == 0:
                    error = min(error, checked(arithmetic.mul(diameter, parameter)/2))
                elif order == 1:
                    error = min(error, checked(arithmetic.mul(arithmetic.mul(diameter, parameter), arithmetic.add(parameter, q))/2))
                if order:
                    inverse_scale = _rational(checked(1/arithmetic.mul(twice_nu, source.time)))
                    coefficients = _coefficients(posterior, centered, inverse_scale, order, radii, extra_costs=costs)
        projected_box = _affine_box(restriction, box, (Fraction(0),)*dimension, Fraction(1), arithmetic)
        affine = _affine_box(restriction, box, model.centers[0], source.time, arithmetic)
        prediction = tuple(_add(_add(value, center), _sum(tuple(row[axis] for row in coefficients[1:])))
                           for axis, (value, center) in enumerate(zip(affine, mean, strict=True)))
        lower_hull = tuple(min(row[axis] for row in model.centers) for axis in range(dimension))
        upper_hull = tuple(max(row[axis] for row in model.centers) for axis in range(dimension))
        lower_box = _affine_box(restriction, box, lower_hull, source.time, arithmetic)
        upper_box = _affine_box(restriction, box, upper_hull, source.time, arithmetic)
        error_upper = _rational(error).hi
        expanded = tuple(_add(value, Interval(-error_upper, error_upper)) for value in prediction)
        enclosure = tuple(value.intersect(Interval(lower.lo, upper.hi))
                          for value, lower, upper in zip(expanded, lower_box, upper_box, strict=True))
        midpoint = tuple(checked(arithmetic.add(Fraction(value.lo), Fraction(value.hi))/2) for value in prediction)
        feasible = tuple(arithmetic.add(anchor, value) for anchor, value in
                         zip(restriction.anchor, _linear(restriction.projector, midpoint, arithmetic), strict=True))
        squared_radius = Fraction(0)
        for interval, middle in zip(prediction, midpoint, strict=True):
            radius = max(abs(arithmetic.add(Fraction(interval.lo), -middle)), abs(arithmetic.add(Fraction(interval.hi), -middle)))
            squared_radius = arithmetic.add(squared_radius, arithmetic.mul(radius, radius))
        point_error = arithmetic.add(error, checked(Fraction(_sqrt_upper(squared_radius, model.budget))))
        twice_time = arithmetic.mul(Fraction(2), source.time)
        gap = checked(arithmetic.mul(error, error)/twice_time)
        point_gap = checked(arithmetic.mul(point_error, point_error)/twice_time)
        from omnibias.core.proof.certificate import encode_interval, make_certificate

        certificate = make_certificate(
            claim="Uniform constrained Gaussian optimum enclosure and feasible-point gap from a forward approximation.",
            payload={
                "type": "gaussian_affine_aging", "restriction_certificate": deepcopy(restriction.certificate),
                "mode": mode, "order": order, "target_box": [encode_interval(value) for value in box],
                "projected_target_box": [encode_interval(value) for value in projected_box],
                "contraction_bound": _pair(q), "aging_parameter": _pair(parameter), "normal_cost_range": _pair(cost_range),
                "posterior": [encode_interval(value) for value in posterior], "relative_mean": [encode_interval(value) for value in mean],
                "correction_coefficients": [[encode_interval(value) for value in row] for row in coefficients],
                "predictor_enclosure": [encode_interval(value) for value in prediction], "enclosure": [encode_interval(value) for value in enclosure],
                "approximation_error_bound": _pair(error), "objective_gap_bound": _pair(gap),
                "feasible_point": [_pair(value) for value in feasible], "feasible_point_error_bound": _pair(point_error),
                "feasible_point_objective_gap_bound": _pair(point_gap), "predictor_box_radius_squared": _pair(squared_radius),
                "coefficient_convention": "Ordinary coefficients of X(v,s/t)-(s/t)*P*v-x0-mu_infinity with original source biases and constraints fixed; c0=0.",
                "argument": "The limiting posterior uses original biases and tangent centers. Relative normal costs enter the first log coefficient along with squared centered tangent distances. Let V=D_T^2+range(c), Q=V/(8*nu*s). Real cost-tilt covariance bounds give errors D_T*Q/2 and D_T*Q*(Q+q_T)/2 at orders zero and one. For the series, a complex inverse-age disk of radius 8*nu/(3*V) has contraction at most 3/5 on the ball of radius D_T/2, yielding coefficient bounds D_T/2*(3*Q)^n and the geometric tail. Identical tangent centers are exactly affine regardless of cost range. The exact finite formula lies in the affine space. Projecting the rational midpoint of its interval enclosure cannot increase distance to that formula, so adding the interval half-diagonal to the analytic error bounds the feasible point's distance to every optimizer. Restricted Hessian <=I/s gives both reported objective-gap bounds.",
                "error_scope": "Exact formula bounds exclude floating tensor error. Feasible-point bounds include interval evaluation and target-box variation. Coordinate enclosures need not contain only feasible points. No derivative bound is inferred from a value error.",
            },
            honesty={"unproven_claim": False, "unconditional_transcendentals": q != 0, "continuum_pde_claim": False},
            meta={"transcend_backend": "not_used" if q == 0 else "mpmath"},
        )
        result = object.__new__(GaussianAffineAging)
        for name, stored in (("restriction", restriction), ("target_box", box), ("projected_target_box", projected_box),
                             ("order", order), ("mode", mode), ("aging_parameter", parameter), ("relative_mean", mean),
                             ("correction_coefficients", coefficients), ("predictor_enclosure", prediction),
                             ("approximation_error_bound", error), ("objective_gap_bound", gap), ("enclosure", enclosure),
                             ("feasible_point", feasible), ("feasible_point_error_bound", point_error),
                             ("feasible_point_objective_gap_bound", point_gap), ("certificate", certificate)):
            object.__setattr__(result, name, stored)
        return result
    except (ArithmeticError, ValueError):
        return None


def certify_gaussian_affine_predictor(
    restriction: GaussianAffineRestriction, target_box: Sequence[Interval], *, order: int = 1,
) -> GaussianAffineAging | None:
    r"""Certify a direct order-zero/one constrained optimizer, requiring q_T<1.

    Q=(D_T²+normal_cost_range)/(8*nu*s) may be large. The exact formula's
    uniform Euclidean error is D_T*Q/2 at order zero, or D_T*Q*(Q+q_T)/2 at
    order one. Both use the original limiting posterior, not the transformed
    fixed-age biases. Query work is O(m*d+d²+r*d) after exact restriction
    setup, including the certificate's r retained constraint rows.
    Invalid types/domains raise; exhausted budgets/nonfinite bounds return
    None. Missing rigorous transcendental support raises except when D_T=0.
    """
    return _build(restriction, target_box, order, "predictor")


def certify_gaussian_affine_aging(
    restriction: GaussianAffineRestriction, target_box: Sequence[Interval], *, order: int = 8,
) -> GaussianAffineAging | None:
    r"""Certify an all-order constrained forward series with normal-cost terms.

    Require Q=(D_T²+normal_cost_range)/(8*nu*s)<1/3, except D_T=0 is exactly
    affine for every Q. The uniform Euclidean tail is D_T/2*(3*Q)^(N+1)/
    (1-3*Q), tightened by the real bounds at orders zero/one. Coefficients
    scale inverse age; original parameters and constraints are fixed.
    No inverse iteration or new factorization occurs after restriction setup.
    Work is O((m+d)*N²+m*d*N+d²+r*d), including certificate handling for r
    retained constraint rows; storage includes the exact dense projector.
    Order is budgeted; interval widening can offset a smaller analytic tail.
    Invalid inputs raise; failed bounds/resources return None. A missing
    rigorous backend raises except for an affine restricted map.
    """
    return _build(restriction, target_box, order, "series")


__all__ = ["GaussianAffineAging", "certify_gaussian_affine_aging", "certify_gaussian_affine_predictor"]
