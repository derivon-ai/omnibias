# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Fixed-age constrained approximations with normal costs in the biases."""

from __future__ import annotations

from collections.abc import Sequence
from copy import deepcopy
from dataclasses import dataclass
from fractions import Fraction
from typing import TYPE_CHECKING

from omnibias.core.gaussian_jet import _Arithmetic
from omnibias.core.verified._gaussian_moments import _multiply
from omnibias.core.verified.gaussian_affine import GaussianAffineRestriction
from omnibias.core.verified.gaussian_affine_aging import _affine_box
from omnibias.core.verified.gaussian_aging import (
    GaussianInverseAging,
    certify_gaussian_inverse_aging,
)
from omnibias.core.verified.gaussian_heat import _rational
from omnibias.core.verified.gaussian_predictor import (
    GaussianInversePredictor,
    certify_gaussian_inverse_predictor,
)
from omnibias.core.verified.interval import Interval
from omnibias.core.verified.inverse_gaussian_affine import _intact, _linear
from omnibias.core.verified.inverse_gaussian_heat import _pair, _sqrt_upper

if TYPE_CHECKING:
    from omnibias.core.proof.certificate import Cert

Box = tuple[Interval, ...]
Vector = tuple[Fraction, ...]
Matrix = tuple[Box, ...]
BaseApproximation = GaussianInverseAging | GaussianInversePredictor
_ZERO = Interval.point(0.0)


@dataclass(frozen=True, init=False)
class GaussianAffineApproximation:
    r"""A fixed-age forward optimizer with exact feasibility and gap bounds.

    The transformed biases absorb normal costs at the declared source age.
    ``correction_coefficients`` are ordinary inverse-age coefficients holding
    those transformed biases fixed; they are NOT the original family's age
    coefficients. A runtime derivative of this finite approximation must
    also differentiate the age-dependent bias transformation.

    ``approximation_error_bound`` and ``objective_gap_bound`` apply to the
    exact finite formula at each target. ``predictor_enclosure`` includes
    interval evaluation and target-box variation, and ``enclosure`` contains
    each exact constrained optimum. The exact rational ``feasible_point``
    satisfies the constraints; its separate error and objective-gap bounds
    include interval widths. Coordinate boxes may contain infeasible points.
    No derivative or separately evaluated tensor is certified here.

    ``base_approximation`` retains the unconstrained certificate at an outward
    projected target box. It is absent when all tangent centers coincide.
    """

    restriction: GaussianAffineRestriction
    target_box: Box
    projected_target_box: Box
    order: int
    correction_coefficients: Matrix
    predictor_enclosure: Box
    approximation_error_bound: Fraction
    objective_gap_bound: Fraction
    enclosure: Box
    feasible_point: Vector
    feasible_point_error_bound: Fraction
    feasible_point_objective_gap_bound: Fraction
    base_approximation: BaseApproximation | None
    certificate: Cert

    def __init__(self) -> None:
        raise TypeError("use certify_gaussian_affine_approximation")


def certify_gaussian_affine_approximation(
    restriction: GaussianAffineRestriction, target_box: Sequence[Interval], *, order: int = 8,
) -> GaussianAffineApproximation | None:
    r"""Certify a constrained forward value using fixed-age transformed biases.

    Let q_T=D_T²/(8*nu*s). Orders zero and one require q_T<1 and have
    Euclidean errors D_T*q_T/2 and D_T*q_T². Higher orders require q_T<1/3
    and have error D_T/2*(3*q_T)**(order+1)/(1-3*q_T). Normal-cost range
    does not enter these convergence tests. Coincident tangent centers give
    an exactly affine formula, subject to the same arithmetic budgets.

    After exact restriction setup, no inverse iteration, covariance matrix
    or factorization is needed. Work is O((m+d)*(N+1)²+m*d*(N+1)+d²+r*d)
    for m centers, dimension d, order N and r retained constraint rows;
    stored data include the dense projector and certificate dependencies.
    Projecting an input box can lose correlations and widen the enclosure.

    Exact restriction transfers the transformed value bound to the original
    constrained minimum at this age. Full runtime age differentiation must
    include db'_j/ds=(c_j-c_0)/(4*nu*s²); the coefficients stored here do not.
    Value error alone does not bound derivative error. Invalid inputs raise;
    failed bounds, damaged dependencies, exhausted budgets or nonfinite
    arithmetic return None. Missing rigorous transcendental support raises
    except for an affine restricted map.
    """
    if not isinstance(restriction, GaussianAffineRestriction):
        raise TypeError("restriction must be a GaussianAffineRestriction")
    if not isinstance(target_box, Sequence):
        raise TypeError("target_box must be a finite sequence of Interval objects")
    model = restriction.model
    box = model._box(target_box)
    if type(order) is not int:
        raise TypeError("order must be an integer, excluding bool")
    if order < 0:
        raise ValueError("order must be nonnegative")
    q = model.contraction_bound
    if order > model.budget.max_order or q >= 1 or (order >= 2 and q >= Fraction(1, 3)):
        return None
    try:
        arithmetic = _Arithmetic(model.budget)
        checked = arithmetic._coefficient
        dimension, count, rows = model.dimension, model.mode_count, len(restriction.constraints)
        arithmetic._entries(96*(count+rows+1)*(dimension+1) + 64*(dimension+1)**2
                            + 64*(count+dimension+1)*(order+1) + 512)
        # Reserve both the composed factory's work and our certificate copy,
        # projection and gap calculations. The nested factories also meter
        # exact intermediates and refuse independently under the same budget.
        arithmetic._work(512*(count+rows+1)*(dimension+1) + 512*(dimension+1)**2 + 1024)
        if q:
            arithmetic._work(256*((count+dimension+1)*(order+1)**2
                                 + (count+1)*(dimension+1)*(order+1)))
        else:
            arithmetic._work(64*(order+1)*(dimension+1))
        if not _intact(restriction.certificate):
            return None
        projected = _affine_box(restriction, box, (Fraction(0),)*dimension, Fraction(1), arithmetic)
        coefficients: Matrix = ((_ZERO,)*dimension,)*(order+1)
        base: BaseApproximation | None = None
        if q == 0:
            # Combine the original target's exact affine endpoints before
            # binary conversion, preserving large center/age cancellation.
            prediction = _affine_box(restriction, box, model.centers[0], model.time, arithmetic)
            enclosure = prediction
            error = Fraction(0)
        else:
            if order < 2:
                base = certify_gaussian_inverse_predictor(model, projected, order=order)
                if base is not None and order:
                    scale = _rational(checked(-1/arithmetic.mul(arithmetic.mul(Fraction(4), model.viscosity), model.time)))
                    coefficients = (coefficients[0], tuple(_multiply(scale, value) for value in base.third_trace))
            else:
                base = certify_gaussian_inverse_aging(model, projected, order=order)
                if base is not None:
                    coefficients = base.correction_coefficients
            if base is None or not _intact(base.certificate):
                return None
            prediction, enclosure, error = base.predictor_enclosure, base.enclosure, base.approximation_error_bound
        midpoint = tuple(checked(arithmetic.add(Fraction(value.lo), Fraction(value.hi))/2) for value in prediction)
        feasible = tuple(arithmetic.add(anchor, value) for anchor, value in
                         zip(restriction.anchor, _linear(restriction.projector, midpoint, arithmetic), strict=True))
        squared_radius = Fraction(0)
        for interval, middle in zip(prediction, midpoint, strict=True):
            radius = max(abs(arithmetic.add(Fraction(interval.lo), -middle)), abs(arithmetic.add(Fraction(interval.hi), -middle)))
            squared_radius = arithmetic.add(squared_radius, arithmetic.mul(radius, radius))
        point_error = arithmetic.add(error, checked(Fraction(_sqrt_upper(squared_radius, model.budget))))
        twice_time = arithmetic.mul(Fraction(2), model.time)
        gap = checked(arithmetic.mul(error, error)/twice_time)
        point_gap = checked(arithmetic.mul(point_error, point_error)/twice_time)
        from omnibias.core.proof.certificate import encode_interval, make_certificate

        certificate = make_certificate(
            claim="Fixed-age constrained Gaussian optimum enclosure and an exactly feasible point with an objective-gap bound.",
            payload={
                "type": "gaussian_affine_approximation", "restriction_certificate": deepcopy(restriction.certificate),
                "base_certificate": None if base is None else deepcopy(base.certificate), "order": order,
                "target_box": [encode_interval(value) for value in box],
                "projected_target_box": [encode_interval(value) for value in projected], "contraction_bound": _pair(q),
                "correction_coefficients": [[encode_interval(value) for value in row] for row in coefficients],
                "predictor_enclosure": [encode_interval(value) for value in prediction],
                "enclosure": [encode_interval(value) for value in enclosure],
                "approximation_error_bound": _pair(error), "objective_gap_bound": _pair(gap),
                "feasible_point": [_pair(value) for value in feasible], "feasible_point_error_bound": _pair(point_error),
                "feasible_point_objective_gap_bound": _pair(point_gap), "predictor_box_radius_squared": _pair(squared_radius),
                "coefficient_convention": "Ordinary scaled inverse-age coefficients with transformed biases held fixed at the declared source age; c0=0. Not original-family age coefficients.",
                "argument": "At the declared age, normal costs are absorbed exactly into the restriction's transformed biases. Its inverse at P*target is the original constrained optimizer. All predictor corrections lie in the tangent span, so the exact finite formula is feasible. The composed forward factory gives uniform Euclidean errors D_T*q_T/2, D_T*q_T^2, or the all-order geometric bound under its stated q_T condition, without a normal-cost-range convergence condition. The projected input box contains every exact P*target, and outward evaluation contains the finite formula there. Orthogonal projection of its rational box midpoint cannot increase distance to that feasible formula. Adding the outward half-diagonal to the analytic error bounds the candidate's distance to every optimizer in the original target box. Restricted Hessian <=I/s then gives both objective gaps as squared distance bounds divided by 2*s. Identical tangent centers are evaluated by exact affine endpoint arithmetic.",
                "error_scope": "Formula bounds exclude separate floating evaluation error. Feasible-point bounds include interval evaluation and target-box variation. Coordinate boxes may contain infeasible points. No derivative certificate or convergence claim for the original-family inverse-age expansion.",
            },
            honesty={"unproven_claim": False, "unconditional_transcendentals": q != 0, "continuum_pde_claim": False},
            meta={"transcend_backend": "not_used" if base is None else base.certificate["meta"]["transcend_backend"]},
        )
        result = object.__new__(GaussianAffineApproximation)
        for name, stored in (("restriction", restriction), ("target_box", box), ("projected_target_box", projected),
                             ("order", order), ("correction_coefficients", coefficients), ("predictor_enclosure", prediction),
                             ("approximation_error_bound", error), ("objective_gap_bound", gap), ("enclosure", enclosure),
                             ("feasible_point", feasible), ("feasible_point_error_bound", point_error),
                             ("feasible_point_objective_gap_bound", point_gap), ("base_approximation", base), ("certificate", certificate)):
            object.__setattr__(result, name, stored)
        return result
    except (ArithmeticError, ValueError):
        return None


__all__ = ["GaussianAffineApproximation", "certify_gaussian_affine_approximation"]
