# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Global Gaussian decisions through a feasible projected-simplex image."""

from __future__ import annotations

import math
from collections.abc import Sequence
from copy import deepcopy
from dataclasses import dataclass
from fractions import Fraction
from typing import TYPE_CHECKING

from omnibias.core.gaussian_jet import _Arithmetic
from omnibias.core.ridge_system import RidgeBudget
from omnibias.core.verified.gaussian_affine import _dot, _validate_vector
from omnibias.core.verified.gaussian_heat import GaussianHeatMap, _finite, _rational
from omnibias.core.verified.global_min import BoxMinimum, MinimumStatus, minimize_box
from omnibias.core.verified.interval import Interval
from omnibias.core.verified.inverse_gaussian_affine import _intact
from omnibias.core.verified.inverse_gaussian_heat import _pair
from omnibias.core.verified.mean_map import _logsum, _strict
from omnibias.core.verified.polyhedral_projection import (
    PolyhedralProjector,
    _project_exact,
    _query_work_bound,
    _source_slots,
)
from omnibias.core.verified.transcend import libm_fallback_used

if TYPE_CHECKING:
    from omnibias.core.proof.certificate import Cert

Rational = int | Fraction
Vector = tuple[Fraction, ...]
Matrix = tuple[Vector, ...]
Box = tuple[Interval, ...]
Bounds = tuple[tuple[Fraction, Fraction], ...]


def _combined_budget(left: RidgeBudget, right: RidgeBudget) -> RidgeBudget:
    return RidgeBudget(max_degree=min(left.max_degree, right.max_degree),
                       max_order=min(left.max_order, right.max_order),
                       max_matrix_entries=min(left.max_matrix_entries, right.max_matrix_entries),
                       max_elimination_work=min(left.max_elimination_work, right.max_elimination_work),
                       max_coefficient_bits=min(left.max_coefficient_bits, right.max_coefficient_bits))


class _ProjectionOracle:
    """Value-only pullback through continuous, possibly nonsmooth projection."""

    def __init__(self, model: GaussianHeatMap, projector: PolyhedralProjector, target: Vector,
                 arithmetic: _Arithmetic) -> None:
        self.model, self.projector, self.target, self.arithmetic = model, projector, target, arithmetic
        self.origin = model.centers[0]
        self.relative = tuple(tuple(arithmetic.add(value, -reference)
                                    for value, reference in zip(row, self.origin, strict=True))
                              for row in model.centers)
        self.shift = tuple(arithmetic.add(reference, arithmetic.mul(model.time, value))
                           for reference, value in zip(self.origin, target, strict=True))
        self.scale = arithmetic.mul(Fraction(2), model.viscosity)
        denominator = arithmetic.mul(self.scale, model.time)
        self.rates = tuple(tuple(arithmetic._coefficient(value/denominator) for value in row)
                           for row in self.relative)
        self.offsets = tuple(arithmetic.add(arithmetic.add(bias, -model.biases[0]),
                             -arithmetic._coefficient(_dot(row, row, arithmetic)/arithmetic.mul(Fraction(2), denominator)))
                             for bias, row in zip(model.biases, self.relative, strict=True))

    def _relative_bounds(self, box: Box) -> Bounds:
        """Exact coordinate ranges of the recursive stick-breaking map."""
        a = self.arithmetic
        bounds = tuple((value, value) for value in self.relative[-1])
        for index in reversed(range(self.model.mode_count-1)):
            endpoints = a.exact(Fraction(box[index].lo)), a.exact(Fraction(box[index].hi))
            result = []
            for vertex, (lower, upper) in zip(self.relative[index], bounds, strict=True):
                values = tuple(a.add(value, a.mul(u, a.add(vertex, -value)))
                               for u in endpoints for value in (lower, upper))
                result.append((min(values), max(values)))
            bounds = tuple(result)
        return bounds

    def point(self, parameters: Sequence[float]) -> Vector:
        a = self.arithmetic
        box = tuple(Interval.point(value) for value in parameters)
        relative = self._relative_bounds(box)
        point = tuple(a.add(shift, bounds[0]) for shift, bounds in zip(self.shift, relative, strict=True))
        projection = _project_exact(self.projector, point, a)
        if projection is None:
            raise ArithmeticError("no feasible exact polyhedral projection")
        return projection[0]

    def image(self, box: Box) -> Bounds:
        """Enclose every projected point with exact rational coordinate bounds."""
        a = self.arithmetic
        relative = self._relative_bounds(box)
        if not self.projector.inequalities:
            return tuple((a.add(shift, lower), a.add(shift, upper))
                         for shift, (lower, upper) in zip(self.shift, relative, strict=True))
        # Use the exact rational midpoint, even when binary64 midpoint rounding
        # would move it. The q-image is enclosed coordinatewise by `relative`.
        midpoint = tuple(a._coefficient(a.add(a.exact(Fraction(axis.lo)), a.exact(Fraction(axis.hi)))/2)
                         for axis in box)
        center_relative = self.relative[-1]
        for index in reversed(range(self.model.mode_count-1)):
            u = midpoint[index]
            center_relative = tuple(a.add(value, a.mul(u, a.add(vertex, -value)))
                                    for vertex, value in zip(self.relative[index], center_relative, strict=True))
        source = tuple(a.add(shift, value) for shift, value in zip(self.shift, center_relative, strict=True))
        projection = _project_exact(self.projector, source, a)
        if projection is None:
            raise ArithmeticError("no feasible exact polyhedral projection")
        deviations = tuple(max(abs(a.add(lower, -center)), abs(a.add(upper, -center)))
                           for (lower, upper), center in zip(relative, center_relative, strict=True))
        radius = a.sum(deviations)
        # ||P(q)-P(q0)||_infinity <= ||q-q0||_2 <= ||q-q0||_1.
        return tuple((a.add(value, -radius), a.add(value, radius)) for value in projection[0])

    def objective(self, physical: Bounds) -> Interval:
        """Original Gaussian objective, with one declared common constant removed."""
        a = self.arithmetic
        a._work(256*(self.model.mode_count+1)*(self.model.dimension+1))
        centered = tuple((a.add(lower, -origin), a.add(upper, -origin))
                         for (lower, upper), origin in zip(physical, self.origin, strict=True))
        polynomial_lower = polynomial_upper = Fraction(0)
        twice_time = a.mul(Fraction(2), self.model.time)
        for (lower, upper), target in zip(centered, self.target, strict=True):
            def at(value: Fraction, linear: Fraction = target) -> Fraction:
                return a.add(a._coefficient(a.mul(value, value)/twice_time), -a.mul(linear, value))

            values = [at(lower), at(upper)]
            vertex = a.mul(self.model.time, target)
            if lower < vertex < upper:
                values.append(at(vertex))
            polynomial_lower = a.add(polynomial_lower, min(values))
            polynomial_upper = a.add(polynomial_upper, max(values))
        logits = []
        for row, offset in zip(self.rates, self.offsets, strict=True):
            lower = upper = offset
            for rate, (left, right) in zip(row, centered, strict=True):
                products = a.mul(rate, left), a.mul(rate, right)
                lower, upper = a.add(lower, min(products)), a.add(upper, max(products))
            logits.append(Interval(_rational(lower).lo, _rational(upper).hi))
        logsum = Interval(_logsum(tuple(value.lo for value in logits)).lo,
                          _logsum(tuple(value.hi for value in logits)).hi)
        polynomial = Interval(_rational(polynomial_lower).lo, _rational(polynomial_upper).hi)
        return _finite(polynomial-_rational(self.scale)*logsum)

    def value(self, box: Box) -> Interval:
        return self.objective(self.image(box))

    def physical_box(self, box: Box) -> Box:
        return tuple(Interval(_rational(lower).lo, _rational(upper).hi) for lower, upper in self.image(box))


@dataclass(frozen=True, init=False)
class GaussianProjectionMinimum:
    r"""Global value bounds and a physical cover of all constrained minimizers.

    The original centers may have any affine rank and any contraction bound.
    `sample_point` is exactly feasible. Physical `candidate_boxes` can contain
    infeasible and nonoptimal points; they enclose the images of every retained
    parameter box. `candidate_parameters` and `parameter_result` refer to a
    stick-breaking cube, not physical coordinates or posterior probabilities.

    Values use F(x)=|x-c|²/(2s)-target.(x-c)-2nu*logsumexp(
    (a_j-c).(x-c)/(2nu*s)+(b_j-b_0)-|a_j-c|²/(4nu*s)), c=a_0.
    This differs from the original target-tilted potential by one common
    constant. No unique optimizer, derivative or general speed claim is made.
    """

    model: GaussianHeatMap
    projector: PolyhedralProjector
    target: Vector
    minimum: Interval
    sample_point: Vector
    sample_value: Interval
    candidate_boxes: tuple[Box, ...]
    candidate_parameters: tuple[Box, ...]
    parameter_result: BoxMinimum
    status: MinimumStatus
    gap_upper: float
    value_tolerance: float
    max_boxes: int
    certificate: Cert

    def __init__(self) -> None:
        raise TypeError("use certify_gaussian_projection_minimum")


def certify_gaussian_projection_minimum(
    model: GaussianHeatMap, projector: PolyhedralProjector, target: Sequence[Rational], *,
    value_tolerance: float = 1e-6, max_boxes: int = 255,
) -> GaussianProjectionMinimum | None:
    r"""Certify a complete global search on a feasible projected-simplex image.

    Every constrained global minimizer obeys x=P_C(s*target+E[a](x)), so it
    lies in P_C(s*target+conv(centers)), a compact subset of the feasible set.
    Stick breaking maps a closed cube onto this hull. Value-only interval
    minimization is valid across projection active-set switches. It does not
    use a smooth gradient or substitute the hull of projected centers.

    All finite Gaussian families are supported, including one mode, duplicate
    and noncollinear centers, at every positive viscosity and age. Projection
    basis count and parameter search dimension can be large. Both source
    budgets apply through their fieldwise minimum; all requested work/storage
    is preflighted before callbacks. Invalid types/domains raise. Damaged
    projector, empty feasible set, exhausted resources or nonfinite arithmetic
    return None, never a partial global certificate. Missing rigorous backend
    or prior conditional-libm history raises RuntimeError, preserving history.
    """
    if not isinstance(model, GaussianHeatMap):
        raise TypeError("model must be a GaussianHeatMap")
    if not isinstance(projector, PolyhedralProjector):
        raise TypeError("projector must be a PolyhedralProjector")
    if projector.dimension != model.dimension:
        raise ValueError("projector and model dimensions must match")
    _validate_vector(target, model.dimension, "target")
    if type(max_boxes) is not int:
        raise TypeError("max_boxes must be an integer excluding bool")
    if max_boxes < 1:
        raise ValueError("max_boxes must be positive")
    if isinstance(value_tolerance, bool) or not isinstance(value_tolerance, (int, float, Fraction)):
        raise TypeError("value_tolerance must be a finite nonnegative real")
    try:
        tolerance = float(value_tolerance)
    except OverflowError as exc:
        raise ValueError("value_tolerance must be finite and nonnegative") from exc
    if not math.isfinite(tolerance) or tolerance < 0:
        raise ValueError("value_tolerance must be finite and nonnegative")
    arithmetic = _Arithmetic(_combined_budget(model.budget, projector.budget))
    dimension, modes = model.dimension, model.mode_count
    parameters = max(1, modes-1)
    try:
        source = 2*_source_slots(projector)+128*(modes+1)*(dimension+1)+64*(dimension+1)**2+4096
        # <=2 value calls/evaluated box, <=one final image/candidate, one
        # sampled projection, plus a root feasibility check. No hidden solves.
        queries = 3*max_boxes+2
        callback_work = _query_work_bound(projector)+512*(modes+1)*(dimension+1)+1024
        arithmetic._entries(source+128*(max_boxes+1)*(dimension+parameters+1)+4096)
        arithmetic.limits._work(source+queries*callback_work+4096)
        arithmetic._work(source)
        arithmetic.exact(max_boxes)
        if not _intact(projector.certificate):
            return None
        goal = tuple(arithmetic.exact(value) for value in target)
        oracle = _ProjectionOracle(model, projector, goal, arithmetic)
        # Establish a feasible image before claiming a nonempty search domain.
        if _project_exact(projector, oracle.shift, arithmetic) is None:
            return None
        if libm_fallback_used():
            raise RuntimeError("Gaussian projection minimization refuses prior conditional libm fallback history")
        with _strict():
            domain = (Interval(0.0, 1.0),)*(modes-1) if modes > 1 else (Interval.point(0.0),)
            result = minimize_box(oracle.value, domain, value_tolerance=tolerance, max_boxes=max_boxes)
            _finite(result.minimum)
            sample = oracle.point(result.point)
            boxes = tuple(oracle.physical_box(box) for box in result.candidates)
            from omnibias.core.proof.certificate import encode_interval, make_certificate

            certificate = make_certificate(
                claim="Global value enclosure and complete physical minimizer cover for a finite Gaussian mixture on a polyhedron.",
                payload={
                    "type": "gaussian_projection_minimum", "projector_certificate": deepcopy(projector.certificate),
                    "centers": [[_pair(value) for value in row] for row in model.centers],
                    "biases": [_pair(value) for value in model.biases], "viscosity": _pair(model.viscosity), "time": _pair(model.time),
                    "target": [_pair(value) for value in goal], "minimum": encode_interval(result.minimum),
                    "sample_point": [_pair(value) for value in sample], "sample_value": encode_interval(result.point_value),
                    "candidate_boxes": [[encode_interval(value) for value in box] for box in boxes],
                    "candidate_parameters": [[encode_interval(value) for value in box] for box in result.candidates],
                    "parameter_point": [value.hex() for value in result.point],
                    "status": result.status, "gap_upper": result.gap_upper.hex(), "value_tolerance": tolerance.hex(),
                    "max_boxes": max_boxes, "evaluated_boxes": result.evaluated_boxes,
                    "value_evaluations": result.value_evaluations, "gradient_evaluations": result.gradient_evaluations,
                    "argument": "The original finite Gaussian target-tilted potential is coercive, so a nonempty closed polyhedron has a global minimizer. At every constrained local minimum, (x-E[a](x))/s-target belongs to the negative polyhedral normal cone. The Euclidean projection characterization therefore gives x=P_C(s*target+E[a](x)). Every global minimizer lies in S=P_C(s*target+conv(centers)), and S is a compact subset of C; minimizing over S preserves the global value and every original global minimizer. The recursive stick-breaking map from [0,1]^(m-1) is continuous and onto the center hull. With one mode use a dummy singleton parameter. Exact rational cube samples map to exact rational projections, giving feasible incumbents. Coordinate ranges of the recursive affine interpolation bound every source image. Euclidean projection is nonexpansive: around the exact midpoint projection, the L1 upper bound for source displacement encloses the entire physical image. With no constraints the source image itself is exact coordinatewise. The original quadratic-minus-logsumexp objective is enclosed over that physical box with exact centered polynomial/logit bounds and rigorous shifted transcendental evaluation. Value-only interval branch-and-bound retains every parameter minimizer through strict lower>upper pruning and requires no differentiability at projection switches. Mapping all retained boxes yields a physical cover of every original global minimizer.",
                    "normalization": "For c=a_0 and r=x-c, F=|r|²/(2s)-target.r-2nu*logsumexp((a_j-c).r/(2nu*s)+(b_j-b_0)-|a_j-c|²/(4nu*s)). The common bias, target translation and heat-normalization constants are omitted.",
                    "scope": "Any finite positive isotropic Gaussian heat mixture and exact closed polyhedral constraints at fixed positive viscosity/time. Covers may contain infeasible/nonoptimal points. No uniqueness, differentiable selection, requested-tolerance completion, universal speed, arbitrary PDE data, or external formal theorem-prover verification claim. Active-basis enumeration and cube search can be combinatorial.",
                }, honesty={"unproven_claim": False, "unconditional_transcendentals": True, "continuum_pde_claim": False},
                meta={"transcend_backend": "mpmath"},
            )
        output = object.__new__(GaussianProjectionMinimum)
        for name, value in (("model", model), ("projector", projector), ("target", goal), ("minimum", result.minimum),
                            ("sample_point", sample), ("sample_value", result.point_value), ("candidate_boxes", boxes),
                            ("candidate_parameters", result.candidates), ("parameter_result", result), ("status", result.status),
                            ("gap_upper", result.gap_upper), ("value_tolerance", tolerance), ("max_boxes", max_boxes),
                            ("certificate", certificate)):
            object.__setattr__(output, name, value)
        return output
    except (ArithmeticError, ValueError):
        return None


__all__ = ["GaussianProjectionMinimum", "certify_gaussian_projection_minimum"]
