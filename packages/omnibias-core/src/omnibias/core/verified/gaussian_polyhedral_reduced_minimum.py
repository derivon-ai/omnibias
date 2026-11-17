# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Global interval minimization of compiled two-center polyhedral pieces."""

from __future__ import annotations

import math
from copy import deepcopy
from dataclasses import dataclass
from fractions import Fraction
from typing import TYPE_CHECKING

from omnibias.core.gaussian_jet import _Arithmetic
from omnibias.core.verified.gaussian_heat import _finite, _rational
from omnibias.core.verified.gaussian_polyhedral_reduction import GaussianPolyhedralReduction
from omnibias.core.verified.global_min import BoxMinimum, MinimumStatus, minimize_box
from omnibias.core.verified.interval import Interval
from omnibias.core.verified.inverse_gaussian_affine import _intact
from omnibias.core.verified.inverse_gaussian_heat import _pair
from omnibias.core.verified.mean_map import _strict
from omnibias.core.verified.transcend import exp_iv, libm_fallback_used, ln_iv, tanh_iv

if TYPE_CHECKING:
    from omnibias.core.proof.certificate import Cert
    from omnibias.core.verified.gaussian_polyhedral_reduction import GaussianPolyhedralPiece

Vector = tuple[Fraction, ...]
Box = tuple[Interval, ...]
_ZERO = Interval.point(0.0)
_ONE = Interval.point(1.0)


def _logcosh_point(value: float, log_two: Interval) -> Interval:
    """Stable even log(cosh), including rigorous exponential underflow."""
    if value == 0.0:
        return _ZERO
    absolute = Interval.point(abs(value))
    exponent = _finite(_rational(Fraction(-2))*absolute)
    exponential = _finite(exp_iv(exponent))
    exponential = Interval(max(0.0, exponential.lo), exponential.hi)
    return _finite(absolute + ln_iv(_finite(_ONE+exponential)) - log_two)


def _logcosh(value: Interval, log_two: Interval) -> Interval:
    lower = 0.0 if value.lo <= 0.0 <= value.hi else min(abs(value.lo), abs(value.hi))
    upper = max(abs(value.lo), abs(value.hi))
    low = _logcosh_point(lower, log_two)
    high = _logcosh_point(upper, log_two)
    # log(cosh) is nonnegative and increasing on [0,infinity).
    return Interval(max(0.0, low.lo), max(0.0, high.hi))


class _PieceOracle:
    def __init__(self, reduction: GaussianPolyhedralReduction, piece: GaussianPolyhedralPiece,
                 arithmetic: _Arithmetic, log_two: Interval) -> None:
        self.piece, self.arithmetic, self.log_two = piece, arithmetic, log_two
        self.lower, upper = piece.search_interval
        self.width = arithmetic.add(upper, -self.lower)
        self.scale = arithmetic.mul(Fraction(2), reduction.model.viscosity)
        g0, g1, g2 = piece.transverse_cost
        self.polynomial = (g0, arithmetic.add(g1, -arithmetic.mul(self.scale, reduction.field)),
                           arithmetic.add(g2, arithmetic.mul(reduction.model.viscosity, reduction.inverse_q)))

    def parameter(self, value: float) -> Fraction:
        a = self.arithmetic
        return a.add(self.lower, a.mul(self.width, a.exact(Fraction(value))))

    def endpoints(self, box: Box) -> tuple[Fraction, Fraction]:
        return self.parameter(box[0].lo), self.parameter(box[0].hi)

    def quadratic(self, lower: Fraction, upper: Fraction) -> Interval:
        a = self.arithmetic
        c0, c1, c2 = self.polynomial

        def at(point: Fraction) -> Fraction:
            return a.add(c0, a.mul(point, a.add(c1, a.mul(c2, point))))

        values = [at(lower), at(upper)]
        if c2:
            vertex = a._coefficient(-c1/a.mul(Fraction(2), c2))
            if lower < vertex < upper:
                values.append(at(vertex))
        return Interval(_rational(min(values)).lo, _rational(max(values)).hi)

    def value(self, box: Box) -> Interval:
        self.arithmetic._work(128)
        lower, upper = self.endpoints(box)
        y = Interval(_rational(lower).lo, _rational(upper).hi)
        return _finite(self.quadratic(lower, upper)-_rational(self.scale)*_logcosh(y, self.log_two))

    def gradient(self, box: Box) -> Box:
        self.arithmetic._work(64)
        if not self.width:
            return (_ZERO,)
        lower, upper = self.endpoints(box)
        y = Interval(_rational(lower).lo, _rational(upper).hi)
        derivative = _finite(_rational(self.piece.scalar_curvature)*y-tanh_iv(y)
                             -_rational(self.piece.scalar_field))
        return (_finite(_rational(self.arithmetic.mul(self.scale, self.width))*derivative),)

    def point(self, parameter: float) -> Vector:
        a = self.arithmetic
        y = self.parameter(parameter)
        return tuple(a.add(c, a.mul(slope, y)) for c, slope in
                     zip(self.piece.point_intercept, self.piece.point_slope, strict=True))

    def physical_box(self, box: Box) -> Box:
        a = self.arithmetic
        lower, upper = self.endpoints(box)
        result = []
        for intercept, slope in zip(self.piece.point_intercept, self.piece.point_slope, strict=True):
            values = (a.add(intercept, a.mul(slope, lower)), a.add(intercept, a.mul(slope, upper)))
            result.append(Interval(_rational(min(values)).lo, _rational(max(values)).hi))
        return tuple(result)


@dataclass(frozen=True, init=False)
class GaussianPolyhedralReducedMinimum:
    """Global reduced objective bounds and a cover of all original minimizers.

    ``minimum`` and ``sample_value`` use the compiler's common reduced
    objective, which differs from the original Gaussian potential by one
    constant independent of the feasible point and piece. ``sample_point``
    is exactly feasible. ``candidate_boxes`` enclose every global minimizer;
    they can overlap and include unresolved nonoptimal points. A value-gap
    tolerance gives neither uniqueness nor a small spatial enclosure.

    ``candidate_parameters`` are binary64 intervals in [0,1], mapped through
    each indexed piece's exact rational search interval. ``piece_results``
    retain the complete bounded optimization diagnostics. Digests establish
    data integrity, not verification by an external formal proof checker.
    """

    reduction: GaussianPolyhedralReduction
    minimum: Interval
    sample_point: Vector
    sample_value: Interval
    candidate_boxes: tuple[Box, ...]
    candidate_piece_indices: tuple[int, ...]
    candidate_parameters: tuple[Interval, ...]
    piece_results: tuple[BoxMinimum, ...]
    status: MinimumStatus
    gap_upper: float
    value_tolerance: float
    max_boxes_per_piece: int
    certificate: Cert

    def __init__(self) -> None:
        raise TypeError("use certify_gaussian_polyhedral_reduced_minimum")


def certify_gaussian_polyhedral_reduced_minimum(
    reduction: GaussianPolyhedralReduction, *, value_tolerance: float = 1e-6,
    max_boxes_per_piece: int = 255,
) -> GaussianPolyhedralReducedMinimum | None:
    """Bound the global minimum across EVERY compiled feasible scalar piece.

    Empty (infeasible) reductions, damaged certificates, exhausted aggregate
    resources and nonfinite interval arithmetic return None. Invalid argument
    types/domains raise. A missing rigorous transcendental backend or prior
    conditional-libm provenance raises RuntimeError. Strict evaluation never
    clears that history. All requested piece work/storage is preflighted before
    callbacks; failure never returns a partial-piece global claim.

    Each piece uses bounded interval branch-and-bound on [0,1], with exact
    rational mapping to its finite search interval. Only a strict lower bound
    above the best feasible upper bound prunes a candidate, preserving ties.
    Budget or resolution exits can return sound coarse covers and value bounds.
    """
    if not isinstance(reduction, GaussianPolyhedralReduction):
        raise TypeError("reduction must be a GaussianPolyhedralReduction")
    if type(max_boxes_per_piece) is not int:
        raise TypeError("max_boxes_per_piece must be an integer excluding bool")
    if max_boxes_per_piece < 1:
        raise ValueError("max_boxes_per_piece must be positive")
    if isinstance(value_tolerance, bool) or not isinstance(value_tolerance, (int, float, Fraction)):
        raise TypeError("value_tolerance must be a finite nonnegative real")
    try:
        tolerance = float(value_tolerance)
    except OverflowError as exc:
        raise ValueError("value_tolerance must be finite and nonnegative") from exc
    if not math.isfinite(tolerance) or tolerance < 0:
        raise ValueError("value_tolerance must be finite and nonnegative")
    if not reduction.pieces:
        return None
    arithmetic = _Arithmetic(reduction.model.budget)
    count, dimension = len(reduction.pieces), reduction.model.dimension
    try:
        source_slots = (64*(len(reduction.inequalities)+count+reduction.model.mode_count+1)*(dimension+1)
                        + 32*(dimension+1)**2+2048)
        # Includes retained BoxMinimum candidates, physical outputs, certificate
        # snapshots and the full heap. Each box has <=2 value+1 gradient calls;
        # an additional value call supports global strict pruning.
        arithmetic._entries(source_slots+64*count*(dimension+1)*(max_boxes_per_piece+1)+4096)
        reduction.model.budget._work(source_slots+1024*count*(dimension+1)*(max_boxes_per_piece+1)+4096)
        arithmetic._work(source_slots)
        arithmetic.exact(max_boxes_per_piece)
        if not _intact(reduction.certificate):
            return None
        if libm_fallback_used():
            raise RuntimeError("reduced Gaussian minimization refuses prior conditional libm fallback history")
        with _strict():
            log_two = _finite(ln_iv(Interval.point(2.0)))
            oracles = tuple(_PieceOracle(reduction, piece, arithmetic, log_two) for piece in reduction.pieces)
            results = tuple(minimize_box(oracle.value, (Interval(0.0, 1.0),), gradient=oracle.gradient,
                                         value_tolerance=tolerance, max_boxes=max_boxes_per_piece) for oracle in oracles)
            best_index = min(range(count), key=lambda index: results[index].point_value.hi)
            best = results[best_index]
            minimum = Interval(min(result.minimum.lo for result in results), best.point_value.hi)
            _finite(minimum)
            sample = oracles[best_index].point(best.point[0])
            boxes, indices, parameters = [], [], []
            for index, (oracle, result) in enumerate(zip(oracles, results, strict=True)):
                if result.minimum.lo > minimum.hi:
                    continue
                for candidate in result.candidates:
                    lower = max(result.minimum.lo, oracle.value(candidate).lo)
                    if lower <= minimum.hi:
                        boxes.append(oracle.physical_box(candidate))
                        indices.append(index)
                        parameters.append(candidate[0])
            if not boxes:
                return None
            gap = 0.0 if minimum.lo == minimum.hi else math.nextafter(minimum.hi-minimum.lo, math.inf)
            status: MinimumStatus = "value_tolerance" if gap <= tolerance else (
                "box_budget" if any(result.status == "box_budget" for result in results) else "resolution_limit")
            from omnibias.core.proof.certificate import encode_interval, make_certificate

            certificate = make_certificate(
                claim="Global reduced Gaussian objective bounds and a cover of all polyhedrally feasible global minimizers.",
                payload={
                    "type": "gaussian_polyhedral_reduced_minimum", "reduction_certificate": deepcopy(reduction.certificate),
                    "minimum": encode_interval(minimum), "sample_point": [_pair(value) for value in sample],
                    "sample_value": encode_interval(best.point_value),
                    "candidate_boxes": [[encode_interval(value) for value in box] for box in boxes],
                    "candidate_piece_indices": indices, "candidate_parameters": [encode_interval(value) for value in parameters],
                    "piece_results": [{"minimum": encode_interval(result.minimum), "point": [value.hex() for value in result.point],
                                       "point_value": encode_interval(result.point_value), "status": result.status,
                                       "candidates": [[encode_interval(value) for value in box] for box in result.candidates],
                                       "evaluated_boxes": result.evaluated_boxes, "value_evaluations": result.value_evaluations,
                                       "gradient_evaluations": result.gradient_evaluations} for result in results],
                    "status": status, "gap_upper": gap.hex(), "value_tolerance": tolerance.hex(),
                    "max_boxes_per_piece": max_boxes_per_piece,
                    "argument": "The reduction supplies a complete cover of transverse projection KKT pieces and finite search intervals containing all piece minima. Map t in [0,1] exactly to y=L+(H-L)t. Enclose the common objective g(y)+2*nu*(lambda*y²/2-log(cosh(y))-w*y) and its derivative 2*nu*(H-L)*(k*y-tanh(y)-h). Stable even log(cosh) uses |y|+log(1+exp(-2|y|))-log(2), with outward arithmetic and rigorous transcendental enclosures. Each bounded interval minimization retains all minimizers. Take the least piece lower bound and a feasible sampled upper bound; discard only candidates with strict lower>upper. Exact affine maps enclose all remaining original minimizers, including ties and piece breakpoints.",
                    "scope": "Reduced objective values differ from the original potential by a common additive constant. No unique minimizer, narrow spatial cover, requested-tolerance completion, differentiable selection, arbitrary mixture, or external formal theorem-prover verification claim.",
                }, honesty={"unproven_claim": False, "unconditional_transcendentals": True, "continuum_pde_claim": False},
                meta={"transcend_backend": "mpmath"},
            )
        output = object.__new__(GaussianPolyhedralReducedMinimum)
        for name, value in (("reduction", reduction), ("minimum", minimum), ("sample_point", sample),
                            ("sample_value", best.point_value), ("candidate_boxes", tuple(boxes)),
                            ("candidate_piece_indices", tuple(indices)), ("candidate_parameters", tuple(parameters)),
                            ("piece_results", results), ("status", status), ("gap_upper", gap),
                            ("value_tolerance", tolerance), ("max_boxes_per_piece", max_boxes_per_piece), ("certificate", certificate)):
            object.__setattr__(output, name, value)
        return output
    except (ArithmeticError, ValueError):
        return None


__all__ = ["GaussianPolyhedralReducedMinimum", "certify_gaussian_polyhedral_reduced_minimum"]
