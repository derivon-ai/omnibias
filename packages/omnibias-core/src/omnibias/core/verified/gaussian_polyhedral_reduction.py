# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Exact scalar reduction of two-center Gaussian decisions on polyhedra."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from fractions import Fraction
from itertools import combinations
from typing import TYPE_CHECKING, TypeVar

from omnibias.core._rational_linear import factor_positive_definite, solve_affine
from omnibias.core.gaussian_jet import _Arithmetic
from omnibias.core.verified.gaussian_affine import _dot, _validate_vector
from omnibias.core.verified.gaussian_heat import GaussianHeatMap
from omnibias.core.verified.inverse_gaussian_affine import _intact, _linear
from omnibias.core.verified.inverse_gaussian_heat import _pair

if TYPE_CHECKING:
    from typing import Any

    from omnibias.core.proof.certificate import Cert

Rational = int | Fraction
Vector = tuple[Fraction, ...]
Matrix = tuple[Vector, ...]


def _preflight(rows: int, dimension: int, arithmetic: _Arithmetic, mode_count: int = 2) -> int:
    """Bound the FULL enumeration before allocating bases or partial results."""
    base = 24*(rows+1)*(dimension+1)+16*(dimension+1)**2+64*mode_count*(dimension+1)+512
    per_piece = 24*dimension+12*rows+96
    per_work = 64*(dimension+1)**3+64*(rows+1)*(dimension+1)+128
    arithmetic._entries(base+per_piece)
    arithmetic._work(64*(rows+mode_count+1)*(dimension+1)+256)
    count = term = 1
    for size in range(1, min(rows, dimension-1)+1):
        term = term*(rows-size+1)//size
        count += term
        arithmetic._entries(base+count*per_piece)
        arithmetic.limits._work(arithmetic.used+count*per_work)
    arithmetic._entries(base+count*per_piece)
    arithmetic.limits._work(arithmetic.used+count*per_work)
    return count


def _intersect(
    lower: Fraction | None, upper: Fraction | None, offset: Fraction, slope: Fraction, arithmetic: _Arithmetic,
) -> tuple[Fraction | None, Fraction | None, bool]:
    """Intersect a closed parameter interval with offset+slope*y>=0."""
    if not slope:
        return lower, upper, offset >= 0
    boundary = arithmetic._coefficient(-offset/slope)
    if slope > 0:
        lower = boundary if lower is None else max(lower, boundary)
    else:
        upper = boundary if upper is None else min(upper, boundary)
    return lower, upper, lower is None or upper is None or lower <= upper


def _clamp(value: Fraction, lower: Fraction | None, upper: Fraction | None) -> Fraction:
    if lower is not None:
        value = max(value, lower)
    return value if upper is None else min(value, upper)


@dataclass(frozen=True, init=False)
class _GaussianScalarPiece:
    """Exact transverse geometry with a bounded scalar-profile search interval."""

    active_indices: tuple[int, ...]
    lower: Fraction | None
    upper: Fraction | None
    point_intercept: Vector
    point_slope: Vector
    transverse_cost: tuple[Fraction, Fraction, Fraction]
    scalar_curvature: Fraction
    scalar_field: Fraction
    search_interval: tuple[Fraction, Fraction]

    def __init__(self) -> None:
        raise TypeError("use a Gaussian scalar reduction compiler")


@dataclass(frozen=True, init=False)
class GaussianPolyhedralPiece(_GaussianScalarPiece):
    r"""One exact KKT validity interval and its transverse quadratic minimum.

    ``lower=None`` means minus infinity, ``upper=None`` means plus infinity.
    Singleton and overlapping intervals are intentional. On this entire
    interval, ``x(y)=point_intercept+point_slope*y`` is the UNIQUE feasible
    point minimizing the transverse quadratic at the specified logit y.
    ``transverse_cost=(g0,g1,g2)`` represents g0+g1*y+g2*y².
    The remaining scalar objective has derivative
    2*nu*(scalar_curvature*y-tanh(y)-scalar_field).

    ``search_interval`` is finite and contains every minimum of this piece's
    objective, including minima at validity endpoints. It is NOT a claim
    that every point in that interval is globally optimal.
    """

    def __init__(self) -> None:
        raise TypeError("use compile_gaussian_polyhedral_reduction")


@dataclass(frozen=True, init=False)
class GaussianPolyhedralReduction:
    r"""Complete exact scalar KKT cover of the original constrained objective.

    For centers c-a,c+a, alpha=|a|² and lambda=2*nu*s/alpha, write
    x=c+lambda*a*(y-delta/2)+r with a.r=0. At fixed y, the original
    target-tilted potential is a constant plus
    ||r-s*P*target||²/(2*s)+2*nu*(lambda*y²/2-log(cosh(y))-field*y).
    Each stored piece minimizes the transverse quadratic on G*x<=h.
    Their union covers every feasible scalar y, without a full-rank or
    interior-point assumption. Overlaps describe the same point and cost.

    Empty ``pieces`` proves the original polyhedron is infeasible. A factory
    return of None instead means unsupported metadata or exhausted budgets.
    A nonempty result reduces GLOBAL minimization to all finite piece search
    intervals; it does not select a stationary branch or assert an optimum.
    """

    model: GaussianHeatMap
    inequalities: Matrix
    upper_bounds: Vector
    target: Vector
    midpoint: Vector
    half_difference: Vector
    projector: Matrix
    inverse_q: Fraction
    bias_contrast: Fraction
    field: Fraction
    pieces: tuple[GaussianPolyhedralPiece, ...]
    enumerated_bases: int
    certificate: Cert

    def __init__(self) -> None:
        raise TypeError("use compile_gaussian_polyhedral_reduction")

    @property
    def feasible(self) -> bool:
        """Exact feasibility, after a COMPLETE bounded enumeration."""
        return bool(self.pieces)

    def _query(self, piece_index: int, parameter: Rational) -> tuple[GaussianPolyhedralPiece, Fraction, _Arithmetic]:
        if type(piece_index) is not int:
            raise TypeError("piece_index must be an integer, excluding bool")
        if not 0 <= piece_index < len(self.pieces):
            raise IndexError("piece_index outside the reduction")
        if isinstance(parameter, bool) or not isinstance(parameter, (int, Fraction)):
            raise TypeError("parameter must be exact int or Fraction, excluding bool")
        arithmetic = _Arithmetic(self.model.budget)
        query_size = (self.model.dimension+1)*(len(self.pieces)+len(self.inequalities)+1)
        arithmetic._entries(32*query_size+16*(self.model.dimension+1)**2+1024)
        arithmetic._work(128*query_size+64*(self.model.dimension+1)**2+512)
        if not _intact(self.certificate):
            raise ValueError("polyhedral Gaussian reduction certificate is damaged")
        value = arithmetic.exact(parameter)
        piece = self.pieces[piece_index]
        if (piece.lower is not None and value < piece.lower) or (piece.upper is not None and value > piece.upper):
            raise ValueError("parameter lies outside the piece's validity interval")
        return piece, value, arithmetic

    def lift(self, piece_index: int, parameter: Rational) -> Vector:
        """Exact minimizing feasible point of the transverse problem at y."""
        piece, value, arithmetic = self._query(piece_index, parameter)
        return tuple(arithmetic.add(a, arithmetic.mul(b, value))
                     for a, b in zip(piece.point_intercept, piece.point_slope, strict=True))

    def transverse_value(self, piece_index: int, parameter: Rational) -> Fraction:
        """Exact transverse quadratic cost g(y); omits the scalar potential."""
        piece, value, arithmetic = self._query(piece_index, parameter)
        g0, g1, g2 = piece.transverse_cost
        return arithmetic.add(g0, arithmetic.mul(value, arithmetic.add(g1, arithmetic.mul(value, g2))))


def _piece_payload(piece: _GaussianScalarPiece) -> dict[str, Any]:
    return {
        "active_indices": list(piece.active_indices),
        "lower": None if piece.lower is None else _pair(piece.lower),
        "upper": None if piece.upper is None else _pair(piece.upper),
        "point_intercept": [_pair(value) for value in piece.point_intercept],
        "point_slope": [_pair(value) for value in piece.point_slope],
        "transverse_cost": [_pair(value) for value in piece.transverse_cost],
        "scalar_curvature": _pair(piece.scalar_curvature), "scalar_field": _pair(piece.scalar_field),
        "search_interval": [_pair(value) for value in piece.search_interval],
    }


_PieceT = TypeVar("_PieceT", bound=_GaussianScalarPiece)


def _compile_pieces(
    model: GaussianHeatMap, matrix: Matrix, upper: Vector, goal: Vector,
    direction: Vector, anchor: Vector, axial: Vector, projector: Matrix,
    axial_scale: Fraction, field: Fraction, mean_bounds: tuple[Fraction, Fraction],
    arithmetic: _Arithmetic, count: int, piece_type: type[_PieceT],
) -> tuple[_PieceT, ...]:
    """Shared complete transverse QP cover for a scalar Gaussian profile.

    ``mean_bounds`` encloses the derivative of its log-partition function.
    The caller preflights the full combinatorial enumeration and metadata.
    Exact rank/factor calls and every piece use the same arithmetic meter.
    """
    rows, dimension = len(matrix), model.dimension
    twice_nu = arithmetic.mul(Fraction(2), model.viscosity)
    preferred = tuple(arithmetic.mul(model.time, value) for value in _linear(projector, goal, arithmetic))
    projected = tuple(_linear(projector, row, arithmetic) for row in matrix)
    offsets = tuple(arithmetic.add(bound, -_dot(row, anchor, arithmetic)) for row, bound in zip(matrix, upper, strict=True))
    slopes = tuple(-_dot(row, axial, arithmetic) for row in matrix)
    candidates: list[_PieceT] = []
    enumerated = 0
    for size in range(min(rows, dimension-1)+1):
        for active in combinations(range(rows), size):
            enumerated += 1
            arithmetic._work(32*(dimension+rows+1))
            selected = tuple(projected[index] for index in active)
            if size:
                if solve_affine(selected, None, arithmetic).rank != size:
                    continue
                gram = tuple(tuple(_dot(left, right, arithmetic) for right in selected) for left in selected)
                factor = factor_positive_definite(gram, arithmetic)
                eta0 = factor.solve(tuple(arithmetic.add(_dot(projected[i], preferred, arithmetic), -offsets[i]) for i in active), arithmetic)
                eta1 = factor.solve(tuple(-slopes[i] for i in active), arithmetic)
            else:
                eta0 = eta1 = ()
            residual0 = tuple(-arithmetic.sum(tuple(arithmetic.mul(eta, row[j]) for eta, row in zip(eta0, selected, strict=True)))
                              for j in range(dimension))
            residual1 = tuple(-arithmetic.sum(tuple(arithmetic.mul(eta, row[j]) for eta, row in zip(eta1, selected, strict=True)))
                              for j in range(dimension))
            point0 = tuple(arithmetic.add(arithmetic.add(base, normal), change)
                           for base, normal, change in zip(anchor, preferred, residual0, strict=True))
            point1 = tuple(arithmetic.add(base, change) for base, change in zip(axial, residual1, strict=True))
            lower: Fraction | None = None
            upper_limit: Fraction | None = None
            valid = True
            for offset, slope in zip(eta0, eta1, strict=True):
                lower, upper_limit, valid = _intersect(lower, upper_limit, offset, slope, arithmetic)
                if not valid:
                    break
            if not valid:
                continue
            for row, bound in zip(matrix, upper, strict=True):
                lower, upper_limit, valid = _intersect(lower, upper_limit,
                    arithmetic.add(bound, -_dot(row, point0, arithmetic)), -_dot(row, point1, arithmetic), arithmetic)
                if not valid:
                    break
            if not valid:
                continue
            # Replay stationarity and selected activity before recording a
            # piece. Exact rank tests never swallow resource exceptions.
            if _dot(direction, residual0, arithmetic) or _dot(direction, residual1, arithmetic):
                raise ArithmeticError("transverse correction left the orthogonal subspace")
            if any(_dot(matrix[i], point0, arithmetic) != upper[i] or _dot(matrix[i], point1, arithmetic) for i in active):
                raise ArithmeticError("selected equality replay failed")
            twice_time = arithmetic.mul(Fraction(2), model.time)
            cost0 = arithmetic._coefficient(_dot(residual0, residual0, arithmetic)/twice_time)
            cost1 = arithmetic._coefficient(_dot(residual0, residual1, arithmetic)/model.time)
            cost2 = arithmetic._coefficient(_dot(residual1, residual1, arithmetic)/twice_time)
            curvature = arithmetic.add(axial_scale, arithmetic._coefficient(cost2/model.viscosity))
            scalar_field = arithmetic.add(field, -arithmetic._coefficient(cost1/twice_nu))
            search = tuple(_clamp(arithmetic._coefficient(arithmetic.add(scalar_field, endpoint)/curvature), lower, upper_limit)
                           for endpoint in mean_bounds)
            piece = object.__new__(piece_type)
            for name, stored in (("active_indices", active), ("lower", lower), ("upper", upper_limit),
                                ("point_intercept", point0), ("point_slope", point1),
                                ("transverse_cost", (cost0, cost1, cost2)), ("scalar_curvature", curvature),
                                ("scalar_field", scalar_field), ("search_interval", search)):
                object.__setattr__(piece, name, stored)
            candidates.append(piece)
    if enumerated != count:
        raise ArithmeticError("incomplete active-basis enumeration")
    return tuple(candidates)


def compile_gaussian_polyhedral_reduction(
    model: GaussianHeatMap, inequalities: Sequence[Sequence[Rational]], upper_bounds: Sequence[Rational],
    target: Sequence[Rational],
) -> GaussianPolyhedralReduction | None:
    r"""Compile all transverse KKT bases for G*x<=h, retaining every piece.

    Exactly two distinct original centers are required; every q>0 is allowed.
    Empty, redundant and zero rows are supported. Equalities can be supplied
    as two opposite inequalities. All data are exact int/Fraction. Invalid
    input types/shapes raise. Budget exhaustion or unsupported families return
    None, never a partial cover or an infeasibility claim. A successfully
    constructed empty cover proves infeasibility. Enumeration is combinatorial
    in the inequality count, with subset/storage/work bounds checked BEFORE
    enumeration. No transcendental evaluation or floating rank test is used.
    """
    if not isinstance(model, GaussianHeatMap):
        raise TypeError("model must be a GaussianHeatMap")
    if not isinstance(inequalities, Sequence) or isinstance(inequalities, (str, bytes)):
        raise TypeError("inequalities must be a finite exact matrix")
    rows, dimension = len(inequalities), model.dimension
    _validate_vector(upper_bounds, rows, "upper_bounds")
    _validate_vector(target, dimension, "target")
    for row in inequalities:
        _validate_vector(row, dimension, "inequality row")
    if model.mode_count != 2:
        return None
    arithmetic = _Arithmetic(model.budget)
    try:
        count = _preflight(rows, dimension, arithmetic)
        arithmetic.exact(count)
        matrix = tuple(tuple(arithmetic.exact(value) for value in row) for row in inequalities)
        upper = tuple(arithmetic.exact(value) for value in upper_bounds)
        goal = tuple(arithmetic.exact(value) for value in target)
        direction = tuple(arithmetic._coefficient(arithmetic.add(plus, -minus)/2)
                          for minus, plus in zip(model.centers[0], model.centers[1], strict=True))
        alpha = _dot(direction, direction, arithmetic)
        if not alpha:
            return None
        midpoint = tuple(arithmetic.add(origin, delta) for origin, delta in zip(model.centers[0], direction, strict=True))
        bias = arithmetic.add(model.biases[1], -model.biases[0])
        twice_nu = arithmetic.mul(Fraction(2), model.viscosity)
        inverse_q = arithmetic._coefficient(arithmetic.mul(twice_nu, model.time)/alpha)
        field = arithmetic._coefficient(arithmetic.mul(model.time,
            arithmetic.add(_dot(direction, goal, arithmetic), arithmetic.mul(model.viscosity, bias)))/alpha)
        projector = tuple(tuple(arithmetic.add(Fraction(int(i == j)),
                               -arithmetic._coefficient(arithmetic.mul(direction[i], direction[j])/alpha))
                               for j in range(dimension)) for i in range(dimension))
        axial = tuple(arithmetic.mul(inverse_q, value) for value in direction)
        half_bias = arithmetic._coefficient(bias/2)
        anchor = tuple(arithmetic.add(c, -arithmetic.mul(a, half_bias)) for c, a in zip(midpoint, axial, strict=True))
        candidates = _compile_pieces(
            model, matrix, upper, goal, direction, anchor, axial, projector,
            inverse_q, field, (Fraction(-1), Fraction(1)), arithmetic, count, GaussianPolyhedralPiece,
        )
        from omnibias.core.proof.certificate import make_certificate

        certificate = make_certificate(
            claim="Complete exact scalar reduction and finite global-minimizer search cover for a two-center Gaussian on a polyhedron.",
            payload={
                "type": "gaussian_polyhedral_reduction", "centers": [[_pair(value) for value in row] for row in model.centers],
                "biases": [_pair(value) for value in model.biases], "viscosity": _pair(model.viscosity), "time": _pair(model.time),
                "inequalities": [[_pair(value) for value in row] for row in matrix], "upper_bounds": [_pair(value) for value in upper],
                "target": [_pair(value) for value in goal], "midpoint": [_pair(value) for value in midpoint],
                "half_difference": [_pair(value) for value in direction], "projector": [[_pair(value) for value in row] for row in projector],
                "inverse_q": _pair(inverse_q), "bias_contrast": _pair(bias), "field": _pair(field),
                "pieces": [_piece_payload(piece) for piece in candidates], "enumerated_bases": count,
                "feasible": bool(candidates),
                "argument": "At logit y, write x=c+lambda*a*(y-delta/2)+r, a.r=0. The original potential minus target.x equals a common constant plus ||r-s*P*target||²/(2*s)+2*nu*(lambda*y²/2-log(cosh(y))-w*y). The fixed-y feasible set is a closed polyhedron. Its nonempty fiber has a unique nearest point to s*P*target. The polyhedral normal-cone optimality condition expresses the correction by nonnegative multipliers on active projected rows. Conic Caratheodory reduces that representation to independent projected rows, at most dimension-1. Every such subset, including the empty set, was enumerated. Its exact Gram solve gives affine correction and multipliers; their nonnegativity and ALL original slacks define the stored closed parameter interval. Hence the intervals cover exactly every feasible fiber, including singleton intervals and degeneracies. Overlapping pieces have the same point and value by strict convexity. The transverse minimum is a convex quadratic on each piece. The remaining derivative is 2*nu*(k*y-tanh(y)-h), k>0. It is strictly negative below (h-1)/k and strictly positive above (h+1)/k. Clamping those two strip endpoints to the full validity interval therefore retains all minima, even for disjoint strips and interval endpoints. The original quadratic-minus-logsumexp potential is coercive, so a nonempty closed polyhedron has at least one global minimizer; all are covered by the finite piece search intervals.",
                "scope": "Exactly two distinct original centers, fixed positive viscosity/time, exact target and finite polyhedral inequalities. No optimum or unique/global stationary branch is selected. Enumeration can be combinatorial. Infeasibility is asserted only for a completed empty cover. Objective values are represented up to one common constant. No continuum inverse PDE, fast general optimizer or formal theorem-prover verification claim.",
            }, honesty={"unproven_claim": False, "continuum_pde_claim": False}, meta={"transcend_backend": "not_used"},
        )
        result = object.__new__(GaussianPolyhedralReduction)
        for result_name, result_value in (("model", model), ("inequalities", matrix), ("upper_bounds", upper), ("target", goal),
                            ("midpoint", midpoint), ("half_difference", direction), ("projector", projector),
                            ("inverse_q", inverse_q), ("bias_contrast", bias), ("field", field),
                            ("pieces", tuple(candidates)), ("enumerated_bases", count), ("certificate", certificate)):
            object.__setattr__(result, result_name, result_value)
        return result
    except (ArithmeticError, ValueError):
        return None


__all__ = ["GaussianPolyhedralPiece", "GaussianPolyhedralReduction", "compile_gaussian_polyhedral_reduction"]
