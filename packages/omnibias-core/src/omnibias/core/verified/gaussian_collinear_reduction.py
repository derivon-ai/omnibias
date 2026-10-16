# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Complete scalar reduction of finite collinear Gaussian mixtures on polyhedra."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from fractions import Fraction
from typing import TYPE_CHECKING

from omnibias.core.gaussian_jet import _Arithmetic
from omnibias.core.verified.gaussian_affine import _dot, _validate_vector
from omnibias.core.verified.gaussian_heat import GaussianHeatMap
from omnibias.core.verified.gaussian_polyhedral_reduction import (
    _compile_pieces,
    _GaussianScalarPiece,
    _piece_payload,
    _preflight,
)
from omnibias.core.verified.inverse_gaussian_affine import _intact
from omnibias.core.verified.inverse_gaussian_heat import _pair

if TYPE_CHECKING:
    from omnibias.core.proof.certificate import Cert

Rational = int | Fraction
Vector = tuple[Fraction, ...]
Matrix = tuple[Vector, ...]


@dataclass(frozen=True, init=False)
class GaussianCollinearPiece(_GaussianScalarPiece):
    r"""One exact transverse KKT interval for a collinear Gaussian profile.

    ``point_intercept+point_slope*y`` is the unique transverse minimizer on
    the full validity interval, with cost g0+g1*y+g2*y². None bounds denote
    infinities; singleton intervals and overlapping bases are intentional.
    The remaining scalar derivative is 2*nu*(k*y-E[t]-h), where posterior
    weights are proportional to exp(t_j*y+beta_j) from the parent reduction.
    ``scalar_curvature`` is k, not the full scalar objective curvature.
    The finite ``search_interval`` contains all minima on this piece.
    """

    def __init__(self) -> None:
        raise TypeError("use compile_gaussian_collinear_reduction")


@dataclass(frozen=True, init=False)
class GaussianCollinearReduction:
    r"""Exact scalar representation and complete finite global search cover.

    Choose c=origin and a=direction, so every original center is c+t_j*a.
    With alpha=|a|², lambda=axial_scale=2*nu*s/alpha and P=I-aa^T/alpha,
    write x=c+lambda*a*y+r, a.r=0. Up to one common additive constant, the
    original target-tilted Gaussian potential is

        ||r-s*P*target||²/(2*s)
        +nu*lambda*y²-2*nu*logsumexp(t_j*y+beta_j)-2*nu*field*y.

    Each piece solves the transverse quadratic exactly. Together their full
    validity intervals cover every feasible y, including degenerate fibers.
    Empty pieces proves infeasibility after completed bounded enumeration;
    None from the factory instead means unsupported geometry or resources.
    Neither a branch nor a unique optimum is selected. ``axial_scale`` is
    coordinate-dependent and is not generally inverse contraction_bound.
    """

    model: GaussianHeatMap
    inequalities: Matrix
    upper_bounds: Vector
    target: Vector
    origin: Vector
    direction: Vector
    projector: Matrix
    axial_scale: Fraction
    field: Fraction
    center_coordinates: Vector
    logit_offsets: Vector
    mean_bounds: tuple[Fraction, Fraction]
    pieces: tuple[GaussianCollinearPiece, ...]
    enumerated_bases: int
    certificate: Cert

    def __init__(self) -> None:
        raise TypeError("use compile_gaussian_collinear_reduction")

    @property
    def feasible(self) -> bool:
        """Exact feasibility after complete bounded enumeration."""
        return bool(self.pieces)

    def _query(self, piece_index: int, parameter: Rational) -> tuple[GaussianCollinearPiece, Fraction, _Arithmetic]:
        if type(piece_index) is not int:
            raise TypeError("piece_index must be an integer, excluding bool")
        if not 0 <= piece_index < len(self.pieces):
            raise IndexError("piece_index outside the reduction")
        if isinstance(parameter, bool) or not isinstance(parameter, (int, Fraction)):
            raise TypeError("parameter must be exact int or Fraction, excluding bool")
        arithmetic = _Arithmetic(self.model.budget)
        query_size = (self.model.dimension+1)*(len(self.pieces)+len(self.inequalities)+self.model.mode_count+1)
        arithmetic._entries(64*query_size+32*(self.model.dimension+1)**2+1024)
        arithmetic._work(256*query_size+64*(self.model.dimension+1)**2+512)
        if not _intact(self.certificate):
            raise ValueError("collinear Gaussian reduction certificate is damaged")
        value = arithmetic.exact(parameter)
        piece = self.pieces[piece_index]
        if (piece.lower is not None and value < piece.lower) or (piece.upper is not None and value > piece.upper):
            raise ValueError("parameter lies outside the piece's validity interval")
        return piece, value, arithmetic

    def lift(self, piece_index: int, parameter: Rational) -> Vector:
        """Exact minimizing feasible point of the transverse problem at y."""
        piece, value, arithmetic = self._query(piece_index, parameter)
        return tuple(arithmetic.add(offset, arithmetic.mul(slope, value))
                     for offset, slope in zip(piece.point_intercept, piece.point_slope, strict=True))

    def transverse_value(self, piece_index: int, parameter: Rational) -> Fraction:
        """Exact g(y), omitting the common scalar potential."""
        piece, value, arithmetic = self._query(piece_index, parameter)
        g0, g1, g2 = piece.transverse_cost
        return arithmetic.add(g0, arithmetic.mul(value, arithmetic.add(g1, arithmetic.mul(value, g2))))


def compile_gaussian_collinear_reduction(
    model: GaussianHeatMap, inequalities: Sequence[Sequence[Rational]], upper_bounds: Sequence[Rational],
    target: Sequence[Rational],
) -> GaussianCollinearReduction | None:
    r"""Reduce every feasible fiber and global minimum of a rank-one mixture.

    Supports any finite number of exactly collinear centers with at least two
    distinct positions, at every positive viscosity and time. Duplicate modes,
    redundant/zero inequalities and equalities encoded by opposite rows are
    supported. The first distinct center relative to the first center fixes
    the scalar coordinate; mode permutations may change that coordinate.
    Common biases cancel exactly before any numerical evaluation.

    Exact int/Fraction metadata is required; invalid types/shapes raise.
    Noncollinear/all-coincident centers or exhausted resources return None.
    A completed empty result proves infeasibility. Full combinatorial
    enumeration and all source metadata are preflighted before geometry;
    resource failure never returns a partial cover. No floating rank test or
    transcendental evaluation is performed. Shared strict sealing applies.
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
    if model.mode_count < 2:
        return None
    arithmetic = _Arithmetic(model.budget)
    try:
        count = _preflight(rows, dimension, arithmetic, model.mode_count)
        arithmetic.exact(count)
        matrix = tuple(tuple(arithmetic.exact(value) for value in row) for row in inequalities)
        upper = tuple(arithmetic.exact(value) for value in upper_bounds)
        goal = tuple(arithmetic.exact(value) for value in target)
        origin = model.centers[0]
        relative = tuple(tuple(arithmetic.add(value, -reference) for value, reference in zip(row, origin, strict=True))
                         for row in model.centers)
        direction = next((row for row in relative if any(row)), None)
        if direction is None:
            return None
        pivot = next(i for i, value in enumerate(direction) if value)
        coordinates = tuple(arithmetic._coefficient(row[pivot]/direction[pivot]) for row in relative)
        if any(value != arithmetic.mul(coordinate, axis)
               for row, coordinate in zip(relative, coordinates, strict=True)
               for value, axis in zip(row, direction, strict=True)):
            return None
        alpha = _dot(direction, direction, arithmetic)
        twice_nu = arithmetic.mul(Fraction(2), model.viscosity)
        axial_scale = arithmetic._coefficient(arithmetic.mul(twice_nu, model.time)/alpha)
        field = arithmetic._coefficient(arithmetic.mul(model.time, _dot(direction, goal, arithmetic))/alpha)
        denominator = arithmetic.mul(arithmetic.mul(Fraction(2), twice_nu), model.time)
        offsets = tuple(arithmetic.add(arithmetic.add(bias, -model.biases[0]),
                        -arithmetic._coefficient(arithmetic.mul(alpha, arithmetic.mul(coordinate, coordinate))/denominator))
                        for bias, coordinate in zip(model.biases, coordinates, strict=True))
        projector = tuple(tuple(arithmetic.add(Fraction(int(i == j)),
                               -arithmetic._coefficient(arithmetic.mul(direction[i], direction[j])/alpha))
                               for j in range(dimension)) for i in range(dimension))
        axial = tuple(arithmetic.mul(axial_scale, value) for value in direction)
        support = min(coordinates), max(coordinates)
        pieces = _compile_pieces(model, matrix, upper, goal, direction, origin, axial, projector,
                                 axial_scale, field, support, arithmetic, count, GaussianCollinearPiece)
        from omnibias.core.proof.certificate import make_certificate

        certificate = make_certificate(
            claim="Complete exact scalar reduction and finite global-minimizer cover for a collinear Gaussian mixture on a polyhedron.",
            payload={
                "type": "gaussian_collinear_reduction", "centers": [[_pair(value) for value in row] for row in model.centers],
                "biases": [_pair(value) for value in model.biases], "viscosity": _pair(model.viscosity), "time": _pair(model.time),
                "inequalities": [[_pair(value) for value in row] for row in matrix], "upper_bounds": [_pair(value) for value in upper],
                "target": [_pair(value) for value in goal], "origin": [_pair(value) for value in origin],
                "direction": [_pair(value) for value in direction], "projector": [[_pair(value) for value in row] for row in projector],
                "axial_scale": _pair(axial_scale), "field": _pair(field),
                "center_coordinates": [_pair(value) for value in coordinates], "logit_offsets": [_pair(value) for value in offsets],
                "mean_bounds": [_pair(value) for value in support], "pieces": [_piece_payload(piece) for piece in pieces],
                "enumerated_bases": count, "feasible": bool(pieces),
                "argument": "Exact replay proves every original center is c+t_j*a with a nonzero. Put alpha=|a|², lambda=2*nu*s/alpha and x=c+lambda*a*y+r, a.r=0. Expanding the Gaussian exponents and completing the transverse square gives the original target-tilted potential, up to one common constant, as ||r-s*P*target||²/(2*s)+nu*lambda*y²-2*nu*logsumexp(t_j*y+beta_j)-2*nu*w*y. Here beta_j=(b_j-b_0)-alpha*t_j²/(4*nu*s) and w=s*a.target/alpha. At each feasible y the strictly convex transverse projection has exactly one minimum. Its polyhedral normal cone represents the correction by nonnegative active projected rows; conic Caratheodory reduces this to an independent subset of at most dimension-1 rows. Every subset, including empty, was enumerated with exact rank and Gram solves. Multiplier nonnegativity and all original slacks give each full closed validity interval, covering all feasible fibers including degeneracies. Overlaps have the same point and cost. On each piece the scalar derivative is 2*nu*(k*y-E[t]-h) with k=lambda+g2/nu>0 and h=w-g1/(2*nu). Positive exponential weights give min(t)<=E[t]<=max(t), so the derivative is strictly negative below (h+min(t))/k and strictly positive above (h+max(t))/k. Clamping both strip endpoints to the full piece interval retains every minimum, including endpoint minima when the strip is disjoint. The original finite quadratic-minus-logsumexp potential is coercive; every nonempty closed polyhedron therefore has a global minimum, and all lie in the finite stored search cover.",
                "scope": "Any finite positive Gaussian mixture whose centers have exact affine rank one, with fixed positive viscosity/time and exact finite polyhedral constraints and target. All-coincident and higher-rank centers are unsupported. No selected optimum, uniqueness, derivative, generic fast optimizer, or external formal theorem-prover verification claim. Enumeration can be combinatorial; reduced objective values omit one common constant.",
            }, honesty={"unproven_claim": False, "continuum_pde_claim": False}, meta={"transcend_backend": "not_used"},
        )
        result = object.__new__(GaussianCollinearReduction)
        for name, value in (("model", model), ("inequalities", matrix), ("upper_bounds", upper), ("target", goal),
                            ("origin", origin), ("direction", direction), ("projector", projector),
                            ("axial_scale", axial_scale), ("field", field), ("center_coordinates", coordinates),
                            ("logit_offsets", offsets), ("mean_bounds", support), ("pieces", pieces),
                            ("enumerated_bases", count), ("certificate", certificate)):
            object.__setattr__(result, name, value)
        return result
    except (ArithmeticError, ValueError):
        return None


__all__ = ["GaussianCollinearPiece", "GaussianCollinearReduction", "compile_gaussian_collinear_reduction"]
