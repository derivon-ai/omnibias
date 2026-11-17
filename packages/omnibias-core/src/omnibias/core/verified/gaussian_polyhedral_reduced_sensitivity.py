# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Stable target sensitivities of certified unique polyhedral Gaussian minima."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from fractions import Fraction
from typing import TYPE_CHECKING

from omnibias.core._rational_linear import factor_positive_definite
from omnibias.core.gaussian_jet import _Arithmetic
from omnibias.core.verified.gaussian_affine import _dot, _validate_vector
from omnibias.core.verified.gaussian_heat import _finite, _rational
from omnibias.core.verified.gaussian_polyhedral_unique import (
    GaussianPolyhedralUnique,
    _minimum_intact,
    _minimum_preflight,
)
from omnibias.core.verified.interval import Interval
from omnibias.core.verified.inverse_gaussian_affine import _intact, _linear
from omnibias.core.verified.inverse_gaussian_heat import _pair
from omnibias.core.verified.mean_map import _strict
from omnibias.core.verified.transcend import exp_iv, libm_fallback_used

if TYPE_CHECKING:
    from collections.abc import Sequence

    from omnibias.core.proof.certificate import Cert

Rational = int | Fraction
Vector = tuple[Fraction, ...]
Matrix = tuple[Vector, ...]
Box = tuple[Interval, ...]
IntervalMatrix = tuple[Box, ...]
_ZERO = Interval.point(0.0)
_ONE = Interval.point(1.0)


def _unique_intact(unique: GaussianPolyhedralUnique) -> bool:
    if not _minimum_intact(unique.minimum) or not _intact(unique.certificate):
        return False
    payload = unique.certificate.get("payload")
    return isinstance(payload, dict) and payload.get("minimum_certificate") == unique.minimum.certificate


def _preflight(unique: GaussianPolyhedralUnique, arithmetic: _Arithmetic) -> None:
    source_entries = _minimum_preflight(unique.minimum, arithmetic)
    reduction = unique.minimum.reduction
    dimension, rows = reduction.model.dimension, len(reduction.inequalities)
    # The uniqueness wrapper retains another minimum-certificate snapshot;
    # reserve its scan and our new nested copy alongside all exact matrices.
    arithmetic._entries(2*source_entries+256*(dimension+1)**2+128*(rows+1)*(dimension+1)+4096)
    arithmetic._work(source_entries+512*(dimension+1)**3+256*(rows+1)*(dimension+1)**2+4096)


def _sech_squared(value: Interval) -> Interval:
    """Stable even sech² using only nonpositive exponential arguments."""
    low = 0.0 if value.lo <= 0.0 <= value.hi else min(abs(value.lo), abs(value.hi))
    high = max(abs(value.lo), abs(value.hi))

    def at(absolute: float) -> Interval:
        if absolute == 0.0:
            return _ONE
        exponent = _finite(Interval.point(-2.0)*Interval.point(absolute))
        exponential = _finite(exp_iv(exponent))
        exponential = Interval(max(0.0, exponential.lo), min(1.0, exponential.hi))
        denominator = _finite(_ONE+exponential)
        result = _finite(Interval.point(4.0)*exponential/(denominator*denominator))
        return Interval(max(0.0, result.lo), min(1.0, result.hi))

    lower, upper = at(high), at(low)
    return Interval(lower.lo, upper.hi)


def _affine_enclosure(offset: Fraction, slope: Fraction, hull: tuple[Fraction, Fraction],
                      arithmetic: _Arithmetic) -> Interval:
    values = tuple(arithmetic.add(offset, arithmetic.mul(slope, endpoint)) for endpoint in hull)
    return Interval(_rational(min(values)).lo, _rational(max(values)).hi)


def _rank_one_entry(base: Fraction, numerator: Fraction, denominator: Interval) -> Interval:
    if not numerator:
        return _rational(base)
    correction = _finite(_rational(numerator)/denominator)
    return correction if not base else _finite(_rational(base)+correction)


@dataclass(frozen=True, init=False)
class GaussianPolyhedralReducedSensitivity:
    r"""Target Jacobian of the original unique minimum on a stable piece.

    Exact compiled matrices are ``base_jacobian=s*R`` and
    ``rank_one_numerator=t*t^T/(2*nu)``. ``target_jacobian`` encloses
    base_jacobian+rank_one_numerator/(k-sech²(y*)) at the certified minimizer.
    All original inactive slacks and selected multipliers have strict positive
    interval lower bounds. ``multipliers`` and ``slacks`` follow the original
    inequality row order; inactive multipliers and selected slacks are zero.

    Derivatives hold with the model and constraints fixed. The certificate
    proves local active-piece stability but gives no numerical perturbation
    radius, parameter gradients, or derivative of a finite numerical solver.
    Certificate digests establish integrity, not an external formal proof.
    """

    unique: GaussianPolyhedralUnique
    piece_index: int
    active_indices: tuple[int, ...]
    parameter_hull: tuple[Fraction, Fraction]
    projected_constraints: Matrix
    gram_inverse: Matrix
    tangent_projector: Matrix
    point_slope: Vector
    scalar_curvature: Fraction
    base_jacobian: Matrix
    rank_one_numerator: Matrix
    denominator: Interval
    target_jacobian: IntervalMatrix
    multipliers: Box
    slacks: Box
    certificate: Cert

    def __init__(self) -> None:
        raise TypeError("use certify_gaussian_polyhedral_reduced_sensitivity")

    def vjp(self, cotangent: Sequence[Rational]) -> Box:
        """Enclose the exact-cotangent target VJP, preserving normal nulls.

        Shape/type errors and damaged proofs or exhausted resources raise.
        Contractions of the compiled rational factors precede conversion;
        this query requires no new transcendental evaluation or linear solve.
        Its contraction uses O(d*r+r²+d) exact scalar operations for r active
        rows, in addition to scanning the stored proof dependencies.
        """
        dimension = self.unique.minimum.reduction.model.dimension
        _validate_vector(cotangent, dimension, "cotangent")
        arithmetic = _Arithmetic(self.unique.minimum.reduction.model.budget)
        _preflight(self.unique, arithmetic)
        payload = self.certificate.get("payload")
        if (not _unique_intact(self.unique) or not _intact(self.certificate) or not isinstance(payload, dict)
                or payload.get("unique_certificate") != self.unique.certificate):
            raise ValueError("reduced sensitivity certificate or nested dependency is damaged")
        vector = tuple(arithmetic.exact(value) for value in cotangent)
        reduction = self.unique.minimum.reduction
        direction = reduction.half_difference
        alpha = _dot(direction, direction, arithmetic)
        normal = arithmetic._coefficient(_dot(direction, vector, arithmetic)/alpha)
        projected = tuple(arithmetic.add(value, -arithmetic.mul(axis, normal))
                          for value, axis in zip(vector, direction, strict=True))
        coordinates = _linear(self.projected_constraints, vector, arithmetic)
        transformed = _linear(self.gram_inverse, coordinates, arithmetic)
        base = tuple(arithmetic.mul(reduction.model.time, arithmetic.add(value,
                     -arithmetic.sum(tuple(arithmetic.mul(coordinate, row[j]) for coordinate, row in
                                           zip(transformed, self.projected_constraints, strict=True)))))
                     for j, value in enumerate(projected))
        contraction = arithmetic._coefficient(_dot(self.point_slope, vector, arithmetic)
                                             /arithmetic.mul(Fraction(2), reduction.model.viscosity))
        numerator = tuple(arithmetic.mul(value, contraction) for value in self.point_slope)
        return tuple(_rank_one_entry(a, b, self.denominator) for a, b in zip(base, numerator, strict=True))


def certify_gaussian_polyhedral_reduced_sensitivity(
    unique: GaussianPolyhedralUnique, piece_index: int,
) -> GaussianPolyhedralReducedSensitivity | None:
    """Certify target derivatives only after global uniqueness and strict KKT.

    The entire uniqueness hull must lie strictly inside the selected piece's
    validity interval. All selected multiplier and inactive slack lower bounds
    must be positive, and k-sech²(y*) must have a positive enclosure. A failed
    condition is inconclusive, including at criticality or active boundaries.
    Redundant active rows are refused. This does not infer differentiability
    from uniqueness alone. No numerical root search is performed.

    Invalid input types/indices raise. Damaged proofs, resource exhaustion,
    nonfinite arithmetic or failed sufficient margins return None. A missing
    rigorous backend or prior conditional-libm history raises RuntimeError;
    strict evaluation preserves that history and the caller's strict flag.
    """
    if not isinstance(unique, GaussianPolyhedralUnique):
        raise TypeError("unique must be a GaussianPolyhedralUnique")
    if type(piece_index) is not int:
        raise TypeError("piece_index must be an integer excluding bool")
    reduction = unique.minimum.reduction
    if not 0 <= piece_index < len(reduction.pieces):
        raise IndexError("piece_index outside the reduction")
    model, piece = reduction.model, reduction.pieces[piece_index]
    arithmetic = _Arithmetic(model.budget)
    try:
        _preflight(unique, arithmetic)
        if not _unique_intact(unique):
            return None
        hull = unique.parameter_hull
        if ((piece.lower is not None and hull[0] <= piece.lower)
                or (piece.upper is not None and hull[1] >= piece.upper)):
            return None
        dimension, active = model.dimension, piece.active_indices
        matrix = tuple(reduction.inequalities[index] for index in active)
        projected = tuple(_linear(reduction.projector, row, arithmetic) for row in matrix)
        if projected:
            gram = tuple(tuple(_dot(left, right, arithmetic) for right in projected) for left in projected)
            factor = factor_positive_definite(gram, arithmetic)
            columns = tuple(factor.solve(tuple(Fraction(int(i == j)) for i in range(len(active))), arithmetic)
                            for j in range(len(active)))
            inverse = tuple(tuple(column[i] for column in columns) for i in range(len(active)))
        else:
            inverse = ()
        # T=M*B; R=P-B^T*T. All normal cancellations happen exactly.
        product = tuple(tuple(arithmetic.sum(tuple(arithmetic.mul(m, row[j]) for m, row in zip(weights, projected, strict=True)))
                              for j in range(dimension)) for weights in inverse)
        projector = tuple(tuple(arithmetic.add(reduction.projector[i][j],
                           -arithmetic.sum(tuple(arithmetic.mul(row[i], transformed[j]) for row, transformed in zip(projected, product, strict=True))))
                                for j in range(dimension)) for i in range(dimension))
        axial = tuple(arithmetic.mul(reduction.inverse_q, value) for value in reduction.half_difference)
        selected_axial = _linear(matrix, axial, arithmetic)
        eta_slope = _linear(inverse, selected_axial, arithmetic)
        slope = tuple(arithmetic.add(axial[j], -arithmetic.sum(tuple(arithmetic.mul(eta, row[j])
                      for eta, row in zip(eta_slope, projected, strict=True)))) for j in range(dimension))
        if slope != piece.point_slope:
            return None
        twice_nu = arithmetic.mul(Fraction(2), model.viscosity)
        curvature = arithmetic._coefficient(_dot(slope, slope, arithmetic)/arithmetic.mul(twice_nu, model.time))
        if curvature != piece.scalar_curvature:
            return None
        anchor = tuple(arithmetic.add(center, -arithmetic.mul(value, arithmetic._coefficient(reduction.bias_contrast/2)))
                       for center, value in zip(reduction.midpoint, axial, strict=True))
        preferred = tuple(arithmetic.mul(model.time, value) for value in _linear(reduction.projector, reduction.target, arithmetic))
        offsets = tuple(arithmetic.add(arithmetic.add(_dot(row, preferred, arithmetic), _dot(original, anchor, arithmetic)),
                                       -reduction.upper_bounds[index])
                        for row, original, index in zip(projected, matrix, active, strict=True))
        eta_offset = _linear(inverse, offsets, arithmetic)
        multipliers = [_ZERO for _ in reduction.inequalities]
        slacks = []
        active_set = set(active)
        for index, offset, change in zip(active, eta_offset, eta_slope, strict=True):
            enclosure = _affine_enclosure(arithmetic._coefficient(offset/model.time),
                                         arithmetic._coefficient(change/model.time), hull, arithmetic)
            if enclosure.lo <= 0:
                return None
            multipliers[index] = enclosure
        for index, (row, bound) in enumerate(zip(reduction.inequalities, reduction.upper_bounds, strict=True)):
            offset = arithmetic.add(bound, -_dot(row, piece.point_intercept, arithmetic))
            change = -_dot(row, slope, arithmetic)
            if index in active_set:
                if offset or change:
                    return None
                slacks.append(_ZERO)
            else:
                enclosure = _affine_enclosure(offset, change, hull, arithmetic)
                if enclosure.lo <= 0:
                    return None
                slacks.append(enclosure)
        if libm_fallback_used():
            raise RuntimeError("reduced Gaussian sensitivity refuses prior conditional libm fallback history")
        with _strict():
            y = Interval(_rational(hull[0]).lo, _rational(hull[1]).hi)
            denominator = _finite(_rational(curvature)-_sech_squared(y))
            if denominator.lo <= 0:
                return None
            base = tuple(tuple(arithmetic.mul(model.time, value) for value in row) for row in projector)
            numerator = tuple(tuple(arithmetic._coefficient(arithmetic.mul(left, right)/twice_nu) for right in slope) for left in slope)
            jacobian = tuple(tuple(_rank_one_entry(a, b, denominator) for a, b in zip(baserow, numrow, strict=True))
                             for baserow, numrow in zip(base, numerator, strict=True))
            from omnibias.core.proof.certificate import encode_interval, make_certificate

            certificate = make_certificate(
                claim="Target Jacobian of the unique global polyhedral Gaussian minimum on a strictly stable active piece.",
                payload={
                    "type": "gaussian_polyhedral_reduced_sensitivity", "unique_certificate": deepcopy(unique.certificate),
                    "piece_index": piece_index, "active_indices": list(active), "parameter_hull": [_pair(value) for value in hull],
                    "projected_constraints": [[_pair(value) for value in row] for row in projected],
                    "gram_inverse": [[_pair(value) for value in row] for row in inverse],
                    "tangent_projector": [[_pair(value) for value in row] for row in projector],
                    "point_slope": [_pair(value) for value in slope], "scalar_curvature": _pair(curvature),
                    "base_jacobian": [[_pair(value) for value in row] for row in base],
                    "rank_one_numerator": [[_pair(value) for value in row] for row in numerator],
                    "denominator": encode_interval(denominator), "target_jacobian": [[encode_interval(value) for value in row] for row in jacobian],
                    "multipliers": [encode_interval(value) for value in multipliers], "slacks": [encode_interval(value) for value in slacks],
                    "argument": "Global uniqueness is certified independently. The unique parameter hull lies strictly inside this exact transverse KKT piece. Its selected multipliers and every inactive original slack have positive lower bounds, so the same active constraints persist locally. For A the selected original rows, B=A*P and M=(B*B^T)^(-1), set R=P-B^T*M*B and t=lambda*a-B^T*M*A*(lambda*a). Then x(y,v)=x0(v)+t*y with derivative dx0/dv=s*R; k=||t||²/(2*nu*s), and the scalar stationarity equation k*y-tanh(y)-h(v)=0 has dh/dv=t/(2*nu). Positive k-sech²(y*) permits the implicit function theorem, giving dx/dv=s*R+t*t^T/[2*nu*(k-sech²(y*))]. The interval denominator encloses the entire certified parameter hull. Exact rational geometry precedes numerical conversion, preserving all constraint-normal annihilations.",
                    "scope": "Target derivatives only, with original model, viscosity, age and constraints fixed. Strict stability is local; no quantitative perturbation radius or parameter derivatives are certified. This is the derivative of the exact global optimizer, not of a finite solver. No external formal proof-checker claim.",
                }, honesty={"unproven_claim": False, "unconditional_transcendentals": True, "continuum_pde_claim": False},
                meta={"transcend_backend": "mpmath"},
            )
        result = object.__new__(GaussianPolyhedralReducedSensitivity)
        for name, value in (("unique", unique), ("piece_index", piece_index), ("active_indices", active), ("parameter_hull", hull),
                            ("projected_constraints", projected), ("gram_inverse", inverse), ("tangent_projector", projector),
                            ("point_slope", slope), ("scalar_curvature", curvature), ("base_jacobian", base),
                            ("rank_one_numerator", numerator), ("denominator", denominator), ("target_jacobian", jacobian),
                            ("multipliers", tuple(multipliers)), ("slacks", tuple(slacks)), ("certificate", certificate)):
            object.__setattr__(result, name, value)
        return result
    except (ArithmeticError, ValueError):
        return None


__all__ = ["GaussianPolyhedralReducedSensitivity", "certify_gaussian_polyhedral_reduced_sensitivity"]
