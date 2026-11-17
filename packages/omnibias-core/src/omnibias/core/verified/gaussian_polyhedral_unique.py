# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Global uniqueness from a complete polyhedral Gaussian minimizer cover."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from fractions import Fraction
from typing import TYPE_CHECKING, Literal

from omnibias.core.gaussian_jet import _Arithmetic
from omnibias.core.verified.gaussian_polyhedral_reduced_minimum import (
    GaussianPolyhedralReducedMinimum,
)
from omnibias.core.verified.inverse_gaussian_affine import _intact
from omnibias.core.verified.inverse_gaussian_heat import _pair
from omnibias.core.verified.transcend import libm_fallback_used, strict_backend

if TYPE_CHECKING:
    from omnibias.core.proof.certificate import Cert

Bounds = tuple[Fraction, Fraction]
Method = Literal["singleton", "strict_convexity", "strong_convexity"]


def _minimum_preflight(minimum: GaussianPolyhedralReducedMinimum, arithmetic: _Arithmetic) -> int:
    """Reserve the full proof dependency, including pruned-piece diagnostics."""
    reduction = minimum.reduction
    dimension = reduction.model.dimension
    pieces = len(reduction.pieces)
    candidates = len(minimum.candidate_parameters)+sum(len(result.candidates) for result in minimum.piece_results)
    slots = (64*(len(reduction.inequalities)+pieces+reduction.model.mode_count+1)*(dimension+1)
             + 32*(dimension+1)**2 + 64*(candidates+pieces+1)*(dimension+1)+4096)
    entries = 2*slots+64*(pieces+1)
    arithmetic._entries(entries)
    arithmetic._work(4*slots+64*(pieces+1))
    return entries


def _minimum_intact(minimum: GaussianPolyhedralReducedMinimum) -> bool:
    if not _intact(minimum.certificate) or not _intact(minimum.reduction.certificate):
        return False
    payload = minimum.certificate.get("payload")
    return isinstance(payload, dict) and payload.get("reduction_certificate") == minimum.reduction.certificate


@dataclass(frozen=True, init=False)
class GaussianPolyhedralUnique:
    r"""A UNIQUE GLOBAL minimizer, with no differentiability assertion.

    ``parameter_hull`` contains every scalar parameter from the complete
    global minimizer cover. All compiled pieces intersecting this ENTIRE hull
    were considered, including pieces discarded by objective bounds. Point
    intersections need no curvature test when the hull has positive length.

    ``curvature_lower_bound`` is normalized: multiply by 2*nu to obtain a
    lower bound for the reduced objective's scalar curvature on its pieces.
    It is None for a singleton proof, which asserts no curvature bound, and
    can be zero for a strict-convexity proof. At k=1, curvature
    vanishes only at y=0, which still permits strict convexity and uniqueness.
    A singleton hull also proves uniqueness because its transverse optimizer
    is unique. Neither case implies a finite target derivative.
    """

    minimum: GaussianPolyhedralReducedMinimum
    parameter_hull: Bounds
    curvature_lower_bound: Fraction | None
    method: Method
    checked_piece_indices: tuple[int, ...]
    piece_parameter_intervals: tuple[Bounds, ...]
    piece_curvature_lower_bounds: tuple[Fraction, ...]
    certificate: Cert

    def __init__(self) -> None:
        raise TypeError("use certify_gaussian_polyhedral_unique")


def certify_gaussian_polyhedral_unique(minimum: GaussianPolyhedralReducedMinimum) -> GaussianPolyhedralUnique | None:
    r"""Prove global uniqueness on the convex hull of ALL remaining candidates.

    For every positive-length intersection of a compiled piece with the hull,
    prove strict convexity of g(y)+2*nu*(lambda*y²/2-log(cosh(y))-w*y).
    If k>=1, k-sech²(y)>=k-1 and is positive except possibly at y=0.
    Otherwise z=min|y| on the intersection gives
    sech²(y)<=1/(1+z²/2)² by cosh(z)>=1+z²/2; a positive rational gap is
    required. Convexity of the transverse value g ensures upward derivative
    jumps at breakpoints, so these pieces join into a strictly convex function.
    This proof does not inspect only the surviving candidate boxes: gaps and
    pruned pieces inside their hull are essential.

    Invalid types raise. Failed sufficient inequalities, damaged/broken proof
    dependencies and exhausted resources return None, never a nonuniqueness
    claim. Uses exact arithmetic only and preserves shared strict sealing
    policy. It does not compute a perturbation radius or target Jacobian.
    """
    if not isinstance(minimum, GaussianPolyhedralReducedMinimum):
        raise TypeError("minimum must be a GaussianPolyhedralReducedMinimum")
    reduction = minimum.reduction
    arithmetic = _Arithmetic(reduction.model.budget)
    try:
        _minimum_preflight(minimum, arithmetic)
        if not _minimum_intact(minimum):
            return None
        if not minimum.candidate_parameters or len(minimum.candidate_piece_indices) != len(minimum.candidate_parameters):
            return None
        intervals: list[Bounds] = []
        for index, parameter in zip(minimum.candidate_piece_indices, minimum.candidate_parameters, strict=True):
            if not 0 <= index < len(reduction.pieces):
                return None
            if not 0 <= parameter.lo <= parameter.hi <= 1:
                return None
            piece = reduction.pieces[index]
            start, end = piece.search_interval
            width = arithmetic.add(end, -start)
            lower = arithmetic.add(start, arithmetic.mul(width, arithmetic.exact(Fraction(parameter.lo))))
            upper = arithmetic.add(start, arithmetic.mul(width, arithmetic.exact(Fraction(parameter.hi))))
            intervals.append((lower, upper))
        hull = min(pair[0] for pair in intervals), max(pair[1] for pair in intervals)
        checked: list[int] = []
        domains: list[Bounds] = []
        curvatures: list[Fraction] = []
        method: Method = "singleton"
        curvature: Fraction | None = None
        if hull[0] < hull[1]:
            for index, piece in enumerate(reduction.pieces):
                lower = hull[0] if piece.lower is None else max(hull[0], piece.lower)
                upper = hull[1] if piece.upper is None else min(hull[1], piece.upper)
                if lower >= upper:
                    continue
                if piece.scalar_curvature >= 1:
                    bound = arithmetic.add(piece.scalar_curvature, Fraction(-1))
                else:
                    distance = Fraction(0) if lower <= 0 <= upper else min(abs(lower), abs(upper))
                    if not distance:
                        return None
                    square = arithmetic.mul(distance, distance)
                    cosh_lower = arithmetic.add(Fraction(1), arithmetic._coefficient(square/2))
                    sech_upper = arithmetic._coefficient(Fraction(1)/arithmetic.mul(cosh_lower, cosh_lower))
                    bound = arithmetic.add(piece.scalar_curvature, -sech_upper)
                    if bound <= 0:
                        return None
                checked.append(index)
                domains.append((lower, upper))
                curvatures.append(bound)
            if not domains:
                return None
            # Replay coverage of the whole hull, not merely curvature on a
            # disconnected collection of retained optimizer boxes.
            arithmetic._work(16*(len(domains)+1)**2)
            cursor = hull[0]
            for lower, upper in sorted(domains):
                if lower > cursor:
                    return None
                cursor = max(cursor, upper)
            if cursor != hull[1]:
                return None
            curvature = min(curvatures)
            method = "strong_convexity" if curvature > 0 else "strict_convexity"
        from omnibias.core.proof.certificate import (
            certificate_is_unconditional,
            certificate_transcend_backend,
            make_certificate,
        )

        if not certificate_is_unconditional(minimum.certificate):
            return None
        provenance = certificate_transcend_backend(minimum.certificate)
        if strict_backend() and libm_fallback_used():
            raise RuntimeError("strict uniqueness sealing refuses prior conditional libm fallback history")

        certificate = make_certificate(
            claim="The original two-center Gaussian objective has exactly one global minimizer on its polyhedron.",
            payload={
                "type": "gaussian_polyhedral_unique", "minimum_certificate": deepcopy(minimum.certificate),
                "parameter_hull": [_pair(value) for value in hull],
                "curvature_lower_bound": None if curvature is None else _pair(curvature),
                "method": method, "checked_piece_indices": checked,
                "piece_parameter_intervals": [[_pair(value) for value in interval] for interval in domains],
                "piece_curvature_lower_bounds": [_pair(value) for value in curvatures],
                "argument": "The source certificate covers ALL global minimizers. Exact affine mapping of every surviving normalized parameter interval puts every minimizing scalar y in the recorded hull. The feasible scalar projection is an interval, so this hull is feasible. Each scalar parameter has exactly one transverse quadratic minimizer. A singleton hull therefore proves uniqueness immediately. Otherwise inspect every compiled piece on the entire hull, including pruned pieces and gaps between candidate boxes, and replay complete interval coverage. On each nontrivial intersection, the reduced scalar curvature is 2*nu*(k-sech²(y)). For k>=1 it is nonnegative and positive except possibly at the isolated point y=0; this gives strict convexity. For k<1, cosh(z)>=1+z²/2 with z the smallest |y| on the intersection gives the exact positive recorded lower bound. The transverse value function is convex because it is a partial minimum of a jointly convex quadratic on a convex polyhedral graph. Its derivative jumps upward at breakpoints. The piecewise strictly convex functions thus join into a strictly convex function on the whole hull. At most one global minimizer can lie there, and existence follows from the source complete global minimum certificate. This proves exactly one original constrained minimizer.",
                "scope": "Uniqueness at the fixed original model, constraints and target. The normalized curvature bound excludes the factor 2*nu and can vanish; it is null for a singleton proof, which asserts no curvature bound. Singleton and critical strict-convexity proofs do not imply differentiability. No numerical perturbation radius, arbitrary mixture, continuum inverse PDE or formal theorem-prover verification claim.",
            }, honesty={"unproven_claim": False, "continuum_pde_claim": False}, meta={"transcend_backend": provenance},
        )
        result = object.__new__(GaussianPolyhedralUnique)
        for name, value in (("minimum", minimum), ("parameter_hull", hull), ("curvature_lower_bound", curvature),
                            ("method", method), ("checked_piece_indices", tuple(checked)),
                            ("piece_parameter_intervals", tuple(domains)), ("piece_curvature_lower_bounds", tuple(curvatures)),
                            ("certificate", certificate)):
            object.__setattr__(result, name, value)
        return result
    except (ArithmeticError, ValueError):
        return None


__all__ = ["GaussianPolyhedralUnique", "certify_gaussian_polyhedral_unique"]
