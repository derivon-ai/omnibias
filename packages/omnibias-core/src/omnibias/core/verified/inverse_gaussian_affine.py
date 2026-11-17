# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Certified Gaussian optimization under exact affine equality constraints."""

from __future__ import annotations

from collections.abc import Sequence
from copy import deepcopy
from dataclasses import dataclass
from fractions import Fraction
from typing import TYPE_CHECKING

from omnibias.core.gaussian_jet import _Arithmetic
from omnibias.core.verified.gaussian_affine import GaussianAffineRestriction
from omnibias.core.verified.gaussian_heat import _float, _rational
from omnibias.core.verified.interval import Interval
from omnibias.core.verified.inverse_gaussian_heat import (
    GaussianInverseCertificate,
    GaussianInverseStatus,
    _inputs,
    _pair,
    certify_gaussian_inverse,
    inverse_gaussian_heat,
)

if TYPE_CHECKING:
    from omnibias.core.proof.certificate import Cert

Rational = int | Fraction
Vector = tuple[Fraction, ...]


def _intact(certificate: Cert) -> bool:
    from omnibias.core.proof.certificate import verify_certificate_digest

    try:
        return verify_certificate_digest(certificate)
    except (TypeError, ValueError, RecursionError):
        return False


def _linear(matrix: tuple[Vector, ...], vector: Vector, arithmetic: _Arithmetic) -> Vector:
    return tuple(arithmetic.sum(tuple(arithmetic.mul(a, b) for a, b in zip(row, vector, strict=True))) for row in matrix)


def _remaining_work(restriction: GaussianAffineRestriction) -> int:
    """Reserve exact projection, certificate copies and checked gap arithmetic."""
    model = restriction.model
    overhead = 1024*(model.mode_count+len(restriction.constraints)+1)*(model.dimension+1)**2 + 1024
    model.budget._work(overhead)
    return model.budget.max_elimination_work-overhead


@dataclass(frozen=True)
class GaussianAffineInverseCertificate:
    """Unique constrained minimum, an exactly feasible point and a gap bound.

    ``enclosure`` contains the unique optimizer; its other points need not be
    feasible. ``feasible_point`` is an exact rational point satisfying the
    original equations. Its Euclidean distance to the optimizer is at most
    ``error_bound`` and its objective gap is at most ``objective_gap_bound``.
    The objective is -2*nu*log(tau_source)-target.x at the fixed source age.
    ``root_certificate`` is absent only for an exactly affine restricted map.
    """

    restriction: GaussianAffineRestriction
    target: Vector
    projected_target: Vector
    root_certificate: GaussianInverseCertificate | None
    feasible_point: Vector
    enclosure: tuple[Interval, ...]
    error_bound: float
    objective_gap_bound: Fraction
    certificate: Cert


@dataclass(frozen=True)
class GaussianAffineInverseResult:
    """Bounded inverse proposal followed by a constrained-minimum certificate.

    A certified result may have ``tolerance_met=False``. An inconclusive
    contraction test does not prove nonuniqueness. Exact infeasibility is
    handled earlier by ``restrict_gaussian_heat``.
    """

    status: GaussianInverseStatus
    iterations: int
    tolerance_met: bool
    certificate: GaussianAffineInverseCertificate | None


def _wrap(
    restriction: GaussianAffineRestriction, target: Vector, projected: Vector,
    root: GaussianInverseCertificate | None,
) -> GaussianAffineInverseCertificate | None:
    model = restriction.model
    if not _intact(restriction.certificate) or (root is not None and not _intact(root.certificate)):
        return None
    try:
        arithmetic = _Arithmetic(model.budget)
        arithmetic._entries(32*(model.mode_count+len(restriction.constraints)+1)*(model.dimension+1)
                            + 16*(model.dimension+1)**2 + 128)
        arithmetic._work(128*(model.mode_count+len(restriction.constraints)+1)*(model.dimension+1)**2 + 256)
        if model.contraction_bound == 0:
            point = tuple(arithmetic.add(arithmetic.mul(model.time, value), center)
                          for value, center in zip(projected, model.centers[0], strict=True))
            enclosure = tuple(_rational(value) for value in point)
            error = 0.0
        else:
            if root is None:
                return None
            physical = tuple(arithmetic.mul(model.time, Fraction(value)) for value in root.scaled_point)
            point = tuple(arithmetic.add(anchor, value) for anchor, value in
                          zip(restriction.anchor, _linear(restriction.projector, physical, arithmetic), strict=True))
            enclosure, error = root.enclosure, root.error_bound
        # Projection onto a closed affine subspace cannot increase distance
        # to the true feasible optimizer. Smoothness on that subspace is 1/s.
        gap = arithmetic._coefficient(arithmetic.mul(Fraction(error), Fraction(error)) /
                                      arithmetic.mul(Fraction(2), model.time))
        from omnibias.core.proof.certificate import encode_interval, make_certificate

        certificate = make_certificate(
            claim="Unique global Gaussian minimum under exact affine equalities, with a feasible point and objective-gap bound.",
            payload={
                "type": "gaussian_affine_inverse", "restriction_certificate": deepcopy(restriction.certificate),
                "target": [_pair(value) for value in target], "projected_target": [_pair(value) for value in projected],
                "root_certificate": None if root is None else deepcopy(root.certificate),
                "feasible_point": [_pair(value) for value in point], "enclosure": [encode_interval(value) for value in enclosure],
                "error_bound": error, "objective_gap_bound": _pair(gap),
                "argument": "On the affine set the projected Gaussian potential differs from the original by a spatial constant. With q_T<1 its tangent Hessian is between (1-q_T)/s and 1/s. The projected-family inverse at P*target automatically lies in the affine set and is its unique global constrained minimizer. Orthogonally projecting the exact physical proposal onto the affine set preserves its root-distance bound. Restricted smoothness bounds its objective gap by error_bound^2/(2*s). An identical-center projected family is solved exactly with no transcendental evaluation.",
                "scope": "The unique root and exact rational feasible_point satisfy the equations; not every point of the enclosing coordinate box does. Fixed original model and heat age; no inequality constraints or general nonconvex optimization claim.",
            },
            honesty={"unproven_claim": False, "unconditional_transcendentals": root is not None, "continuum_pde_claim": False},
            meta={"transcend_backend": "not_used" if root is None else root.certificate["meta"]["transcend_backend"]},
        )
        return GaussianAffineInverseCertificate(restriction, target, projected, root, point, enclosure, error, gap, certificate)
    except (ArithmeticError, ValueError):
        return None


def certify_gaussian_affine_inverse(
    restriction: GaussianAffineRestriction, target: Sequence[Rational], scaled_point: Sequence[float],
) -> GaussianAffineInverseCertificate | None:
    """Certify an arbitrary y=x/s proposal against the exact affine problem.

    The proposal need not satisfy the equations. The returned rational point
    is its orthogonal projection, with the certified error unchanged. An
    exactly affine restricted map instead returns its exact optimizer.
    Target data are exact integers/Fractions; scaled-point entries are finite
    binary64 values. Invalid inputs raise; failed bounds/resources return None.
    """
    if not isinstance(restriction, GaussianAffineRestriction):
        raise TypeError("restriction must be a GaussianAffineRestriction")
    frozen, point = _inputs(restriction.model, target, scaled_point)
    try:
        remaining = _remaining_work(restriction)
        projected = restriction.project_target(frozen)
        if restriction.model.contraction_bound >= 1:
            return None
        if restriction.model.contraction_bound == 0:
            return _wrap(restriction, frozen, projected, None)
        if remaining < restriction.model._evaluation_work() + 64*restriction.dimension + 64:
            return None
        root = certify_gaussian_inverse(restriction.model, projected, point)
        return None if root is None else _wrap(restriction, frozen, projected, root)
    except (ArithmeticError, ValueError):
        return None


def inverse_gaussian_affine(
    restriction: GaussianAffineRestriction, target: Sequence[Rational], *, initial: Sequence[float] | None = None,
    max_iterations: int = 128, absolute_tolerance: float = 1e-10,
) -> GaussianAffineInverseResult:
    """Find and certify the unique affine-constrained Gaussian optimizer.

    Require only the projected contraction bound q_T<1; the ambient source
    potential can be nonconvex. Initial coordinates are scaled y=x/s, as for
    ``inverse_gaussian_heat``. The exact affine case requires no iterations or
    transcendental backend. Missing rigorous support otherwise raises.
    """
    if not isinstance(restriction, GaussianAffineRestriction):
        raise TypeError("restriction must be a GaussianAffineRestriction")
    if type(max_iterations) is not int:
        raise TypeError("max_iterations must be an integer, excluding bool")
    if max_iterations < 0:
        raise ValueError("max_iterations must be nonnegative")
    tolerance = _float(absolute_tolerance)
    if tolerance <= 0:
        raise ValueError("absolute_tolerance must be positive")
    model = restriction.model
    frozen, point = _inputs(model, target, (0.0,)*model.dimension if initial is None else initial)
    if model.contraction_bound >= 1:
        return GaussianAffineInverseResult("outside_contraction_regime", 0, False, None)
    try:
        remaining = _remaining_work(restriction)
        projected = restriction.project_target(frozen)
        if model.contraction_bound == 0:
            certificate = _wrap(restriction, frozen, projected, None)
            return GaussianAffineInverseResult("certified" if certificate is not None else "not_certified", 0,
                                               certificate is not None, certificate)
        evaluation_work = model._evaluation_work() + 64*model.dimension + 64
        iterations = min(max_iterations, remaining//(2*evaluation_work)-1)
        if iterations < 0:
            return GaussianAffineInverseResult("not_certified", 0, False, None)
        inverse = inverse_gaussian_heat(model, projected, initial=point, max_iterations=iterations,
                                        absolute_tolerance=tolerance)
        certificate = None if inverse.certificate is None else _wrap(restriction, frozen, projected, inverse.certificate)
        return GaussianAffineInverseResult("certified" if certificate is not None else "not_certified", inverse.iterations,
                                           certificate is not None and inverse.tolerance_met, certificate)
    except (ArithmeticError, ValueError):
        return GaussianAffineInverseResult("not_certified", 0, False, None)


__all__ = ["GaussianAffineInverseCertificate", "GaussianAffineInverseResult",
           "certify_gaussian_affine_inverse", "inverse_gaussian_affine"]
