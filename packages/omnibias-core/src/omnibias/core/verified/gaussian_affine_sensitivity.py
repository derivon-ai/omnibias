# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""True constrained target sensitivities from forward and adjoint proposals."""

from __future__ import annotations

from collections.abc import Sequence
from copy import deepcopy
from dataclasses import dataclass
from fractions import Fraction
from typing import TYPE_CHECKING

from omnibias.core.gaussian_jet import _Arithmetic
from omnibias.core.verified.gaussian_affine import GaussianAffineRestriction
from omnibias.core.verified.gaussian_heat import _float, _rational
from omnibias.core.verified.gaussian_sensitivity import (
    GaussianTargetVJPCertificate,
    certify_gaussian_target_vjp,
)
from omnibias.core.verified.interval import Interval
from omnibias.core.verified.inverse_gaussian_affine import _intact, _linear
from omnibias.core.verified.inverse_gaussian_heat import _pair, _sqrt_upper

if TYPE_CHECKING:
    from omnibias.core.proof.certificate import Cert

Vector = tuple[Fraction, ...]
BinaryVector = tuple[float, ...]
Box = tuple[Interval, ...]


def _binary_vector(values: Sequence[float], dimension: int, name: str) -> BinaryVector:
    if not isinstance(values, Sequence):
        raise TypeError(f"{name} must be a finite sequence")
    if len(values) != dimension:
        raise ValueError(f"{name} dimension must match the restriction")
    return tuple(_float(value) for value in values)


def _norm_upper(values: Vector, arithmetic: _Arithmetic) -> Fraction:
    squared = arithmetic.sum(tuple(arithmetic.mul(value, value) for value in values))
    return arithmetic._coefficient(Fraction(_sqrt_upper(squared, arithmetic.limits)))


@dataclass(frozen=True, init=False)
class GaussianAffineTargetVJPCertificate:
    r"""Enclose the true constrained inverse target VJP at a declared target.

    For S=D_target X_C and the fixed declared binary64 ``upstream`` g,
    ``enclosure`` contains S.T*g and ||value-S.T*g|| <= ``error_bound``.
    S is symmetric, so this is also a target JVP certificate. ``value`` is
    exact rational and tangent. Normally it is P*proposal; for an affine
    inverse or exactly normal upstream it is the exact sensitivity instead.
    ``projected_proposal`` always retains P*proposal, even in those shortcuts.

    The bound includes the root/adjoint residual and both projection-related
    rounding corrections. It certifies the exact inverse derivative, not a
    finite formula's autodiff value. The upstream is held fixed; errors in
    a separately computed loss gradient, or derivatives with respect to
    constraints, centers, biases, viscosity and age, are outside this claim.

    ``base_certificate`` is absent for exact shortcuts. Coordinate radii
    exploit tangency; the enclosing box need not contain only tangent vectors.
    """

    restriction: GaussianAffineRestriction
    target: Vector
    projected_target: Vector
    scaled_point: BinaryVector
    upstream: BinaryVector
    projected_upstream: Vector
    proposal: BinaryVector
    projected_proposal: Vector
    value: Vector
    enclosure: Box
    error_bound: Fraction
    base_error_bound: Fraction
    upstream_rounding_error_bound: Fraction
    proposal_rounding_error_bound: Fraction
    coordinate_error_bounds: Vector
    base_certificate: GaussianTargetVJPCertificate | None
    certificate: Cert

    def __init__(self) -> None:
        raise TypeError("use certify_gaussian_affine_target_vjp")


def certify_gaussian_affine_target_vjp(
    restriction: GaussianAffineRestriction, target: Sequence[int | Fraction], scaled_point: Sequence[float],
    upstream: Sequence[float], proposal: Sequence[float],
) -> GaussianAffineTargetVJPCertificate | None:
    r"""Certify a proposed target derivative of the true constrained optimum.

    Target data are exact integers/Fractions. The root proposal is y=x/s;
    scaled_point, upstream and proposal are finite binary64 vectors (exactly
    representable integers also accepted). The root proposal need not be
    feasible. Neither proposal is trusted as a solution or a derivative.
    Require q_T<1, including for exact-null and exactly affine shortcuts.

    Set gT=P*g exactly, ghat=round(gT), aT=P*proposal exactly, and
    z=round(aT+s*(I-P)*ghat). The existing matrix-free certificate gives
    ||z-J'*ghat||<=beta for the transformed inverse at P*target. With
    S=P*J'*P and ||S||<=s/(1-q_T), an error bound about aT is
    beta+s/(1-q_T)*||P*(ghat-gT)||+||P*(z-proposal)||.
    All projections and squared norms use bounded exact rationals. Norm
    bounds round outward. Coordinate errors are at most sqrt(P_ii) times
    the Euclidean bound because the error is tangent. If gT=0, the true
    sensitivity is exactly zero. If D_T=0, it is exactly s*gT. Both cases
    require no root, posterior or adjoint evaluation and ignore the proposals
    after validating their input contracts.

    After restriction setup, work/storage are O(m*d+d²+r*d), including the
    dense projector and retained certificate dependencies. No inverse
    iteration, covariance/Jacobian matrix construction or factorization is
    used. Invalid input types/domains raise; failed contraction, damaged
    dependencies, exhausted budgets and nonfinite arithmetic return None.
    A missing rigorous transcendental backend raises except for shortcuts.
    A digest is not a theorem-prover verification.
    """
    if not isinstance(restriction, GaussianAffineRestriction):
        raise TypeError("restriction must be a GaussianAffineRestriction")
    model = restriction.model
    dimension = model.dimension
    if not isinstance(target, Sequence):
        raise TypeError("target must be a finite sequence of exact integers or Fractions")
    if len(target) != dimension:
        raise ValueError("target dimension must match the restriction")
    if any(isinstance(value, bool) or not isinstance(value, (int, Fraction)) for value in target):
        raise TypeError("target entries must be exact integers or Fractions, excluding bool")
    point = _binary_vector(scaled_point, dimension, "scaled_point")
    direction = _binary_vector(upstream, dimension, "upstream")
    proposed = _binary_vector(proposal, dimension, "proposal")
    q = model.contraction_bound
    if q >= 1:
        return None
    try:
        arithmetic = _Arithmetic(model.budget)
        checked = arithmetic._coefficient
        count, rows = model.mode_count, len(restriction.constraints)
        arithmetic._entries(128*(count+rows+1)*(dimension+1) + 128*(dimension+1)**2 + 512)
        arithmetic._work(512*(count+rows+1)*(dimension+1) + 512*(dimension+1)**2 + 2048)
        if not _intact(restriction.certificate):
            return None
        frozen = tuple(arithmetic.exact(value) for value in target)
        projected_target = _linear(restriction.projector, frozen, arithmetic)
        exact_direction = tuple(checked(Fraction(value)) for value in direction)
        exact_proposed = tuple(checked(Fraction(value)) for value in proposed)
        tangent_direction = _linear(restriction.projector, exact_direction, arithmetic)
        tangent_proposal = _linear(restriction.projector, exact_proposed, arithmetic)
        base: GaussianTargetVJPCertificate | None = None
        base_error = upstream_error = proposal_error = Fraction(0)
        rounded_direction: BinaryVector = ()
        completed: Vector = ()
        rounded_proposal: BinaryVector = ()
        if q == 0 or not any(tangent_direction):
            value = tuple(arithmetic.mul(model.time, component) for component in tangent_direction)
        else:
            # Reserve the composed residual proof's root evaluation, matrix-
            # free action and bounded scalar arithmetic, in addition to our
            # already reserved projection/certificate overhead.
            arithmetic._work(4*model._evaluation_work() + 512*(count+1)*(dimension+1) + 512*dimension + 1024)
            rounded_direction = tuple(_float(float(component)) for component in tangent_direction)
            exact_rounded_direction = tuple(checked(Fraction(component)) for component in rounded_direction)
            projected_rounded_direction = _linear(restriction.projector, exact_rounded_direction, arithmetic)
            completed = tuple(arithmetic.add(center, arithmetic.mul(model.time, arithmetic.add(rounded, -tangent)))
                              for center, rounded, tangent in zip(tangent_proposal, exact_rounded_direction,
                                                                  projected_rounded_direction, strict=True))
            rounded_proposal = tuple(_float(float(component)) for component in completed)
            base = certify_gaussian_target_vjp(model, projected_target, point, rounded_direction, rounded_proposal)
            if base is None or not _intact(base.certificate) or not _intact(base.root_certificate.certificate):
                return None
            base_error = checked(Fraction(base.error_bound))
            upstream_delta = tuple(arithmetic.add(rounded, -exact) for rounded, exact in
                                   zip(projected_rounded_direction, tangent_direction, strict=True))
            inverse_margin = checked(model.time/arithmetic.add(Fraction(1), -q))
            upstream_error = arithmetic.mul(inverse_margin, _norm_upper(upstream_delta, arithmetic))
            exact_rounded_proposal = tuple(checked(Fraction(component)) for component in rounded_proposal)
            projected_rounded_proposal = _linear(restriction.projector, exact_rounded_proposal, arithmetic)
            proposal_delta = tuple(arithmetic.add(rounded, -exact) for rounded, exact in
                                   zip(projected_rounded_proposal, tangent_proposal, strict=True))
            proposal_error = _norm_upper(proposal_delta, arithmetic)
            value = tangent_proposal
        error = arithmetic.add(arithmetic.add(base_error, upstream_error), proposal_error)
        coordinate_errors = tuple(arithmetic.mul(error, checked(Fraction(_sqrt_upper(row[axis], model.budget))))
                                  for axis, row in enumerate(restriction.projector)) if error else (Fraction(0),)*dimension
        # Exact endpoints avoid extra interval widening on structural zeros.
        enclosure = tuple(Interval(_rational(arithmetic.add(center, -radius)).lo,
                                   _rational(arithmetic.add(center, radius)).hi)
                          for center, radius in zip(value, coordinate_errors, strict=True))
        from omnibias.core.proof.certificate import encode_interval, make_certificate

        certificate = make_certificate(
            claim="Enclosure of the true target VJP of the unique affine-constrained Gaussian optimum.",
            payload={
                "type": "gaussian_affine_target_vjp", "restriction_certificate": deepcopy(restriction.certificate),
                "base_certificate": None if base is None else deepcopy(base.certificate),
                "target": [_pair(component) for component in frozen],
                "projected_target": [_pair(component) for component in projected_target], "scaled_point": list(point),
                "upstream": list(direction), "projected_upstream": [_pair(component) for component in tangent_direction],
                "proposal": list(proposed), "projected_proposal": [_pair(component) for component in tangent_proposal],
                "rounded_upstream": list(rounded_direction), "completed_proposal": [_pair(component) for component in completed],
                "rounded_proposal": list(rounded_proposal), "value": [_pair(component) for component in value],
                "contraction_bound": _pair(q), "base_error_bound": _pair(base_error),
                "upstream_rounding_error_bound": _pair(upstream_error), "proposal_rounding_error_bound": _pair(proposal_error),
                "error_bound": _pair(error), "coordinate_error_bounds": [_pair(component) for component in coordinate_errors],
                "enclosure": [encode_interval(interval) for interval in enclosure],
                "argument": "Exact restriction identifies the constrained target Jacobian as S=P*J'*P. Transformed centered Gaussian directions are tangent, so J' commutes with P, acts as s*I on its normal complement, and ||S||<=s/(1-q_T). Exact tangent projection removes irrelevant normal upstream components before any floating conversion. The base root-plus-adjoint residual certificate controls z-J'*ghat. Applying P is nonexpansive, and upstream conversion and recentering add s/(1-q_T)*||P*(ghat-P*g)|| and ||P*(z-proposal)||. The resulting error about P*proposal is tangent, hence coordinate i has bound sqrt(P_ii) times its Euclidean bound. Exact P*g=0 gives zero; zero tangent diameter gives s*P*g. Those shortcuts need no root or adjoint evaluation.",
                "scope": "The upstream is the exact value of the declared finite binary64 vector and is held fixed. The claim concerns the true constrained target JVP/VJP, not a finite formula's derivative, a full Jacobian, parameter derivatives, or error in an externally computed loss gradient. The rational value is tangent; not all points of its enclosing coordinate box are tangent.",
            },
            honesty={"unproven_claim": False, "unconditional_transcendentals": base is not None, "continuum_pde_claim": False},
            meta={"transcend_backend": "not_used" if base is None else base.certificate["meta"]["transcend_backend"]},
        )
        result = object.__new__(GaussianAffineTargetVJPCertificate)
        for name, stored in (("restriction", restriction), ("target", frozen), ("projected_target", projected_target),
                             ("scaled_point", point), ("upstream", direction), ("projected_upstream", tangent_direction),
                             ("proposal", proposed), ("projected_proposal", tangent_proposal), ("value", value),
                             ("enclosure", enclosure), ("error_bound", error), ("base_error_bound", base_error),
                             ("upstream_rounding_error_bound", upstream_error), ("proposal_rounding_error_bound", proposal_error),
                             ("coordinate_error_bounds", coordinate_errors), ("base_certificate", base), ("certificate", certificate)):
            object.__setattr__(result, name, stored)
        return result
    except (ArithmeticError, ValueError):
        return None


__all__ = ["GaussianAffineTargetVJPCertificate", "certify_gaussian_affine_target_vjp"]
