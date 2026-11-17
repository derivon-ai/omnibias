# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Forced inverse PDE and target sensitivities on affine Gaussian constraints."""

from __future__ import annotations

from collections.abc import Sequence
from copy import deepcopy
from dataclasses import dataclass
from fractions import Fraction
from typing import TYPE_CHECKING

from omnibias.core.gaussian_jet import _Arithmetic
from omnibias.core.verified._gaussian_moments import _add, _multiply, gaussian_moments
from omnibias.core.verified.gaussian_affine import GaussianAffineRestriction
from omnibias.core.verified.gaussian_evolution import (
    GaussianInverseEvolution,
    certify_gaussian_inverse_evolution,
)
from omnibias.core.verified.gaussian_heat import _finite, _float, _rational
from omnibias.core.verified.interval import Interval
from omnibias.core.verified.inverse_gaussian_affine import _intact
from omnibias.core.verified.inverse_gaussian_heat import _pair, _sqrt_upper

if TYPE_CHECKING:
    from omnibias.core.proof.certificate import Cert

Box = tuple[Interval, ...]
Matrix = tuple[Box, ...]
_ZERO = Interval.point(0.0)


def _action(matrix: Matrix, vector: Box) -> Box:
    output = []
    for row in matrix:
        total = _ZERO
        for entry, value in zip(row, vector, strict=True):
            total = _add(total, _multiply(entry, value))
        output.append(total)
    return tuple(output)


@dataclass(frozen=True, init=False)
class GaussianAffineEvolution:
    """Target Jacobian and original-model heat-age evolution on an affine set.

    The ambient ``target_jacobian`` is P*J_projected*P and is singular when
    constraints are present. The hodograph PDE is understood in tangent
    coordinates, not by inverting this ambient matrix. ``normal_forcing``
    accounts for the age dependence of the projected Gaussian bias shifts.
    ``time_derivative`` keeps the original centers, biases, constraints, target
    and viscosity fixed. All boxes refer to one fixed heat age.
    """

    restriction: GaussianAffineRestriction
    target_box: Box
    projected_target_box: Box
    spatial_enclosure: Box
    target_jacobian: Matrix
    normal_cost_covariance: Box
    diffusion: Box
    normal_forcing: Box
    time_derivative: Box
    correction_norm_bound: Fraction
    base_evolution: GaussianInverseEvolution
    certificate: Cert

    def __init__(self) -> None:
        raise TypeError("use certify_gaussian_affine_evolution")


def certify_gaussian_affine_evolution(
    restriction: GaussianAffineRestriction, target_box: Sequence[Interval], *,
    max_iterations: int = 128, absolute_tolerance: float = 1e-12,
) -> GaussianAffineEvolution | None:
    r"""Certify constrained target sensitivities and the forced inverse PDE.

    Let c_j be squared normal distances of the original centers to the affine
    set, r_j=P(a_j-mu), C_T=E[r*r^T], T_T=E[r*|r|²], K=E[r*c_j],
    and H_T=I-C_T/(2*nu*s). With q_T<1 the constrained inverse obeys
    X_s=P*v+H_T^(-1)*(T_T+K)/(4*nu*s²). The T_T term is tangent
    hodograph diffusion; K is essential normal-cost forcing. Common normal
    costs cancel, so exact cost differences suffice. Their range gives
    ||X_s-P*v|| <= (D_T³+D_T*range(c))/(16*nu*s²*(1-q_T)).

    Compose a certified projected-model inverse/Jacobian with outward moments
    and matrix actions. The exact projector preserves the affine geometry;
    coordinate boxes need not consist entirely of feasible points. No full
    third-order tensor is formed. The core path stores dense d-by-d matrices
    and enforces aggregate resource bounds. Projection of a target box can
    lose correlations and widen the resulting enclosure.

    Invalid inputs raise. Failed contraction, damaged dependencies, exhausted
    resources or nonfinite bounds return None. Missing rigorous transcendental
    support raises except for an exactly affine restricted map.
    """
    if not isinstance(restriction, GaussianAffineRestriction):
        raise TypeError("restriction must be a GaussianAffineRestriction")
    if not isinstance(target_box, Sequence):
        raise TypeError("target_box must be a finite sequence of Interval objects")
    model = restriction.model
    box = model._box(target_box)
    if type(max_iterations) is not int:
        raise TypeError("max_iterations must be an integer, excluding bool")
    if max_iterations < 0:
        raise ValueError("max_iterations must be nonnegative")
    tolerance = _float(absolute_tolerance)
    if tolerance <= 0:
        raise ValueError("absolute_tolerance must be positive")
    if model.contraction_bound >= 1 or not _intact(restriction.certificate):
        return None
    try:
        arithmetic = _Arithmetic(model.budget)
        checked = arithmetic._coefficient
        dimension, count = model.dimension, model.mode_count
        arithmetic._entries(128*(count+len(restriction.constraints)+1)*(dimension+1) + 128*(dimension+1)**2 + 512)
        # Reserve the composed evolution's noniterative work, including its
        # moments, LDLT, all inverse columns, certificate copy and our actions.
        arithmetic._work(1024*(count+1)*(dimension+1)**2 + 1024*(dimension+1)**3 + 1024)
        iterations = 0
        if model.contraction_bound:
            evaluation_work = model._evaluation_work() + 64*dimension + 64
            remaining = model.budget.max_elimination_work-arithmetic.used
            iterations = min(max_iterations, remaining//(4*evaluation_work)-1)
            if iterations < 0:
                return None
            # Two evaluations per iteration is an upper bound; leave half
            # the remaining budget for checked rational intermediates here.
            arithmetic._work((2*iterations+2)*evaluation_work)
        projector = tuple(tuple(_rational(value) for value in row) for row in restriction.projector)
        projected_box = _action(projector, box)
        base = certify_gaussian_inverse_evolution(model, projected_box, max_iterations=iterations,
                                                  absolute_tolerance=tolerance)
        if base is None:
            return None
        covariance: Box = (_ZERO,)*dimension
        forcing = covariance
        forcing_bound = Fraction(0)
        if model.contraction_bound == 0:
            jacobian = tuple(tuple(_rational(arithmetic.mul(model.time, value)) for value in row)
                             for row in restriction.projector)
        else:
            # Both projections express the derivative of the constrained map
            # with respect to the original ambient target, including its nulls.
            right = tuple(_action(projector, row) for row in base.target_jacobian)
            columns = tuple(_action(projector, tuple(row[j] for row in right)) for j in range(dimension))
            jacobian = tuple(tuple(column[i] for column in columns) for i in range(dimension))
            costs = restriction.normal_cost_differences
            cost_range = arithmetic.add(max(costs), -min(costs))
            if cost_range:
                moments = gaussian_moments(model, base.spatial_enclosure, arithmetic=arithmetic)
                relative = tuple(tuple(arithmetic.add(value, -reference) for value, reference in
                                       zip(row, model.centers[0], strict=True)) for row in model.centers)
                total = [_ZERO]*dimension
                # K is unchanged by a common cost, already removed exactly.
                for probability, row, cost in zip(moments.posterior, relative, costs, strict=True):
                    weight = _multiply(probability, _rational(cost))
                    for axis, (value, mean) in enumerate(zip(row, moments.relative_mean, strict=True)):
                        centered = _ZERO if mean.lo == mean.hi and value == Fraction(mean.lo) else _finite(_rational(value)-mean)
                        total[axis] = _add(total[axis], _multiply(weight, centered))
                covariance = tuple(total)
                # J'=s*H_T^-1, so its action needs one additional factor 1/s.
                denominator = arithmetic.mul(arithmetic.mul(Fraction(4), model.viscosity),
                                             arithmetic.mul(arithmetic.mul(model.time, model.time), model.time))
                rhs = tuple(_multiply(_rational(checked(1/denominator)), value) for value in covariance)
                forcing = _action(base.target_jacobian, rhs)
                diameter = checked(Fraction(_sqrt_upper(model.diameter_squared, model.budget)))
                bound_denominator = arithmetic.mul(arithmetic.mul(Fraction(16), model.viscosity),
                    arithmetic.mul(arithmetic.mul(model.time, model.time), arithmetic.add(Fraction(1), -model.contraction_bound)))
                forcing_bound = checked(arithmetic.mul(diameter, cost_range)/bound_denominator)
                upper = _rational(forcing_bound).hi
                forcing = tuple(value.intersect(Interval(-upper, upper)) for value in forcing)
        # Both exact corrections are tangent, independently of rounding in
        # the ambient inverse-Jacobian enclosure. Intersect with that identity.
        diffusion = tuple(value.intersect(projected) for value, projected in
                          zip(base.diffusion, _action(projector, base.diffusion), strict=True))
        forcing = tuple(value.intersect(projected) for value, projected in
                        zip(forcing, _action(projector, forcing), strict=True))
        bound = arithmetic.add(base.diffusion_norm_bound, forcing_bound)
        time_derivative = tuple(_add(_add(target, diffusion), extra) for target, diffusion, extra in
                                zip(projected_box, diffusion, forcing, strict=True))
        from omnibias.core.proof.certificate import encode_interval, make_certificate

        certificate = make_certificate(
            claim="Uniform constrained Gaussian target Jacobian and forced heat-age evolution at fixed affine constraints.",
            payload={
                "type": "gaussian_affine_evolution", "restriction_certificate": deepcopy(restriction.certificate),
                "projected_evolution_certificate": deepcopy(base.certificate),
                "target_box": [encode_interval(value) for value in box],
                "projected_target_box": [encode_interval(value) for value in projected_box],
                "spatial_enclosure": [encode_interval(value) for value in base.spatial_enclosure],
                "target_jacobian": [[encode_interval(value) for value in row] for row in jacobian],
                "normal_cost_covariance": [encode_interval(value) for value in covariance],
                "diffusion": [encode_interval(value) for value in diffusion],
                "normal_forcing": [encode_interval(value) for value in forcing],
                "time_derivative": [encode_interval(value) for value in time_derivative],
                "normal_forcing_norm_bound": _pair(forcing_bound), "correction_norm_bound": _pair(bound),
                "argument": "The fixed-age constrained inverse is the projected-family inverse at P*v, giving target Jacobian P*J'*P. Its relative log weights vary with heat age by normal_cost_differences/(4*nu*s^2); their posterior mean derivative is K/(4*nu*s^2). Adding this chain-rule term to the fixed-bias hodograph equation gives X_s=P*v+H_T^-1*(T_T+K)/(4*nu*s^2). Positive range-variance covariance bounds yield ||K||<=D_T*range(c)/4. Outward posterior moments and the certified J'=s*H_T^-1 enclose its action, tightened by the uniform spectral bound. The homogeneous diffusion remains the tangent hodograph contraction.",
                "scope": "Original centers, biases, affine equations, viscosity and target held fixed; every target in target_box at one fixed heat age. The ambient target Jacobian can be singular; its inverse is not used to state a full-space hodograph equation.",
            },
            honesty={"unproven_claim": False, "unconditional_transcendentals": model.contraction_bound != 0,
                     "continuum_pde_claim": False},
            meta={"transcend_backend": base.certificate["meta"]["transcend_backend"]},
        )
        result = object.__new__(GaussianAffineEvolution)
        for name, stored in (("restriction", restriction), ("target_box", box), ("projected_target_box", projected_box),
                             ("spatial_enclosure", base.spatial_enclosure), ("target_jacobian", jacobian),
                             ("normal_cost_covariance", covariance), ("diffusion", diffusion),
                             ("normal_forcing", forcing), ("time_derivative", time_derivative),
                             ("correction_norm_bound", bound), ("base_evolution", base), ("certificate", certificate)):
            object.__setattr__(result, name, stored)
        return result
    except (ArithmeticError, ValueError):
        return None


__all__ = ["GaussianAffineEvolution", "certify_gaussian_affine_evolution"]
