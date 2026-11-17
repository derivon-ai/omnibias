# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Uniform inverse Jacobian and hodograph evolution on Gaussian target boxes."""

from __future__ import annotations

from collections.abc import Sequence
from copy import deepcopy
from dataclasses import dataclass
from fractions import Fraction
from typing import TYPE_CHECKING

from omnibias.core._rational_linear import RationalLDLT, factor_positive_definite
from omnibias.core.gaussian_jet import _Arithmetic
from omnibias.core.verified._gaussian_moments import gaussian_moments
from omnibias.core.verified.gaussian_heat import GaussianHeatMap, _finite, _float, _rational
from omnibias.core.verified.interval import Interval
from omnibias.core.verified.inverse_gaussian_heat import (
    GaussianInverseCertificate,
    _pair,
    _sqrt_upper,
    inverse_gaussian_heat,
)

if TYPE_CHECKING:
    from omnibias.core.proof.certificate import Cert

Box = tuple[Interval, ...]
Matrix = tuple[Box, ...]
Vector = tuple[Fraction, ...]


def _intersect(left: Interval, right: Interval) -> Interval:
    return _finite(Interval(max(left.lo, right.lo), min(left.hi, right.hi)))


def _midpoint(value: Interval, arithmetic: _Arithmetic) -> Fraction:
    return arithmetic._coefficient(arithmetic.add(Fraction(value.lo), Fraction(value.hi)) / 2)


def _solve_enclosure(
    matrix: Matrix, rhs: Box, proposal: Vector, margin: Fraction, arithmetic: _Arithmetic,
) -> tuple[Box, Fraction]:
    """Residual bound for a proposal; every actual Gaussian H >= margin*I."""
    residual = []
    for row, wanted in zip(matrix, rhs, strict=True):
        total = Interval.point(0.0)
        for entry, value in zip(row, proposal, strict=True):
            total = _finite(total + _finite(entry * _rational(value)))
        residual.append(_finite(total - wanted))
    squared = Fraction(0)
    for entry in residual:
        magnitude = max(abs(Fraction(entry.lo)), abs(Fraction(entry.hi)))
        squared = arithmetic.add(squared, arithmetic.mul(magnitude, magnitude))
    error = arithmetic._coefficient(Fraction(_sqrt_upper(squared, arithmetic.limits)) / margin)
    upper = _rational(error).hi
    return tuple(_finite(_rational(value) + Interval(-upper, upper)) for value in proposal), error


@dataclass(frozen=True, init=False)
class GaussianInverseEvolution:
    r"""Enclosures for every target in a box at one fixed positive heat age.

    ``target_jacobian[i][j]`` encloses dX_i/dv_j. ``time_derivative`` holds
    targets, centers, biases and viscosity fixed while the heat age varies.
    ``diffusion`` encloses nu*(G G^T):D_v²X, where G=(D_vX)^(-1).
    It is computed by a third-moment contraction, without a full Hessian.
    ``diffusion_norm_bound`` is a global Euclidean bound, valid for all targets.
    No boundary/initial fitting or arbitrary PDE solution is asserted.
    """

    model: GaussianHeatMap
    target_box: Box
    spatial_enclosure: Box
    target_jacobian: Matrix
    third_trace: Box
    diffusion: Box
    time_derivative: Box
    diffusion_norm_bound: Fraction
    root_certificate: GaussianInverseCertificate | None
    certificate: Cert

    def __init__(self) -> None:
        raise TypeError("use certify_gaussian_inverse_evolution")


def certify_gaussian_inverse_evolution(
    model: GaussianHeatMap, target_box: Sequence[Interval], *,
    max_iterations: int = 128, absolute_tolerance: float = 1e-12,
) -> GaussianInverseEvolution | None:
    r"""Enclose inverse-map derivatives uniformly over a finite target box.

    Require q=D²/(8*nu*s)<1. A certified inverse of the exact box midpoint,
    enlarged by the global inverse Lipschitz constant s/(1-q), covers all
    inverse points. Positive posterior moments on this spatial box enclose
    C=Cov(a), T=E[(a-mu)*|a-mu|²] and H=I-C/(2*nu*s).

    Since H >= (1-q)I, an arbitrary exact proposed solve z with residual
    H*z-b proves ||z-H^(-1)b|| <= ||residual||/(1-q). Apply this to columns
    of H^(-1) and to b=T/(4*nu*s²). Then D_vX=s*H^(-1) and
    X_s=v+H^(-1)T/(4*nu*s²). The latter correction also obeys the global
    Euclidean bound D³/(16*nu*s²*(1-q)), used to tighten its coordinates.

    Time differentiation holds the original centers, biases, viscosity and
    target fixed. The target box is at one fixed age; this is not a time slab.
    The tolerance controls only the midpoint inverse, not final box widths.
    Coarse or zero-iteration solves can still yield valid wide enclosures.
    Invalid inputs raise. Inapplicable contraction, resource exhaustion and
    unavailable finite enclosures return None. A missing rigorous backend
    raises except for identical centers, which are handled exactly.
    """
    if not isinstance(model, GaussianHeatMap):
        raise TypeError("model must be a GaussianHeatMap")
    if not isinstance(target_box, Sequence):
        raise TypeError("target_box must be a finite sequence of Interval objects")
    box = model._box(target_box)
    if type(max_iterations) is not int:
        raise TypeError("max_iterations must be an integer, excluding bool")
    if max_iterations < 0:
        raise ValueError("max_iterations must be nonnegative")
    tolerance = _float(absolute_tolerance)
    if tolerance <= 0:
        raise ValueError("absolute_tolerance must be positive")
    q = model.contraction_bound
    if q >= 1:
        return None
    try:
        arithmetic = _Arithmetic(model.budget)
        checked = arithmetic._coefficient
        dimension, count = model.dimension, model.mode_count
        arithmetic._entries(64 * (count + 1) * (dimension + 1) + 64 * (dimension + 1)**2 + 256)
        arithmetic._work(128 * (count + 1) * (dimension + 1)**2 + 128 * (dimension + 1)**3 + 256)
        margin = arithmetic.add(Fraction(1), -q)
        inverse_margin = checked(1 / margin)
        midpoint = tuple(_midpoint(value, arithmetic) for value in box)
        root: GaussianInverseCertificate | None = None
        jacobian_errors: tuple[Fraction, ...] = (Fraction(0),) * dimension
        diffusion_error = Fraction(0)
        radius_squared = Fraction(0)
        displacement = Fraction(0)
        diffusion_norm_bound = Fraction(0)
        h_matrix: Matrix = tuple(tuple(Interval.point(float(i == j)) for j in range(dimension))
                                 for i in range(dimension))
        inverse_proposals = tuple(tuple(Fraction(int(i == j)) for i in range(dimension))
                                  for j in range(dimension))
        proposed_diffusion = (Fraction(0),) * dimension
        third_trace: Box = (Interval.point(0.0),) * dimension
        rhs = third_trace
        diffusion = third_trace
        time_derivative = box
        if q == 0:
            spatial = tuple(Interval(_rational(arithmetic.add(arithmetic.mul(model.time, Fraction(value.lo)), center)).lo,
                                     _rational(arithmetic.add(arithmetic.mul(model.time, Fraction(value.hi)), center)).hi)
                            for value, center in zip(box, model.centers[0], strict=True))
            jacobian = tuple(tuple(_rational(model.time) if i == j else Interval.point(0.0)
                                   for j in range(dimension)) for i in range(dimension))
        else:
            evaluation_work = model._evaluation_work() + 64 * dimension + 64
            remaining = model.budget.max_elimination_work - arithmetic.used
            iterations = min(max_iterations, remaining // (2 * evaluation_work) - 1)
            if iterations < 0:
                return None
            inverse = inverse_gaussian_heat(model, midpoint, max_iterations=iterations, absolute_tolerance=tolerance)
            arithmetic._work((2 * inverse.iterations + 2) * evaluation_work)
            root = inverse.certificate
            if root is None:
                return None
            for value, center in zip(box, midpoint, strict=True):
                radius = max(abs(arithmetic.add(Fraction(value.lo), -center)),
                             abs(arithmetic.add(Fraction(value.hi), -center)))
                radius_squared = arithmetic.add(radius_squared, arithmetic.mul(radius, radius))
            target_radius = checked(Fraction(_sqrt_upper(radius_squared, model.budget)))
            displacement = arithmetic.mul(arithmetic.mul(model.time, inverse_margin), target_radius)
            expansion = _rational(displacement).hi
            covers = []
            for axis, (value, center_box) in enumerate(zip(box, root.enclosure, strict=True)):
                lipschitz_cover = _finite(center_box + Interval(-expansion, expansion))
                lower = arithmetic.add(arithmetic.mul(model.time, Fraction(value.lo)), min(a[axis] for a in model.centers))
                upper = arithmetic.add(arithmetic.mul(model.time, Fraction(value.hi)), max(a[axis] for a in model.centers))
                covers.append(_intersect(lipschitz_cover, Interval(_rational(lower).lo, _rational(upper).hi)))
            spatial = tuple(covers)
            moments = gaussian_moments(model, spatial, arithmetic=arithmetic)
            third_trace = moments.third_trace
            covariance_scale = _rational(checked(1 / arithmetic.mul(arithmetic.mul(Fraction(2), model.viscosity), model.time)))
            h_offdiag = _rational(checked(q / 2)).hi
            h_rows = []
            for i, row in enumerate(moments.covariance):
                entries = []
                for j, value in enumerate(row):
                    enclosure = _finite(Interval.point(float(i == j)) - _finite(value * covariance_scale))
                    spectral = Interval(_rational(margin).lo, 1.0) if i == j else Interval(-h_offdiag, h_offdiag)
                    entries.append(_intersect(enclosure, spectral))
                h_rows.append(tuple(entries))
            h_matrix = tuple(h_rows)
            # The midpoint is only a proposal. An indefinite midpoint or an
            # inaccurate factorization never substitutes for the residual proof.
            middle = tuple(tuple(checked(arithmetic.add(_midpoint(h_matrix[i][j], arithmetic),
                                                       _midpoint(h_matrix[j][i], arithmetic)) / 2)
                                 for j in range(dimension)) for i in range(dimension))
            factor: RationalLDLT | None
            try:
                factor = factor_positive_definite(middle, arithmetic)
            except ValueError as failure:
                if str(failure) != "exact matrix is not positive definite":
                    raise
                factor = None
            inverse_columns = []
            proposals = []
            errors = []
            offdiag = _rational(checked(arithmetic.mul(q, inverse_margin) / 2)).hi
            for j in range(dimension):
                unit = tuple(Fraction(int(i == j)) for i in range(dimension))
                proposed = unit if factor is None else factor.solve(unit, arithmetic)
                proposals.append(proposed)
                column, error = _solve_enclosure(h_matrix, tuple(_rational(value) for value in unit), proposed, margin, arithmetic)
                inverse_columns.append(tuple(_intersect(value, Interval(1.0, _rational(inverse_margin).hi)
                                             if i == j else Interval(-offdiag, offdiag)) for i, value in enumerate(column)))
                errors.append(error)
            jacobian_errors = tuple(errors)
            inverse_proposals = tuple(proposals)
            jacobian = tuple(tuple(_finite(_intersect(inverse_columns[j][i], inverse_columns[i][j]) * _rational(model.time))
                                   for j in range(dimension)) for i in range(dimension))
            denominator = arithmetic.mul(arithmetic.mul(Fraction(4), model.viscosity), arithmetic.mul(model.time, model.time))
            rhs = tuple(_finite(value * _rational(checked(1 / denominator))) for value in third_trace)
            rhs_midpoint = tuple(_midpoint(value, arithmetic) for value in rhs)
            proposed_diffusion = rhs_midpoint if factor is None else factor.solve(rhs_midpoint, arithmetic)
            diffusion, diffusion_error = _solve_enclosure(h_matrix, rhs, proposed_diffusion, margin, arithmetic)
            diameter_sixth = arithmetic.mul(arithmetic.mul(model.diameter_squared, model.diameter_squared), model.diameter_squared)
            numerator = checked(Fraction(_sqrt_upper(diameter_sixth, model.budget)))
            diffusion_norm_bound = checked(numerator / arithmetic.mul(arithmetic.mul(Fraction(4), denominator), margin))
            global_upper = _rational(diffusion_norm_bound).hi
            diffusion = tuple(_intersect(value, Interval(-global_upper, global_upper)) for value in diffusion)
            time_derivative = tuple(_finite(value + correction) for value, correction in zip(box, diffusion, strict=True))
        from omnibias.core.proof.certificate import encode_interval, make_certificate

        certificate = make_certificate(
            claim="Uniform inverse target Jacobian and fixed-target heat-age derivative on a Gaussian target box.",
            payload={
                "type": "gaussian_inverse_evolution", "centers": [[_pair(value) for value in row] for row in model.centers],
                "biases": [_pair(value) for value in model.biases], "viscosity": _pair(model.viscosity), "time": _pair(model.time),
                "contraction_bound": _pair(q), "target_box": [encode_interval(value) for value in box],
                "midpoint_target": [_pair(value) for value in midpoint], "target_radius_squared": _pair(radius_squared),
                "inverse_displacement_bound": _pair(displacement),
                "root_certificate": None if root is None else deepcopy(root.certificate),
                "spatial_enclosure": [encode_interval(value) for value in spatial],
                "scaled_velocity_jacobian": [[encode_interval(value) for value in row] for row in h_matrix],
                "target_jacobian": [[encode_interval(value) for value in row] for row in jacobian],
                "inverse_column_proposals": [[_pair(value) for value in row] for row in inverse_proposals],
                "inverse_column_error_bounds": [_pair(value) for value in jacobian_errors],
                "third_trace": [encode_interval(value) for value in third_trace],
                "diffusion_rhs": [encode_interval(value) for value in rhs],
                "diffusion_proposal": [_pair(value) for value in proposed_diffusion],
                "diffusion": [encode_interval(value) for value in diffusion],
                "diffusion_solve_error_bound": _pair(diffusion_error), "diffusion_norm_bound": _pair(diffusion_norm_bound),
                "time_derivative": [encode_interval(value) for value in time_derivative],
                "argument": "The global inverse Lipschitz bound s/(1-q) transports one midpoint root enclosure to every target in the box. Positive Gaussian posterior moments on the spatial cover enclose C and T=E[(a-mu)*|a-mu|^2]. H=I-C/(2*nu*s)>=(1-q)I bounds the inverse error by the residual norm divided by 1-q. D_vX=s*H^(-1). With centers, biases, viscosity and targets held fixed, differentiating X=s*v+mu yields X_s=v+H^(-1)*T/(4*nu*s^2). This correction equals nu*(G*G^T):D_v^2 X for G=(D_vX)^(-1). Range-variance bounds give ||T||<=D^3/4, hence the global correction bound. Symmetric spectral coordinate bounds and exact affine cases tighten the enclosures.",
                "scope": "Every real target in target_box at this fixed heat age; no boundary-data fitting, time slab or arbitrary PDE solution claim.",
            },
            honesty={"unproven_claim": False, "unconditional_transcendentals": q != 0, "continuum_pde_claim": False},
            meta={"transcend_backend": "not_used" if q == 0 else "mpmath"},
        )
        result = object.__new__(GaussianInverseEvolution)
        for name, stored in (("model", model), ("target_box", box), ("spatial_enclosure", spatial),
                             ("target_jacobian", jacobian), ("third_trace", third_trace), ("diffusion", diffusion),
                             ("time_derivative", time_derivative), ("diffusion_norm_bound", diffusion_norm_bound),
                             ("root_certificate", root), ("certificate", certificate)):
            object.__setattr__(result, name, stored)
        return result
    except (ArithmeticError, ValueError):
        return None


__all__ = ["GaussianInverseEvolution", "certify_gaussian_inverse_evolution"]
