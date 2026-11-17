# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Forward constrained Gaussian predictors with original-family age dependence."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import NamedTuple

import torch
from torch import Tensor

from .gaussian_heat import _norm, _tensor


class GaussianConstrainedAgingResult(NamedTuple):
    """Numerical forward formula and detached diagnostics, not certificates.

    Coefficients are ordinary scaled inverse-age Taylor coefficients with
    shape (order+1,...,d) and structural c0=0. Error estimates omit floating
    evaluation error; feasibility omits QR/projection error. q_T and Q are
    shared scalars, while error, feasibility and validity have batch shape.
    """

    value: Tensor
    correction_coefficients: Tensor
    approximation_error_bound: Tensor
    contraction_bound: Tensor
    aging_parameter: Tensor
    feasibility_residual: Tensor
    constraint_reciprocal_condition: Tensor
    valid: Tensor


def gaussian_heat_inverse_constrained_predictor(
    target: Tensor, centers: Tensor | Sequence[Sequence[float]], biases: Tensor | Sequence[float],
    constraint_matrix: Tensor | Sequence[Sequence[float]], constraint_rhs: Tensor | Sequence[float],
    viscosity: Tensor | float = 1.0, time: Tensor | float = 1.0, *, order: int = 1,
    constraint_rcond: float | None = None, feasibility_tolerance: float | None = None,
) -> GaussianConstrainedAgingResult:
    r"""Evaluate the order-0/1 constrained inverse predictor without roots.

    The limiting posterior uses ORIGINAL biases. Order one subtracts
    E[A*(|A|²+normal_cost_difference)]/(4*nu*time), including the original
    model's normal-energy age dependence. Require q_T=D_T²/(8*nu*time)<1.
    With Q=(D_T²+range(normal_cost))/(8*nu*time), Euclidean truncation bounds
    are D_T*Q/2 and D_T*Q*(Q+q_T)/2 for orders 0 and 1. Q may exceed one.
    Order is static 0 or 1. No inverse iteration or implicit solve is used.

    Target is float32/64 (...,d); shared centers, biases, C(r,d), rhs(r),
    viscosity and time retain gradients, including higher derivatives.
    Constraints must have numerical full row rank after row equilibration;
    constraint_rcond defaults to sqrt(epsilon). A reduced QR constructs the
    tangent geometry with no dense projector. Feasibility tolerance defaults
    to 64*epsilon in physical normal coordinates. All options are static.
    Autodiff differentiates this finite approximation, not the true inverse.
    Diagnostics are detached numerical estimates, not rigorous certificates
    of error, exact feasibility or derivative accuracy. Invalid entries give
    NaN value/coefficients/error and false validity. N orders cost
    O((m+d)*N²+m*d*N) per target after shared QR and diameter preprocessing;
    the recurrence uses no further factorization and autodiff may retain more
    storage. JAX supports jit/vmap; Torch preserves device and double backward.
    """
    return _evaluate(target, centers, biases, constraint_matrix, constraint_rhs, viscosity, time,
                     order=order, series=False, constraint_rcond=constraint_rcond, feasibility_tolerance=feasibility_tolerance)


def gaussian_heat_inverse_constrained_aging(
    target: Tensor, centers: Tensor | Sequence[Sequence[float]], biases: Tensor | Sequence[float],
    constraint_matrix: Tensor | Sequence[Sequence[float]], constraint_rhs: Tensor | Sequence[float],
    viscosity: Tensor | float = 1.0, time: Tensor | float = 1.0, *, order: int = 8,
    constraint_rcond: float | None = None, feasibility_tolerance: float | None = None,
) -> GaussianConstrainedAgingResult:
    r"""Evaluate the constrained inverse's original-family aging series.

    With physical age s, the coefficients expand
    X_C(v,s/t)-(s/t)*Pv-mu_infinity in ordinary powers of t, where
    mu_infinity is the limiting affine mean using ORIGINAL biases.
    Value evaluates the truncation at t=1. Nonzero tangent diameter requires
    Q=(D_T²+range(normal_cost))/(8*nu*s)<1/3. The Euclidean tail is
    D_T/2*(3Q)**(order+1)/(1-3Q), tightened by the real order-0/1 bounds.
    Coincident tangent centers bypass the Q restriction. Order is a static
    nonnegative integer. This is a finite formula, not an implicit inverse.

    Target is float32/64 (...,d); shared centers, biases, C(r,d), rhs(r),
    viscosity and time retain gradients, including higher derivatives.
    Constraints must have numerical full row rank after row equilibration;
    constraint_rcond defaults to sqrt(epsilon). A reduced QR constructs the
    tangent geometry with no dense projector. Feasibility tolerance defaults
    to 64*epsilon in physical normal coordinates. All options are static.
    Autodiff differentiates this finite approximation, not the true inverse.
    Diagnostics are detached numerical estimates, not rigorous certificates
    of error, exact feasibility or derivative accuracy. Invalid entries give
    NaN value/coefficients/error and false validity. N orders cost
    O((m+d)*N²+m*d*N) per target after shared QR and diameter preprocessing;
    the recurrence uses no further factorization and autodiff may retain more
    storage. JAX supports jit/vmap; Torch preserves device and double backward.
    """
    return _evaluate(target, centers, biases, constraint_matrix, constraint_rhs, viscosity, time,
                     order=order, series=True, constraint_rcond=constraint_rcond, feasibility_tolerance=feasibility_tolerance)


def _evaluate(
    target: Tensor, centers: Tensor | Sequence[Sequence[float]], biases: Tensor | Sequence[float],
    constraint_matrix: Tensor | Sequence[Sequence[float]], constraint_rhs: Tensor | Sequence[float],
    viscosity: Tensor | float = 1.0, time: Tensor | float = 1.0, *, order: int, series: bool,
    constraint_rcond: float | None, feasibility_tolerance: float | None, resummed: bool = False,
) -> GaussianConstrainedAgingResult:
    if type(order) is not int:
        raise TypeError("order must be an integer")
    if order < 0 or (not series and order not in (0, 1)):
        raise ValueError("order must be nonnegative for aging and 0 or 1 for predictor")
    if not isinstance(target, Tensor) or target.dtype not in (torch.float32, torch.float64):
        raise TypeError("target must be a float32 or float64 Tensor")
    if target.ndim < 1 or target.shape[-1] < 1:
        raise ValueError("target must have shape (..., positive dimension)")
    epsilon = torch.finfo(target.dtype).eps
    if constraint_rcond is None:
        constraint_rcond = math.sqrt(epsilon)
    if feasibility_tolerance is None:
        feasibility_tolerance = 64 * epsilon
    for name, option in (("constraint_rcond", constraint_rcond),
                         ("feasibility_tolerance", feasibility_tolerance)):
        if type(option) not in (float, int) or not math.isfinite(option) or option <= 0:
            raise ValueError(f"{name} must be a finite positive scalar")
    if constraint_rcond >= 1:
        raise ValueError("constraint_rcond must be less than one")
    locations, offsets = _tensor(centers, target, "centers"), _tensor(biases, target, "biases")
    matrix, rhs = _tensor(constraint_matrix, target, "constraint_matrix"), _tensor(constraint_rhs, target, "constraint_rhs")
    nu, age = _tensor(viscosity, target, "viscosity"), _tensor(time, target, "time")
    dimension = target.shape[-1]
    if locations.ndim != 2 or locations.shape[0] < 1 or locations.shape[1] != dimension:
        raise ValueError("centers must have shape (positive mode count, target dimension)")
    if offsets.shape != (locations.shape[0],):
        raise ValueError("biases must have one entry per center")
    if matrix.ndim != 2 or matrix.shape[1] != dimension or matrix.shape[0] > dimension:
        raise ValueError("constraint_matrix must have shape (r, dimension) with 0<=r<=dimension")
    if rhs.shape != (matrix.shape[0],):
        raise ValueError("constraint_rhs must have one entry per constraint")
    if nu.ndim or age.ndim:
        raise ValueError("viscosity and time must be scalars")
    rows = matrix.shape[0]
    if rows:
        constraints_valid = torch.isfinite(matrix).all() & torch.isfinite(rhs).all()
        matrix = torch.where(torch.isfinite(matrix), matrix, torch.zeros_like(matrix))
        rhs = torch.where(torch.isfinite(rhs), rhs, torch.zeros_like(rhs))
        scales = matrix.abs().amax(-1)
        safe_scales = torch.where(scales > 0, scales, torch.ones_like(scales))
        matrix, rhs = matrix / safe_scales[:, None], rhs / safe_scales
        constraints_valid = constraints_valid & (scales > 0).all() & torch.isfinite(rhs).all()
        with torch.no_grad():
            singular = torch.linalg.svdvals(matrix)
            reciprocal = singular[-1] / torch.where(singular[0] > 0, singular[0], torch.ones_like(singular[0]))
            condition = torch.where(constraints_valid, reciprocal, torch.full_like(reciprocal, float("nan")))
        constraints_valid = constraints_valid & torch.isfinite(reciprocal) & (reciprocal > constraint_rcond)
        identity_rows = torch.eye(rows, dimension, dtype=target.dtype, device=target.device)
        matrix = torch.where(constraints_valid, matrix, identity_rows)
        rhs = torch.where(constraints_valid, rhs, torch.zeros_like(rhs))
        normal_basis, triangular = torch.linalg.qr(matrix.mT, mode="reduced")
        eta = torch.linalg.solve_triangular(triangular.mT, rhs[:, None], upper=False)[:, 0]
        constraints_valid = (constraints_valid & torch.isfinite(normal_basis).all() & torch.isfinite(triangular).all()
                             & torch.isfinite(eta).all())
        normal_basis = torch.where(constraints_valid, normal_basis, identity_rows.mT)
        eta = torch.where(constraints_valid, eta, torch.zeros_like(eta))
    else:
        normal_basis, eta = matrix.mT, rhs
        condition = target.new_ones(())
        constraints_valid = torch.ones((), dtype=torch.bool, device=target.device)

    shared = (constraints_valid & torch.isfinite(locations).all() & torch.isfinite(offsets).all()
              & torch.isfinite(nu) & (nu > 0) & torch.isfinite(age) & (age > 0))
    locations = torch.where(torch.isfinite(locations), locations, torch.zeros_like(locations))
    offsets = torch.where(torch.isfinite(offsets), offsets, torch.zeros_like(offsets))
    nu = torch.where(torch.isfinite(nu) & (nu > 0), nu, torch.ones_like(nu))
    age = torch.where(torch.isfinite(age) & (age > 0), age, torch.ones_like(age))
    query_valid = torch.isfinite(target).all(-1)
    query = torch.where(torch.isfinite(target), target, torch.zeros_like(target))
    relative = locations - locations[0]
    normal_coordinates = relative @ normal_basis
    normal_offset = locations[0] @ normal_basis - eta
    if rows == dimension:
        anchor = eta @ normal_basis.mT
        projected_relative, projected_query = relative * 0, query * 0
    else:
        anchor = locations[0] - normal_offset @ normal_basis.mT
        projected_relative = relative - normal_coordinates @ normal_basis.mT
        projected_query = query - (query @ normal_basis) @ normal_basis.mT
    offsets = offsets - offsets[0]
    cost = normal_coordinates.square().sum(-1) + 2 * (normal_coordinates * normal_offset).sum(-1)
    denominator = 8 * nu * age
    if resummed:
        # This shift stays in the differentiable graph. Only the auxiliary
        # coefficient variable freezes it; runtime age/viscosity AD does not.
        offsets = offsets - cost / (denominator / 2)
    geometry_valid = (shared & torch.isfinite(relative).all() & torch.isfinite(normal_coordinates).all()
                      & torch.isfinite(normal_offset).all() & torch.isfinite(anchor).all()
                      & torch.isfinite(projected_relative).all())
    with torch.no_grad():
        diameter2 = torch.stack([(projected_relative-row).square().sum(-1).amax() for row in projected_relative]).amax()
        affine = diameter2 == 0
        cost_range = torch.zeros_like(denominator) if resummed else cost.amax() - cost.amin()
        q = diameter2 / denominator
        aging = (diameter2 + cost_range) / denominator
        q = torch.where(affine, torch.zeros_like(q), q)
        parameters_finite = (torch.isfinite(diameter2) & torch.isfinite(cost_range) & torch.isfinite(denominator)
                             & (denominator > 0) & torch.isfinite(q) & torch.isfinite(aging))
        allowed = parameters_finite & (affine | (aging < 1/3 if series else q < 1))
        bound_q = torch.where(parameters_finite, q, torch.zeros_like(q))
        bound_aging = torch.where(parameters_finite, aging, torch.zeros_like(aging))
        diameter = torch.sqrt(diameter2)
        real_bound = diameter * bound_aging / 2 if order == 0 else diameter * bound_aging * (bound_aging+bound_q) / 2
        safe_aging = torch.where(bound_aging < 1/3, bound_aging, torch.zeros_like(bound_aging))
        tail = diameter / 2 * (3*safe_aging).pow(order+1) / (1-3*safe_aging)
        estimate = torch.minimum(tail, real_bound) if series and order <= 1 else (tail if series else real_bound)
        estimate = torch.where(affine, torch.zeros_like(estimate), estimate)
    if rows == dimension:
        # No tangent directions remain even when the constraint matrix varies.
        zero = (locations*0).sum() + (offsets*0).sum() + nu*0 + age*0
        value = anchor + projected_query + zero
        stacked = torch.stack([value*0 for _ in range(order+1)])
        valid = geometry_valid & query_valid & parameters_finite
    else:
        logits = projected_query @ projected_relative.mT / (2*nu) + offsets
        valid = geometry_valid & query_valid & allowed & torch.isfinite(logits).all(-1) & torch.isfinite(cost).all()
        probabilities = torch.softmax(torch.where(torch.isfinite(logits), logits, torch.zeros_like(logits)), dim=-1)
        mean = probabilities @ projected_relative
        centered = projected_relative - mean[..., None, :]
        coefficients = [torch.zeros_like(query)]
        logarithms = [torch.zeros_like(probabilities)]
        exponentials = [torch.ones_like(probabilities)]
        partitions = [torch.ones_like(query_valid, dtype=target.dtype)]
        for n in range(1, order+1):
            logarithm = (-(centered.square().sum(-1)+(0 if resummed else cost))/(denominator/2) if n == 1
                         else (centered*coefficients[n-1][..., None, :]).sum(-1)/(denominator/4))
            logarithms.append(logarithm)
            exponential = torch.zeros_like(probabilities)
            for k in range(1, n+1):
                exponential = exponential + (k/n)*logarithms[k]*exponentials[n-k]
            exponentials.append(exponential)
            weighted = probabilities*exponential
            partitions.append(weighted.sum(-1))
            coefficient = (weighted[..., :, None]*centered).sum(-2)
            for k in range(1, n+1):
                coefficient = coefficient-partitions[k][..., None]*coefficients[n-k]
            coefficients.append(coefficient)
            valid = (valid & torch.isfinite(logarithm).all(-1) & torch.isfinite(exponential).all(-1)
                     & torch.isfinite(partitions[-1]) & torch.isfinite(coefficient).all(-1))
        stacked = torch.stack(coefficients)
        value = anchor + age*projected_query + mean + stacked.sum(0)
    with torch.no_grad():
        feasibility = _norm(value @ normal_basis-eta) if rows else torch.zeros_like(query_valid, dtype=target.dtype)
        valid = (valid & torch.isfinite(value).all(-1) & torch.isfinite(estimate)
                 & torch.isfinite(feasibility) & (feasibility <= feasibility_tolerance))
        q = torch.where(geometry_valid, q, torch.full_like(q, float("inf")))
        aging = torch.where(geometry_valid, aging, torch.full_like(aging, float("inf")))
        feasibility = torch.where(valid, feasibility, torch.full_like(feasibility, float("inf")))
    return GaussianConstrainedAgingResult(
        torch.where(valid[..., None], value, torch.full_like(value, float("nan"))),
        torch.where(valid[None, ..., None], stacked, torch.full_like(stacked, float("nan"))),
        torch.where(valid, estimate, torch.full_like(valid, float("nan"), dtype=target.dtype)),
        q, aging, feasibility, condition, valid,
    )


__all__ = ["GaussianConstrainedAgingResult", "gaussian_heat_inverse_constrained_aging", "gaussian_heat_inverse_constrained_predictor"]
