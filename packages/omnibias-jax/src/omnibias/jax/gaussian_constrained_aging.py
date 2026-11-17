# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Forward constrained Gaussian predictors with original-family age dependence."""

from __future__ import annotations

import math
from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax import Array
from jax.scipy.linalg import solve_triangular
from jax.typing import ArrayLike

from .gaussian_heat import _real_array


class GaussianConstrainedAgingResult(NamedTuple):
    """Numerical forward formula and detached diagnostics, not certificates.

    Coefficients are ordinary scaled inverse-age Taylor coefficients with
    shape (order+1,...,d) and structural c0=0. Error estimates omit floating
    evaluation error; feasibility omits QR/projection error. q_T and Q are
    shared scalars, while error, feasibility and validity have batch shape.
    """

    value: Array
    correction_coefficients: Array
    approximation_error_bound: Array
    contraction_bound: Array
    aging_parameter: Array
    feasibility_residual: Array
    constraint_reciprocal_condition: Array
    valid: Array


def gaussian_heat_inverse_constrained_predictor(
    target: Array, centers: ArrayLike, biases: ArrayLike, constraint_matrix: ArrayLike, constraint_rhs: ArrayLike,
    viscosity: ArrayLike = 1.0, time: ArrayLike = 1.0, *, order: int = 1,
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
    target: Array, centers: ArrayLike, biases: ArrayLike, constraint_matrix: ArrayLike, constraint_rhs: ArrayLike,
    viscosity: ArrayLike = 1.0, time: ArrayLike = 1.0, *, order: int = 8,
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
    target: Array, centers: ArrayLike, biases: ArrayLike, constraint_matrix: ArrayLike, constraint_rhs: ArrayLike,
    viscosity: ArrayLike = 1.0, time: ArrayLike = 1.0, *, order: int, series: bool,
    constraint_rcond: float | None, feasibility_tolerance: float | None, resummed: bool = False,
) -> GaussianConstrainedAgingResult:
    if type(order) is not int:
        raise TypeError("order must be an integer")
    if order < 0 or (not series and order not in (0, 1)):
        raise ValueError("order must be nonnegative for aging and 0 or 1 for predictor")
    target = jnp.asarray(target)
    if target.dtype not in (jnp.float32, jnp.float64):
        raise TypeError("target must have dtype float32 or float64")
    if target.ndim < 1 or target.shape[-1] < 1:
        raise ValueError("target must have shape (..., positive dimension)")
    epsilon = 2.0**(-23 if target.dtype == jnp.dtype("float32") else -52)
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
    locations, offsets = _real_array(centers, target.dtype), _real_array(biases, target.dtype)
    matrix, rhs = _real_array(constraint_matrix, target.dtype), _real_array(constraint_rhs, target.dtype)
    nu, age = _real_array(viscosity, target.dtype), _real_array(time, target.dtype)
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
        constraints_valid = jnp.isfinite(matrix).all() & jnp.isfinite(rhs).all()
        matrix, rhs = jnp.where(jnp.isfinite(matrix), matrix, 0.0), jnp.where(jnp.isfinite(rhs), rhs, 0.0)
        scales = jnp.max(jnp.abs(matrix), axis=-1)
        safe_scales = jnp.where(scales > 0, scales, 1.0)
        matrix, rhs = matrix / safe_scales[:, None], rhs / safe_scales
        constraints_valid = constraints_valid & (scales > 0).all() & jnp.isfinite(rhs).all()
        singular = jnp.linalg.svd(jax.lax.stop_gradient(matrix), compute_uv=False)
        reciprocal = singular[-1] / jnp.where(singular[0] > 0, singular[0], 1.0)
        condition = jnp.where(constraints_valid, reciprocal, jnp.nan)
        constraints_valid = constraints_valid & jnp.isfinite(reciprocal) & (reciprocal > constraint_rcond)
        identity_rows = jnp.eye(rows, dimension, dtype=target.dtype)
        matrix, rhs = jnp.where(constraints_valid, matrix, identity_rows), jnp.where(constraints_valid, rhs, 0.0)
        normal_basis, triangular = jnp.linalg.qr(matrix.T, mode="reduced")
        eta = solve_triangular(triangular.T, rhs, lower=True)
        constraints_valid = (constraints_valid & jnp.isfinite(normal_basis).all() & jnp.isfinite(triangular).all()
                             & jnp.isfinite(eta).all())
        normal_basis, eta = jnp.where(constraints_valid, normal_basis, identity_rows.T), jnp.where(constraints_valid, eta, 0.0)
    else:
        normal_basis, eta = matrix.T, rhs
        condition = jnp.ones((), dtype=target.dtype)
        constraints_valid = jnp.asarray(True)

    shared = (constraints_valid & jnp.isfinite(locations).all() & jnp.isfinite(offsets).all()
              & jnp.isfinite(nu) & (nu > 0) & jnp.isfinite(age) & (age > 0))
    locations, offsets = jnp.where(jnp.isfinite(locations), locations, 0.0), jnp.where(jnp.isfinite(offsets), offsets, 0.0)
    nu, age = jnp.where(jnp.isfinite(nu) & (nu > 0), nu, 1.0), jnp.where(jnp.isfinite(age) & (age > 0), age, 1.0)
    query_valid = jnp.isfinite(target).all(-1)
    query = jnp.where(jnp.isfinite(target), target, 0.0)
    relative = locations - locations[0]
    normal_coordinates = relative @ normal_basis
    normal_offset = locations[0] @ normal_basis - eta
    if rows == dimension:
        anchor = eta @ normal_basis.T
        projected_relative, projected_query = relative*0, query*0
    else:
        anchor = locations[0] - normal_offset @ normal_basis.T
        projected_relative = relative - normal_coordinates @ normal_basis.T
        projected_query = query - (query @ normal_basis) @ normal_basis.T
    offsets = offsets-offsets[0]
    cost = jnp.sum(normal_coordinates**2, axis=-1)+2*jnp.sum(normal_coordinates*normal_offset, axis=-1)
    denominator = 8*nu*age
    if resummed:
        # Freeze only with respect to the auxiliary series variable; preserve
        # runtime differentiation of the original model's age-dependent shift.
        offsets = offsets-cost/(denominator/2)
    geometry_valid = (shared & jnp.isfinite(relative).all() & jnp.isfinite(normal_coordinates).all()
                      & jnp.isfinite(normal_offset).all() & jnp.isfinite(anchor).all()
                      & jnp.isfinite(projected_relative).all())
    detached_relative = jax.lax.stop_gradient(projected_relative)
    def diameter_step(index: int, maximum: Array) -> Array:
        return jnp.maximum(maximum, jnp.sum((detached_relative-detached_relative[index])**2, axis=-1).max())
    diameter2 = jax.lax.fori_loop(0, locations.shape[0], diameter_step, jnp.zeros((), dtype=target.dtype))
    affine = diameter2 == 0
    cost_range = jnp.zeros_like(denominator) if resummed else jax.lax.stop_gradient(jnp.max(cost)-jnp.min(cost))
    diagnostic_denominator = jax.lax.stop_gradient(denominator)
    q = jnp.where(affine, 0.0, diameter2/diagnostic_denominator)
    aging = (diameter2+cost_range)/diagnostic_denominator
    parameters_finite = (jnp.isfinite(diameter2) & jnp.isfinite(cost_range) & jnp.isfinite(diagnostic_denominator)
                         & (diagnostic_denominator > 0) & jnp.isfinite(q) & jnp.isfinite(aging))
    allowed = parameters_finite & (affine | (aging < 1/3 if series else q < 1))
    bound_q, bound_aging = jnp.where(parameters_finite, q, 0.0), jnp.where(parameters_finite, aging, 0.0)
    diameter = jnp.sqrt(diameter2)
    real_bound = diameter*bound_aging/2 if order == 0 else diameter*bound_aging*(bound_aging+bound_q)/2
    safe_aging = jnp.where(bound_aging < 1/3, bound_aging, 0.0)
    tail = diameter/2*(3*safe_aging)**(order+1)/(1-3*safe_aging)
    estimate = jnp.minimum(tail, real_bound) if series and order <= 1 else (tail if series else real_bound)
    estimate = jnp.where(affine, 0.0, estimate)
    if rows == dimension:
        zero = jnp.sum(locations*0)+jnp.sum(offsets*0)+nu*0+age*0
        value = anchor+projected_query+zero
        stacked = jnp.stack([value*0 for _ in range(order+1)])
        valid = geometry_valid & query_valid & parameters_finite
    else:
        logits = projected_query @ projected_relative.T/(2*nu)+offsets
        valid = geometry_valid & query_valid & allowed & jnp.isfinite(logits).all(-1) & jnp.isfinite(cost).all()
        probabilities = jax.nn.softmax(jnp.where(jnp.isfinite(logits), logits, 0.0), axis=-1)
        mean = probabilities @ projected_relative
        centered = projected_relative-mean[..., None, :]
        coefficients = [jnp.zeros_like(query)]
        logarithms = [jnp.zeros_like(probabilities)]
        exponentials = [jnp.ones_like(probabilities)]
        partitions = [jnp.ones_like(query_valid, dtype=target.dtype)]
        for n in range(1, order+1):
            logarithm = (-(jnp.sum(centered**2, axis=-1)+(0 if resummed else cost))/(denominator/2) if n == 1
                         else jnp.sum(centered*coefficients[n-1][..., None, :], axis=-1)/(denominator/4))
            logarithms.append(logarithm)
            exponential = jnp.zeros_like(probabilities)
            for k in range(1, n+1):
                exponential = exponential+(k/n)*logarithms[k]*exponentials[n-k]
            exponentials.append(exponential)
            weighted = probabilities*exponential
            partitions.append(jnp.sum(weighted, axis=-1))
            coefficient = jnp.sum(weighted[..., :, None]*centered, axis=-2)
            for k in range(1, n+1):
                coefficient = coefficient-partitions[k][..., None]*coefficients[n-k]
            coefficients.append(coefficient)
            valid = (valid & jnp.isfinite(logarithm).all(-1) & jnp.isfinite(exponential).all(-1)
                     & jnp.isfinite(partitions[-1]) & jnp.isfinite(coefficient).all(-1))
        stacked = jnp.stack(coefficients)
        value = anchor+age*projected_query+mean+jnp.sum(stacked, axis=0)
    if rows:
        normal_residual = value @ normal_basis-eta
        scale = jnp.max(jnp.abs(normal_residual), axis=-1)
        safe_scale = jnp.where(scale > 0, scale, 1.0)
        feasibility = scale*jnp.sqrt(jnp.sum((normal_residual/safe_scale[..., None])**2, axis=-1))
    else:
        feasibility = jnp.zeros_like(query_valid, dtype=target.dtype)
    valid = (valid & jnp.isfinite(value).all(-1) & jnp.isfinite(estimate)
             & jnp.isfinite(feasibility) & (feasibility <= feasibility_tolerance))
    q, aging = jnp.where(geometry_valid, q, jnp.inf), jnp.where(geometry_valid, aging, jnp.inf)
    return GaussianConstrainedAgingResult(
        jnp.where(valid[..., None], value, jnp.nan), jnp.where(valid[None, ..., None], stacked, jnp.nan),
        jax.lax.stop_gradient(jnp.where(valid, estimate, jnp.nan)), jax.lax.stop_gradient(q),
        jax.lax.stop_gradient(aging), jax.lax.stop_gradient(jnp.where(valid, feasibility, jnp.inf)),
        jax.lax.stop_gradient(condition), jax.lax.stop_gradient(valid),
    )


__all__ = ["GaussianConstrainedAgingResult", "gaussian_heat_inverse_constrained_aging", "gaussian_heat_inverse_constrained_predictor"]
