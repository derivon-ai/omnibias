# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""JIT-compatible Gaussian inverse optimization on affine constraints."""

from __future__ import annotations

import math
from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax import Array
from jax.scipy.linalg import solve_triangular
from jax.typing import ArrayLike

from .gaussian_heat import _real_array, gaussian_heat_inverse


class GaussianConstrainedInverseResult(NamedTuple):
    """Numerical constrained optimum and diagnostics, not certificates.

    ``feasibility_residual`` is the computed norm ||Q.T@value-eta||, which
    is physical distance to the affine plane only in exact arithmetic.
    ``residual_bound`` concerns the numerical projected mixture. Neither
    diagnostic encloses transformation error or proves exact feasibility.
    """

    value: Array
    converged: Array
    residual_bound: Array
    contraction_bound: Array
    feasibility_residual: Array
    constraint_reciprocal_condition: Array
    constraints_valid: Array


def gaussian_heat_inverse_constrained(
    target: Array, centers: ArrayLike, biases: ArrayLike, constraint_matrix: ArrayLike, constraint_rhs: ArrayLike,
    viscosity: ArrayLike = 1.0, time: ArrayLike = 1.0, *, max_iterations: int = 128,
    tolerance: float | None = None, adjoint: str = "auto", constraint_rcond: float | None = None,
    feasibility_tolerance: float | None = None,
) -> GaussianConstrainedInverseResult:
    r"""Minimize the Gaussian potential minus target·x under Cx=rhs.

    Target ``(...,d)`` is float32/64. Shared centers ``(m,d)``, biases ``(m,)``,
    C ``(r,d)``, rhs ``(r,)`` and positive scalar viscosity/time are cast to
    its dtype. Require 0<=r<=d and numerical full row rank; redundant rows
    refuse, without claiming infeasibility. All seven inputs remain trainable
    under jit/vmap and higher autodiff. The derivative chains through the
    implicit base inverse and the age/viscosity-dependent normal-energy shift.

    Row equilibration divides C and rhs by each row's largest absolute entry.
    Detached singular values require sigma_min/sigma_max > constraint_rcond,
    default sqrt(dtype epsilon). A reduced QR uses O(d*r+r²+m*d) storage,
    with no dense d-by-d projector. The projected center diameter D_T gives
    q_T=D_T²/(8*nu*time)<1; no contraction assumption on the original diameter
    is needed. Empty constraints delegate to the base inverse. For r=d the
    unique affine point is computed directly. Controls are static metadata.

    ``tolerance`` defaults to 64*dtype epsilon for the numerical inverse
    residual. ``feasibility_tolerance`` defaults to the same physical distance
    threshold. All diagnostics are stopped-gradient and omit floating-point
    transformation error: they are NOT interval certificates or guarantees
    of exact Cx=rhs. Failed entries have NaN values. Invalid arithmetic has
    infinite residual diagnostics. QR preprocessing costs O(d*r²+r³+m*d*r),
    followed by the existing inverse with its selectable adjoint strategy.
    """
    target = jnp.asarray(target)
    if target.dtype not in (jnp.float32, jnp.float64):
        raise TypeError("target must have dtype float32 or float64")
    if target.ndim < 1 or target.shape[-1] < 1:
        raise ValueError("target must have shape (..., positive dimension)")
    if type(max_iterations) is not int:
        raise TypeError("max_iterations must be an integer")
    if max_iterations < 0:
        raise ValueError("max_iterations must be nonnegative")
    if adjoint not in ("auto", "dense", "low_rank"):
        raise ValueError("adjoint must be auto, dense or low_rank")
    epsilon = 2.0**(-23 if target.dtype == jnp.dtype("float32") else -52)
    if tolerance is None:
        tolerance = 64 * epsilon
    if constraint_rcond is None:
        constraint_rcond = math.sqrt(epsilon)
    if feasibility_tolerance is None:
        feasibility_tolerance = tolerance
    for name, option in (("tolerance", tolerance), ("constraint_rcond", constraint_rcond),
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
    if rows == 0:
        base = gaussian_heat_inverse(target, locations, offsets, nu, age, max_iterations=max_iterations,
                                     tolerance=tolerance, adjoint=adjoint)
        return GaussianConstrainedInverseResult(*base, jnp.where(base.converged, jnp.zeros_like(base.residual_bound), jnp.inf),
                                                jnp.ones((), dtype=target.dtype), jnp.asarray(True))

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

    shared = (constraints_valid & jnp.isfinite(locations).all() & jnp.isfinite(offsets).all()
              & jnp.isfinite(nu) & (nu > 0) & jnp.isfinite(age) & (age > 0))
    locations, offsets = jnp.where(jnp.isfinite(locations), locations, 0.0), jnp.where(jnp.isfinite(offsets), offsets, 0.0)
    nu, age = jnp.where(jnp.isfinite(nu) & (nu > 0), nu, 1.0), jnp.where(jnp.isfinite(age) & (age > 0), age, 1.0)
    query_valid = jnp.isfinite(target).all(-1)
    query = jnp.where(jnp.isfinite(target), target, 0.0)
    if rows == dimension:
        zero = jnp.sum(locations * 0) + jnp.sum(offsets * 0) + nu * 0 + age * 0
        value = eta @ normal_basis.T + query * 0 + zero
        root_converged = jnp.ones_like(query_valid)
        residual = jnp.zeros_like(query_valid, dtype=target.dtype)
        contraction = jnp.zeros((), dtype=target.dtype)
    else:
        relative = locations - locations[0]
        normal_coordinates = relative @ normal_basis
        normal_offset = locations[0] @ normal_basis - eta
        anchor = locations[0] - normal_offset @ normal_basis.T
        projected = anchor + (relative - normal_coordinates @ normal_basis.T)
        denominator = 4 * nu * age
        normal_energy = jnp.sum(normal_coordinates**2, axis=-1) + 2 * jnp.sum(normal_coordinates * normal_offset, axis=-1)
        shifted = (offsets - offsets[0]) - normal_energy / denominator
        projected_query = query - (query @ normal_basis) @ normal_basis.T
        shared = (shared & jnp.isfinite(relative).all() & jnp.isfinite(normal_coordinates).all()
                  & jnp.isfinite(normal_offset).all() & jnp.isfinite(projected).all()
                  & jnp.isfinite(denominator) & (denominator > 0) & jnp.isfinite(normal_energy).all()
                  & jnp.isfinite(shifted).all())
        base = gaussian_heat_inverse(projected_query, projected, shifted, nu, age,
                                     max_iterations=max_iterations, tolerance=tolerance, adjoint=adjoint)
        value, root_converged, residual, contraction = base
    finite = jnp.isfinite(value).all(-1)
    normal_residual = value @ normal_basis - eta
    scale = jnp.max(jnp.abs(normal_residual), axis=-1)
    safe_scale = jnp.where(scale > 0, scale, 1.0)
    feasibility = scale * jnp.sqrt(jnp.sum((normal_residual / safe_scale[..., None])**2, axis=-1))
    valid = shared & query_valid & root_converged & finite & jnp.isfinite(feasibility)
    feasibility = jnp.where(valid, feasibility, jnp.inf)
    residual_ok = shared & query_valid & (~root_converged | (finite & jnp.isfinite(feasibility)))
    residual = jnp.where(residual_ok, residual, jnp.inf)
    contraction = jnp.where(shared, contraction, jnp.inf)
    converged = valid & (feasibility <= feasibility_tolerance)
    return GaussianConstrainedInverseResult(
        jnp.where(converged[..., None], value, jnp.nan), jax.lax.stop_gradient(converged),
        jax.lax.stop_gradient(residual), jax.lax.stop_gradient(contraction), jax.lax.stop_gradient(feasibility),
        jax.lax.stop_gradient(condition), jax.lax.stop_gradient(constraints_valid),
    )


__all__ = ["GaussianConstrainedInverseResult", "gaussian_heat_inverse_constrained"]
