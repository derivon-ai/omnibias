# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Trainable Gaussian inverse optimization on affine equality constraints."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import NamedTuple

import torch
from torch import Tensor

from .gaussian_heat import _norm, _tensor, gaussian_heat_inverse


class GaussianConstrainedInverseResult(NamedTuple):
    """Numerical constrained optimum and diagnostics, not certificates.

    Feasibility is the computed norm ``||Q.T@value-eta||``. In exact
    arithmetic this is distance to the affine plane; floating-point QR and
    projection error are not enclosed. The residual bound concerns the
    numerical projected mixture, not the exact original constraints.
    """

    value: Tensor
    converged: Tensor
    residual_bound: Tensor
    contraction_bound: Tensor
    feasibility_residual: Tensor
    constraint_reciprocal_condition: Tensor
    constraints_valid: Tensor


def gaussian_heat_inverse_constrained(
    target: Tensor, centers: Tensor | Sequence[Sequence[float]], biases: Tensor | Sequence[float],
    constraint_matrix: Tensor | Sequence[Sequence[float]], constraint_rhs: Tensor | Sequence[float],
    viscosity: Tensor | float = 1.0, time: Tensor | float = 1.0, *, max_iterations: int = 128,
    tolerance: float | None = None, adjoint: str = "auto", constraint_rcond: float | None = None,
    feasibility_tolerance: float | None = None,
) -> GaussianConstrainedInverseResult:
    r"""Minimize the Gaussian potential minus target·x subject to Cx=rhs.

    Target shape is ``(...,d)`` with float32/64 dtype. Shared centers ``(m,d)``,
    biases ``(m,)``, constraints ``(r,d)``, rhs ``(r,)`` and positive scalar
    viscosity/time are converted to its dtype/device without detaching gradients.
    Require 0<=r<=d and numerical full row rank. Redundant constraints refuse;
    this is not a proof of infeasibility. All seven inputs support backward
    and double backward on accepted points. Derivatives use the implicit base
    inverse, including the age/viscosity-dependent normal-energy bias shift.

    Each constraint row and rhs are divided by the row's largest absolute
    entry. A detached SVD requires sigma_min/sigma_max > constraint_rcond,
    default sqrt(dtype epsilon). A reduced QR then projects through r-dimensional
    coordinates, without a d-by-d projector. The projected center diameter D_T
    gives q_T=D_T²/(8*nu*time), which must be <1; the original diameter may be
    larger. ``adjoint`` controls the unchanged base inverse's linear solve.
    Empty constraints delegate to it; r=d uses the unique affine point directly.

    ``tolerance`` defaults to 64*dtype epsilon and controls the base inverse's
    numerical physical residual estimate. ``feasibility_tolerance`` defaults
    to the same value and bounds the computed normal-coordinate residual.
    These diagnostics omit floating-point transformation error and are NOT
    rigorous bounds for exact Cx=rhs. Diagnostics are stopped-gradient. Failed
    entries have NaN values; invalid arithmetic has infinite residuals. Fixed
    preprocessing costs O(d*r²+r³+m*d*r) and O(d*r+r²+m*d) storage, plus the
    base inverse. No iteration tape is retained for the implicit inverse.
    """
    if not isinstance(target, Tensor) or target.dtype not in (torch.float32, torch.float64):
        raise TypeError("target must be a float32 or float64 Tensor")
    if target.ndim < 1 or target.shape[-1] < 1:
        raise ValueError("target must have shape (..., positive dimension)")
    if type(max_iterations) is not int:
        raise TypeError("max_iterations must be an integer")
    if max_iterations < 0:
        raise ValueError("max_iterations must be nonnegative")
    if adjoint not in ("auto", "dense", "low_rank"):
        raise ValueError("adjoint must be auto, dense or low_rank")
    epsilon = torch.finfo(target.dtype).eps
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
    if rows == 0:
        base = gaussian_heat_inverse(target, locations, offsets, nu, age, max_iterations=max_iterations,
                                     tolerance=tolerance, adjoint=adjoint)
        return GaussianConstrainedInverseResult(*base, torch.where(base.converged, torch.zeros_like(base.residual_bound), float("inf")),
                                                target.new_ones(()), torch.ones((), dtype=torch.bool, device=target.device))

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

    shared = (constraints_valid & torch.isfinite(locations).all() & torch.isfinite(offsets).all()
              & torch.isfinite(nu) & (nu > 0) & torch.isfinite(age) & (age > 0))
    locations = torch.where(torch.isfinite(locations), locations, torch.zeros_like(locations))
    offsets = torch.where(torch.isfinite(offsets), offsets, torch.zeros_like(offsets))
    nu = torch.where(torch.isfinite(nu) & (nu > 0), nu, torch.ones_like(nu))
    age = torch.where(torch.isfinite(age) & (age > 0), age, torch.ones_like(age))
    query_valid = torch.isfinite(target).all(-1)
    query = torch.where(torch.isfinite(target), target, torch.zeros_like(target))
    if rows == dimension:
        # Keep zero derivatives connected to every live model/target input.
        zero = (locations * 0).sum() + (offsets * 0).sum() + nu * 0 + age * 0
        value = eta @ normal_basis.mT + query * 0 + zero
        root_converged = torch.ones_like(query_valid)
        residual = torch.zeros_like(query_valid, dtype=target.dtype)
        contraction = target.new_zeros(())
    else:
        relative = locations - locations[0]
        normal_coordinates = relative @ normal_basis
        normal_offset = locations[0] @ normal_basis - eta
        anchor = locations[0] - normal_offset @ normal_basis.mT
        projected = anchor + (relative - normal_coordinates @ normal_basis.mT)
        denominator = 4 * nu * age
        normal_energy = normal_coordinates.square().sum(-1) + 2 * (normal_coordinates * normal_offset).sum(-1)
        shifted = (offsets - offsets[0]) - normal_energy / denominator
        projected_query = query - (query @ normal_basis) @ normal_basis.mT
        shared = (shared & torch.isfinite(relative).all() & torch.isfinite(normal_coordinates).all()
                  & torch.isfinite(normal_offset).all() & torch.isfinite(projected).all()
                  & torch.isfinite(denominator) & (denominator > 0) & torch.isfinite(normal_energy).all()
                  & torch.isfinite(shifted).all())
        base = gaussian_heat_inverse(projected_query, projected, shifted, nu, age,
                                     max_iterations=max_iterations, tolerance=tolerance, adjoint=adjoint)
        value, root_converged, residual, contraction = base
    with torch.no_grad():
        finite = torch.isfinite(value).all(-1)
        normal_residual = value @ normal_basis - eta
        feasibility = _norm(normal_residual)
        valid = shared & query_valid & root_converged & finite & torch.isfinite(feasibility)
        feasibility = torch.where(valid, feasibility, torch.full_like(feasibility, float("inf")))
        residual_ok = shared & query_valid & (~root_converged | (finite & torch.isfinite(feasibility)))
        residual = torch.where(residual_ok, residual, torch.full_like(residual, float("inf")))
        contraction = torch.where(shared, contraction, torch.full_like(contraction, float("inf")))
        converged = valid & (feasibility <= feasibility_tolerance)
    return GaussianConstrainedInverseResult(
        torch.where(converged[..., None], value, torch.full_like(value, float("nan"))), converged,
        residual, contraction, feasibility, condition, constraints_valid.detach(),
    )


__all__ = ["GaussianConstrainedInverseResult", "gaussian_heat_inverse_constrained"]
