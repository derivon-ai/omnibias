# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Fixed-age constrained Gaussian approximations with normal costs resummed."""

from __future__ import annotations

from collections.abc import Sequence
from typing import NamedTuple

from torch import Tensor

from .gaussian_constrained_aging import _evaluate


class GaussianConstrainedApproximationResult(NamedTuple):
    """Numerical fixed-age approximation, auxiliary coefficients and diagnostics.

    Coefficients have shape ``(order+1,...,d)`` with c0=0. They freeze the
    transformed biases in an auxiliary series, and are NOT coefficients of
    the original constrained model's physical-age expansion. The error bound
    excludes floating-point evaluation error; feasibility excludes QR error.
    Diagnostics are detached estimates, not rigorous interval certificates.
    """

    value: Tensor
    correction_coefficients: Tensor
    approximation_error_bound: Tensor
    contraction_bound: Tensor
    feasibility_residual: Tensor
    constraint_reciprocal_condition: Tensor
    valid: Tensor


def gaussian_heat_inverse_constrained_approximation(
    target: Tensor, centers: Tensor | Sequence[Sequence[float]], biases: Tensor | Sequence[float],
    constraint_matrix: Tensor | Sequence[Sequence[float]], constraint_rhs: Tensor | Sequence[float],
    viscosity: Tensor | float = 1.0, time: Tensor | float = 1.0, *, order: int = 8,
    constraint_rcond: float | None = None, feasibility_tolerance: float | None = None,
) -> GaussianConstrainedApproximationResult:
    r"""Approximate the constrained inverse after resumming normal costs.

    At the requested age s, reduced QR projects onto Cx=rhs and transforms
    log weights to ``b_j-b_0-normal_cost_difference_j/(4*nu*s)``. Ordinary
    correction coefficients expand an auxiliary inverse-age variable with
    these transformed biases held fixed. Value sums them at that variable=1.
    Runtime autograd differentiates the whole finite formula, INCLUDING the
    bias shift's age and viscosity dependence. These gradients are neither
    true inverse derivatives nor the auxiliary frozen-bias age derivative.

    For tangent center diameter D_T and q_T=D_T²/(8*nu*s), orders 0/1 require
    q_T<1 with Euclidean truncation estimates D_T*q_T/2 and D_T*q_T². Higher
    orders require q_T<1/3 with estimate D_T/2*(3*q_T)**(order+1)/(1-3*q_T).
    Coincident tangent centers are affine and bypass the cutoff. Nonfinite
    arithmetic still refuses. The original normal-cost range places no
    restriction on these analytical bounds. Returned floating bounds omit
    rounding and projection error and are NOT certificates.

    Target is float32/64 (...,d); shared centers (m,d), biases (m,), C(r,d),
    rhs(r), scalar viscosity/time are cast to its dtype/device. All seven
    inputs support gradients and double backward. Numerical full row rank
    is required after equilibration, with reciprocal-condition threshold
    sqrt(epsilon) by default. Feasibility tolerance defaults to 64*epsilon
    in physical normal coordinates; it does not guarantee exact Cx=rhs.
    Order is a static nonnegative integer. Invalid entries return NaN values,
    coefficients and error estimates, with false validity. After QR there is
    no inverse solve or Jacobian factorization. The order-N recurrence uses
    O((m+d)*N²+m*d*N) work per target, plus geometry preprocessing.
    """
    if type(order) is not int:
        raise TypeError("order must be a nonnegative integer")
    result = _evaluate(target, centers, biases, constraint_matrix, constraint_rhs, viscosity, time,
                       order=order, series=order >= 2, constraint_rcond=constraint_rcond,
                       feasibility_tolerance=feasibility_tolerance, resummed=True)
    return GaussianConstrainedApproximationResult(
        result.value, result.correction_coefficients, result.approximation_error_bound, result.contraction_bound,
        result.feasibility_residual, result.constraint_reciprocal_condition, result.valid,
    )


__all__ = ["GaussianConstrainedApproximationResult", "gaussian_heat_inverse_constrained_approximation"]
