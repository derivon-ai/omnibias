# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Fixed-age constrained Gaussian approximations with normal costs resummed."""

from __future__ import annotations

from typing import NamedTuple

from jax import Array
from jax.typing import ArrayLike

from .gaussian_constrained_aging import _evaluate


class GaussianConstrainedApproximationResult(NamedTuple):
    """Finite approximation, auxiliary coefficients and numerical diagnostics.

    Coefficients have shape (order+1,...,d), with c0=0, and freeze transformed
    biases only in an auxiliary series. They are NOT physical-age coefficients
    of the original model. Error and feasibility diagnostics omit floating
    evaluation/projection error and are not rigorous interval certificates.
    """

    value: Array
    correction_coefficients: Array
    approximation_error_bound: Array
    contraction_bound: Array
    feasibility_residual: Array
    constraint_reciprocal_condition: Array
    valid: Array


def gaussian_heat_inverse_constrained_approximation(
    target: Array, centers: ArrayLike, biases: ArrayLike, constraint_matrix: ArrayLike, constraint_rhs: ArrayLike,
    viscosity: ArrayLike = 1.0, time: ArrayLike = 1.0, *, order: int = 8,
    constraint_rcond: float | None = None, feasibility_tolerance: float | None = None,
) -> GaussianConstrainedApproximationResult:
    r"""Approximate the constrained inverse at one age with normal costs resummed.

    Reduced QR projects onto Cx=rhs. The runtime transformed biases are
    ``b_j-b_0-normal_cost_difference_j/(4*nu*time)``. Correction coefficients
    use an auxiliary inverse-age variable holding those transformed biases
    fixed; value sums the series at one. Runtime AD includes the live bias
    shift and supports all seven parameters, higher derivatives, jit/vmap.
    It differentiates a finite approximation, not the true inverse. In
    particular its age derivative differs from a frozen-bias series derivative.

    With tangent diameter D_T and q_T=D_T²/(8*nu*time), orders 0/1 require
    q_T<1 and return Euclidean truncation estimates D_T*q_T/2 and D_T*q_T².
    Higher orders require q_T<1/3 and use D_T/2*(3*q_T)**(order+1)/(1-3*q_T).
    Coincident tangent centers bypass the cutoff as an affine case, while
    nonfinite arithmetic refuses. The original normal-cost range does not
    enter these bounds. Floating diagnostics exclude rounding/projection error
    and are NOT exact-feasibility, error or derivative certificates.

    Target is float32/64 (...,d). Shared centers (m,d), biases (m,), C(r,d),
    rhs(r) and positive scalar viscosity/time are cast to its dtype. Require
    numerical full row rank after equilibration; constraint_rcond defaults
    to sqrt(epsilon), feasibility_tolerance to 64*epsilon in physical normal
    coordinates. Order is a static nonnegative integer. Failed entries return
    NaN values, coefficients and errors, with false validity. Diagnostics are
    stopped-gradient. No inverse solve or Jacobian factorization follows QR;
    N orders cost O((m+d)*N²+m*d*N) per target after geometry preprocessing.
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
