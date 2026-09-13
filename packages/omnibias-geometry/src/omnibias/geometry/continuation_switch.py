# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Numerical switching at a supported simple equilibrium branch point.

This is a finite-map predictor/corrector, not a certified bifurcation theorem.
The supplied parent tangent must belong to a known solution branch. Neither
finite derivatives nor a corrected seed establish global branch completeness.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Literal

import numpy as np
from numpy.typing import NDArray
from omnibias.geometry.continuation import ImplicitFamily, branch_tangent

Array = NDArray[np.float64]


@dataclass(frozen=True)
class BranchSwitchResult:
    status: Literal["switched", "inconclusive"]
    seed: Array | None
    tangent: Array | None
    branch_direction: Array | None
    branch_curvature: Array | None
    step: float
    residual: float
    reason: str
    certified: bool = False


def switch_equilibrium_branch(
    family: ImplicitFamily,
    point: Array,
    parent_tangent: Array,
    *,
    step: float = 0.01,
    side: int = 1,
    tol: float = 1e-10,
    rtol: float = 1e-8,
    max_corrector: int = 20,
    max_halvings: int = 8,
) -> BranchSwitchResult:
    """Construct and correct an off-parent seed from F, DF, D²F, and D³F.

    For F:R^(n+1)->R^n, the supported point has rank DF=n-1. Its kernel
    contains the supplied parent tangent t and a transverse direction v.
    The mixed crossing ``left_null @ D²F[t,v]`` must be nonzero. The parent
    tangent must change the final parameter and satisfy the quadratic cone.

    The second direction solves the quadratic cone in span(t,v); a cubic
    solvability equation gives its second derivative. Newton correction fixes
    the transverse coordinate at ``side*step``, scales the critical residual
    by that nonzero coordinate, and stays in a bounded neighborhood. This avoids
    mistaking a tiny unscaled residual near the parent for a switched branch.

    ``side`` selects the sign of v, oriented by its largest coordinate. The
    returned tangent points away from the branch point and can be passed to
    ``continue_branch``. Unsupported ranks, missing derivatives, degenerate
    crossings, and failed correctors return an explicit inconclusive result.
    Supplied callbacks must be mutually consistent derivatives of a C³ family.
    """
    y = np.asarray(point, dtype=float)
    t = np.asarray(parent_tangent, dtype=float)
    if (
        y.ndim != 1
        or len(y) < 2
        or t.shape != y.shape
        or not np.isfinite(y).all()
        or not np.isfinite(t).all()
        or np.linalg.norm(t) == 0
    ):
        raise ValueError("finite point and nonzero matching parent tangent required")
    if (
        step <= 0
        or tol <= 0
        or rtol <= 0
        or not isfinite(step + tol + rtol)
        or side not in (-1, 1)
        or max_corrector < 1
        or max_halvings < 0
    ):
        raise ValueError("positive finite step/tolerances, side +/-1, and valid budgets required")
    n = len(y) - 1
    t = t / np.linalg.norm(t)
    direction: Array | None = None
    curvature: Array | None = None
    residual = float("inf")

    def refuse(reason: str) -> BranchSwitchResult:
        return BranchSwitchResult(
            "inconclusive", None, None, direction, curvature, 0.0, residual, reason
        )

    f = np.asarray(family.value(y), dtype=float)
    j = np.asarray(family.jacobian(y), dtype=float)
    if (
        f.shape != (n,)
        or j.shape != (n, n + 1)
        or not np.isfinite(f).all()
        or not np.isfinite(j).all()
    ):
        raise ValueError("finite residual (n,) and Jacobian (n,n+1) required")
    residual = float(np.linalg.norm(f, np.inf))
    if residual > tol:
        return refuse("point_does_not_solve_residual")
    u, singular, vh = np.linalg.svd(j, full_matrices=True)
    threshold = rtol * max(1.0, float(singular[0]))
    rank = int(np.count_nonzero(singular > threshold))
    if rank != n - 1:
        return refuse("unsupported_rank_or_higher_codimension")
    if np.linalg.norm(j @ t) > threshold or abs(t[-1]) <= rtol:
        return refuse("parent_tangent_not_supported")
    if family.second is None or family.third is None:
        return refuse("second_and_third_derivative_providers_required")
    h = np.asarray(family.second(y), dtype=float)
    third = np.asarray(family.third(y), dtype=float)
    if (
        h.shape != (n, n + 1, n + 1)
        or third.shape != (n, n + 1, n + 1, n + 1)
        or not np.isfinite(h).all()
        or not np.isfinite(third).all()
    ):
        raise ValueError(
            "finite second and third derivative tensors with all parameter axes required"
        )
    kernel = vh[-2:]
    projected = kernel - (kernel @ t)[:, None] * t
    v = projected[int(np.argmax(np.linalg.norm(projected, axis=1)))].copy()
    v /= np.linalg.norm(v)
    if v[int(np.argmax(np.abs(v)))] < 0:
        v = -v
    left = u[:, -1]
    hleft = np.einsum("i,ijk->jk", left, h)
    aa, bb, cc = float(t @ hleft @ t), float(t @ hleft @ v), float(v @ hleft @ v)
    coefficient_tol = rtol * max(1.0, float(np.linalg.norm(hleft)))
    if abs(aa) > coefficient_tol:
        return refuse("parent_tangent_fails_quadratic_cone")
    if abs(bb) <= coefficient_tol:
        return refuse("degenerate_mixed_quadratic_crossing")
    direction = v - cc / (2 * bb) * t
    hdd = np.einsum("ijk,j,k->i", h, direction, direction)
    tddd = np.einsum("ijkl,j,k,l->i", third, direction, direction, direction)
    # J*w=-D²F[d,d], v*w=0, and 3*left*D²F[d,w]+left*D³F[d,d,d]=0.
    range_rows = u[:, : n - 1].T
    matrix = np.vstack((range_rows @ j, v, direction @ hleft))
    rhs = np.concatenate((-range_rows @ hdd, [0.0, -float(left @ tddd) / 3]))
    try:
        curvature = np.linalg.solve(matrix, rhs)
    except np.linalg.LinAlgError:
        return refuse("degenerate_cubic_predictor_system")
    if (
        not np.isfinite(curvature).all()
        or np.linalg.norm(j @ curvature + hdd) > 10 * coefficient_tol
    ):
        return refuse("inconsistent_derivative_predictor")
    for shrink in range(max_halvings + 1):
        amplitude = side * step * 0.5**shrink
        if abs(amplitude) < np.finfo(float).tiny:
            return refuse("step_underflow")
        candidate = y + amplitude * direction + 0.5 * amplitude**2 * curvature
        transform = np.vstack((range_rows, left / abs(amplitude)))
        radius = 4 * abs(amplitude) * max(1.0, float(np.linalg.norm(direction)))

        def augmented(
            z: Array, weighting: Array = transform, coordinate: float = amplitude
        ) -> Array:
            return np.concatenate((weighting @ family.value(z), [float(v @ (z - y) - coordinate)]))

        converged = False
        for _ in range(max_corrector):
            if np.linalg.norm(candidate - y) > radius:
                break
            values = augmented(candidate)
            if not np.isfinite(values).all():
                break
            error = float(np.linalg.norm(values, np.inf))
            if error <= tol:
                converged = True
                break
            tangent_matrix = np.vstack((transform @ family.jacobian(candidate), v))
            try:
                delta = np.linalg.solve(tangent_matrix, -values)
            except np.linalg.LinAlgError:
                break
            for backtrack in range(12):
                trial = candidate + 0.5**backtrack * delta
                trial_error = float(np.linalg.norm(augmented(trial), np.inf))
                if (
                    np.isfinite(trial_error)
                    and trial_error < error
                    and np.linalg.norm(trial - y) <= radius
                ):
                    candidate = trial
                    break
            else:
                break
        if not converged and np.linalg.norm(candidate - y) <= radius:
            final_error = float(np.linalg.norm(augmented(candidate), np.inf))
            converged = np.isfinite(final_error) and final_error <= tol
        residual = float(np.linalg.norm(family.value(candidate), np.inf))
        if (
            converged
            and residual <= tol
            and side * float(v @ (candidate - y)) >= abs(amplitude) / 2
        ):
            try:
                tangent = branch_tangent(family, candidate, side * direction)
            except ValueError:
                continue
            return BranchSwitchResult(
                "switched",
                candidate,
                tangent,
                direction,
                curvature,
                amplitude,
                residual,
                "corrected_transverse_seed",
            )
    return refuse("off_parent_corrector_failed")


__all__ = ["BranchSwitchResult", "switch_equilibrium_branch"]
