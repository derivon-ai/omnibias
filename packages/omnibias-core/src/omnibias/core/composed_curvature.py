# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Composed-curvature joint Newton algebra (theory 08-02).

The order-2 chain rule ``(f circ g)'' = f''(g) (g')^2 + f'(g) g''`` is
the exact coupling between consecutive layers. A Newton step on the
joint block ``(W_{ell-1}, W_ell)`` can leave a critical point that is a
minimum of the current-layer slice and a saddle of the pair.

This module is backend-free: tanh derivatives come from
:func:`omnibias.core.polynomials.tanh_polynomial_coeffs` (the same
tower the backends evaluate), and the small dense eigen / Newton solve
is shared so torch and jax take the same subspace step on the same
floats. Tensor HVPs live in ``omnibias.{torch,jax}.optim_composed``.

Jets come from the founding bias collapse (``delta -> 0``). No
temperature collapse appears. Escape is from a *slice* critical point,
not a global min of a deep nest, and not CCF stretch.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

from omnibias.core.polynomials import tanh_polynomial_coeffs

Matrix = tuple[tuple[float, ...], ...]
Vector = tuple[float, ...]

_SLICE_PD_FLOOR = 1e-12
_JACOBI_TOL = 1e-15
_JACOBI_SWEEPS = 64
_GE_PIVOT = 1e-18


def _horner(coeffs: Sequence[float], x: float) -> float:
    acc = 0.0
    for c in reversed(coeffs):
        acc = acc * x + float(c)
    return acc


def eval_tanh_derivative(z: float, n: int) -> float:
    """``tanh^(n)(z)`` from the shared Legendre-style coefficients."""
    t = math.tanh(z)
    return _horner(tanh_polynomial_coeffs(n), t)


def reject_full_parameter_jacobian(
    n_directions: int,
    n_params: int,
    allow_full: bool,
) -> None:
    """G4: refuse a full-parameter Jacobian unless ``allow_full`` is set."""
    if n_directions < 1:
        raise ValueError(f"n_directions must be >= 1, got {n_directions}")
    if n_params < 2:
        raise ValueError(
            f"composed curvature needs at least two parameters (prev + curr), "
            f"got n_params={n_params}"
        )
    if n_directions >= n_params and not allow_full:
        raise ValueError(
            f"n_directions={n_directions} >= n_params={n_params}; "
            "pass allow_full=True to opt into a full-block Jacobian "
            "(the API rejects silent P x P materialisation)"
        )


@dataclass(frozen=True)
class ComposedCurvatureConfig:
    """Subspace budget and Newton / escape knobs (theory 08-02)."""

    n_directions: int = 4
    include_residual_hess: bool = True
    escape_tol: float = 0.0
    damping: float = 1e-6
    allow_full: bool = False

    def __post_init__(self) -> None:
        if self.n_directions < 1:
            raise ValueError(f"n_directions must be >= 1, got {self.n_directions}")
        if self.escape_tol < 0.0 or not math.isfinite(self.escape_tol):
            raise ValueError(
                f"escape_tol must be a finite number >= 0, got {self.escape_tol}"
            )
        if self.damping < 0.0 or not math.isfinite(self.damping):
            raise ValueError(f"damping must be a finite number >= 0, got {self.damping}")


@dataclass(frozen=True)
class ComposedCurvatureReport:
    """Diagnostics of one joint / escape step."""

    lambda_min_slice: float
    lambda_min_joint: float
    escaped: bool
    step_norm: float


@dataclass(frozen=True)
class SubspaceStep:
    """Newton or negative-curvature coefficients in the ``k``-direction basis."""

    coeffs: tuple[float, ...]
    lambda_min_slice: float
    lambda_min_joint: float
    escaped: bool


def chain_rule_mse_blocks(
    h: float,
    h_w: float,
    h_ww: float,
    v: float,
) -> tuple[float, float, float]:
    r"""Hessian blocks of ``L = 1/2 (v h - 1)^2`` from the order-2 chain rule.

    Spec §5 writes ``(v h - 1)^2``; the half is the PINN convention
    ``1/2 ||r||^2`` and does not change eigenvalue *signs*. Returns
    ``(H_ww, H_wv, H_vv)``.
    """
    residual = v * h - 1.0
    h_vv = h * h
    h_wv = h_w * (v * h + residual)
    h_ww = (v * h_w) ** 2 + residual * v * h_ww
    return (h_ww, h_wv, h_vv)


def scalar_nest_hessian(
    w: float,
    v: float,
    x: float = 1.0,
) -> tuple[float, float, float]:
    """Closed-form Hessian of the spec §5 scalar nest at ``(w, v)``.

    ``g(w) = tanh(w x)``, ``L = 1/2 (v g - 1)^2``. Derivatives of ``tanh``
    use :func:`eval_tanh_derivative`.
    """
    z = w * x
    h = eval_tanh_derivative(z, 0)
    h_w = x * eval_tanh_derivative(z, 1)
    h_ww = (x * x) * eval_tanh_derivative(z, 2)
    return chain_rule_mse_blocks(h, h_w, h_ww, v)


def symmetrize(matrix: Sequence[Sequence[float]]) -> Matrix:
    """``(H + H^T) / 2`` as nested tuples."""
    n = len(matrix)
    if n == 0:
        return ()
    if any(len(row) != n for row in matrix):
        raise ValueError("matrix must be square")
    out: list[tuple[float, ...]] = []
    for i in range(n):
        row = []
        for j in range(n):
            row.append(0.5 * (float(matrix[i][j]) + float(matrix[j][i])))
        out.append(tuple(row))
    return tuple(out)


def eigh_symmetric(
    matrix: Sequence[Sequence[float]],
    *,
    tol: float = _JACOBI_TOL,
    max_sweeps: int = _JACOBI_SWEEPS,
) -> tuple[tuple[float, ...], Matrix]:
    """Jacobi eigen-decomposition of a small symmetric matrix.

    Returns eigenvalues (ascending) and a matching column-orthonormal
    eigenvector matrix ``V`` (``V[i][j]`` is row ``i``, column ``j``).
    """
    h = symmetrize(matrix)
    n = len(h)
    if n == 0:
        return (), ()
    a = [list(row) for row in h]
    v = [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]
    scale = max((abs(a[i][j]) for i in range(n) for j in range(n)), default=1.0)
    thresh = tol * (1.0 + scale)
    for _ in range(max_sweeps):
        p, q, off = 0, 1, 0.0
        for i in range(n):
            for j in range(i + 1, n):
                mag = abs(a[i][j])
                if mag > off:
                    off = mag
                    p, q = i, j
        if off <= thresh:
            break
        app = a[p][p]
        aqq = a[q][q]
        apq = a[p][q]
        theta = 0.5 * (aqq - app) / apq
        t = math.copysign(1.0, theta) / (abs(theta) + math.sqrt(1.0 + theta * theta))
        c = 1.0 / math.sqrt(1.0 + t * t)
        s = t * c
        tau_cs = s / (1.0 + c)
        for k in range(n):
            if k == p or k == q:
                continue
            akp = a[k][p]
            akq = a[k][q]
            a[k][p] = akp - s * (akq + tau_cs * akp)
            a[p][k] = a[k][p]
            a[k][q] = akq + s * (akp - tau_cs * akq)
            a[q][k] = a[k][q]
        a[p][p] = app - t * apq
        a[q][q] = aqq + t * apq
        a[p][q] = 0.0
        a[q][p] = 0.0
        for k in range(n):
            vkp = v[k][p]
            vkq = v[k][q]
            v[k][p] = c * vkp - s * vkq
            v[k][q] = s * vkp + c * vkq
    evals = [a[i][i] for i in range(n)]
    order = sorted(range(n), key=lambda i: evals[i])
    sorted_evals = tuple(evals[i] for i in order)
    sorted_vecs = tuple(tuple(v[row][order[col]] for col in range(n)) for row in range(n))
    return sorted_evals, sorted_vecs


def solve_dense(
    matrix: Sequence[Sequence[float]],
    rhs: Sequence[float],
    *,
    damping: float = 0.0,
) -> Vector:
    """Gaussian elimination with partial pivoting on ``(H + damping I) x = b``."""
    n = len(rhs)
    if len(matrix) != n or any(len(row) != n for row in matrix):
        raise ValueError("matrix and rhs length must match a square system")
    a = [
        [float(matrix[i][j]) + (damping if i == j else 0.0) for j in range(n)]
        for i in range(n)
    ]
    b = [float(x) for x in rhs]
    for k in range(n):
        pivot = k
        best = abs(a[k][k])
        for i in range(k + 1, n):
            mag = abs(a[i][k])
            if mag > best:
                best = mag
                pivot = i
        if best < _GE_PIVOT:
            a[k][k] = _GE_PIVOT if a[k][k] >= 0.0 else -_GE_PIVOT
            pivot = k
        if pivot != k:
            a[k], a[pivot] = a[pivot], a[k]
            b[k], b[pivot] = b[pivot], b[k]
        diag = a[k][k]
        for i in range(k + 1, n):
            factor = a[i][k] / diag
            for j in range(k, n):
                a[i][j] -= factor * a[k][j]
            b[i] -= factor * b[k]
    x = [0.0] * n
    for i in range(n - 1, -1, -1):
        acc = b[i]
        for j in range(i + 1, n):
            acc -= a[i][j] * x[j]
        x[i] = acc / a[i][i]
    return tuple(x)


def select_composed_step(
    h_slice: Sequence[Sequence[float]],
    h_joint: Sequence[Sequence[float]],
    g_joint: Sequence[float],
    config: ComposedCurvatureConfig | None = None,
) -> SubspaceStep:
    """Damped joint Newton, or a unit negative-curvature direction if escaped.

    The driver applies the subspace coefficients through ``Q`` and may
    then pick the *length* with spec 03-12. A negative-curvature
    quadratic has no interior minimum; the coefficients here are a unit
    descent eigenvector, not an unbounded Newton step.
    """
    cfg = config if config is not None else ComposedCurvatureConfig()
    slice_evals, _ = eigh_symmetric(h_slice)
    joint_evals, joint_vecs = eigh_symmetric(h_joint)
    if not slice_evals or not joint_evals:
        raise ValueError("H_slice and H_joint must be non-empty")
    if len(g_joint) != len(joint_evals):
        raise ValueError(
            f"g_joint length {len(g_joint)} != joint size {len(joint_evals)}"
        )
    lam_s = float(slice_evals[0])
    lam_j = float(joint_evals[0])
    slice_pd = lam_s >= -_SLICE_PD_FLOOR
    escaped = bool(slice_pd and lam_j < -cfg.escape_tol)
    k = len(joint_evals)
    if escaped:
        vec = [float(joint_vecs[i][0]) for i in range(k)]
        dot = sum(vec[i] * float(g_joint[i]) for i in range(k))
        if dot > 0.0:
            vec = [-x for x in vec]
        nrm = math.sqrt(sum(x * x for x in vec))
        coeffs = tuple(x / nrm for x in vec) if nrm > 0.0 else tuple(0.0 for _ in vec)
    else:
        neg_g = tuple(-float(g) for g in g_joint)
        coeffs = solve_dense(h_joint, neg_g, damping=cfg.damping)
    return SubspaceStep(
        coeffs=coeffs,
        lambda_min_slice=lam_s,
        lambda_min_joint=lam_j,
        escaped=escaped,
    )


__all__ = [
    "ComposedCurvatureConfig",
    "ComposedCurvatureReport",
    "SubspaceStep",
    "chain_rule_mse_blocks",
    "eigh_symmetric",
    "eval_tanh_derivative",
    "reject_full_parameter_jacobian",
    "scalar_nest_hessian",
    "select_composed_step",
    "solve_dense",
    "symmetrize",
]
