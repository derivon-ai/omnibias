# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""The ``eps -> 0`` rank / regularization collapse (JAX differentiable register).

Tikhonov-regularized solving collapses onto the Moore-Penrose / minimum-norm
solution as the damping vanishes:

.. math:: x_\eps = (A + \eps I)^{-1} b \;\xrightarrow[\eps \to 0]{}\; A^{+} b.

Taken naively the ``eps -> 0`` limit **blows up** whenever ``A`` is
rank-deficient or ill-conditioned. This module does the limit *stably* and pairs
it with the rigorous conditioning certificate in
:mod:`omnibias.core.verified.conditioning`:

* :func:`regularized_solve` -- ``(A + eps I)^{-1} b`` (the generic damped solve
  that :func:`omnibias.curvature.damped_solve` and
  :func:`omnibias.curvature.mse_newton_step` now delegate to).
* :func:`min_norm_solve` -- the collapse limit done right: a symmetric
  eigendecomposition with small-eigenvalue truncation yields ``A^+ b`` with no
  blow-up.
* :func:`regularization_path` -- the measured homotopy ``x(eps)`` over a grid
  (the data-driven curve behind an ``eps`` / ``rcond`` choice).
* :func:`rank_collapse` -- the high-level entry: pick a certified damping for a
  target condition number (or take the min-norm limit), solve, and attach a
  sealed :func:`~omnibias.core.verified.conditioning.conditioning_certificate`.

Collapse honesty
----------------
This ``eps -> 0`` collapse is a **distinct** limit from the founding multi-bias
``delta -> 0`` derivative collapse (:mod:`omnibias.torch.unit`) and from the
``beta -> inf`` feasibility penalty used in the discrete-optimization packages --
same spirit (a parameter driven to a limit that recovers a canonical object),
different parameter, never conflated. Nothing here rides the ``sigma`` derivative
tower: the omnibias value is the *stable collapse* plus the *rigorous
certificate*. The solves themselves (``jnp.linalg.solve`` / ``eigh``) are ordinary
numerical (LAPACK-class) operations; only the conditioning enclosure is verified.
The input matrix may of course come from omnibias closed-form curvature
(:func:`omnibias.curvature.mse_gauss_newton_fisher`,
:func:`omnibias.curvature.glm_fisher.glm_fisher`,
:func:`omnibias.curvature.mse_loss_hessian`).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from omnibias.core.verified.conditioning import (
    certified_damping,
    conditioning_certificate,
)


def _validate_square_system(matrix: Array, rhs: Array) -> int:
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"matrix must be square (P, P), got {tuple(matrix.shape)}")
    if rhs.ndim != 1 or rhs.shape[0] != matrix.shape[0]:
        raise ValueError(
            f"rhs must be (P,) with P = {matrix.shape[0]}, got {tuple(rhs.shape)}"
        )
    return int(matrix.shape[0])


def _default_rcond(matrix: Array) -> float:
    """Numerical-rank cutoff ``max(shape) * eps(dtype)`` (numpy-``pinv``-style)."""
    eps = float(np.finfo(matrix.dtype).eps)
    return float(max(matrix.shape)) * eps


def _to_float_rows(matrix: Array) -> list[list[float]]:
    """Detach a JAX matrix to nested Python floats for the pure-numpy core verifier."""
    return cast("list[list[float]]", np.asarray(matrix, dtype=np.float64).tolist())


def regularized_solve(matrix: Array, rhs: Array, *, eps: float = 1e-3) -> Array:
    r"""The Tikhonov-regularized solve ``(A + eps I)^{-1} b``.

    The generic damped solve at the heart of the ``eps -> 0`` collapse: for
    ``eps > 0`` it is well-posed even when ``A`` is singular. ``matrix`` is a
    ``(P, P)`` symmetric matrix and ``rhs`` a ``(P,)`` vector.
    :func:`omnibias.curvature.damped_solve` and
    :func:`omnibias.curvature.mse_newton_step` delegate here, so they are
    bit-identical to this routine at the same damping.
    """
    p = _validate_square_system(matrix, rhs)
    if eps < 0.0:
        raise ValueError(f"eps must be >= 0, got {eps}")
    damped = matrix + eps * jnp.eye(p, dtype=matrix.dtype)
    out: Array = jnp.linalg.solve(damped, rhs)
    return out


def min_norm_solve(matrix: Array, rhs: Array, *, rcond: float | None = None) -> Array:
    r"""The ``eps -> 0`` limit done right: the minimum-norm / pseudoinverse solution.

    Computes ``A^+ b`` via a symmetric eigendecomposition with small-eigenvalue
    truncation (eigenvalues with ``|lambda| <= rcond * max|lambda|`` are treated as
    zero and dropped from the inverse), so the limit is taken *stably* instead of
    letting ``(A + eps I)^{-1} b`` blow up. ``matrix`` must be symmetric. ``rcond``
    defaults to ``max(P) * eps(dtype)``.
    """
    _validate_square_system(matrix, rhs)
    w, vecs = jnp.linalg.eigh(matrix)
    rc = _default_rcond(matrix) if rcond is None else float(rcond)
    cutoff = rc * jnp.max(jnp.abs(w))
    keep = jnp.abs(w) > cutoff
    safe_w = jnp.where(keep, w, 1.0)
    inv_w = jnp.where(keep, 1.0 / safe_w, 0.0)
    out: Array = vecs @ (inv_w * (vecs.T @ rhs))
    return out


def numerical_rank(matrix: Array, *, rcond: float | None = None) -> int:
    r"""The numerical rank of a symmetric ``matrix`` at the ``rcond`` cutoff."""
    _validate_square_system(matrix, matrix[:, 0])
    w = jnp.linalg.eigvalsh(matrix)
    rc = _default_rcond(matrix) if rcond is None else float(rcond)
    cutoff = rc * jnp.max(jnp.abs(w))
    return int(jnp.sum(jnp.abs(w) > cutoff))


def regularization_path(matrix: Array, rhs: Array, eps_grid: Array) -> Array:
    r"""Stacked Tikhonov solutions ``x(eps)`` over ``eps_grid`` (the collapse homotopy).

    Returns a ``(len(eps_grid), P)`` array whose ``k``-th row is
    ``(A + eps_grid[k] I)^{-1} b`` -- the measured curve that a data-driven ``eps``
    or ``rcond`` choice is read off from.
    """
    p = _validate_square_system(matrix, rhs)
    grid = jnp.asarray(eps_grid)
    if grid.ndim != 1:
        raise ValueError(f"eps_grid must be 1-D, got shape {tuple(grid.shape)}")
    eye = jnp.eye(p, dtype=matrix.dtype)

    def solve_one(e: Array) -> Array:
        row: Array = jnp.linalg.solve(matrix + e * eye, rhs)
        return row

    out: Array = jax.vmap(solve_one)(grid)
    return out


@dataclass(frozen=True)
class CollapseResult:
    """The outcome of a rank / regularization collapse.

    Attributes
    ----------
    solution:
        The solved direction ``x`` (min-norm ``A^+ b`` or certified-damped
        ``(A + eps I)^{-1} b``).
    effective_rank:
        The numerical rank of ``A`` at the ``rcond`` cutoff.
    eps:
        The damping actually used (``0.0`` on the pure min-norm path).
    certificate:
        The sealed conditioning certificate, or ``None`` when ``certify=False``.
    """

    solution: Array
    effective_rank: int
    eps: float
    certificate: dict[str, Any] | None


def rank_collapse(
    matrix: Array,
    rhs: Array,
    *,
    target_condition: float | None = None,
    rcond: float | None = None,
    certify: bool = True,
) -> CollapseResult:
    r"""High-level ``eps -> 0`` collapse: min-norm limit or certified-damped solve.

    Two modes:

    * ``target_condition is None`` -- take the collapse limit directly with
      :func:`min_norm_solve` (``eps = 0``, ``A^+ b``).
    * ``target_condition`` set -- pick the smallest damping ``eps`` that provably
      brings ``kappa(A + eps I)`` at or below the target (via
      :func:`omnibias.core.verified.conditioning.certified_damping`) and solve
      ``(A + eps I)^{-1} b``.

    When ``certify`` is set the result carries a sealed
    :func:`~omnibias.core.verified.conditioning.conditioning_certificate`
    (``lambda_min`` / ``lambda_max`` / ``kappa`` enclosures + the ``lambda_min > 0``
    pivot data). ``theorem_prover_verified`` is *not* asserted here -- it is earned
    only by driving the sealed certificate through the Lean bridge.
    """
    _validate_square_system(matrix, rhs)
    if target_condition is None:
        eps = 0.0
        solution = min_norm_solve(matrix, rhs, rcond=rcond)
    else:
        rows = _to_float_rows(matrix)
        eps = certified_damping(rows, target_condition=float(target_condition))
        solution = regularized_solve(matrix, rhs, eps=eps)
    rank = numerical_rank(matrix, rcond=rcond)
    certificate: dict[str, Any] | None = None
    if certify:
        rows = _to_float_rows(matrix)
        certificate = conditioning_certificate(
            rows,
            target_condition=target_condition,
            eps=(eps if target_condition is not None else None),
        )
    return CollapseResult(
        solution=solution, effective_rank=rank, eps=float(eps), certificate=certificate
    )


__all__ = [
    "CollapseResult",
    "min_norm_solve",
    "numerical_rank",
    "rank_collapse",
    "regularization_path",
    "regularized_solve",
]
