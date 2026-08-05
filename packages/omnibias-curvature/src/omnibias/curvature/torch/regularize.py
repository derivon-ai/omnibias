# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""The ``eps -> 0`` rank / regularization collapse (PyTorch differentiable register).

The bit-identical PyTorch twin of :mod:`omnibias.curvature.regularize`. Tikhonov
solving collapses onto the Moore-Penrose / minimum-norm solution as the damping
vanishes, ``(A + eps I)^{-1} b -> A^+ b``; naively taking ``eps -> 0`` blows up
whenever ``A`` is rank-deficient. This module takes that limit *stably* and pairs
it with the rigorous conditioning certificate in
:mod:`omnibias.core.verified.conditioning`:

* :func:`regularized_solve` -- ``(A + eps I)^{-1} b``.
* :func:`min_norm_solve` -- the collapse limit done right (eigen-truncated ``A^+ b``).
* :func:`regularization_path` -- the measured homotopy ``x(eps)`` over a grid.
* :func:`rank_collapse` -- certified-damping (target condition number) or min-norm
  entry point, attaching a sealed conditioning certificate.

The certified damping and the sealed certificate are produced by the shared
pure-Python core verifier, so for the same matrix they are **bit-identical** to
the JAX twin (the numerical solves match to a calibrated tolerance).

Collapse honesty
----------------
This ``eps -> 0`` collapse is a **distinct** limit from the founding multi-bias
``delta -> 0`` derivative collapse and from the ``beta -> inf`` feasibility
penalty of the discrete-optimization packages -- same spirit, different
parameter, never conflated. Nothing here rides the ``sigma`` derivative tower; the
solves are ordinary numerical (LAPACK-class) operations and only the conditioning
enclosure is verified.

Requires the optional ``torch`` extra (``pip install "omnibias-curvature[torch]"``);
it is not imported by the JAX-only top-level :mod:`omnibias.curvature`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast

import torch
from omnibias.core.verified.conditioning import (
    certified_damping,
    conditioning_certificate,
)
from torch import Tensor


def _validate_square_system(matrix: Tensor, rhs: Tensor) -> int:
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"matrix must be square (P, P), got {tuple(matrix.shape)}")
    if rhs.ndim != 1 or rhs.shape[0] != matrix.shape[0]:
        raise ValueError(
            f"rhs must be (P,) with P = {matrix.shape[0]}, got {tuple(rhs.shape)}"
        )
    return int(matrix.shape[0])


def _default_rcond(matrix: Tensor) -> float:
    """Numerical-rank cutoff ``max(shape) * eps(dtype)`` (numpy-``pinv``-style)."""
    eps = float(torch.finfo(matrix.dtype).eps)
    return float(max(matrix.shape)) * eps


def _to_float_rows(matrix: Tensor) -> list[list[float]]:
    """Detach a torch matrix to nested Python floats for the pure-numpy core verifier."""
    return cast("list[list[float]]", matrix.detach().to(torch.float64).cpu().tolist())


def regularized_solve(matrix: Tensor, rhs: Tensor, *, eps: float = 1e-3) -> Tensor:
    r"""The Tikhonov-regularized solve ``(A + eps I)^{-1} b`` (see the JAX twin)."""
    p = _validate_square_system(matrix, rhs)
    if eps < 0.0:
        raise ValueError(f"eps must be >= 0, got {eps}")
    damped = matrix + eps * torch.eye(p, dtype=matrix.dtype, device=matrix.device)
    out: Tensor = torch.linalg.solve(damped, rhs)
    return out


def min_norm_solve(matrix: Tensor, rhs: Tensor, *, rcond: float | None = None) -> Tensor:
    r"""The ``eps -> 0`` limit done right: the minimum-norm / pseudoinverse solution.

    Computes ``A^+ b`` via a symmetric eigendecomposition with small-eigenvalue
    truncation, so the limit is taken *stably* instead of letting
    ``(A + eps I)^{-1} b`` blow up. ``matrix`` must be symmetric.
    """
    _validate_square_system(matrix, rhs)
    w, vecs = torch.linalg.eigh(matrix)
    rc = _default_rcond(matrix) if rcond is None else float(rcond)
    cutoff = rc * torch.max(torch.abs(w))
    keep = torch.abs(w) > cutoff
    safe_w = torch.where(keep, w, torch.ones_like(w))
    inv_w = torch.where(keep, 1.0 / safe_w, torch.zeros_like(w))
    out: Tensor = vecs @ (inv_w * (vecs.T @ rhs))
    return out


def numerical_rank(matrix: Tensor, *, rcond: float | None = None) -> int:
    r"""The numerical rank of a symmetric ``matrix`` at the ``rcond`` cutoff."""
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"matrix must be square (P, P), got {tuple(matrix.shape)}")
    w = torch.linalg.eigvalsh(matrix)
    rc = _default_rcond(matrix) if rcond is None else float(rcond)
    cutoff = rc * torch.max(torch.abs(w))
    return int(torch.sum(torch.abs(w) > cutoff))


def regularization_path(matrix: Tensor, rhs: Tensor, eps_grid: Tensor) -> Tensor:
    r"""Stacked Tikhonov solutions ``x(eps)`` over ``eps_grid`` (the collapse homotopy)."""
    p = _validate_square_system(matrix, rhs)
    grid = torch.as_tensor(eps_grid)
    if grid.ndim != 1:
        raise ValueError(f"eps_grid must be 1-D, got shape {tuple(grid.shape)}")
    eye = torch.eye(p, dtype=matrix.dtype, device=matrix.device)
    rows = [torch.linalg.solve(matrix + float(e) * eye, rhs) for e in grid]
    return torch.stack(rows)


@dataclass(frozen=True)
class CollapseResult:
    """The outcome of a rank / regularization collapse (torch twin of the JAX result)."""

    solution: Tensor
    effective_rank: int
    eps: float
    certificate: dict[str, Any] | None


def rank_collapse(
    matrix: Tensor,
    rhs: Tensor,
    *,
    target_condition: float | None = None,
    rcond: float | None = None,
    certify: bool = True,
) -> CollapseResult:
    r"""High-level ``eps -> 0`` collapse: min-norm limit or certified-damped solve.

    ``target_condition is None`` takes the collapse limit with :func:`min_norm_solve`
    (``eps = 0``); otherwise the smallest certified damping meeting the target
    condition number is used. When ``certify`` is set the result carries a sealed
    :func:`~omnibias.core.verified.conditioning.conditioning_certificate` -- identical
    to the JAX twin's for the same matrix (both come from the shared core verifier).
    ``theorem_prover_verified`` is not asserted here.
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
