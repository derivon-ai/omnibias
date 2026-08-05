# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
r"""Ergonomic constructors for the sparse-recovery problems from ``(A, b)`` data.

:func:`sparse_least_squares` builds the penalised form (a fixed ``lambda``);
:func:`cardinality_constrained` calibrates ``lambda`` from a target cardinality ``k`` via
the standard "each retained column must earn at least ``lambda``" argument, giving a
``lambda`` whose penalised optimum tends to select about ``k`` columns. Both return a
:class:`~omnibias.discrete.sparse.problem.SupportSelectionProblem` (Fork A, the certified
QUBO surrogate) by default, or the :class:`~omnibias.discrete.sparse.problem.BestSubsetProblem`
(Fork B, continuous coefficients) when ``continuous=True``.
"""

from __future__ import annotations

import numpy as np
from omnibias.discrete.sparse.problem import BestSubsetProblem, SupportSelectionProblem


def sparse_least_squares(
    A: object,
    b: object,
    lam: float,
    *,
    continuous: bool = False,
    name: str | None = None,
) -> SupportSelectionProblem | BestSubsetProblem:
    r"""Sparse least squares with a fixed ``l_0`` penalty ``lambda``.

    Parameters
    ----------
    A, b:
        Design matrix ``(m, n)`` and target ``(m,)``.
    lam:
        Nonnegative cardinality penalty ``lambda`` (larger -> sparser).
    continuous:
        Return the continuous-coefficient :class:`BestSubsetProblem` (Fork B) instead of
        the pseudo-Boolean :class:`SupportSelectionProblem` (Fork A, the default).
    name:
        Optional label.
    """
    if continuous:
        return BestSubsetProblem(A=np.asarray(A, dtype=float), b=np.asarray(b, dtype=float),
                                 lam=float(lam), name=name)
    return SupportSelectionProblem(A=np.asarray(A, dtype=float), b=np.asarray(b, dtype=float),
                                   lam=float(lam), name=name)


def cardinality_constrained(
    A: object,
    b: object,
    k: int,
    *,
    continuous: bool = False,
    name: str | None = None,
) -> SupportSelectionProblem | BestSubsetProblem:
    r"""Sparse least squares calibrated toward a target cardinality ``k`` (approximate).

    A column is worth keeping only if it lowers the residual by more than ``lambda``. The
    per-column marginal gain of the ``j``-th feature at the empty support is
    ``1/2 (A_j^T b)^2 / ||A_j||^2``; taking ``lambda`` between the ``k``-th and
    ``(k+1)``-th largest of these gains biases the penalised optimum toward roughly ``k``
    selected columns. This is a *soft* target (the penalised problem is not a hard
    cardinality constraint); the certificate always reports the true realised gap.

    Parameters
    ----------
    A, b:
        Design matrix ``(m, n)`` and target ``(m,)``.
    k:
        Target support size (``1 <= k <= n``).
    continuous, name:
        See :func:`sparse_least_squares`.
    """
    am = np.asarray(A, dtype=float)
    bv = np.asarray(b, dtype=float).reshape(-1)
    n = am.shape[1]
    if not (1 <= int(k) <= n):
        raise ValueError(f"k must satisfy 1 <= k <= n = {n}, got {k}")
    col_sq = np.sum(am * am, axis=0)
    col_sq = np.where(col_sq > 0.0, col_sq, 1.0)
    gains = 0.5 * (am.T @ bv) ** 2 / col_sq
    order = np.sort(gains)[::-1]
    kk = int(k)
    if kk < n:
        lam = 0.5 * (float(order[kk - 1]) + float(order[kk]))
    else:
        lam = 0.5 * float(order[kk - 1])
    return sparse_least_squares(am, bv, lam, continuous=continuous, name=name)


__all__ = ["cardinality_constrained", "sparse_least_squares"]
