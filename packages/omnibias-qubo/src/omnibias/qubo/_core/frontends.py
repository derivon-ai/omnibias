# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
r"""Canonical combinatorial problems encoded as QUBO instances.

* :func:`max_cut` -- the maximum cut of a weighted graph. With the arc weights ``W``,
  the QUBO energy is ``E(x) = -cut(x)`` where
  ``cut(x) = sum_{i<j} W_ij [x_i != x_j]``; minimizing ``E`` maximizes the cut, so
  ``cut(x) = -problem.energy(x)``. (This is the encoding behind the Goemans-Williamson
  SDP and the natural target for the SOS / Lasserre certified bound.)
* :func:`max_independent_set` -- a maximum independent set. The QUBO
  ``E(x) = -sum_i x_i + penalty * sum_{(i,j) in E} x_i x_j`` rewards chosen vertices and
  penalizes chosen edges; any ``penalty >= 1`` makes the minimum a valid independent set
  (``penalty = 2`` by default), and ``|set| = -problem.energy(x)`` there.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from omnibias.qubo.problem import QUBOProblem

FloatArray = NDArray[np.float64]


def _symmetric_zero_diag(matrix: object, *, what: str) -> FloatArray:
    m = np.asarray(matrix, dtype=float)
    if m.ndim != 2 or m.shape[0] != m.shape[1]:
        raise ValueError(f"{what} must be a square (n, n) matrix, got shape {m.shape}")
    if m.shape[0] < 1:
        raise ValueError(f"{what} must have at least one vertex")
    sym: FloatArray = 0.5 * (m + m.T)
    np.fill_diagonal(sym, 0.0)
    return sym


def max_cut(weights: object, *, name: str | None = None) -> QUBOProblem:
    r"""A ``QUBOProblem`` whose energy is ``-cut(x)`` for the weighted graph ``weights``.

    ``weights`` is a symmetric ``(n, n)`` (nonnegative) adjacency / weight matrix; the
    diagonal is ignored. The decoded point's cut value is ``-problem.energy(x)``, and
    the certified lower bound on the energy is a certified *upper* bound on the max cut.
    """
    w = _symmetric_zero_diag(weights, what="weights")
    degree = w.sum(axis=1)
    return QUBOProblem(Q=w, c=-degree, const=0.0, name=name if name is not None else "max_cut")


def max_independent_set(
    adjacency: object, *, penalty: float = 2.0, name: str | None = None
) -> QUBOProblem:
    r"""A ``QUBOProblem`` whose minimum is a maximum independent set of ``adjacency``.

    ``adjacency`` is a symmetric ``(n, n)`` 0/1 matrix; ``penalty`` (``>= 1``, default
    ``2``) is the per-edge conflict weight that keeps the optimum edge-free. The chosen
    set is ``{i : x_i = 1}`` and its size is ``-problem.energy(x)`` at a valid solution.
    """
    if penalty < 1.0:
        raise ValueError("penalty must be >= 1 so the minimum is a valid independent set")
    a = _symmetric_zero_diag(adjacency, what="adjacency")
    q = 0.5 * penalty * a
    c = -np.ones(a.shape[0])
    return QUBOProblem(Q=q, c=c, const=0.0, name=name if name is not None else "max_independent_set")


__all__ = [
    "max_cut",
    "max_independent_set",
]
