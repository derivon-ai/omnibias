# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
r"""Standard-form LP systems for the matching / flow / matroid polytopes.

Each builder returns a :class:`PolytopeSystem` -- the min-space objective ``c``, the
equalities ``A_eq x = b_eq``, the non-box inequalities ``A_ineq x <= b_ineq``, and the
variable box ``x_lower <= x <= x_upper`` -- the single description shared by the
differentiable relaxation and the ``lp_dual_lower_bound`` certificate. Every polytope
here is **integral** (its vertices are the integer feasible points), so the LP relaxation
is exact and the LP-dual lower bound is tight:

* :func:`assignment_system` -- Birkhoff polytope (row / column sums ``= 1``);
* :func:`transport_system` -- transportation polytope (row sums ``= supply``, column
  sums ``= demand``);
* :func:`min_cost_flow_system` -- arc-flow polytope (node conservation, arc capacities);
* :func:`matroid_system` -- the independent-set polytope's rank inequalities.

Everything is minimization; a max-weight matroid problem uses ``c = -weights``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:  # pragma: no cover - typing only
    from omnibias.combinatorics._core.matroids import Matroid

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class PolytopeSystem:
    r"""Standard-form LP data for an (integral) combinatorial polytope.

    Attributes
    ----------
    name:
        Polytope label (``"assignment"`` / ``"transport"`` / ``"min_cost_flow"`` /
        ``"matroid"``).
    n_vars:
        Number of decision variables.
    c:
        The minimization objective ``c`` (a max-weight problem stores ``-weights``).
    A_eq, b_eq:
        Equality block ``A_eq x = b_eq`` (may have zero rows).
    A_ineq, b_ineq:
        Non-box inequality block ``A_ineq x <= b_ineq`` (may have zero rows; the box is
        carried separately in ``x_lower`` / ``x_upper``).
    x_lower, x_upper:
        Finite variable box containing the feasible set (required by the LP certificate).
    shape:
        Optional ``(rows, cols)`` for matrix-structured problems (assignment / transport),
        so the flat variable vector can be reshaped back to a matrix.
    """

    name: str
    n_vars: int
    c: FloatArray
    A_eq: FloatArray
    b_eq: FloatArray
    A_ineq: FloatArray
    b_ineq: FloatArray
    x_lower: FloatArray
    x_upper: FloatArray
    shape: tuple[int, int] | None = None


def _empty_rows(n_vars: int) -> tuple[FloatArray, FloatArray]:
    return np.zeros((0, n_vars), dtype=float), np.zeros((0,), dtype=float)


def assignment_system(cost: FloatArray) -> PolytopeSystem:
    r"""Birkhoff polytope for an ``n x n`` assignment cost (row / column sums ``= 1``)."""
    c = np.asarray(cost, dtype=float)
    if c.ndim != 2 or c.shape[0] != c.shape[1]:
        raise ValueError(f"assignment cost must be square (n, n); got {c.shape}")
    n = int(c.shape[0])
    n_vars = n * n

    def idx(i: int, j: int) -> int:
        return i * n + j

    rows: list[FloatArray] = []
    rhs: list[float] = []
    for i in range(n):  # row sums = 1
        row = np.zeros(n_vars)
        for j in range(n):
            row[idx(i, j)] = 1.0
        rows.append(row)
        rhs.append(1.0)
    for j in range(n):  # column sums = 1
        row = np.zeros(n_vars)
        for i in range(n):
            row[idx(i, j)] = 1.0
        rows.append(row)
        rhs.append(1.0)

    a_ineq, b_ineq = _empty_rows(n_vars)
    return PolytopeSystem(
        name="assignment",
        n_vars=n_vars,
        c=c.reshape(-1).copy(),
        A_eq=np.asarray(rows, dtype=float),
        b_eq=np.asarray(rhs, dtype=float),
        A_ineq=a_ineq,
        b_ineq=b_ineq,
        x_lower=np.zeros(n_vars),
        x_upper=np.ones(n_vars),
        shape=(n, n),
    )


def transport_system(cost: FloatArray, supply: FloatArray, demand: FloatArray) -> PolytopeSystem:
    r"""Transportation polytope: row sums ``= supply``, column sums ``= demand``."""
    c = np.asarray(cost, dtype=float)
    a = np.asarray(supply, dtype=float)
    b = np.asarray(demand, dtype=float)
    if c.ndim != 2:
        raise ValueError(f"transport cost must be a matrix (m, n); got {c.shape}")
    m, n = int(c.shape[0]), int(c.shape[1])
    if a.shape != (m,) or b.shape != (n,):
        raise ValueError(f"supply must be ({m},) and demand ({n},); got {a.shape}, {b.shape}")
    if not np.isclose(a.sum(), b.sum()):
        raise ValueError("transport problem must be balanced: sum(supply) == sum(demand)")
    if np.any(a < 0) or np.any(b < 0):
        raise ValueError("supply and demand must be nonnegative")
    n_vars = m * n

    def idx(i: int, j: int) -> int:
        return i * n + j

    rows: list[FloatArray] = []
    rhs: list[float] = []
    for i in range(m):  # row (supply) marginals
        row = np.zeros(n_vars)
        for j in range(n):
            row[idx(i, j)] = 1.0
        rows.append(row)
        rhs.append(float(a[i]))
    for j in range(n):  # column (demand) marginals
        row = np.zeros(n_vars)
        for i in range(m):
            row[idx(i, j)] = 1.0
        rows.append(row)
        rhs.append(float(b[j]))

    x_upper = np.zeros(n_vars)
    for i in range(m):
        for j in range(n):
            x_upper[idx(i, j)] = min(float(a[i]), float(b[j]))

    a_ineq, b_ineq = _empty_rows(n_vars)
    return PolytopeSystem(
        name="transport",
        n_vars=n_vars,
        c=c.reshape(-1).copy(),
        A_eq=np.asarray(rows, dtype=float),
        b_eq=np.asarray(rhs, dtype=float),
        A_ineq=a_ineq,
        b_ineq=b_ineq,
        x_lower=np.zeros(n_vars),
        x_upper=x_upper,
        shape=(m, n),
    )


def min_cost_flow_system(
    n_nodes: int,
    arcs: tuple[tuple[int, int], ...],
    cost: FloatArray,
    capacity: FloatArray,
    source: int,
    sink: int,
    value: float,
) -> PolytopeSystem:
    r"""Arc-flow polytope: node conservation with net supply ``value`` at ``source``.

    Sends ``value`` units from ``source`` to ``sink`` at minimum cost. The node balance is
    ``+value`` at the source, ``-value`` at the sink, ``0`` elsewhere; arc flows lie in
    ``[0, capacity]``. Integral whenever ``value`` and ``capacity`` are integral.
    """
    c = np.asarray(cost, dtype=float)
    cap = np.asarray(capacity, dtype=float)
    e = len(arcs)
    if c.shape != (e,) or cap.shape != (e,):
        raise ValueError(f"cost and capacity must both be ({e},); got {c.shape}, {cap.shape}")
    if not (0 <= source < n_nodes and 0 <= sink < n_nodes) or source == sink:
        raise ValueError("source and sink must be distinct valid node indices")

    balance = np.zeros(n_nodes)
    balance[source] = float(value)
    balance[sink] = -float(value)

    # Conservation: out-flow(v) - in-flow(v) = balance[v] for every node v.
    rows = np.zeros((n_nodes, e), dtype=float)
    for arc_i, (u, v) in enumerate(arcs):
        rows[u, arc_i] += 1.0  # leaves u
        rows[v, arc_i] -= 1.0  # enters v

    a_ineq, b_ineq = _empty_rows(e)
    return PolytopeSystem(
        name="min_cost_flow",
        n_vars=e,
        c=c.copy(),
        A_eq=rows,
        b_eq=balance,
        A_ineq=a_ineq,
        b_ineq=b_ineq,
        x_lower=np.zeros(e),
        x_upper=cap.copy(),
        shape=None,
    )


def matroid_system(weights: FloatArray, matroid: Matroid) -> PolytopeSystem:
    r"""Independent-set polytope for a max-weight matroid problem (``c = -weights``).

    Minimizing ``c^T x = -weights^T x`` over the (integral) rank-inequality polytope is
    max-weight independent set; greedy solves it exactly.
    """
    w = np.asarray(weights, dtype=float)
    n = matroid.ground_size
    if w.shape != (n,):
        raise ValueError(f"weights must be ({n},) for this matroid; got {w.shape}")
    a_ineq, b_ineq = matroid.polytope_constraints()
    a_ineq = np.asarray(a_ineq, dtype=float).reshape(-1, n)
    b_ineq = np.asarray(b_ineq, dtype=float).reshape(-1)
    eq_a, eq_b = _empty_rows(n)
    return PolytopeSystem(
        name="matroid",
        n_vars=n,
        c=(-w).copy(),
        A_eq=eq_a,
        b_eq=eq_b,
        A_ineq=a_ineq,
        b_ineq=b_ineq,
        x_lower=np.zeros(n),
        x_upper=np.ones(n),
        shape=None,
    )


__all__ = [
    "PolytopeSystem",
    "assignment_system",
    "matroid_system",
    "min_cost_flow_system",
    "transport_system",
]
