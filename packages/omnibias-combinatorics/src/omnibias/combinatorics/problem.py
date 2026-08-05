# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
r"""Problem containers, the gap-shaped certificate, and the DiscreteProblem adapter.

Backend-agnostic frozen dataclasses (like ``omnibias.routing.RoutingProblem``): each
exposes ``n`` and a :meth:`system` returning the standard-form :class:`PolytopeSystem`
shared by the differentiable relaxation and the certificate. Everything is minimization
(a max-weight matroid problem carries ``c = -weights``); the decoded objective is the
*upper* bound, the LP dual the *lower* bound.

The :class:`AnnealSchedule` beta-homotopy is re-exported from ``omnibias.discrete`` (the
shared substrate). :meth:`AssignmentProblem.to_discrete_problem` returns an
``omnibias.discrete`` ``DiscreteProblem`` view (penalized pseudo-Boolean energy +
``to_polynomial``) so the substrate's ``decode`` / ``brute_force_min`` / ``certify_gap`` /
``anneal_descent`` stay usable on the primal-polytope-native problems.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray
from omnibias.combinatorics._core.matroids import Matroid
from omnibias.combinatorics._core.polytopes import (
    PolytopeSystem,
    assignment_system,
    matroid_system,
    min_cost_flow_system,
    transport_system,
)
from omnibias.discrete import AnnealSchedule

if TYPE_CHECKING:  # pragma: no cover - typing only
    from omnibias.sos import Polynomial

FloatArray = NDArray[np.float64]

_TINY = 1e-12


@dataclass(frozen=True)
class AssignmentProblem:
    r"""A linear assignment (min-cost perfect bipartite matching): an ``n x n`` cost matrix.

    Minimizing ``sum_ij cost[i, j] x[i, j]`` over permutation matrices; the LP relaxation
    is the (integral) Birkhoff polytope, so the LP-dual certificate is tight. Solved
    exactly by the Hungarian algorithm.
    """

    cost: FloatArray
    name: str | None = None

    def __post_init__(self) -> None:
        c = np.asarray(self.cost, dtype=float)
        if c.ndim != 2 or c.shape[0] != c.shape[1]:
            raise ValueError(f"assignment cost must be square (n, n); got {c.shape}")
        object.__setattr__(self, "cost", c)

    @property
    def n(self) -> int:
        return int(self.cost.shape[0])

    def system(self) -> PolytopeSystem:
        return assignment_system(self.cost)

    def to_discrete_problem(self, penalty: float | None = None) -> _AssignmentDiscreteProblem:
        r"""An ``omnibias.discrete`` ``DiscreteProblem`` view over ``{0, 1}^(n^2)``.

        Energy = assignment cost + a quadratic penalty on the row / column-sum equalities,
        so ``omnibias.discrete.decode`` / ``brute_force_min`` / ``certify_gap`` /
        ``anneal_descent`` all apply. ``penalty`` defaults to ``2 * max|cost| * n + 1``,
        large enough that any single constraint violation outweighs the cost range.
        """
        pen = (
            2.0 * float(np.max(np.abs(self.cost))) * self.n + 1.0
            if penalty is None
            else float(penalty)
        )
        return _AssignmentDiscreteProblem(self.cost, pen)


@dataclass(frozen=True)
class TransportProblem:
    r"""A balanced transportation problem: cost ``(m, n)``, ``supply`` ``(m,)``, ``demand`` ``(n,)``.

    Minimizing ``sum_ij cost[i, j] x[i, j]`` with row sums ``= supply`` and column sums
    ``= demand`` (``sum supply == sum demand``). The transportation polytope is integral
    for integral marginals; solved exactly by LP (scipy HiGHS).
    """

    cost: FloatArray
    supply: FloatArray
    demand: FloatArray
    name: str | None = None

    def __post_init__(self) -> None:
        c = np.asarray(self.cost, dtype=float)
        a = np.asarray(self.supply, dtype=float)
        b = np.asarray(self.demand, dtype=float)
        if c.ndim != 2:
            raise ValueError(f"transport cost must be a matrix (m, n); got {c.shape}")
        if a.shape != (c.shape[0],) or b.shape != (c.shape[1],):
            raise ValueError("supply / demand shapes must match the cost matrix")
        if not np.isclose(a.sum(), b.sum()):
            raise ValueError("transport must be balanced: sum(supply) == sum(demand)")
        object.__setattr__(self, "cost", c)
        object.__setattr__(self, "supply", a)
        object.__setattr__(self, "demand", b)

    @property
    def n(self) -> int:
        return int(self.cost.shape[0] * self.cost.shape[1])

    def system(self) -> PolytopeSystem:
        return transport_system(self.cost, self.supply, self.demand)


@dataclass(frozen=True)
class MinCostFlowProblem:
    r"""A min-cost flow: route ``value`` units from ``source`` to ``sink`` at minimum cost.

    ``arcs`` are directed ``(u, v)`` node pairs; ``cost`` / ``capacity`` are per-arc. The
    arc-flow polytope (node conservation + capacities) is integral for integral ``value``
    / ``capacity``; solved exactly by LP. Use :func:`omnibias.combinatorics.max_flow_value`
    to route the maximum feasible amount (min-cost max-flow).
    """

    n_nodes: int
    arcs: tuple[tuple[int, int], ...]
    cost: FloatArray
    capacity: FloatArray
    source: int
    sink: int
    value: float
    name: str | None = None

    def __post_init__(self) -> None:
        arcs = tuple((int(u), int(v)) for (u, v) in self.arcs)
        c = np.asarray(self.cost, dtype=float)
        cap = np.asarray(self.capacity, dtype=float)
        if c.shape != (len(arcs),) or cap.shape != (len(arcs),):
            raise ValueError("cost and capacity must have one entry per arc")
        if not (0 <= self.source < self.n_nodes and 0 <= self.sink < self.n_nodes):
            raise ValueError("source / sink must be valid node indices")
        if self.source == self.sink:
            raise ValueError("source and sink must be distinct")
        if self.value < 0:
            raise ValueError("flow value must be nonnegative")
        object.__setattr__(self, "arcs", arcs)
        object.__setattr__(self, "cost", c)
        object.__setattr__(self, "capacity", cap)

    @property
    def n(self) -> int:
        return len(self.arcs)

    def system(self) -> PolytopeSystem:
        return min_cost_flow_system(
            self.n_nodes, self.arcs, self.cost, self.capacity, self.source, self.sink, self.value
        )


@dataclass(frozen=True)
class MatroidProblem:
    r"""Max-weight independent set of a matroid: element ``weights`` + a :class:`Matroid`.

    Internally minimized as ``c = -weights`` over the (integral) independent-set polytope,
    so the certified min-space objective ``lower <= c^T x <= objective`` corresponds to a
    tight bound on the max weight ``-c^T x``. Solved exactly by greedy.
    """

    weights: FloatArray
    matroid: Matroid
    name: str | None = None

    def __post_init__(self) -> None:
        w = np.asarray(self.weights, dtype=float)
        if w.shape != (self.matroid.ground_size,):
            raise ValueError(
                f"weights must be ({self.matroid.ground_size},); got {w.shape}"
            )
        object.__setattr__(self, "weights", w)

    @property
    def n(self) -> int:
        return int(self.matroid.ground_size)

    def system(self) -> PolytopeSystem:
        return matroid_system(self.weights, self.matroid)


@dataclass(frozen=True)
class CombinatorialCertificate:
    r"""A rigorous optimality-gap certificate for a decoded combinatorial solution.

    Combines a rigorous **lower** bound on the minimum (min-space) objective -- the LP
    dual, sealed by the outward-rounded Neumaier-Shcherbina bound in
    :func:`omnibias.convex.lp_dual_lower_bound` -- with the decoded solution's objective as
    the **upper** bound. The optimum is provably sandwiched
    ``lower_bound <= optimum <= objective``; the gap certifies how close to optimal the
    solution is. There is no exactness field: because these polytopes are **integral** the
    LP relaxation is exact and the gap is *tight* (``~0``), but the certificate reports the
    gap, it never asserts it.

    Attributes
    ----------
    lower_bound:
        Rigorous lower bound on the minimum objective for this instance.
    objective:
        The decoded solution's (min-space) objective -- the certified upper bound.
    polytope:
        Which polytope produced the bound (``"assignment"`` / ``"transport"`` /
        ``"min_cost_flow"`` / ``"matroid"``).
    method:
        ``"lp_dual"`` when interval-sealed, ``"lp_float"`` for the valid-but-unsealed
        float LP value.
    certified:
        ``True`` iff ``lower_bound`` is the outward-rounded interval-sealed bound.
    """

    lower_bound: float
    objective: float
    polytope: str
    method: str
    certified: bool

    @property
    def absolute_gap(self) -> float:
        """Certified absolute optimality gap ``objective - lower_bound`` (``>= 0``)."""
        return self.objective - self.lower_bound

    @property
    def relative_gap(self) -> float:
        """Certified relative gap ``(objective - lower_bound) / max(|lower_bound|, tiny)``."""
        return self.absolute_gap / max(abs(self.lower_bound), _TINY)

    @property
    def is_sound(self) -> bool:
        """Whether the sandwich holds (``lower_bound <= objective`` within rounding)."""
        return self.lower_bound <= self.objective + 1e-9


@dataclass(frozen=True)
class _AssignmentDiscreteProblem:
    r"""``omnibias.discrete.DiscreteProblem`` view of an assignment over ``{0, 1}^(n^2)``.

    Energy is the assignment cost plus a quadratic penalty on the row / column-sum
    equalities; both the batched ``energy`` and ``to_polynomial`` agree on the cube.
    """

    cost: FloatArray
    penalty: float

    @property
    def n(self) -> int:
        return int(self.cost.shape[0] ** 2)

    def energy(self, x: object) -> float | FloatArray:
        side = int(self.cost.shape[0])
        xv = np.asarray(x, dtype=float)
        single = xv.ndim == 1
        matrix = xv.reshape(1, -1) if single else xv
        m = matrix.reshape(matrix.shape[0], side, side)
        base = np.einsum("bij,ij->b", m, self.cost)
        row_res = m.sum(axis=2) - 1.0  # (b, side)
        col_res = m.sum(axis=1) - 1.0
        pen = self.penalty * (np.sum(row_res**2, axis=1) + np.sum(col_res**2, axis=1))
        total = base + pen
        return float(total[0]) if single else total

    def to_polynomial(self) -> Polynomial:
        from omnibias.sos import Polynomial

        side = int(self.cost.shape[0])
        n = side * side

        def var(i: int, j: int) -> Polynomial:
            return Polynomial.variable(i * side + j, n)

        poly = Polynomial.zero(n)
        for i in range(side):
            for j in range(side):
                poly = poly + var(i, j) * float(self.cost[i, j])
        for i in range(side):  # penalty * (sum_j x_ij - 1)^2
            row = Polynomial.constant(-1.0, n)
            for j in range(side):
                row = row + var(i, j)
            poly = poly + (row * row) * self.penalty
        for j in range(side):  # penalty * (sum_i x_ij - 1)^2
            col = Polynomial.constant(-1.0, n)
            for i in range(side):
                col = col + var(i, j)
            poly = poly + (col * col) * self.penalty
        return poly


__all__ = [
    "AnnealSchedule",
    "AssignmentProblem",
    "CombinatorialCertificate",
    "MatroidProblem",
    "MinCostFlowProblem",
    "TransportProblem",
]
