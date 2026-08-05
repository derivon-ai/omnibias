# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
r"""Decoders, exact classical baselines, and the small-instance brute-force oracle (numpy).

Three roles, all returning ``(x, objective)`` with ``x`` the flat variable vector (a
polytope vertex) and ``objective`` its min-space value:

* :func:`decode` -- round a (relaxed) heatmap to a feasible vertex: the *upper* bound
  used to train through the layer. It follows the heatmap, so a poor (untrained) heatmap
  yields a worse-but-valid vertex.
* :func:`classical_optimum` -- the exact, poly-time classical baseline (Hungarian /
  scipy-HiGHS LP / greedy). These problems are in **P**, so this is a legitimate exact
  solver and the best-in-class baseline (not a P = NP claim).
* :func:`brute_force_min` -- the exponential vertex-enumeration oracle (permutations /
  independent sets), used only to self-check the certificate sandwich on small instances.

:func:`solve_lp` returns the exact LP primal and (sign-consistent) duals; it is shared
with the certificate.
"""

from __future__ import annotations

from itertools import permutations
from typing import TYPE_CHECKING, Any, TypeAlias

import numpy as np
from numpy.typing import NDArray
from omnibias.combinatorics._core.matroids import independent_sets
from omnibias.combinatorics._core.polytopes import PolytopeSystem

if TYPE_CHECKING:  # pragma: no cover - typing only
    from omnibias.combinatorics.problem import (
        AssignmentProblem,
        MatroidProblem,
        MinCostFlowProblem,
        TransportProblem,
    )

    Problem: TypeAlias = AssignmentProblem | TransportProblem | MinCostFlowProblem | MatroidProblem

FloatArray = NDArray[np.float64]

_MAX_BRUTE_PERM_N = 8  # n! permutations
_MAX_BRUTE_GROUND = 18  # 2^ground_size independent sets


class DecodeError(RuntimeError):
    """Raised when a linear program cannot be solved for a decode / baseline."""


def solve_lp(system: PolytopeSystem) -> tuple[FloatArray, float, FloatArray, FloatArray]:
    r"""Exact LP primal ``x``, objective, and duals ``(lambda >= 0, nu)`` (scipy HiGHS).

    Returns multipliers in the Lagrangian sign convention used by
    :func:`omnibias.convex.lp_dual_lower_bound` (``lambda >= 0`` for ``A_ineq x <= b``),
    which is the negation of HiGHS' ``d(obj)/d(rhs)`` marginals.
    """
    try:
        from scipy.optimize import linprog
    except ImportError as exc:  # pragma: no cover - scipy is a core dependency
        raise DecodeError("scipy is required for the exact LP solve: pip install scipy") from exc

    a_ub = system.A_ineq if system.A_ineq.shape[0] else None
    b_ub = system.b_ineq if system.b_ineq.shape[0] else None
    a_eq = system.A_eq if system.A_eq.shape[0] else None
    b_eq = system.b_eq if system.b_eq.shape[0] else None
    bounds = list(zip(system.x_lower.tolist(), system.x_upper.tolist(), strict=True))
    res = linprog(system.c, A_ub=a_ub, b_ub=b_ub, A_eq=a_eq, b_eq=b_eq, bounds=bounds, method="highs")
    if not res.success:
        raise DecodeError(f"LP solve failed: {res.message}")
    lam = (
        -np.asarray(res.ineqlin.marginals, dtype=float)
        if a_ub is not None
        else np.zeros(0, dtype=float)
    )
    nu = (
        -np.asarray(res.eqlin.marginals, dtype=float)
        if a_eq is not None
        else np.zeros(0, dtype=float)
    )
    return np.asarray(res.x, dtype=float), float(res.fun), lam, nu


def _assignment_from_matrix(scores: FloatArray, cost: FloatArray) -> tuple[FloatArray, float]:
    """Hungarian assignment maximizing ``scores``; returns (flat 0/1 matrix, true cost)."""
    from scipy.optimize import linear_sum_assignment

    n = cost.shape[0]
    row, col = linear_sum_assignment(-np.asarray(scores, dtype=float))
    x = np.zeros((n, n), dtype=float)
    x[row, col] = 1.0
    objective = float(cost[row, col].sum())
    return x.reshape(-1), objective


def _greedy_matroid(order_scores: FloatArray, problem: MatroidProblem) -> tuple[FloatArray, float]:
    """Greedy max-weight independent set, scanning elements by ``order_scores`` desc."""
    w = np.asarray(problem.weights, dtype=float)
    matroid = problem.matroid
    chosen: set[int] = set()
    for e in np.argsort(-np.asarray(order_scores, dtype=float)):
        e = int(e)
        if w[e] <= 0.0:
            continue
        if matroid.is_independent(frozenset(chosen | {e})):
            chosen.add(e)
    x = np.zeros(matroid.ground_size, dtype=float)
    for e in chosen:
        x[e] = 1.0
    return x, float(-(w @ x))


def classical_optimum(problem: Problem) -> tuple[FloatArray, float]:
    r"""The exact poly-time optimum (best-in-class baseline) as ``(x, objective)``.

    Hungarian for assignment, scipy-HiGHS LP for transport / flow, greedy for matroids.
    """
    from omnibias.combinatorics.problem import (
        AssignmentProblem,
        MatroidProblem,
        MinCostFlowProblem,
        TransportProblem,
    )

    if isinstance(problem, AssignmentProblem):
        return _assignment_from_matrix(-problem.cost, problem.cost)
    if isinstance(problem, MatroidProblem):
        return _greedy_matroid(problem.weights, problem)
    if isinstance(problem, TransportProblem | MinCostFlowProblem):
        x, obj, _, _ = solve_lp(problem.system())
        return x, obj
    raise TypeError(f"unknown problem type {type(problem).__name__}")


def decode(problem: Problem, relaxed: object | None = None) -> tuple[FloatArray, float]:
    r"""Round a (relaxed) heatmap to a feasible vertex -- the certified *upper* bound.

    For assignment / matroid the decode *follows* ``relaxed`` (so a poor heatmap gives a
    worse-but-valid vertex, and training through the relaxation improves it). Transport /
    flow decode to the exact LP vertex (integral; these are in P). With ``relaxed=None``
    the decode falls back to the true objective.
    """
    from omnibias.combinatorics.problem import (
        AssignmentProblem,
        MatroidProblem,
        MinCostFlowProblem,
        TransportProblem,
    )

    if isinstance(problem, AssignmentProblem):
        if relaxed is None:
            return _assignment_from_matrix(-problem.cost, problem.cost)
        scores = np.asarray(relaxed, dtype=float).reshape(problem.n, problem.n)
        # Follow the heatmap, but break its (near-)degenerate ties with the true cost so a
        # fractional Birkhoff face rounds to its cheapest vertex, not an index-order one.
        tie = problem.cost / (float(np.mean(np.abs(problem.cost))) + 1e-12)
        return _assignment_from_matrix(scores - 1e-6 * tie, problem.cost)
    if isinstance(problem, MatroidProblem):
        order = problem.weights if relaxed is None else np.asarray(relaxed, dtype=float).reshape(-1)
        return _greedy_matroid(order, problem)
    if isinstance(problem, TransportProblem | MinCostFlowProblem):
        x, obj, _, _ = solve_lp(problem.system())
        return x, obj
    raise TypeError(f"unknown problem type {type(problem).__name__}")


def brute_force_min(problem: Problem) -> tuple[FloatArray, float]:
    r"""Exact optimum by enumerating feasible vertices -- **exponential**, small instances only.

    Enumerates ``n!`` permutations (assignment) or ``2^ground_size`` independent sets
    (matroid). Raises for transport / flow (their integral vertices are not enumerated --
    use :func:`classical_optimum`, exact in poly time as these problems are in P).
    """
    from omnibias.combinatorics.problem import (
        AssignmentProblem,
        MatroidProblem,
    )

    if isinstance(problem, AssignmentProblem):
        n = problem.n
        if n > _MAX_BRUTE_PERM_N:
            raise ValueError(
                f"brute_force_min enumerates n! permutations; n={n} exceeds the "
                f"{_MAX_BRUTE_PERM_N} cap. Use classical_optimum (Hungarian, exact)."
            )
        cost = problem.cost
        best_perm: tuple[int, ...] | None = None
        best = np.inf
        for perm in permutations(range(n)):
            val = float(sum(cost[i, perm[i]] for i in range(n)))
            if val < best:
                best, best_perm = val, perm
        assert best_perm is not None
        x = np.zeros((n, n), dtype=float)
        for i, j in enumerate(best_perm):
            x[i, j] = 1.0
        return x.reshape(-1), best
    if isinstance(problem, MatroidProblem):
        g = problem.matroid.ground_size
        if g > _MAX_BRUTE_GROUND:
            raise ValueError(
                f"brute_force_min enumerates 2^ground_size sets; ground_size={g} exceeds the "
                f"{_MAX_BRUTE_GROUND} cap. Use classical_optimum (greedy, exact)."
            )
        w = np.asarray(problem.weights, dtype=float)
        best_set: frozenset[int] = frozenset()
        best = 0.0  # the empty independent set has objective 0
        for s in independent_sets(problem.matroid):
            val = float(-sum(w[e] for e in s))
            if val < best:
                best, best_set = val, s
        x = np.zeros(g, dtype=float)
        for e in best_set:
            x[e] = 1.0
        return x, best
    raise ValueError(
        f"brute_force_min does not enumerate {type(problem).__name__} vertices; "
        "use classical_optimum (exact LP / max-flow, poly-time)."
    )


def max_flow_value(
    n_nodes: int,
    arcs: tuple[tuple[int, int], ...],
    capacity: Any,
    source: int,
    sink: int,
) -> float:
    r"""Classical maximum ``source -> sink`` throughput via scipy ``csgraph`` (integer caps).

    The exact max-flow value; use it to route the maximum feasible amount in a
    :class:`~omnibias.combinatorics.problem.MinCostFlowProblem` (min-cost max-flow), and as
    the classical baseline the arc-flow LP must match.
    """
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import maximum_flow

    cap = np.asarray(capacity, dtype=float)
    if not np.allclose(cap, np.round(cap)):
        raise ValueError("scipy maximum_flow needs integer capacities; got non-integer values")
    data = np.round(cap).astype(np.int64)
    rows = np.array([u for (u, _) in arcs], dtype=np.int64)
    cols = np.array([v for (_, v) in arcs], dtype=np.int64)
    graph = csr_matrix((data, (rows, cols)), shape=(n_nodes, n_nodes))
    result = maximum_flow(graph, int(source), int(sink))
    return float(result.flow_value)


__all__ = [
    "DecodeError",
    "brute_force_min",
    "classical_optimum",
    "decode",
    "max_flow_value",
    "solve_lp",
]
