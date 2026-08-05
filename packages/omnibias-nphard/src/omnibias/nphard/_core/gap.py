# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
r"""Generalized Assignment Problem (GAP) as a QUBO-form ``DiscreteProblem``.

GAP assigns each of ``T`` tasks to exactly one of ``A`` agents to minimise
``sum_{a,t} cost[a,t] x[a,t]`` subject to per-agent capacity
``sum_t resource[a,t] x[a,t] <= capacity[a]``. It is **NP-hard** (it contains bin
packing / multiple-knapsack). We encode it over ``A*T`` assignment bits plus per-agent
**binary slack bits** ``s`` (the standard QUBO inequality encoding, turning
``used <= cap`` into the equality ``used + slack = cap`` with ``slack in {0..cap}``
represented in a bounded binary weight set), minimising over ``x in {0, 1}^n``

.. math::
    E = \sum_{a,t} c_{at} x_{at}
      + \lambda_{\mathrm{one}} \sum_t \bigl(\textstyle\sum_a x_{at} - 1\bigr)^2
      + \lambda_{\mathrm{cap}} \sum_a \bigl(\textstyle\sum_t r_{at} x_{at}
                                    + \textstyle\sum_k w_{ak} s_{ak} - b_a\bigr)^2 .

A capacity-feasible assignment (with slack set to ``cap - used``) has zero penalty, so
its energy equals the linear assignment cost. Resources / capacities must be
non-negative integers for the exact slack encoding.
"""

from __future__ import annotations

import itertools
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray
from omnibias.qubo import QUBOProblem

if TYPE_CHECKING:  # pragma: no cover - typing only
    from omnibias.sos import Polynomial

FloatArray = NDArray[np.float64]


def slack_weights(capacity: int) -> list[int]:
    r"""Bounded binary weights representing every integer in ``[0, capacity]`` with no gaps.

    Uses the classic ``1, 2, 4, ..., 2^{m-1}, capacity - (2^m - 1)`` trick, so
    ``sum`` of any bit subset ranges over ``0..capacity`` contiguously.
    """
    cap = int(capacity)
    if cap <= 0:
        return []
    weights: list[int] = []
    k = 0
    while (1 << (k + 1)) - 1 <= cap:
        weights.append(1 << k)
        k += 1
    remainder = cap - ((1 << k) - 1)
    if remainder > 0:
        weights.append(remainder)
    return weights


def encode_slack(value: int, weights: list[int]) -> list[int]:
    """Greedy (largest-first) bit assignment realising ``value`` in a bounded weight set."""
    bits = [0] * len(weights)
    remaining = int(value)
    for idx in sorted(range(len(weights)), key=lambda i: -weights[i]):
        if weights[idx] <= remaining:
            bits[idx] = 1
            remaining -= weights[idx]
    return bits


@dataclass(frozen=True)
class GAPProblem:
    r"""A GAP instance encoded as a QUBO over assignment + binary slack bits.

    Attributes
    ----------
    cost:
        ``(A, T)`` assignment cost ``cost[a, t]``.
    resource:
        ``(A, T)`` non-negative integer resource use ``resource[a, t]``.
    capacity:
        ``(A,)`` non-negative integer agent capacities.
    penalty_one:
        The task one-hot penalty ``lambda_one``.
    penalty_cap:
        The capacity-equality penalty ``lambda_cap``.
    name:
        Optional label.
    """

    cost: FloatArray
    resource: FloatArray
    capacity: FloatArray
    penalty_one: float
    penalty_cap: float
    name: str | None = None

    def __post_init__(self) -> None:
        cost = np.asarray(self.cost, dtype=float)
        resource = np.asarray(self.resource, dtype=float)
        capacity = np.asarray(self.capacity, dtype=float).reshape(-1)
        if cost.ndim != 2:
            raise ValueError(f"cost must be (A, T), got {cost.shape}")
        if resource.shape != cost.shape:
            raise ValueError(f"resource {resource.shape} must match cost {cost.shape}")
        if capacity.shape[0] != cost.shape[0]:
            raise ValueError(f"capacity must have length A={cost.shape[0]}, got {capacity.shape[0]}")
        if np.any(resource < 0) or np.any(capacity < 0):
            raise ValueError("resource and capacity must be non-negative")
        if not np.allclose(resource, np.round(resource)) or not np.allclose(
            capacity, np.round(capacity)
        ):
            raise ValueError("resource / capacity must be integer-valued for the slack encoding")
        object.__setattr__(self, "cost", cost)
        object.__setattr__(self, "resource", resource)
        object.__setattr__(self, "capacity", capacity)
        object.__setattr__(self, "penalty_one", float(self.penalty_one))
        object.__setattr__(self, "penalty_cap", float(self.penalty_cap))

    @property
    def n_agents(self) -> int:
        return int(self.cost.shape[0])

    @property
    def n_tasks(self) -> int:
        return int(self.cost.shape[1])

    def _slack(self) -> tuple[list[list[int]], list[int]]:
        """Per-agent slack weights and their start offsets in the variable vector."""
        cache = self.__dict__.get("_slack_cache")
        if cache is None:
            weights = [slack_weights(int(c)) for c in self.capacity]
            offsets: list[int] = []
            off = self.n_agents * self.n_tasks
            for w in weights:
                offsets.append(off)
                off += len(w)
            cache = (weights, offsets)
            object.__setattr__(self, "_slack_cache", cache)
        slack: tuple[list[list[int]], list[int]] = cache
        return slack

    @property
    def n(self) -> int:
        """The number of binary variables (assignment + slack bits)."""
        weights, _ = self._slack()
        return self.n_agents * self.n_tasks + sum(len(w) for w in weights)

    def _arrays(self) -> tuple[FloatArray, FloatArray, float]:
        cache = self.__dict__.get("_qubo_arrays")
        if cache is None:
            cache = self._build_arrays()
            object.__setattr__(self, "_qubo_arrays", cache)
        arrays: tuple[FloatArray, FloatArray, float] = cache
        return arrays

    def _build_arrays(self) -> tuple[FloatArray, FloatArray, float]:
        A, T = self.n_agents, self.n_tasks
        weights, offsets = self._slack()
        n = self.n
        q = np.zeros((n, n))
        c = np.zeros(n)
        const = 0.0
        # linear assignment objective
        for a in range(A):
            for t in range(T):
                c[a * T + t] += self.cost[a, t]
        # task one-hot penalty: lambda_one * sum_t (sum_a x[a,t] - 1)^2
        for t in range(T):
            u = np.zeros(n)
            for a in range(A):
                u[a * T + t] = 1.0
            q += self.penalty_one * np.outer(u, u)
            c += self.penalty_one * (-2.0) * u
            const += self.penalty_one
        # capacity penalty: lambda_cap * sum_a (r_a . x + w_a . s - cap_a)^2
        for a in range(A):
            g = np.zeros(n)
            for t in range(T):
                g[a * T + t] = self.resource[a, t]
            for k, w in enumerate(weights[a]):
                g[offsets[a] + k] = float(w)
            cap = float(self.capacity[a])
            q += self.penalty_cap * np.outer(g, g)
            c += self.penalty_cap * (-2.0) * cap * g
            const += self.penalty_cap * cap * cap
        return q, c, const

    def energy(self, x: object) -> float | FloatArray:
        r"""The QUBO energy (assignment cost + one-hot + capacity penalties)."""
        q, c, const = self._arrays()
        xv = np.asarray(x, dtype=float)
        quad = np.sum((xv @ q) * xv, axis=-1)
        lin = xv @ c
        out = quad + lin + const
        return float(out) if xv.ndim == 1 else out

    def assignment_cost(self, assignment: Sequence[int] | NDArray[np.intp]) -> float:
        r"""The linear cost ``sum_t cost[assignment[t], t]`` of a task->agent assignment."""
        return float(sum(self.cost[int(a), t] for t, a in enumerate(assignment)))

    def is_feasible(self, assignment: Sequence[int] | NDArray[np.intp]) -> bool:
        r"""Whether a task->agent assignment respects every agent capacity."""
        used = np.zeros(self.n_agents)
        for t, a in enumerate(assignment):
            used[int(a)] += self.resource[int(a), t]
        return bool(np.all(used <= self.capacity + 1e-9))

    def full_x(self, assignment: Sequence[int] | NDArray[np.intp]) -> FloatArray:
        r"""Full binary vector (assignment + slack bits solving ``slack = cap - used``).

        Slack bits are set only for feasible agents; an over-capacity agent's slack stays
        zero (leaving a positive capacity penalty, correctly flagging infeasibility).
        """
        A, T = self.n_agents, self.n_tasks
        weights, offsets = self._slack()
        x = np.zeros(self.n)
        used = np.zeros(A)
        for t, a in enumerate(assignment):
            x[int(a) * T + t] = 1.0
            used[int(a)] += self.resource[int(a), t]
        for a in range(A):
            slack = int(round(float(self.capacity[a] - used[a])))
            if 0 <= slack <= int(self.capacity[a]):
                bits = encode_slack(slack, weights[a])
                for k, b in enumerate(bits):
                    x[offsets[a] + k] = float(b)
        return x

    def flip_deltas(self, x: object) -> FloatArray:
        r"""Closed-form single-bit flip deltas (delegated to the QUBO fast path)."""
        deltas: FloatArray = self.to_qubo().flip_deltas(x)
        return deltas

    def to_qubo(self) -> QUBOProblem:
        r"""The equivalent :class:`omnibias.qubo.QUBOProblem`."""
        q, c, const = self._arrays()
        return QUBOProblem(Q=q, c=c, const=const, name=self.name or "gap")

    def to_polynomial(self) -> Polynomial:
        r"""The energy as an :class:`omnibias.sos.Polynomial` (via the QUBO)."""
        return self.to_qubo().to_polynomial()


def gap(
    cost: object,
    resource: object,
    capacity: object,
    *,
    penalty_one: float | None = None,
    penalty_cap: float | None = None,
    name: str | None = None,
) -> GAPProblem:
    r"""Build a :class:`GAPProblem`; penalties default to the safe ``sum|cost| + 1``."""
    cost_arr = np.asarray(cost, dtype=float)
    scale = float(np.abs(cost_arr).sum()) + 1.0
    return GAPProblem(
        cost_arr,
        np.asarray(resource, dtype=float),
        np.asarray(capacity, dtype=float),
        scale if penalty_one is None else penalty_one,
        scale if penalty_cap is None else penalty_cap,
        name,
    )


# --------------------------------------------------------------------------------------
# structured decoder + named classical baseline + exact exponential oracle
# --------------------------------------------------------------------------------------


def _greedy_feasible(problem: GAPProblem, order: NDArray[np.intp]) -> list[int]:
    """Assign tasks (in ``order``) to the min-cost agent that still has capacity."""
    A, T = problem.n_agents, problem.n_tasks
    used = np.zeros(A)
    assignment = [-1] * T
    for t in order:
        t = int(t)
        best_a, best_c = -1, np.inf
        for a in range(A):
            if used[a] + problem.resource[a, t] <= problem.capacity[a] + 1e-9:
                if problem.cost[a, t] < best_c:
                    best_a, best_c = a, float(problem.cost[a, t])
        if best_a < 0:  # no feasible agent: least-overflowing fallback
            best_a = int(np.argmin(used + problem.resource[:, t] - problem.capacity))
        assignment[t] = best_a
        used[best_a] += problem.resource[best_a, t]
    return assignment


def _repair(problem: GAPProblem, assignment: list[int]) -> list[int]:
    """Move tasks off over-capacity agents to the cheapest feasible agent (best-effort)."""
    A, T = problem.n_agents, problem.n_tasks
    assignment = list(assignment)
    for _ in range(2 * T):
        used = np.zeros(A)
        for t, a in enumerate(assignment):
            used[a] += problem.resource[a, t]
        over = [a for a in range(A) if used[a] > problem.capacity[a] + 1e-9]
        if not over:
            break
        a = over[0]
        tasks = [t for t in range(T) if assignment[t] == a]
        moved = False
        for t in sorted(tasks, key=lambda t: -problem.resource[a, t]):
            cands = [
                b
                for b in range(A)
                if b != a and used[b] + problem.resource[b, t] <= problem.capacity[b] + 1e-9
            ]
            if cands:
                b = min(cands, key=lambda b: problem.cost[b, t])
                assignment[t] = b
                moved = True
                break
        if not moved:
            break
    return assignment


def gap_decode(
    problem: GAPProblem, *, relaxed: object | None = None
) -> tuple[tuple[int, ...], float]:
    r"""Decode to a capacity-feasible assignment (argmax + capacity repair) then set slacks.

    ``relaxed`` is the soft assignment over all ``n`` bits; its ``A*T`` assignment block is
    argmax-ed per task, then capacity-repaired. Returns the full binary ``x`` (assignment +
    slack) and its energy -- a heuristic *upper* bound.
    """
    A, T = problem.n_agents, problem.n_tasks
    if relaxed is None:
        order = np.argsort(-problem.resource.max(axis=0))  # hardest (largest) tasks first
        assignment = _greedy_feasible(problem, order)
    else:
        heat = np.asarray(relaxed, dtype=float).reshape(-1)[: A * T].reshape(A, T)
        assignment = [int(np.argmax(heat[:, t])) for t in range(T)]
    assignment = _repair(problem, assignment)
    x = problem.full_x(assignment)
    return tuple(int(v) for v in x), float(problem.energy(x))


def gap_lp_lower_bound(problem: GAPProblem) -> float:
    r"""LP-relaxation lower bound on the GAP cost (``scipy.optimize.linprog``, a baseline).

    Relaxes ``x in {0, 1}`` to ``x in [0, 1]`` with the one-hot equalities and capacity
    inequalities; the LP optimum is a valid *lower* bound on the integer optimum (it is a
    heuristic bound here, **not** the rigorous certificate -- that is
    :func:`omnibias.nphard.certify_gap`).
    """
    from scipy.optimize import linprog

    A, T = problem.n_agents, problem.n_tasks
    c = problem.cost.reshape(-1)
    a_eq = np.zeros((T, A * T))
    for t in range(T):
        for a in range(A):
            a_eq[t, a * T + t] = 1.0
    b_eq = np.ones(T)
    a_ub = np.zeros((A, A * T))
    for a in range(A):
        for t in range(T):
            a_ub[a, a * T + t] = problem.resource[a, t]
    b_ub = problem.capacity
    res = linprog(c, A_ub=a_ub, b_ub=b_ub, A_eq=a_eq, b_eq=b_eq, bounds=(0.0, 1.0))
    return float(res.fun) if res.success else -np.inf


def _ortools_optimum(problem: GAPProblem) -> tuple[tuple[int, ...], float] | None:
    """Exact GAP via OR-Tools CP-SAT if installed, else ``None`` (optional baseline)."""
    try:
        from ortools.sat.python import cp_model
    except ModuleNotFoundError:
        return None
    A, T = problem.n_agents, problem.n_tasks
    model = cp_model.CpModel()
    xv = {(a, t): model.NewBoolVar(f"x_{a}_{t}") for a in range(A) for t in range(T)}
    for t in range(T):
        model.Add(sum(xv[a, t] for a in range(A)) == 1)
    for a in range(A):
        model.Add(
            sum(int(problem.resource[a, t]) * xv[a, t] for t in range(T))
            <= int(problem.capacity[a])
        )
    scale = 1000
    model.Minimize(
        sum(int(round(problem.cost[a, t] * scale)) * xv[a, t] for a in range(A) for t in range(T))
    )
    solver = cp_model.CpSolver()
    status = solver.Solve(model)
    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return None
    assignment = [next(a for a in range(A) if solver.Value(xv[a, t])) for t in range(T)]
    x = problem.full_x(assignment)
    return tuple(int(v) for v in x), float(problem.energy(x))


def gap_classical(problem: GAPProblem) -> tuple[tuple[int, ...], float]:
    r"""Named baseline: OR-Tools CP-SAT exact if installed, else a greedy feasible upper bound.

    GAP is NP-hard; without the optional ``ortools`` extra this returns the greedy
    (resource-descending, min-cost feasible) heuristic -- a valid *upper* bound, not a
    guaranteed optimum. See :func:`gap_lp_lower_bound` for the LP lower bound.
    """
    exact = _ortools_optimum(problem)
    if exact is not None:
        return exact
    order = np.argsort(-problem.resource.max(axis=0))
    assignment = _repair(problem, _greedy_feasible(problem, order))
    x = problem.full_x(assignment)
    return tuple(int(v) for v in x), float(problem.energy(x))


def gap_brute_force(
    problem: GAPProblem, *, max_evaluations: int = 200_000
) -> tuple[tuple[int, ...], float]:
    r"""Exact optimum by enumerating all ``A^T`` feasible assignments (**exponential**).

    Guarded by ``A^T <= max_evaluations``. Returns the min-cost capacity-feasible
    assignment; raises if none is feasible. The exponential oracle used only to
    self-check the sandwich on tiny instances.
    """
    A, T = problem.n_agents, problem.n_tasks
    if A**T > max_evaluations:
        raise ValueError(f"brute force is exponential (A^T={A**T}); exceeds {max_evaluations}")
    best_assignment: tuple[int, ...] | None = None
    best_cost = np.inf
    for assignment in itertools.product(range(A), repeat=T):
        if problem.is_feasible(assignment) and problem.assignment_cost(assignment) < best_cost:
            best_assignment, best_cost = assignment, problem.assignment_cost(assignment)
    if best_assignment is None:
        raise ValueError("no capacity-feasible assignment exists for this GAP instance")
    x = problem.full_x(best_assignment)
    return tuple(int(v) for v in x), float(problem.energy(x))


__all__ = [
    "GAPProblem",
    "encode_slack",
    "gap",
    "gap_brute_force",
    "gap_classical",
    "gap_decode",
    "gap_lp_lower_bound",
    "slack_weights",
]
