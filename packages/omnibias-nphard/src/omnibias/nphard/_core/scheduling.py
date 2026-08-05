# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
r"""Parallel-machine scheduling (load balancing) as a QUBO-form ``DiscreteProblem``.

Assign ``J`` jobs with processing times ``p`` to ``M`` identical machines to balance the
loads. Minimising the makespan ``max_k load_k`` is **NP-hard** (multiprocessor
scheduling; number partitioning for ``M = 2``); we minimise the smooth **load-variance
proxy** ``sum_k load_k^2`` (minimised exactly when loads are equal, and a monotone proxy
that pushes down the makespan), which is a QUBO. Over ``J*M`` one-hot bits
``x[j,k] = [job j -> machine k]``,

.. math::
    E(x) = \sum_k \Bigl(\sum_j p_j x_{jk}\Bigr)^2
         + \lambda \sum_j \Bigl(\sum_k x_{jk} - 1\Bigr)^2,

with interaction ``Q = kron(p p^T, I_M) + lambda * kron(I_J, ones_M)``. A valid one-hot
assignment has zero penalty, so its energy is the sum of squared machine loads.
"""

from __future__ import annotations

import itertools
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

import numpy as np
from numpy.typing import NDArray
from omnibias.qubo import QUBOProblem

if TYPE_CHECKING:  # pragma: no cover - typing only
    from omnibias.sos import Polynomial

FloatArray = NDArray[np.float64]


def scheduling_qubo_arrays(
    processing: FloatArray, machines: int, penalty: float
) -> tuple[FloatArray, FloatArray, float]:
    r"""The scheduling QUBO ``(Q, c, const)``: load-variance interaction + one-hot penalty."""
    n_jobs = int(processing.shape[0])
    eye_m = np.eye(machines)
    outer = np.outer(processing, processing)
    # cast: np.kron's stub widens the dtype to floating[Any]; the inputs are float64.
    interaction: FloatArray = cast("FloatArray", np.kron(outer, eye_m))
    q_pen: FloatArray = cast("FloatArray", np.kron(np.eye(n_jobs), np.ones((machines, machines))))
    q: FloatArray = interaction + penalty * q_pen
    c: FloatArray = -2.0 * penalty * np.ones(n_jobs * machines)
    return q, c, penalty * float(n_jobs)


@dataclass(frozen=True)
class SchedulingProblem:
    r"""Parallel-machine load balancing encoded as a QUBO over ``J*M`` one-hot bits.

    Attributes
    ----------
    processing:
        ``(J,)`` job processing times ``p``.
    machines:
        The number of identical machines ``M``.
    penalty:
        The job one-hot penalty ``lambda`` (:func:`schedule` picks a safe default).
    name:
        Optional label.
    """

    processing: FloatArray
    machines: int
    penalty: float
    name: str | None = None

    def __post_init__(self) -> None:
        processing = np.asarray(self.processing, dtype=float).reshape(-1)
        if processing.shape[0] < 1:
            raise ValueError("scheduling needs at least one job")
        if int(self.machines) < 1:
            raise ValueError("scheduling needs at least one machine")
        object.__setattr__(self, "processing", processing)
        object.__setattr__(self, "machines", int(self.machines))
        object.__setattr__(self, "penalty", float(self.penalty))

    @property
    def n_jobs(self) -> int:
        return int(self.processing.shape[0])

    @property
    def n(self) -> int:
        """The number of binary variables (``J*M``)."""
        return self.n_jobs * self.machines

    def _arrays(self) -> tuple[FloatArray, FloatArray, float]:
        cache = self.__dict__.get("_qubo_arrays")
        if cache is None:
            cache = scheduling_qubo_arrays(self.processing, self.machines, self.penalty)
            object.__setattr__(self, "_qubo_arrays", cache)
        arrays: tuple[FloatArray, FloatArray, float] = cache
        return arrays

    def loads(self, assignment: Sequence[int] | NDArray[np.intp]) -> FloatArray:
        r"""The per-machine loads of a job->machine assignment."""
        out = np.zeros(self.machines)
        for j, k in enumerate(assignment):
            out[int(k)] += self.processing[j]
        return out

    def makespan(self, assignment: Sequence[int] | NDArray[np.intp]) -> float:
        r"""The makespan ``max_k load_k`` of a job->machine assignment (the true objective)."""
        return float(self.loads(assignment).max())

    def objective(self, x: object) -> float | FloatArray:
        r"""The pure load-variance objective ``sum_k load_k^2`` (no penalty)."""
        xv = np.asarray(x, dtype=float)
        interaction = np.kron(np.outer(self.processing, self.processing), np.eye(self.machines))
        quad = np.sum((xv @ interaction) * xv, axis=-1)
        return float(quad) if xv.ndim == 1 else quad

    def energy(self, x: object) -> float | FloatArray:
        r"""The QUBO energy (load variance + one-hot penalty)."""
        q, c, const = self._arrays()
        xv = np.asarray(x, dtype=float)
        quad = np.sum((xv @ q) * xv, axis=-1)
        lin = xv @ c
        out = quad + lin + const
        return float(out) if xv.ndim == 1 else out

    def flip_deltas(self, x: object) -> FloatArray:
        r"""Closed-form single-bit flip deltas (delegated to the QUBO fast path)."""
        deltas: FloatArray = self.to_qubo().flip_deltas(x)
        return deltas

    def to_qubo(self) -> QUBOProblem:
        r"""The equivalent :class:`omnibias.qubo.QUBOProblem`."""
        q, c, const = self._arrays()
        return QUBOProblem(Q=q, c=c, const=const, name=self.name or "scheduling")

    def to_polynomial(self) -> Polynomial:
        r"""The energy as an :class:`omnibias.sos.Polynomial` (via the QUBO)."""
        return self.to_qubo().to_polynomial()


def schedule(
    processing: object, machines: int, *, penalty: float | None = None, name: str | None = None
) -> SchedulingProblem:
    r"""Build a :class:`SchedulingProblem`; ``penalty`` defaults to the safe ``2*max(p)*sum(p) + 1``.

    Leaving a job unassigned drops the load-variance objective by at most
    ``2*max(p)*sum(p)``, so this penalty keeps every job assigned (the minimiser is a valid
    one-hot schedule); it is much smaller than the loose ``(sum p)^2``. Penalty size affects
    only the certified-gap tightness, never a decoded feasible solution's energy.
    """
    processing_arr = np.asarray(processing, dtype=float).reshape(-1)
    if penalty is None:
        penalty = float(2.0 * processing_arr.max() * processing_arr.sum()) + 1.0
    return SchedulingProblem(processing_arr, int(machines), penalty, name)


# --------------------------------------------------------------------------------------
# structured decoder + named classical baseline + exact exponential oracle
# --------------------------------------------------------------------------------------


def assignment_to_x(assignment: Sequence[int] | NDArray[np.intp], machines: int) -> FloatArray:
    r"""The ``J*M`` binary indicator of a job->machine assignment."""
    n_jobs = len(assignment)
    x = np.zeros((n_jobs, machines))
    for j, k in enumerate(assignment):
        x[j, int(k)] = 1.0
    return x.reshape(-1)


def _lpt(problem: SchedulingProblem) -> list[int]:
    """Longest-Processing-Time greedy: assign the longest job to the least-loaded machine."""
    loads = np.zeros(problem.machines)
    assignment = [0] * problem.n_jobs
    for j in np.argsort(-problem.processing):
        k = int(np.argmin(loads))
        assignment[int(j)] = k
        loads[k] += problem.processing[int(j)]
    return assignment


def _local_moves(problem: SchedulingProblem, assignment: list[int]) -> tuple[list[int], float]:
    """Best-improvement single-job re-assignment on the load-variance energy."""
    best = list(assignment)
    best_e = float(problem.energy(assignment_to_x(best, problem.machines)))
    improved = True
    while improved:
        improved = False
        for j in range(problem.n_jobs):
            cur = best[j]
            for k in range(problem.machines):
                if k == cur:
                    continue
                cand = list(best)
                cand[j] = k
                e = float(problem.energy(assignment_to_x(cand, problem.machines)))
                if e < best_e - 1e-12:
                    best, best_e, improved = cand, e, True
    return best, best_e


def scheduling_decode(
    problem: SchedulingProblem, *, relaxed: object | None = None
) -> tuple[tuple[int, ...], float]:
    r"""Decode a relaxed heatmap to a one-hot schedule (argmax) then LPT-style local repair.

    ``relaxed`` is the ``(J, M)`` (or flat) soft assignment; ``None`` starts from the LPT
    construction. Returns the binary ``x in {0, 1}^{J*M}`` and its energy (a heuristic
    *upper* bound).
    """
    if relaxed is None:
        assignment = _lpt(problem)
    else:
        heat = np.asarray(relaxed, dtype=float).reshape(problem.n_jobs, problem.machines)
        assignment = [int(np.argmax(heat[j])) for j in range(problem.n_jobs)]
    assignment, energy = _local_moves(problem, assignment)
    x = assignment_to_x(assignment, problem.machines)
    return tuple(int(v) for v in x), energy


def scheduling_classical(problem: SchedulingProblem) -> tuple[tuple[int, ...], float]:
    r"""Named baseline: the Longest-Processing-Time (LPT) greedy list-scheduling heuristic.

    LPT is the classical ``4/3 - 1/(3M)`` makespan approximation. It is a *heuristic*
    (scheduling is NP-hard), returning a valid *upper* bound, not a guaranteed optimum.
    """
    assignment = _lpt(problem)
    x = assignment_to_x(assignment, problem.machines)
    return tuple(int(v) for v in x), float(problem.energy(x))


def scheduling_brute_force(
    problem: SchedulingProblem, *, max_evaluations: int = 200_000
) -> tuple[tuple[int, ...], float]:
    r"""Exact optimum by enumerating all ``M^J`` assignments (**exponential**; guarded).

    The exponential oracle used only to self-check the certified sandwich on tiny
    instances (``M^J <= max_evaluations``).
    """
    n_jobs, m = problem.n_jobs, problem.machines
    if m**n_jobs > max_evaluations:
        raise ValueError(f"brute force is exponential (M^J={m**n_jobs}); exceeds {max_evaluations}")
    best_assignment: tuple[int, ...] | None = None
    best_e = np.inf
    for assignment in itertools.product(range(m), repeat=n_jobs):
        e = float(problem.energy(assignment_to_x(assignment, m)))
        if e < best_e:
            best_assignment, best_e = assignment, e
    assert best_assignment is not None
    x = assignment_to_x(best_assignment, m)
    return tuple(int(v) for v in x), best_e


__all__ = [
    "SchedulingProblem",
    "assignment_to_x",
    "schedule",
    "scheduling_brute_force",
    "scheduling_classical",
    "scheduling_decode",
    "scheduling_qubo_arrays",
]
