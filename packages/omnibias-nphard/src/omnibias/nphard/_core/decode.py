# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
r"""Family-dispatching decoder, named classical baseline, and exact exponential oracle.

Each NP-hard family ships its own structure-preserving decoder / named classical
heuristic / brute-force oracle (in :mod:`.qap`, :mod:`.gap`, :mod:`.scheduling`); this
module dispatches the three uniform entry points on the problem type so callers write
``decode(problem, relaxed=...)`` regardless of family. All three return
``(assignment, energy)`` where ``assignment`` is the binary point ``x in {0, 1}^n`` (the
QUBO variable space, so it feeds straight into :func:`omnibias.nphard.certify_gap`).
"""

from __future__ import annotations

from typing import TypeAlias

from omnibias.nphard._core.gap import (
    GAPProblem,
    gap_brute_force,
    gap_classical,
    gap_decode,
)
from omnibias.nphard._core.qap import (
    QAPProblem,
    qap_brute_force,
    qap_classical,
    qap_decode,
)
from omnibias.nphard._core.scheduling import (
    SchedulingProblem,
    scheduling_brute_force,
    scheduling_classical,
    scheduling_decode,
)

Problem: TypeAlias = QAPProblem | GAPProblem | SchedulingProblem


def decode(problem: Problem, *, relaxed: object | None = None) -> tuple[tuple[int, ...], float]:
    r"""Decode a relaxed heatmap to a feasible solution (structure-preserving, per family).

    QAP -> Hungarian + 2-opt; GAP -> argmax + capacity repair; scheduling -> argmax + LPT
    repair. ``relaxed`` is the soft assignment from the differentiable relaxation (``None``
    falls back to the family's constructive heuristic). Returns the binary ``x`` and its
    energy -- a heuristic *upper* bound on the optimum (never claimed optimal).
    """
    out: tuple[tuple[int, ...], float]
    if isinstance(problem, QAPProblem):
        out = qap_decode(problem, relaxed=relaxed)
    elif isinstance(problem, GAPProblem):
        out = gap_decode(problem, relaxed=relaxed)
    elif isinstance(problem, SchedulingProblem):
        out = scheduling_decode(problem, relaxed=relaxed)
    else:
        raise TypeError(f"unsupported problem type {type(problem).__name__}")
    return out


def classical_optimum(problem: Problem) -> tuple[tuple[int, ...], float]:
    r"""The named classical *heuristic* baseline per family (a valid upper bound).

    QAP -> ``scipy.optimize.quadratic_assignment`` (FAQ + 2-opt); scheduling -> LPT; GAP ->
    OR-Tools CP-SAT if installed else greedy. These families are **NP-hard**, so this is a
    strong heuristic baseline, **not** a guaranteed exact optimum (contrast the P-class
    ``omnibias-combinatorics``, whose ``classical_optimum`` is exact).
    """
    out: tuple[tuple[int, ...], float]
    if isinstance(problem, QAPProblem):
        out = qap_classical(problem)
    elif isinstance(problem, GAPProblem):
        out = gap_classical(problem)
    elif isinstance(problem, SchedulingProblem):
        out = scheduling_classical(problem)
    else:
        raise TypeError(f"unsupported problem type {type(problem).__name__}")
    return out


def brute_force_min(problem: Problem) -> tuple[tuple[int, ...], float]:
    r"""Exact optimum by full enumeration -- **exponential** (``dim!`` / ``A^T`` / ``M^J``).

    The exact oracle used only to self-check the certified sandwich on tiny instances; it
    is guarded to small sizes per family and raises on larger ones. Not a poly-time
    solver -- these problems are NP-hard.
    """
    out: tuple[tuple[int, ...], float]
    if isinstance(problem, QAPProblem):
        out = qap_brute_force(problem)
    elif isinstance(problem, GAPProblem):
        out = gap_brute_force(problem)
    elif isinstance(problem, SchedulingProblem):
        out = scheduling_brute_force(problem)
    else:
        raise TypeError(f"unsupported problem type {type(problem).__name__}")
    return out


__all__ = ["Problem", "brute_force_min", "classical_optimum", "decode"]
