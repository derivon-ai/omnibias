# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
r"""The exact (numpy) end-to-end pipeline: continuous greedy -> rounding -> polish.

:func:`maximize` runs continuous greedy to a fractional ``p*``, rounds it to a feasible
integral set (``pipage`` or ``swap``), and polishes with a feasibility-preserving local
search. It returns a :class:`~omnibias.submodular.problem.SubmodularSolution` carrying the
achieved value and the fractional witness, from which
:func:`~omnibias.submodular.certify.certify_submodular_gap` builds the certified sandwich.
This is a certified *heuristic* with an a-priori ``(1 - 1/e)`` guarantee -- never an
exact-optimality claim.
"""

from __future__ import annotations

import numpy as np
from omnibias.submodular._core.continuous import continuous_greedy
from omnibias.submodular._core.greedy import local_search
from omnibias.submodular._core.rounding import pipage_round, swap_round
from omnibias.submodular.problem import (
    ContinuousGreedySchedule,
    SubmodularProblem,
    SubmodularSolution,
)


def maximize(
    problem: SubmodularProblem,
    *,
    schedule: ContinuousGreedySchedule | None = None,
    rounding: str = "pipage",
    polish: bool = True,
    seed: int = 0,
) -> SubmodularSolution:
    r"""Continuous greedy + rounding (+ polish) -> a feasible :class:`SubmodularSolution`.

    Parameters
    ----------
    problem:
        The monotone submodular maximization instance.
    schedule:
        Continuous-greedy hyperparameters (defaults to eval quality).
    rounding:
        ``"pipage"`` (deterministic, guarantees ``f(S) >= F(p*)``) or ``"swap"``
        (randomized, seeded; ``E[f(S)] >= F(p*)``).
    polish:
        Whether to run the feasibility-preserving add / swap local search afterward
        (only ever increases the value).
    seed:
        Seed for ``"swap"`` rounding (ignored by ``"pipage"``).
    """
    sched = schedule or ContinuousGreedySchedule()
    p_star, bases = continuous_greedy(problem.function, problem.matroid, steps=sched.steps)
    if rounding == "pipage":
        selection, value = pipage_round(problem.function, problem.matroid, p_star)
    elif rounding == "swap":
        selection, value = swap_round(problem.function, problem.matroid, bases, seed=seed)
    else:
        raise ValueError(f"unknown rounding {rounding!r}; choose 'pipage' or 'swap'")
    if polish:
        selection, value = local_search(
            problem.function, problem.matroid, np.asarray(selection, dtype=float)
        )
    fractional_value = float(problem.function.multilinear(p_star))
    return SubmodularSolution(
        selection=selection,
        value=value,
        fractional=p_star,
        fractional_value=fractional_value,
        rounding=rounding,
    )


__all__ = ["maximize"]
