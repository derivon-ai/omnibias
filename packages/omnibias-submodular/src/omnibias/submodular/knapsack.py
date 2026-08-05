# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
r"""Public knapsack-constrained submodular maximization surface.

Re-exports the numpy knapsack engine from :mod:`omnibias.submodular._core.knapsack`
(:class:`KnapsackConstraint`, :func:`cost_benefit_greedy`, :func:`knapsack_maximize`,
:func:`brute_force_max_knapsack`, :func:`certify_knapsack_gap`) and adds the ergonomic
:func:`budgeted` front-end that runs Sviridenko's ``(1 - 1/e)`` maximizer end-to-end.

A knapsack constraint is *not* a matroid, so this path is separate from the continuous-greedy
matroid pipeline; there is no differentiable twin this tier (numpy-only, honestly labelled).
"""

from __future__ import annotations

import numpy as np
from omnibias.submodular._core.knapsack import (
    KnapsackConstraint,
    brute_force_max_knapsack,
    certify_knapsack_gap,
    cost_benefit_greedy,
    knapsack_maximize,
)
from omnibias.submodular.functions import SubmodularFunction
from omnibias.submodular.problem import SubmodularSolution


def budgeted(
    function: SubmodularFunction,
    costs: object,
    budget: float,
    *,
    enumerate_size: int = 3,
) -> SubmodularSolution:
    r"""Maximize a monotone submodular ``function`` under a budget -> a :class:`SubmodularSolution`.

    Builds the :class:`KnapsackConstraint` and runs Sviridenko's partial-enumeration greedy
    (:func:`knapsack_maximize`, ``(1 - 1/e)``). Certify the gap with
    ``certify_knapsack_gap(function, KnapsackConstraint(costs, budget), sol.selection)``.
    """
    constraint = KnapsackConstraint(np.asarray(costs, dtype=float), budget)
    if constraint.n != function.n:
        raise ValueError(f"costs length {constraint.n} != function ground set {function.n}")
    selection, value = knapsack_maximize(function, constraint, enumerate_size=enumerate_size)
    return SubmodularSolution(selection=selection, value=value)


__all__ = [
    "KnapsackConstraint",
    "brute_force_max_knapsack",
    "budgeted",
    "certify_knapsack_gap",
    "cost_benefit_greedy",
    "knapsack_maximize",
]
