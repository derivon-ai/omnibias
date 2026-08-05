# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
r"""Ergonomic constructors for the shipped monotone submodular problems.

* :func:`max_coverage` -- maximum (weighted) coverage: choose ``k`` of ``n`` candidate
  sets to cover the most (weighted) universe elements.
* :func:`facility_location` -- choose ``k`` of ``n`` facilities to maximize summed
  best-facility service ``sum_j w_j max_{i in S} M[j, i]``.
* :func:`budget_additive` -- maximize the concave-of-modular ``min(sum_{i in S} a_i, B)``
  subject to a matroid constraint.
* :func:`log_det_dpp` -- choose ``k`` diverse items under a determinantal (DPP) kernel,
  maximizing ``log det(I + K_S)`` (a *greedy-path* problem: no differentiable twin, solved
  by lazy / stochastic greedy).

Each returns a :class:`~omnibias.submodular.problem.SubmodularProblem` under a cardinality
(:class:`~omnibias.submodular.matroid.UniformMatroid`) constraint by default; pass an
explicit ``matroid`` (e.g. a :class:`~omnibias.submodular.matroid.PartitionMatroid`) to
override.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import numpy as np
from omnibias.submodular.functions import (
    BudgetAdditive,
    Coverage,
    FacilityLocation,
    LogDeterminant,
)
from omnibias.submodular.matroid import Matroid, UniformMatroid
from omnibias.submodular.problem import SubmodularProblem


def _resolve_matroid(n: int, k: int | None, matroid: Matroid | None) -> Matroid:
    if matroid is not None:
        if matroid.n != n:
            raise ValueError(f"matroid ground set {matroid.n} != number of elements {n}")
        return matroid
    if k is None:
        raise ValueError("provide either k (cardinality) or an explicit matroid")
    return UniformMatroid(n, k)


def max_coverage(
    sets: Sequence[Iterable[int]],
    *,
    universe: int | None = None,
    k: int | None = None,
    weights: object | None = None,
    matroid: Matroid | None = None,
    name: str | None = None,
) -> SubmodularProblem:
    r"""A max-coverage ``SubmodularProblem``: pick sets to cover the weighted universe.

    ``sets[i]`` lists the universe elements covered by candidate set ``i``; ``universe``
    is the number of elements (inferred from the largest index if omitted); ``weights``
    are optional per-element weights.
    """
    set_lists = [sorted({int(e) for e in s}) for s in sets]
    n = len(set_lists)
    if n < 1:
        raise ValueError("need at least one candidate set")
    max_elem = max((s[-1] for s in set_lists if s), default=-1)
    m = (max_elem + 1) if universe is None else int(universe)
    if m < 1:
        raise ValueError("universe must have at least one element")
    membership = np.zeros((m, n), dtype=float)
    for i, elements in enumerate(set_lists):
        for e in elements:
            if not 0 <= e < m:
                raise ValueError(f"element {e} out of range for universe size {m}")
            membership[e, i] = 1.0
    label = name if name is not None else "max_coverage"
    function = Coverage(membership, weights, name=label)
    return SubmodularProblem(function, _resolve_matroid(n, k, matroid), name=label)


def facility_location(
    gains: object,
    *,
    k: int | None = None,
    weights: object | None = None,
    matroid: Matroid | None = None,
    name: str | None = None,
) -> SubmodularProblem:
    r"""A facility-location ``SubmodularProblem`` for the client-by-facility ``gains``."""
    label = name if name is not None else "facility_location"
    function = FacilityLocation(np.asarray(gains, dtype=float), weights, name=label)
    return SubmodularProblem(function, _resolve_matroid(function.n, k, matroid), name=label)


def budget_additive(
    ground: object,
    budget: float,
    *,
    k: int | None = None,
    matroid: Matroid | None = None,
    name: str | None = None,
) -> SubmodularProblem:
    r"""A budget-additive ``SubmodularProblem`` ``min(sum_{i in S} a_i, budget)``."""
    label = name if name is not None else "budget_additive"
    function = BudgetAdditive(np.asarray(ground, dtype=float), budget, name=label)
    return SubmodularProblem(function, _resolve_matroid(function.n, k, matroid), name=label)


def log_det_dpp(
    kernel: object,
    *,
    k: int | None = None,
    matroid: Matroid | None = None,
    name: str | None = None,
) -> SubmodularProblem:
    r"""A log-determinant / DPP ``SubmodularProblem`` ``log det(I + K_S)`` (greedy-path).

    ``kernel`` is a symmetric PSD ``(n, n)`` similarity matrix. The problem is *greedy-path*
    (no closed-form multilinear extension, so no differentiable twin): maximize it with
    :func:`~omnibias.submodular.lazy_greedy` / :func:`~omnibias.submodular.stochastic_greedy`
    and certify with :func:`~omnibias.submodular.certify_submodular_gap`.
    """
    label = name if name is not None else "log_det_dpp"
    function = LogDeterminant(np.asarray(kernel, dtype=float), name=label)
    return SubmodularProblem(function, _resolve_matroid(function.n, k, matroid), name=label)


__all__ = [
    "budget_additive",
    "facility_location",
    "log_det_dpp",
    "max_coverage",
]
