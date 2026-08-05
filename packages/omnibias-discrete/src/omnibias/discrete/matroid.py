# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
r"""Shared, representation-neutral matroid independence / rank kernel.

The public home of the canonical independence and rank definitions for the three
matroid families the stack shares -- **uniform**, **partition**, and **graphic** -- on
the mathematical ``frozenset[int]`` representation. Both matroid lenses build on it:

* :mod:`omnibias.combinatorics` (polytope / LP-certification lens) delegates its
  ``frozenset`` independence / rank here directly;
* :mod:`omnibias.submodular` (greedy / soft-oracle lens) thresholds its ``0/1`` vectors
  to the selected index set and delegates the independence question here.

Routing both through one kernel keeps the two graphic matroids from drifting on
acyclicity and the two partition matroids from drifting on the capacity rule. The
specialized surfaces (rank inequalities, linear-maximization oracle, differentiable soft
basis) stay in their own packages.
"""

from __future__ import annotations

from omnibias.discrete._core.matroid import (
    MatroidCore,
    graphic_independent,
    graphic_rank,
    independent_sets,
    partition_independent,
    partition_rank,
    uniform_independent,
    uniform_rank,
)

__all__ = [
    "MatroidCore",
    "graphic_independent",
    "graphic_rank",
    "independent_sets",
    "partition_independent",
    "partition_rank",
    "uniform_independent",
    "uniform_rank",
]
