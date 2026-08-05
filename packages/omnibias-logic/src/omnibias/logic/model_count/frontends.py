# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
r"""Ergonomic constructor for :class:`ModelCountProblem` from raw clauses.

:func:`model_count` accepts DIMACS-style clauses -- sequences of signed **1-based** integers
(``+k`` for ``x_{k-1}``, ``-k`` for its negation) -- with optional per-variable literal
weights, and returns a :class:`~omnibias.logic.model_count.problem.ModelCountProblem` ready
for :func:`omnibias.logic.count_enclosure` and the substrate decoder / oracle. CNF parsing is
reused from :func:`omnibias.discrete.maxsat.max_sat` (the clauses become hard constraints).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import numpy as np
from omnibias.discrete.maxsat import max_sat
from omnibias.logic.model_count.problem import ModelCountProblem

if TYPE_CHECKING:  # pragma: no cover - typing only
    from numpy.typing import ArrayLike


def model_count(
    clauses: Sequence[Sequence[int]],
    weights: ArrayLike | None = None,
    *,
    n_vars: int | None = None,
    name: str | None = None,
) -> ModelCountProblem:
    r"""Build a :class:`ModelCountProblem` (a #SAT / weighted-model-counting instance).

    Parameters
    ----------
    clauses:
        A sequence of clauses, each a sequence of nonzero signed 1-based literals. Every
        clause is a **hard** constraint for counting.
    weights:
        Optional per-variable literal weights of shape ``(n, 2)`` with
        ``weights[i] = [w_i(0), w_i(1)]`` (nonnegative). Omit for plain (unweighted) ``#SAT``.
    n_vars:
        Number of variables; inferred from the largest ``|literal|`` when omitted.
    name:
        Optional label.
    """
    base = max_sat(clauses, n_vars=n_vars, name=name)
    w = None if weights is None else np.asarray(weights, dtype=float)
    return ModelCountProblem(cnf=base.cnf, weights=w, name=name)


__all__ = ["model_count"]
