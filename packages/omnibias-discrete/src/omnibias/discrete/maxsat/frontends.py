# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
r"""Ergonomic constructor for :class:`MaxSATProblem` from raw clauses.

:func:`max_sat` accepts DIMACS-style clauses -- sequences of signed **1-based** integers
(``+k`` for ``x_{k-1}``, ``-k`` for its negation) -- with optional per-clause weights,
and returns a :class:`~omnibias.discrete.maxsat.problem.MaxSATProblem` ready for the
shared relax / decode / certify pipeline.
"""

from __future__ import annotations

from collections.abc import Sequence

from omnibias.discrete.maxsat.problem import Clause, MaxSATProblem, WeightedCNF


def max_sat(
    clauses: Sequence[Sequence[int]],
    weights: Sequence[float] | None = None,
    *,
    n_vars: int | None = None,
    name: str | None = None,
) -> MaxSATProblem:
    r"""Build a :class:`MaxSATProblem` from clauses (and optional weights).

    Parameters
    ----------
    clauses:
        A sequence of clauses, each a sequence of nonzero signed 1-based literals.
    weights:
        Optional per-clause nonnegative weights (default: all ``1.0`` -- unweighted /
        MaxSAT). Length must match ``clauses``.
    n_vars:
        Number of variables; inferred from the largest ``|literal|`` when omitted.
    name:
        Optional label.
    """
    parsed = [tuple(int(literal) for literal in clause) for clause in clauses]
    inferred = max((abs(literal) for clause in parsed for literal in clause), default=0)
    n = inferred if n_vars is None else int(n_vars)
    if n < 1:
        raise ValueError("at least one variable is required (empty problem)")
    if n < inferred:
        raise ValueError(f"n_vars={n} is smaller than the largest literal {inferred}")

    if weights is None:
        ws = [1.0] * len(parsed)
    else:
        ws = [float(w) for w in weights]
        if len(ws) != len(parsed):
            raise ValueError(f"got {len(ws)} weights for {len(parsed)} clauses")

    clause_objs = tuple(Clause(literals=cl, weight=w) for cl, w in zip(parsed, ws, strict=True))
    return MaxSATProblem(cnf=WeightedCNF(n_vars=n, clauses=clause_objs), name=name)


__all__ = ["max_sat"]
