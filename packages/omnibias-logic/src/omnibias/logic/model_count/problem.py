# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
r"""(Weighted) #SAT / model counting as a pseudo-Boolean :class:`DiscreteProblem`.

Counting the satisfying assignments of a CNF -- and its weighted generalisation (weighted
model counting: sum over satisfying ``x`` of ``prod_i w_i(x_i)``) -- is ``#P``-hard, so
there is no poly-time exact counter here. :class:`ModelCountProblem` carries the *formula*
(a :class:`~omnibias.discrete.maxsat.problem.WeightedCNF`, whose clauses are treated as
**hard** constraints for counting) plus optional per-variable literal weights, and reuses
the ``omnibias-discrete`` substrate seam:

* it implements ``n`` / ``energy`` / ``to_polynomial`` by delegating to an internal
  **hard-clause** :class:`~omnibias.discrete.maxsat.problem.MaxSATProblem` whose energy is
  the number of violated clauses -- ``0`` iff the assignment is a model -- so the substrate
  decoder (:func:`omnibias.discrete.decode`) and exact oracle
  (:func:`omnibias.discrete.brute_force_min`) work on it unchanged;
* :func:`exact_model_count` is the exact ``O(2^n)`` count (built on the pure-Python
  :mod:`omnibias.boolean` truth-table primitives) -- the small-``n`` oracle that
  self-checks the certified :func:`omnibias.logic.count_enclosure` sandwich.

Literals use the DIMACS convention: signed **1-based** integers, ``+k`` for ``x_{k-1}`` and
``-k`` for its negation.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from typing import TYPE_CHECKING, cast

import numpy as np
from numpy.typing import NDArray
from omnibias.discrete.maxsat.problem import Clause, MaxSATProblem, WeightedCNF

if TYPE_CHECKING:  # pragma: no cover - typing only
    from collections.abc import Sequence

    from omnibias.sos import Polynomial

FloatArray = NDArray[np.float64]

#: Cap for the exponential exact oracle (``O(2^n)``); mirrors the substrate's brute force.
_MAX_EXACT_N = 20


def _clause_satisfied(bits: Sequence[int], literals: tuple[int, ...]) -> bool:
    """Whether ``bits`` (LSB-first ``x_0..x_{n-1}``) satisfies a single clause."""
    for literal in literals:
        i = abs(literal) - 1
        if (literal > 0 and bits[i] == 1) or (literal < 0 and bits[i] == 0):
            return True
    return False


def _formula_satisfied(bits: Sequence[int], clauses: tuple[Clause, ...]) -> bool:
    """Whether ``bits`` satisfies every clause of the CNF (a model)."""
    return all(_clause_satisfied(bits, clause.literals) for clause in clauses)


@dataclass(frozen=True)
class ModelCountProblem:
    r"""A (weighted) #SAT instance: a CNF of hard clauses + optional literal weights.

    Attributes
    ----------
    cnf:
        The :class:`~omnibias.discrete.maxsat.problem.WeightedCNF` whose satisfying
        assignments are counted. Clause weights are **ignored** for counting (every clause
        is a hard constraint); only the literals matter.
    weights:
        Optional per-variable literal weights, shape ``(n, 2)`` with
        ``weights[i] = [w_i(0), w_i(1)]`` (nonnegative). When omitted the count is the plain
        (unweighted) number of models; when present the weighted model count
        ``sum_{models} prod_i w_i(x_i)``.
    name:
        Optional label.
    """

    cnf: WeightedCNF
    weights: FloatArray | None = None
    name: str | None = None

    def __post_init__(self) -> None:
        n = int(self.cnf.n_vars)
        if self.weights is not None:
            w = np.asarray(self.weights, dtype=float)
            if w.shape != (n, 2):
                raise ValueError(f"weights must have shape ({n}, 2), got {w.shape}")
            if np.any(w < 0.0):
                raise ValueError("literal weights must be nonnegative")
            object.__setattr__(self, "weights", w)
        # Cache the hard-clause MaxSATProblem seam (clause weights forced to 1.0): its
        # energy is the count of violated clauses, so energy == 0 iff x is a model.
        hard = tuple(Clause(literals=clause.literals, weight=1.0) for clause in self.cnf.clauses)
        maxsat = MaxSATProblem(cnf=WeightedCNF(n_vars=n, clauses=hard), name=self.name)
        object.__setattr__(self, "_maxsat", maxsat)

    @property
    def n(self) -> int:
        return int(self.cnf.n_vars)

    @property
    def is_weighted(self) -> bool:
        """Whether per-variable literal weights were supplied."""
        return self.weights is not None

    @property
    def as_maxsat(self) -> MaxSATProblem:
        """The internal hard-clause :class:`MaxSATProblem` (the ``DiscreteProblem`` seam)."""
        maxsat: MaxSATProblem = self._maxsat  # type: ignore[attr-defined]
        return maxsat

    def energy(self, x: object) -> float | FloatArray:
        r"""Number of violated (hard) clauses at ``x`` -- ``0`` iff ``x`` is a model.

        Delegates to the internal hard-clause :class:`MaxSATProblem`; supports a single
        point ``(n,)`` and a batch ``(m, n)``.
        """
        return cast("float | FloatArray", self.as_maxsat.energy(x))

    def is_model(self, x: object) -> bool:
        """Whether the single binary point ``x`` satisfies every clause."""
        return float(self.as_maxsat.energy(x)) == 0.0

    def to_polynomial(self) -> Polynomial:
        """The hard-clause violation energy as an :class:`omnibias.sos.Polynomial`."""
        return self.as_maxsat.to_polynomial()

    def weight_fractions(self) -> list[tuple[Fraction, Fraction]]:
        r"""Exact per-variable ``(w_i(0), w_i(1))`` as :class:`~fractions.Fraction` pairs.

        Unweighted problems use ``(1, 1)`` per variable (so ``Z0 = 2^n`` and the count is an
        integer). Weighted problems convert each stored float to its exact rational value, so
        the enclosure and the oracle agree bit-for-bit.
        """
        n = self.n
        if self.weights is None:
            return [(Fraction(1), Fraction(1)) for _ in range(n)]
        w = self.weights
        return [(Fraction(float(w[i, 0])), Fraction(float(w[i, 1]))) for i in range(n)]


def exact_model_count(problem: ModelCountProblem, *, max_n: int = _MAX_EXACT_N) -> float:
    r"""Exact (weighted) model count by enumerating all ``2^n`` assignments.

    Exponential (``O(2^n)``); intended as the small-``n`` oracle that self-checks the
    certified :func:`omnibias.logic.count_enclosure` sandwich. Built on the pure-Python
    :mod:`omnibias.boolean` truth-table primitives. Raises :class:`ValueError` for
    ``n > max_n``. Returns the count as a ``float`` (integer-valued when unweighted).
    """
    from omnibias.boolean import all_assignments, truth_table_from_callable

    n = problem.n
    if n > max_n:
        raise ValueError(
            f"exact_model_count is exponential (O(2^n)); n={n} exceeds the {max_n} cap. "
            "Use count_enclosure for a certified enclosure instead."
        )
    clauses = problem.cnf.clauses
    if problem.weights is None:
        table = truth_table_from_callable(
            lambda *bits: 1 if _formula_satisfied(bits, clauses) else 0, n
        )
        return float(sum(table))

    fracs = problem.weight_fractions()
    total = Fraction(0)
    for bits in all_assignments(n):
        if _formula_satisfied(bits, clauses):
            prod = Fraction(1)
            for i in range(n):
                prod *= fracs[i][bits[i]]
            total += prod
    return float(total)


__all__ = ["ModelCountProblem", "exact_model_count"]
