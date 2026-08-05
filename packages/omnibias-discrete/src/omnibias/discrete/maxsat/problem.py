# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
r"""Weighted MaxSAT as a pseudo-Boolean :class:`DiscreteProblem`.

A weighted CNF is minimized by its **weighted-violation energy**: a clause
``(l_1 or ... or l_k)`` with weight ``w`` contributes ``w`` iff every literal is
falsified, i.e. ``w * prod_j f_j(x)`` where ``f_j = 1 - x_i`` for a positive literal
``x_i`` and ``f_j = x_i`` for a negated one ``~x_i``. Summing over clauses gives a
nonnegative pseudo-Boolean polynomial whose minimum is the minimum total weight of
violated clauses (``0`` iff the instance is satisfiable). Literals use the DIMACS
convention: signed **1-based** integers, ``+k`` for ``x_{k-1}`` and ``-k`` for its
negation.

:class:`MaxSATProblem` implements the substrate's ``DiscreteProblem`` seam, so it plugs
straight into :func:`omnibias.discrete.decode`, :func:`omnibias.discrete.certify_gap`,
and the annealed relaxation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:  # pragma: no cover - typing only
    from omnibias.sos import Polynomial

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class Clause:
    r"""A weighted CNF clause: DIMACS signed 1-based ``literals`` and a ``weight``."""

    literals: tuple[int, ...]
    weight: float = 1.0

    def __post_init__(self) -> None:
        lits = tuple(int(literal) for literal in self.literals)
        if any(literal == 0 for literal in lits):
            raise ValueError("literals must be nonzero signed integers (DIMACS 1-based)")
        if len({abs(literal) for literal in lits}) != len(lits):
            raise ValueError("a clause must not repeat a variable")
        if self.weight < 0.0:
            raise ValueError("clause weight must be nonnegative")
        object.__setattr__(self, "literals", lits)
        object.__setattr__(self, "weight", float(self.weight))


@dataclass(frozen=True)
class WeightedCNF:
    r"""A weighted CNF over ``n_vars`` variables (``x_0, ..., x_{n_vars-1}``)."""

    n_vars: int
    clauses: tuple[Clause, ...]

    def __post_init__(self) -> None:
        if self.n_vars < 1:
            raise ValueError("n_vars must be >= 1")
        for clause in self.clauses:
            for literal in clause.literals:
                if abs(literal) > self.n_vars:
                    raise ValueError(
                        f"literal {literal} references a variable outside 1..{self.n_vars}"
                    )


@dataclass(frozen=True)
class MaxSATProblem:
    r"""Weighted MaxSAT as a minimization of the weighted-violation energy.

    Attributes
    ----------
    cnf:
        The :class:`WeightedCNF` to minimize the violated weight of.
    name:
        Optional label.
    """

    cnf: WeightedCNF
    name: str | None = None

    @property
    def n(self) -> int:
        return int(self.cnf.n_vars)

    def energy(self, x: object) -> float | FloatArray:
        r"""Total violated weight at one point ``(n,)`` or a batch ``(m, n)``.

        ``0`` iff ``x`` satisfies every clause; otherwise the sum of the weights of the
        clauses ``x`` falsifies.
        """
        xv = np.asarray(x, dtype=float)
        single = xv.ndim == 1
        matrix = xv.reshape(1, -1) if single else xv
        total = np.zeros(matrix.shape[0])
        for clause in self.cnf.clauses:
            factor = np.ones(matrix.shape[0])
            for literal in clause.literals:
                col = matrix[:, abs(literal) - 1]
                factor = factor * ((1.0 - col) if literal > 0 else col)
            total = total + clause.weight * factor
        return float(total[0]) if single else total

    def to_polynomial(self) -> Polynomial:
        r"""The weighted-violation energy as an :class:`omnibias.sos.Polynomial`."""
        from omnibias.sos import Polynomial

        n = self.n
        poly = Polynomial.zero(n)
        for clause in self.cnf.clauses:
            term = Polynomial.constant(1.0, n)
            for literal in clause.literals:
                xi = Polynomial.variable(abs(literal) - 1, n)
                term = term * ((1.0 - xi) if literal > 0 else xi)
            poly = poly + term * clause.weight
        return poly

    def grad_scale(self) -> float:
        r"""A conservative step ``scale`` for the annealed relaxation.

        The violation factors lie in ``[0, 1]`` so ``|grad_x E|_inf <= total_weight``;
        ``2 * total_weight * max_clause_len`` over-estimates the gradient magnitude (a
        larger scale only shrinks the descent step, never destabilises it).
        """
        total_weight = sum(clause.weight for clause in self.cnf.clauses)
        max_len = max((len(clause.literals) for clause in self.cnf.clauses), default=1)
        return max(1.0, 2.0 * float(total_weight) * float(max_len))


__all__ = ["Clause", "MaxSATProblem", "WeightedCNF"]
