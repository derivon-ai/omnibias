# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
r"""Exact (weighted) #SAT by a compact component-caching #DPLL counter.

:func:`count_models_exact` returns the **exact** (weighted) model count -- ``int`` for plain
``#SAT``, :class:`~fractions.Fraction` for weighted model counting. It is a small
Cachet-style exhaustive DPLL counter:

* **unit propagation** -- a length-1 clause forces its variable (multiplying the running
  weight by that literal's weight);
* **connected-component decomposition** -- clauses over disjoint variable sets multiply their
  counts independently, and variables absent from the residual formula are *free* and
  contribute their weight sum ``w(0) + w(1)``;
* **component caching** -- each residual component's count is memoised by its (canonical)
  clause set, so isomorphic subproblems are counted once;
* **branching** -- on the remaining variables (respecting an optional external ``branch_order``
  for the relaxation warm start).

Honest scope: exact model counting is ``#P``-hard, so this is **exponential in the worst
case**. Unit propagation + component caching make it far cheaper than the naive ``2^n``
enumeration on structured instances, but it is not a poly-time counter. Pure-Python: it never
imports a tensor backend (a precomputed ``branch_order`` is passed in).
"""

from __future__ import annotations

from collections import defaultdict
from fractions import Fraction
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - typing only
    from collections.abc import Sequence

    from omnibias.logic.model_count.problem import ModelCountProblem

Clauses = frozenset[frozenset[int]]


class CountBudgetExceeded(Exception):
    """Raised when the DPLL counter exceeds its branch-node budget (router fallback signal)."""

    def __init__(self, nodes: int, budget: int) -> None:
        super().__init__(f"exact counter exceeded its node budget ({nodes} > {budget})")
        self.nodes = nodes
        self.budget = budget


def _vars_of(clauses: Clauses) -> frozenset[int]:
    """The set of variables (1-based) appearing in ``clauses``."""
    return frozenset(abs(literal) for clause in clauses for literal in clause)


def _assign(clauses: Clauses, literal: int) -> Clauses | None:
    """Condition ``clauses`` on ``literal`` being true; ``None`` if a clause becomes empty."""
    out: set[frozenset[int]] = set()
    for clause in clauses:
        if literal in clause:
            continue  # clause satisfied
        if -literal in clause:
            reduced = clause - {-literal}
            if not reduced:
                return None  # empty clause -> conflict, this branch has 0 models
            out.add(reduced)
        else:
            out.add(clause)
    return frozenset(out)


def _components(clauses: Clauses) -> list[Clauses]:
    """Partition ``clauses`` into connected components over the variable co-occurrence graph."""
    parent: dict[int, int] = {}

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for clause in clauses:
        variables = [abs(literal) for literal in clause]
        for v in variables:
            parent.setdefault(v, v)
        for v in variables[1:]:
            union(variables[0], v)

    groups: dict[int, set[frozenset[int]]] = defaultdict(set)
    for clause in clauses:
        groups[find(abs(next(iter(clause))))].add(clause)
    return [frozenset(group) for group in groups.values()]


class _Counter:
    """Mutable DPLL counting state (weights, cache, branch order, node budget)."""

    def __init__(
        self,
        fracs: list[tuple[Fraction, Fraction]],
        node_budget: int | None,
        branch_order: list[int] | None,
    ) -> None:
        self.fracs = fracs
        self.node_budget = node_budget
        self.nodes = 0
        self.order = branch_order
        self.cache: dict[Clauses, Fraction] = {}

    def _wsum(self, v: int) -> Fraction:
        w0, w1 = self.fracs[v - 1]
        return w0 + w1

    def _wval(self, v: int, value: int) -> Fraction:
        return self.fracs[v - 1][value]

    def count(self, clauses: Clauses, active: frozenset[int]) -> Fraction:
        """Weighted count over ``active`` (vars of ``clauses`` are a subset of ``active``)."""
        if not clauses:
            prod = Fraction(1)
            for v in active:
                prod *= self._wsum(v)
            return prod
        cvars = _vars_of(clauses)
        cached = self.cache.get(clauses)
        base = cached if cached is not None else self._core(clauses, cvars)
        if cached is None:
            self.cache[clauses] = base
        prod = base
        for v in active - cvars:  # free variables contribute their weight sum
            prod *= self._wsum(v)
        return prod

    def _core(self, clauses: Clauses, active: frozenset[int]) -> Fraction:
        for clause in clauses:
            if not clause:
                return Fraction(0)  # an empty clause is unsatisfiable
        unit = next((next(iter(c)) for c in clauses if len(c) == 1), None)
        if unit is not None:
            conditioned = _assign(clauses, unit)
            if conditioned is None:
                return Fraction(0)
            v, value = abs(unit), (1 if unit > 0 else 0)
            return self._wval(v, value) * self.count(conditioned, active - {v})
        components = _components(clauses)
        if len(components) > 1:
            prod = Fraction(1)
            for component in components:
                prod *= self.count(component, _vars_of(component))
            return prod
        self.nodes += 1
        if self.node_budget is not None and self.nodes > self.node_budget:
            raise CountBudgetExceeded(self.nodes, self.node_budget)
        v = self._pick(clauses)
        rest = active - {v}
        total = Fraction(0)
        low = _assign(clauses, -v)
        if low is not None:
            total += self._wval(v, 0) * self.count(low, rest)
        high = _assign(clauses, v)
        if high is not None:
            total += self._wval(v, 1) * self.count(high, rest)
        return total

    def _pick(self, clauses: Clauses) -> int:
        cvars = _vars_of(clauses)
        if self.order:
            for v in self.order:
                if v in cvars:
                    return v
        freq: dict[int, int] = defaultdict(int)
        for clause in clauses:
            for literal in clause:
                freq[abs(literal)] += 1
        return max(cvars, key=lambda v: (freq[v], -v))


def count_models_exact(
    problem: ModelCountProblem,
    *,
    branch_order: Sequence[int] | None = None,
    node_budget: int | None = None,
) -> int | Fraction:
    r"""Exact (weighted) model count of ``problem`` -- ``int`` unweighted, ``Fraction`` weighted.

    Parameters
    ----------
    problem:
        The :class:`~omnibias.logic.model_count.problem.ModelCountProblem` to count.
    branch_order:
        Optional preferred branching order (1-based variables, most-informative first, e.g.
        from the annealed relaxation warm start). Affects *speed only* -- the exact count is
        invariant to it.
    node_budget:
        Optional cap on branch nodes; exceeding it raises :class:`CountBudgetExceeded` so a
        router can fall back to the certified enclosure. ``None`` means unbounded.
    """
    n = problem.n
    fracs = problem.weight_fractions()
    clauses: Clauses = frozenset(
        frozenset(int(literal) for literal in clause.literals) for clause in problem.cnf.clauses
    )
    order = None if branch_order is None else [int(v) for v in branch_order]
    counter = _Counter(fracs, node_budget, order)
    result = counter.count(clauses, frozenset(range(1, n + 1)))
    return result if problem.is_weighted else int(result)


__all__ = ["CountBudgetExceeded", "count_models_exact"]
