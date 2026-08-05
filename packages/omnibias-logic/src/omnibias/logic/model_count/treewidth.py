# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
r"""Exact (weighted) #SAT by bounded-treewidth variable elimination (bucket DP).

:func:`treewidth_model_count` computes the **exact** (weighted) model count by variable
elimination on the CNF's factor graph: each clause is a ``{0, 1}`` factor over its variables
(``0`` only on its single falsifying assignment), each variable carries its weight factor
``[w(0), w(1)]``, and summing the whole product over all assignments is the (weighted) count.
Eliminating variables one at a time along an order multiplies the factors that mention a
variable and sums it out; the cost is ``O(2^{w+1})`` per bucket where ``w`` is the
**induced width** of that order.

For instances of bounded treewidth this is polynomial and exact -- one of the two genuinely
tractable exact-counting regimes (the other being the affine/XOR fragment). It is **weighted**
by construction, so it also serves the weighted exact case.

Honest scope:

* the elimination order comes from a **min-fill heuristic**, so the reported ``width`` is an
  honest upper bound on the treewidth, *not* a certified-optimal one;
* cost is exponential in that width, so above ``max_width`` we raise
  :class:`TreewidthTooLarge` (the router then falls back to a cheaper regime) rather than
  attempt an intractable elimination.
"""

from __future__ import annotations

import itertools
from collections import defaultdict
from dataclasses import dataclass
from fractions import Fraction
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - typing only
    from omnibias.logic.model_count.problem import ModelCountProblem


class TreewidthTooLarge(Exception):
    """Raised when the heuristic induced width exceeds ``max_width`` (router fallback signal)."""

    def __init__(self, width: int, max_width: int) -> None:
        super().__init__(f"heuristic treewidth {width} exceeds max_width {max_width}")
        self.width = width
        self.max_width = max_width


@dataclass(frozen=True)
class _Factor:
    """A table factor over ``scope`` (sorted 1-based vars): ``bits -> Fraction`` value."""

    scope: tuple[int, ...]
    table: dict[tuple[int, ...], Fraction]


def _clause_factor(literals: tuple[int, ...]) -> _Factor:
    """The indicator factor of a clause: ``0`` on its unique falsifying point, else ``1``."""
    scope = tuple(sorted(abs(literal) for literal in literals))
    falsifying = {abs(literal): (0 if literal > 0 else 1) for literal in literals}
    table: dict[tuple[int, ...], Fraction] = {}
    for combo in itertools.product((0, 1), repeat=len(scope)):
        assignment = dict(zip(scope, combo, strict=True))
        falsified = all(assignment[v] == falsifying[v] for v in scope)
        table[combo] = Fraction(0) if falsified else Fraction(1)
    return _Factor(scope, table)


def _weight_factor(v: int, weights: tuple[Fraction, Fraction]) -> _Factor:
    """The unary weight factor ``[w(0), w(1)]`` for variable ``v``."""
    return _Factor((v,), {(0,): weights[0], (1,): weights[1]})


def _multiply(factors: list[_Factor]) -> _Factor:
    """Product of ``factors`` over the union of their scopes."""
    scope_set: set[int] = set()
    for factor in factors:
        scope_set.update(factor.scope)
    scope = tuple(sorted(scope_set))
    table: dict[tuple[int, ...], Fraction] = {}
    for combo in itertools.product((0, 1), repeat=len(scope)):
        assignment = dict(zip(scope, combo, strict=True))
        value = Fraction(1)
        for factor in factors:
            value *= factor.table[tuple(assignment[v] for v in factor.scope)]
            if value == 0:
                break
        table[combo] = value
    return _Factor(scope, table)


def _sum_out(factor: _Factor, v: int) -> _Factor:
    """Marginalise variable ``v`` out of ``factor``."""
    idx = factor.scope.index(v)
    scope = factor.scope[:idx] + factor.scope[idx + 1 :]
    table: dict[tuple[int, ...], Fraction] = defaultdict(lambda: Fraction(0))
    for combo, value in factor.table.items():
        table[combo[:idx] + combo[idx + 1 :]] += value
    return _Factor(scope, dict(table))


def _min_fill_order(adjacency: dict[int, set[int]]) -> tuple[list[int], int]:
    """A min-fill elimination order + its induced width (an upper bound on the treewidth)."""
    adj = {v: set(neighbors) for v, neighbors in adjacency.items()}
    remaining = set(adj)
    order: list[int] = []
    width = 0
    while remaining:
        best: int | None = None
        best_fill = -1
        best_degree = -1
        for v in remaining:
            neighbors = adj[v] & remaining
            neighbor_list = list(neighbors)
            fill = sum(
                1
                for i in range(len(neighbor_list))
                for j in range(i + 1, len(neighbor_list))
                if neighbor_list[j] not in adj[neighbor_list[i]]
            )
            if best is None or fill < best_fill or (fill == best_fill and len(neighbors) < best_degree):
                best, best_fill, best_degree = v, fill, len(neighbors)
        assert best is not None
        neighbors = adj[best] & remaining
        width = max(width, len(neighbors))
        neighbor_list = list(neighbors)
        for i in range(len(neighbor_list)):
            for j in range(i + 1, len(neighbor_list)):
                adj[neighbor_list[i]].add(neighbor_list[j])
                adj[neighbor_list[j]].add(neighbor_list[i])
        remaining.discard(best)
        order.append(best)
    return order, width


def treewidth_model_count(
    problem: ModelCountProblem, *, max_width: int = 18
) -> tuple[int | Fraction, int]:
    r"""Exact (weighted) model count via bucket elimination + the heuristic width used.

    Returns ``(count, width)`` where ``count`` is ``int`` (unweighted) or
    :class:`~fractions.Fraction` (weighted) and ``width`` is the min-fill induced width.
    Raises :class:`TreewidthTooLarge` when ``width > max_width`` (cost would be ``~2^{width}``).
    """
    n = problem.n
    fracs = problem.weight_fractions()

    def free_product() -> Fraction:
        prod = Fraction(1)
        for v in range(1, n + 1):
            if v not in adjacency:
                prod *= fracs[v - 1][0] + fracs[v - 1][1]
        return prod

    adjacency: dict[int, set[int]] = {}
    for clause in problem.cnf.clauses:
        variables = [abs(literal) for literal in clause.literals]
        for v in variables:
            adjacency.setdefault(v, set())
        for i in range(len(variables)):
            for j in range(i + 1, len(variables)):
                adjacency[variables[i]].add(variables[j])
                adjacency[variables[j]].add(variables[i])

    if not adjacency:  # no clauses -> every variable is free
        result = free_product()
        return (result if problem.is_weighted else int(result)), 0

    order, width = _min_fill_order(adjacency)
    if width > max_width:
        raise TreewidthTooLarge(width, max_width)

    factors = [_clause_factor(clause.literals) for clause in problem.cnf.clauses]
    for v in order:
        bucket = [factor for factor in factors if v in factor.scope]
        factors = [factor for factor in factors if v not in factor.scope]
        bucket.append(_weight_factor(v, fracs[v - 1]))
        factors.append(_sum_out(_multiply(bucket), v))

    result = free_product()
    for factor in factors:  # every remaining factor is a scalar (empty scope)
        result *= factor.table[()]
    return (result if problem.is_weighted else int(result)), width


__all__ = ["TreewidthTooLarge", "treewidth_model_count"]
