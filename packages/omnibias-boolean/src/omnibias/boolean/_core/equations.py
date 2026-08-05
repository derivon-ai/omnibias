# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Boolean equation solving: the eliminant and reproductive general solutions.

A Boolean equation is written in the homogeneous form ``phi(x) = 0`` where the
*constraint* ``phi`` is ``1`` exactly on the violated assignments (e.g.
``phi = lhs XOR rhs``). Solving the equation means describing ``{x : phi(x) = 0}``.

Two classical objects (Boole, Loewenheim, Rudeanu):

* the **eliminant** of ``x_i`` -- the condition ``phi|_{x_i=0} \wedge phi|_{x_i=1}``
  on the remaining variables under which *no* value of ``x_i`` works; the equation
  is solvable for ``x_i`` exactly where this is ``0``;
* a **reproductive general solution** -- a formula
  ``x_i = \neg E_1 \wedge (E_0 \vee c)`` (with ``E_0 = phi|_{x_i=0}``,
  ``E_1 = phi|_{x_i=1}``) carrying a free parameter ``c``, the "constant of
  integration": it returns the forced value where ``x_i`` is determined and ``c``
  where ``x_i`` is free, and reproduces any actual solution when ``c`` is set to it.

``solve_system`` chains this by Boole's successive elimination; it is exact but
worst-case exponential, so it is intended for small systems.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

from omnibias.boolean._core.derivative import restrict
from omnibias.boolean._core.truth_table import (
    TruthTable,
    check_truth_table,
    index_of,
    num_vars,
)


def equation_from_callables(
    lhs: Callable[..., int], rhs: Callable[..., int], n: int
) -> TruthTable:
    """Constraint ``phi = lhs XOR rhs`` (``0`` where ``lhs == rhs``)."""
    from omnibias.boolean._core.truth_table import all_assignments

    rows = []
    for bits in all_assignments(n):
        rows.append(1 if bool(lhs(*bits)) != bool(rhs(*bits)) else 0)
    return tuple(rows)


def constraint_from_predicate(pred: Callable[..., bool], n: int) -> TruthTable:
    """Constraint that is ``0`` exactly where ``pred(x)`` is satisfied."""
    from omnibias.boolean._core.truth_table import all_assignments

    return tuple(0 if pred(*bits) else 1 for bits in all_assignments(n))


def eliminant(constraint: TruthTable, i: int) -> TruthTable:
    """Eliminant of ``x_i``: ``(phi|_{x_i=0} AND phi|_{x_i=1})`` over the rest.

    The equation ``constraint = 0`` has a solution in ``x_i`` for a given setting
    of the other variables iff this ``(n-1)``-variable function is ``0`` there.
    """
    e0 = restrict(constraint, i, 0)
    e1 = restrict(constraint, i, 1)
    return tuple(a & b for a, b in zip(e0, e1, strict=False))


def _full_index(rest_vars: Sequence[int], var: int, rest_bits: int, xi: int, n: int) -> int:
    x = 0
    for pos, orig in enumerate(rest_vars):
        if (rest_bits >> pos) & 1:
            x |= 1 << orig
    if xi:
        x |= 1 << var
    return x


@dataclass(frozen=True)
class BooleanSolution:
    """Reproductive general solution of ``constraint = 0`` for one variable.

    :attr:`solution` is a truth table over ``(rest..., c)`` -- the ``n-1`` remaining
    variables (in original order) followed by the free parameter ``c`` in the
    most-significant slot -- whose value is the chosen ``x_var``.
    :attr:`consistency` is ``1`` where a solution exists.
    """

    var: int
    n: int
    rest_vars: tuple[int, ...]
    consistency: TruthTable
    solution: TruthTable

    def value(self, rest_bits: int, c: int) -> int:
        """Chosen ``x_var`` for a given setting of the rest and the parameter ``c``."""
        return self.solution[rest_bits | ((c & 1) << (self.n - 1))]

    def satisfies(self, constraint: TruthTable) -> bool:
        """Check the defining property: every consistent ``(rest, c)`` solves ``phi=0``."""
        for r in range(1 << (self.n - 1)):
            if not self.consistency[r]:
                continue
            for c in (0, 1):
                xi = self.value(r, c)
                x = _full_index(self.rest_vars, self.var, r, xi, self.n)
                if constraint[x] != 0:
                    return False
        return True


def solve_for(constraint: TruthTable, i: int) -> BooleanSolution:
    """Reproductive general solution of ``constraint = 0`` for variable ``x_i``."""
    check_truth_table(constraint)
    n = num_vars(constraint)
    if not 0 <= i < n:
        raise ValueError(f"variable index {i} out of range for n={n}")
    e0 = restrict(constraint, i, 0)
    e1 = restrict(constraint, i, 1)
    rest_vars = tuple(j for j in range(n) if j != i)
    m = 1 << (n - 1)
    consistency = tuple(0 if (e0[r] and e1[r]) else 1 for r in range(m))
    sol = [0] * (1 << n)
    for r in range(m):
        for c in (0, 1):
            chosen = 1 if ((not e1[r]) and (e0[r] or c)) else 0
            sol[r | (c << (n - 1))] = chosen
    return BooleanSolution(
        var=i, n=n, rest_vars=rest_vars, consistency=consistency, solution=tuple(sol)
    )


def system_constraint(constraints: Sequence[TruthTable]) -> TruthTable:
    """Combine equations ``phi_k = 0`` into one constraint (their logical OR)."""
    if not constraints:
        raise ValueError("need at least one constraint")
    n = num_vars(constraints[0])
    for c in constraints:
        if num_vars(c) != n:
            raise ValueError("all constraints must have the same number of variables")
    size = 1 << n
    out = [0] * size
    for c in constraints:
        for x in range(size):
            out[x] |= c[x]
    return tuple(out)


def solution_set(constraint: TruthTable) -> list[int]:
    """All assignment indices ``x`` with ``constraint[x] == 0`` (brute force)."""
    check_truth_table(constraint)
    return [x for x, v in enumerate(constraint) if v == 0]


def is_satisfiable(constraint: TruthTable) -> bool:
    """``True`` iff the equation ``constraint = 0`` has at least one solution."""
    return any(v == 0 for v in constraint)


@dataclass(frozen=True)
class GeneralSolution:
    """Parametric general solution of a system via Boole's successive elimination.

    For each variable ``x_k`` (eliminated in increasing order) we keep the cofactor
    pair ``(A_k, B_k)`` over the later variables; back-substitution turns any
    parameter vector ``c in {0,1}^n`` into a solution, and sweeping ``c`` over the
    cube enumerates the whole solution set (reproductively).
    """

    n: int
    cofactors: tuple[tuple[TruthTable, TruthTable], ...]
    consistent: bool

    def reconstruct(self, params: Sequence[int]) -> tuple[int, ...]:
        """Back-substitute a parameter vector ``c`` into a concrete assignment."""
        if len(params) != self.n:
            raise ValueError(f"need {self.n} parameters, got {len(params)}")
        xs = [0] * self.n
        for k in range(self.n - 1, -1, -1):
            a_tab, b_tab = self.cofactors[k]
            rest_index = 0
            for pos in range(self.n - 1 - k):
                if xs[k + 1 + pos]:
                    rest_index |= 1 << pos
            a = a_tab[rest_index]
            b = b_tab[rest_index]
            c = params[k] & 1
            xs[k] = 1 if ((not b) and (a or c)) else 0
        return tuple(xs)

    def enumerate_solutions(self) -> list[int]:
        """All solution indices obtained by sweeping the free parameters."""
        if not self.consistent:
            return []
        sols = set()
        for p in range(1 << self.n):
            params = tuple((p >> k) & 1 for k in range(self.n))
            sols.add(index_of(self.reconstruct(params)))
        return sorted(sols)


def solve_system(constraints: Sequence[TruthTable]) -> GeneralSolution:
    """Reproductive general solution of a Boolean system by successive elimination."""
    phi = system_constraint(constraints)
    n = num_vars(phi)
    if n < 1:
        raise ValueError("system must have at least one variable")
    cofactors: list[tuple[TruthTable, TruthTable]] = []
    cur = phi
    for _ in range(n):
        a_tab = restrict(cur, 0, 0)
        b_tab = restrict(cur, 0, 1)
        cofactors.append((a_tab, b_tab))
        cur = tuple(a & b for a, b in zip(a_tab, b_tab, strict=False))
    final_const = cur[0]
    return GeneralSolution(n=n, cofactors=tuple(cofactors), consistent=(final_const == 0))


__all__ = [
    "BooleanSolution",
    "GeneralSolution",
    "constraint_from_predicate",
    "eliminant",
    "equation_from_callables",
    "is_satisfiable",
    "solution_set",
    "solve_for",
    "solve_system",
    "system_constraint",
]
