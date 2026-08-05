# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Boolean equation solving: eliminant, reproductive solve_for, and systems."""

from __future__ import annotations

import random

from omnibias.boolean._core.equations import (
    eliminant,
    equation_from_callables,
    solution_set,
    solve_for,
    solve_system,
)
from omnibias.boolean._core.truth_table import index_of

# Equation: x0 & x1 = x2  (variables x0, x1, x2 == y). Solve for x1 given x0, x2.
AND_EQ = equation_from_callables(lambda a, b, y: a & b, lambda a, b, y: y, 3)


def test_eliminant_and_equation() -> None:
    # Solve x0 & x1 = y for x1; eliminant over (x0, y) is "no x1 works".
    elim = eliminant(AND_EQ, 1)
    # rest vars are (x0, y) with x0 = bit0, y = bit1 (reduced order).
    # No x1 works exactly when y=1 and x0=0  -> rest index = 0b10 = 2.
    assert elim[2] == 1
    assert sum(elim) == 1


def test_solve_for_is_a_valid_reproductive_solution() -> None:
    sol = solve_for(AND_EQ, 1)
    # Plugging the solution back satisfies the original equation everywhere consistent.
    assert sol.satisfies(AND_EQ)


def test_solve_for_matches_known_branches() -> None:
    sol = solve_for(AND_EQ, 1)
    # rest = (x0, y): index = x0 | (y<<1). When x0=1 the AND forces x1 = y.
    # x0=1, y=0 -> x1 must be 0 (any c).
    r = index_of((1, 0))  # x0=1, y=0 over rest (x0,y)
    assert sol.value(r, 0) == 0 and sol.value(r, 1) == 0
    # x0=1, y=1 -> x1 must be 1.
    r = index_of((1, 1))
    assert sol.value(r, 0) == 1 and sol.value(r, 1) == 1
    # x0=0, y=0 -> x1 is free (equals the parameter c).
    r = index_of((0, 0))
    assert sol.value(r, 0) == 0 and sol.value(r, 1) == 1


def test_solve_system_enumerates_solution_set() -> None:
    rng = random.Random(5)
    for n in (1, 2, 3, 4):
        for _ in range(25):
            # A random system of 1-3 constraints.
            k = rng.randint(1, 3)
            constraints = [
                tuple(rng.randint(0, 1) for _ in range(1 << n)) for _ in range(k)
            ]
            from omnibias.boolean._core.equations import system_constraint

            phi = system_constraint(constraints)
            gen = solve_system(constraints)
            assert gen.enumerate_solutions() == solution_set(phi)


def test_inconsistent_system_is_empty() -> None:
    # x0 = 0 and x0 = 1 simultaneously.
    c1 = (0, 1)  # phi = x0 (violated when x0=1) -> forces x0=0
    c2 = (1, 0)  # phi = ~x0 (violated when x0=0) -> forces x0=1
    gen = solve_system([c1, c2])
    assert not gen.consistent
    assert gen.enumerate_solutions() == []
