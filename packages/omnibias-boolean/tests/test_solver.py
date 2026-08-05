# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Differentiable solver (torch): GF(2) fast-path and annealed propose-and-verify."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from omnibias.boolean._core.systems import gf2_solve  # noqa: E402
from omnibias.boolean.torch.ops.solver import (  # noqa: E402
    BooleanSystem,
    brute_force_solutions,
    solve,
)


def test_gf2_fast_path_linear_unique() -> None:
    # x0 ^ x1 = 1 and x1 = 1  ->  x1 = 1, x0 = 0 (unique).
    sys = BooleanSystem.from_predicates(
        [lambda a, b: (a ^ b) == 1, lambda a, b: b == 1], n=2
    )
    assert sys.is_linear()
    res = solve(sys)
    assert res.method == "gf2"
    assert res.verified
    assert res.assignment == (0, 1)


def test_gf2_enumerates_full_solution_space() -> None:
    # Parity x0 ^ x1 ^ x2 = 0 has a 2-dimensional null space (4 even-parity points).
    sys = BooleanSystem.from_predicates([lambda a, b, c: (a ^ b ^ c) == 0], n=3)
    gsol = gf2_solve(sys.constraints)
    assert gsol is not None and gsol.consistent
    from omnibias.boolean._core.truth_table import index_of

    bf = sorted(index_of(b) for b in brute_force_solutions(sys))
    assert gsol.enumerate_solutions() == bf
    assert len(bf) == 4


def test_inconsistent_linear_system() -> None:
    sys = BooleanSystem.from_predicates(
        [lambda a: a == 0, lambda a: a == 1], n=1
    )
    res = solve(sys)
    assert res.method == "gf2"
    assert not res.verified
    assert res.assignment is None


def test_anneal_solves_nonlinear_system() -> None:
    # x0 & x1 = 1  and  x1 | x2 = 1  ->  x0 = 1, x1 = 1, x2 free. Nonlinear (AND).
    sys = BooleanSystem.from_predicates(
        [lambda a, b, c: (a & b) == 1, lambda a, b, c: (b | c) == 1], n=3
    )
    assert not sys.is_linear()
    res = solve(sys, steps=200, restarts=12, seed=0)
    assert res.method == "anneal"
    assert res.verified
    assert res.assignment is not None and sys.verify(res.assignment)


def test_solve_runs_across_sizes() -> None:
    # Cliff smoke: the solver returns a well-formed result regardless of success.
    for n in (2, 3, 4):
        sys = BooleanSystem.from_predicates(
            [lambda *x: sum(x) % 2 == 0], n=n
        )
        res = solve(sys, steps=60, restarts=3)
        assert res.assignment is None or len(res.assignment) == n
