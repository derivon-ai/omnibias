# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Linear dynamic equations: forward recursion vs variation of constants."""

from __future__ import annotations

import math

import pytest
from omnibias.timescale import (
    delta_derivative,
    h_integers,
    hilger_exponential,
    quantum,
    solve_linear_dynamic,
    variation_of_constants,
)


def test_homogeneous_solution_is_hilger_exponential() -> None:
    H = h_integers(0.5)
    traj = solve_linear_dynamic(0.6, 0.0, 1.0, H, 0.0, 2.0)
    for t, y in traj:
        assert y == pytest.approx(hilger_exponential(0.6, t, 0.0, H))


def test_recursion_matches_variation_of_constants() -> None:
    H = h_integers(0.25)
    p = lambda t: 0.5 - 0.1 * t  # noqa: E731
    r = lambda t: math.sin(t)  # noqa: E731
    traj = solve_linear_dynamic(p, r, 2.0, H, 0.0, 2.0)
    for t, y in traj:
        assert y == pytest.approx(variation_of_constants(p, r, 2.0, t, H, 0.0))


def test_solution_satisfies_the_equation() -> None:
    H = h_integers(0.5)
    p, r = 0.4, 1.2
    traj = dict(solve_linear_dynamic(p, r, 0.5, H, 0.0, 3.0))
    y = lambda t: traj[min(traj, key=lambda s: abs(s - t))]  # noqa: E731
    for t in (0.0, 0.5, 1.0, 1.5, 2.0):
        # y^Delta = p y + r on the mesh.
        assert delta_derivative(y, t, H) == pytest.approx(p * y(t) + r)


def test_quantum_dynamic_equation() -> None:
    Q = quantum(2.0)
    traj = solve_linear_dynamic(0.1, 0.0, 1.0, Q, 1.0, 8.0)
    for t, yv in traj:
        assert yv == pytest.approx(hilger_exponential(0.1, t, 1.0, Q))


def test_reals_rejected() -> None:
    from omnibias.timescale import reals

    with pytest.raises(ValueError):
        solve_linear_dynamic(1.0, 0.0, 1.0, reals(), 0.0, 1.0)
