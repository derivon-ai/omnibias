# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
"""Standard-form LP systems: shapes, feasibility of integral vertices, integrality."""

from __future__ import annotations

import numpy as np
import pytest
from omnibias.combinatorics import (
    AssignmentProblem,
    GraphicMatroid,
    MatroidProblem,
    MinCostFlowProblem,
    PartitionMatroid,
    TransportProblem,
    UniformMatroid,
    max_flow_value,
)
from omnibias.combinatorics._core.decode import solve_lp


def test_assignment_system_shape_and_vertex() -> None:
    cost = np.arange(9, dtype=float).reshape(3, 3)
    sys = AssignmentProblem(cost).system()
    assert sys.name == "assignment"
    assert sys.n_vars == 9
    assert sys.A_eq.shape == (6, 9)  # 3 row + 3 column sums
    assert np.allclose(sys.b_eq, 1.0)
    assert sys.A_ineq.shape == (0, 9)  # only the box, carried separately
    assert np.allclose(sys.x_lower, 0.0) and np.allclose(sys.x_upper, 1.0)
    # a permutation matrix is a feasible vertex: A_eq x = 1
    perm = np.eye(3)[[1, 2, 0]].reshape(-1)
    assert np.allclose(sys.A_eq @ perm, sys.b_eq)


def test_transport_system_marginals() -> None:
    cost = np.ones((2, 3))
    supply = np.array([2.0, 1.0])
    demand = np.array([1.0, 1.0, 1.0])
    sys = TransportProblem(cost, supply, demand).system()
    assert sys.n_vars == 6
    assert sys.A_eq.shape == (5, 6)  # 2 supply + 3 demand rows
    assert np.allclose(sys.b_eq, np.concatenate([supply, demand]))
    plan = np.array([[1.0, 1.0, 0.0], [0.0, 0.0, 1.0]]).reshape(-1)
    assert np.allclose(sys.A_eq @ plan, sys.b_eq)


def test_transport_unbalanced_rejected() -> None:
    with pytest.raises(ValueError, match="balanced"):
        TransportProblem(np.ones((2, 2)), np.array([1.0, 1.0]), np.array([1.0, 2.0]))


def test_flow_system_conservation() -> None:
    arcs = ((0, 1), (0, 2), (1, 3), (2, 3))
    cost = np.array([1.0, 1.0, 1.0, 1.0])
    cap = np.array([2.0, 2.0, 2.0, 2.0])
    prob = MinCostFlowProblem(4, arcs, cost, cap, source=0, sink=3, value=2.0)
    sys = prob.system()
    assert sys.n_vars == 4
    assert sys.A_eq.shape == (4, 4)  # one conservation row per node
    # net balance: +2 at source 0, -2 at sink 3, 0 elsewhere
    assert np.allclose(sys.b_eq, np.array([2.0, 0.0, 0.0, -2.0]))
    flow = np.array([1.0, 1.0, 1.0, 1.0])  # split 1 unit down each path
    assert np.allclose(sys.A_eq @ flow, sys.b_eq)


def test_uniform_matroid_system() -> None:
    w = np.array([3.0, 1.0, 2.0, -1.0])
    sys = MatroidProblem(w, UniformMatroid(4, 2)).system()
    assert sys.n_vars == 4
    assert np.allclose(sys.c, -w)  # min-space
    assert sys.A_ineq.shape == (1, 4)
    assert np.allclose(sys.A_ineq, 1.0) and np.allclose(sys.b_ineq, 2.0)
    x = np.array([1.0, 0.0, 1.0, 0.0])  # an independent set of size 2
    assert np.all(sys.A_ineq @ x <= sys.b_ineq + 1e-12)


def test_partition_matroid_system() -> None:
    mat = PartitionMatroid(groups=((0, 1), (2, 3)), caps=(1, 1))
    sys = MatroidProblem(np.ones(4), mat).system()
    assert sys.A_ineq.shape == (2, 4)
    assert np.allclose(sys.A_ineq @ np.array([1.0, 0.0, 1.0, 0.0]), np.array([1.0, 1.0]))


def test_graphic_matroid_forest_polytope_bounds_a_tree() -> None:
    edges = ((0, 1), (1, 2), (2, 0), (2, 3))  # triangle + pendant
    mat = GraphicMatroid(4, edges)
    sys = MatroidProblem(np.ones(4), mat).system()
    # a spanning tree (3 edges, no cycle) must satisfy every forest inequality
    tree = np.array([1.0, 1.0, 0.0, 1.0])
    assert np.all(sys.A_ineq @ tree <= sys.b_ineq + 1e-12)
    # the full triangle {0,1,2} violates the |T|-1 bound for T = {0,1,2}
    cycle = np.array([1.0, 1.0, 1.0, 0.0])
    assert np.any(sys.A_ineq @ cycle > sys.b_ineq + 1e-9)


@pytest.mark.parametrize("seed", range(5))
def test_lp_relaxation_is_integral(seed: int) -> None:
    """The LP optimum over each polytope is integral (the polytopes are integral)."""
    rng = np.random.default_rng(seed)
    # assignment
    xa, _, _, _ = solve_lp(AssignmentProblem(rng.random((5, 5))).system())
    assert np.allclose(xa, np.round(xa), atol=1e-6)
    # transport (integer marginals)
    sup = np.array([2.0, 2.0])
    dem = np.array([1.0, 1.0, 2.0])
    xt, _, _, _ = solve_lp(TransportProblem(rng.random((2, 3)), sup, dem).system())
    assert np.allclose(xt, np.round(xt), atol=1e-6)
    # matroid
    xm, _, _, _ = solve_lp(MatroidProblem(rng.standard_normal(6), UniformMatroid(6, 3)).system())
    assert np.allclose(xm, np.round(xm), atol=1e-6)


def test_max_flow_value_matches_capacity_cut() -> None:
    arcs = ((0, 1), (0, 2), (1, 3), (2, 3), (1, 2))
    cap = np.array([3.0, 2.0, 2.0, 3.0, 1.0])
    # min cut around the source is cap(0->1)+cap(0->2) = 5
    assert max_flow_value(4, arcs, cap, 0, 3) == pytest.approx(5.0)
