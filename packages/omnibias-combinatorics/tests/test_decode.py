# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
"""Decoders, exact classical baselines, and the brute-force oracle: correctness."""

from __future__ import annotations

import numpy as np
import pytest
from omnibias.combinatorics import (
    AssignmentProblem,
    GraphicMatroid,
    MatroidProblem,
    MinCostFlowProblem,
    TransportProblem,
    UniformMatroid,
    brute_force_min,
    classical_optimum,
    decode,
    max_flow_value,
)

K = 8


@pytest.mark.parametrize("seed", range(K))
def test_assignment_classical_equals_brute_and_scipy(seed: int) -> None:
    """Hungarian == brute force == the independent scipy LAP optimum."""
    from scipy.optimize import linear_sum_assignment

    cost = np.random.default_rng(seed).random((6, 6))
    _, opt = classical_optimum(AssignmentProblem(cost))
    _, bf = brute_force_min(AssignmentProblem(cost))
    r, c = linear_sum_assignment(cost)
    assert opt == pytest.approx(bf, abs=1e-9)
    assert opt == pytest.approx(float(cost[r, c].sum()), abs=1e-9)


@pytest.mark.parametrize("seed", range(K))
def test_matroid_greedy_equals_brute(seed: int) -> None:
    """Greedy (the matroid oracle) == brute force over independent sets."""
    rng = np.random.default_rng(seed)
    for mat in (UniformMatroid(8, 3), GraphicMatroid(4, ((0, 1), (1, 2), (2, 0), (2, 3)))):
        w = rng.standard_normal(mat.ground_size)
        prob = MatroidProblem(w, mat)
        _, opt = classical_optimum(prob)
        _, bf = brute_force_min(prob)
        assert opt == pytest.approx(bf, abs=1e-9)


def test_transport_lp_equals_brute_on_tiny_integer_instance() -> None:
    """The transport LP optimum equals brute force over integral 2x2 tables."""
    cost = np.array([[4.0, 1.0], [2.0, 3.0]])
    supply = np.array([2.0, 1.0])
    demand = np.array([1.0, 2.0])
    _, opt = classical_optimum(TransportProblem(cost, supply, demand))
    best = np.inf
    for x00 in range(3):  # enumerate integral tables consistent with the marginals
        x01 = supply[0] - x00
        x10 = demand[0] - x00
        x11 = supply[1] - x10
        table = np.array([[x00, x01], [x10, x11]])
        if np.all(table >= -1e-9) and np.allclose(table.sum(1), supply) and np.allclose(table.sum(0), demand):
            best = min(best, float((table * cost).sum()))
    assert opt == pytest.approx(best, abs=1e-9)


@pytest.mark.parametrize("seed", range(K))
def test_min_cost_flow_matches_lp_maxflow_and_csgraph(seed: int) -> None:
    """The csgraph max-flow value equals an independent LP max-throughput computation."""
    from scipy.optimize import linprog

    arcs = ((0, 1), (0, 2), (1, 3), (2, 3), (1, 2))
    rng = np.random.default_rng(seed)
    cap = rng.integers(1, 5, size=len(arcs)).astype(float)
    mf = max_flow_value(4, arcs, cap, 0, 3)
    # LP: maximize net out-flow of the source s.t. conservation at 1,2 and 0<=f<=cap
    n_nodes, e = 4, len(arcs)
    a_eq = np.zeros((n_nodes, e))
    for i, (u, v) in enumerate(arcs):
        a_eq[u, i] += 1.0
        a_eq[v, i] -= 1.0
    a_eq = a_eq[[1, 2]]  # conservation only at interior nodes
    src_out = np.array([1.0 if u == 0 else (-1.0 if v == 0 else 0.0) for (u, v) in arcs])
    res = linprog(-src_out, A_eq=a_eq, b_eq=np.zeros(2), bounds=list(zip([0.0] * e, cap.tolist(), strict=False)), method="highs")
    assert (-res.fun) == pytest.approx(mf, abs=1e-6)


@pytest.mark.parametrize("seed", range(K))
def test_decode_matches_classical_with_sharp_relaxation(seed: int) -> None:
    """A well-annealed relaxation decodes to the exact classical optimum (integral polytope).

    Uses well-separated integer costs so the optimum is unique and the sharp entropic
    coupling pins to it exactly (near-ties would leave a fractional coupling).
    """
    pytest.importorskip("jax")
    from omnibias.combinatorics import AnnealSchedule
    from omnibias.combinatorics.jax import assignment_relaxation, matroid_relaxation

    heavy = AnnealSchedule(beta0=0.5, beta_growth=1.7, stages=16, steps=80)
    cost = np.random.default_rng(seed).integers(0, 100, size=(6, 6)).astype(float)
    prob = AssignmentProblem(cost)
    P = np.asarray(assignment_relaxation(cost, heavy))
    _, decoded = decode(prob, relaxed=P)
    _, opt = classical_optimum(prob)
    assert decoded == pytest.approx(opt, abs=1e-9)

    w = np.random.default_rng(seed + 100).standard_normal(8)
    mprob = MatroidProblem(w, UniformMatroid(8, 3))
    r = np.asarray(matroid_relaxation(w, UniformMatroid(8, 3), heavy))
    _, mdec = decode(mprob, relaxed=r)
    _, mopt = classical_optimum(mprob)
    assert mdec == pytest.approx(mopt, abs=1e-9)


@pytest.mark.parametrize("seed", range(K))
def test_decoded_is_a_valid_upper_bound(seed: int) -> None:
    """However it is guided, the decoded objective is a feasible upper bound (>= optimum)."""
    rng = np.random.default_rng(seed)
    cost = rng.random((6, 6))
    prob = AssignmentProblem(cost)
    _, opt = classical_optimum(prob)
    guide = rng.random((6, 6))  # an arbitrary (bad) heatmap
    _, decoded = decode(prob, relaxed=guide)
    assert decoded >= opt - 1e-9


def test_brute_force_labelled_and_capped() -> None:
    assert "exponential" in (brute_force_min.__doc__ or "").lower()
    with pytest.raises(ValueError, match="exceeds"):
        brute_force_min(AssignmentProblem(np.zeros((12, 12))))
    with pytest.raises(ValueError, match="does not enumerate"):
        brute_force_min(MinCostFlowProblem(2, ((0, 1),), np.array([1.0]), np.array([1.0]), 0, 1, 1.0))
