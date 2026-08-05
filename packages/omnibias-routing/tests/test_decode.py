# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
"""Backend-agnostic decoder + exact Held-Karp oracle."""

from __future__ import annotations

import itertools

import numpy as np
import pytest
from omnibias.routing import (
    RoutingProblem,
    decode_tour,
    held_karp_dp,
    is_valid_tour,
    nearest_neighbor,
    tour_cost,
    two_opt,
)


def _euclidean(n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    pts = rng.random((n, 2))
    d = pts[:, None, :] - pts[None, :, :]
    return np.sqrt(np.sum(d * d, axis=-1))


def _brute_force(cost: np.ndarray) -> float:
    """Exact optimum by permutation enumeration (tiny n) -- oracle for the oracle."""
    n = cost.shape[0]
    best = np.inf
    for perm in itertools.permutations(range(1, n)):
        tour = (0, *perm)
        best = min(best, tour_cost(tour, cost))
    return float(best)


@pytest.mark.parametrize("seed", range(6))
def test_held_karp_matches_brute_force(seed: int) -> None:
    cost = _euclidean(6, seed)
    tour, opt = held_karp_dp(cost)
    assert is_valid_tour(tour, 6)
    assert opt == pytest.approx(_brute_force(cost), abs=1e-9)
    assert tour_cost(tour, cost) == pytest.approx(opt, abs=1e-9)


def test_held_karp_asymmetric() -> None:
    """The DP is correct for a directed (asymmetric) cost matrix, not just Euclidean."""
    rng = np.random.default_rng(0)
    cost = rng.random((7, 7))
    np.fill_diagonal(cost, 0.0)
    _, opt = held_karp_dp(cost)
    assert opt == pytest.approx(_brute_force(cost), abs=1e-9)


def test_held_karp_cap_raises() -> None:
    with pytest.raises(ValueError, match="exponential"):
        held_karp_dp(np.zeros((19, 19)))


@pytest.mark.parametrize("seed", range(5))
def test_decode_returns_valid_tour_at_or_above_optimum(seed: int) -> None:
    cost = _euclidean(8, seed)
    _, opt = held_karp_dp(cost)
    tour, c = decode_tour(cost)
    assert is_valid_tour(tour, 8)
    assert c == pytest.approx(tour_cost(tour, cost), abs=1e-9)
    assert c >= opt - 1e-9  # a heuristic tour never beats the exact optimum


def test_two_opt_never_worsens() -> None:
    cost = _euclidean(9, 3)
    start = nearest_neighbor(cost, 0)
    refined, rc = two_opt(start, cost)
    assert is_valid_tour(refined, 9)
    assert rc <= tour_cost(start, cost) + 1e-12


def test_decode_follows_heatmap() -> None:
    """A heatmap concentrated on the optimal tour's arcs guides the decoder to it."""
    cost = _euclidean(8, 1)
    opt_tour, opt = held_karp_dp(cost)
    heat = np.zeros((8, 8))
    for i in range(8):
        heat[opt_tour[i], opt_tour[(i + 1) % 8]] = 1.0
    tour, c = decode_tour(cost, heat=heat)
    assert is_valid_tour(tour, 8)
    assert c == pytest.approx(opt, abs=1e-9)


def test_from_coords_symmetric() -> None:
    prob = RoutingProblem.from_coords(np.random.default_rng(0).random((5, 2)))
    assert prob.symmetric
    assert prob.n == 5


def test_routing_problem_rejects_tiny() -> None:
    with pytest.raises(ValueError, match="at least 3 cities"):
        RoutingProblem(cost=np.zeros((2, 2)))
