# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
"""max-cut and max-independent-set QUBO encodings."""

from __future__ import annotations

import numpy as np
import pytest
from omnibias.qubo import brute_force_min, max_cut, max_independent_set


def _random_graph(n: int, seed: int, *, weighted: bool) -> np.ndarray:
    rng = np.random.default_rng(seed)
    upper = rng.random((n, n)) if weighted else (rng.random((n, n)) < 0.5).astype(float)
    a = np.triu(upper, 1)
    return a + a.T


def test_max_cut_energy_is_minus_cut() -> None:
    rng = np.random.default_rng(0)
    n = 6
    w = _random_graph(n, 1, weighted=True)
    prob = max_cut(w)
    for _ in range(50):
        x = rng.integers(0, 2, size=n).astype(float)
        cut = sum(w[i, j] * (x[i] != x[j]) for i in range(n) for j in range(i + 1, n))
        assert abs(float(prob.energy(x)) - (-cut)) < 1e-9


def test_max_cut_optimum_on_triangle() -> None:
    w = np.array([[0.0, 1.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 0.0]])
    prob = max_cut(w)
    _, e_min = brute_force_min(prob)
    assert abs(-e_min - 2.0) < 1e-9  # the triangle's max cut is 2


def test_max_independent_set_optimum_is_independent() -> None:
    for seed in range(5):
        n = 6
        adj = _random_graph(n, seed, weighted=False)
        prob = max_independent_set(adj, penalty=2.0)
        assignment, e_min = brute_force_min(prob)
        chosen = [i for i, v in enumerate(assignment) if v == 1]
        # no edge inside the chosen set
        assert all(adj[i, j] == 0 for i in chosen for j in chosen if i < j)
        # size equals -energy at the optimum
        assert abs(-e_min - len(chosen)) < 1e-9


def test_max_independent_set_penalty_validation() -> None:
    with pytest.raises(ValueError, match="penalty"):
        max_independent_set(np.zeros((3, 3)), penalty=0.5)
