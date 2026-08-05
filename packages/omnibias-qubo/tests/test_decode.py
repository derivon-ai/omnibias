# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
"""Rounding, 1-flip local search, and the exact oracle."""

from __future__ import annotations

import numpy as np
import pytest
from omnibias.qubo import (
    QUBOProblem,
    brute_force_min,
    decode_qubo,
    is_binary,
    one_flip_descent,
    round_relaxed,
)


def _random_qubo(n: int, seed: int) -> QUBOProblem:
    rng = np.random.default_rng(seed)
    m = rng.standard_normal((n, n))
    return QUBOProblem(m + m.T, rng.standard_normal(n), const=0.4)


def test_round_relaxed_threshold() -> None:
    x = np.array([0.2, 0.7, 0.5, 0.9, 0.4999])
    assert list(round_relaxed(x)) == [0.0, 1.0, 1.0, 1.0, 0.0]


def test_one_flip_descent_reaches_a_local_minimum() -> None:
    rng = np.random.default_rng(0)
    n = 7
    prob = _random_qubo(n, 0)
    x0 = rng.integers(0, 2, size=n).astype(float)
    e0 = float(prob.energy(x0))
    x, e = one_flip_descent(prob, x0)
    assert is_binary(x)
    assert e <= e0 + 1e-12  # never increases the energy
    # No single flip improves on the returned point (1-flip local optimum).
    for i in range(n):
        y = x.copy()
        y[i] = 1.0 - y[i]
        assert float(prob.energy(y)) >= e - 1e-9


def test_decode_is_an_upper_bound_and_finds_optimum_when_tiny() -> None:
    for seed in range(6):
        prob = _random_qubo(3, seed)
        _, e_min = brute_force_min(prob)
        assignment, e_dec = decode_qubo(prob, n_starts=16, seed=seed)
        assert set(assignment) <= {0, 1}
        assert e_dec >= e_min - 1e-9  # sound: never below the true minimum
        assert abs(e_dec - e_min) < 1e-9  # exhaustive-enough starts recover the optimum


def test_brute_force_matches_exhaustive_scan() -> None:
    prob = _random_qubo(4, 11)
    assignment, e_min = brute_force_min(prob)
    energies = [
        float(prob.energy(np.array([(b >> k) & 1 for k in range(4)], dtype=float)))
        for b in range(1 << 4)
    ]
    assert abs(e_min - min(energies)) < 1e-12
    assert abs(float(prob.energy(np.array(assignment, dtype=float))) - e_min) < 1e-12


def test_brute_force_cap_is_enforced() -> None:
    prob = QUBOProblem(np.zeros((21, 21)))
    with pytest.raises(ValueError, match="exponential"):
        brute_force_min(prob)
