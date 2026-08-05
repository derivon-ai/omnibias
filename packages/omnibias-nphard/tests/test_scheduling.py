# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
"""Parallel-machine scheduling as a QUBO-form DiscreteProblem: load variance + LPT."""

from __future__ import annotations

import numpy as np
import pytest
from omnibias.nphard import SchedulingProblem, schedule
from omnibias.nphard._core.scheduling import (
    assignment_to_x,
    scheduling_brute_force,
    scheduling_classical,
    scheduling_decode,
)


def _random_schedule(jobs: int, machines: int, seed: int) -> SchedulingProblem:
    rng = np.random.default_rng(seed)
    return schedule(rng.integers(1, 20, size=jobs).astype(float), machines)


def test_energy_matches_to_polynomial_on_random_points() -> None:
    pytest.importorskip("omnibias.sos")
    rng = np.random.default_rng(0)
    prob = _random_schedule(5, 2, 1)
    poly = prob.to_polynomial()
    for _ in range(20):
        x = rng.standard_normal(prob.n)
        assert abs(poly.evaluate(list(x)) - float(prob.energy(x))) < 1e-9


def test_to_qubo_energy_matches_on_binary_points() -> None:
    rng = np.random.default_rng(1)
    prob = _random_schedule(5, 3, 2)
    qubo = prob.to_qubo()
    assert qubo.n == prob.n
    for _ in range(40):
        x = rng.integers(0, 2, size=prob.n).astype(float)
        assert abs(float(prob.energy(x)) - float(qubo.energy(x))) < 1e-9


def test_one_hot_schedule_energy_is_sum_of_squared_loads() -> None:
    """A valid one-hot assignment pays no penalty; energy == sum_k load_k^2."""
    prob = _random_schedule(6, 3, 3)
    rng = np.random.default_rng(3)
    for _ in range(20):
        assignment = rng.integers(0, prob.machines, size=prob.n_jobs)
        x = assignment_to_x(assignment, prob.machines)
        loads = prob.loads(assignment)
        assert abs(float(prob.energy(x)) - float(np.sum(loads**2))) < 1e-9


def test_decoder_returns_a_one_hot_schedule() -> None:
    for seed in range(5):
        prob = _random_schedule(7, 3, seed)
        heat = np.random.default_rng(seed).random((prob.n_jobs, prob.machines))
        x, energy = scheduling_decode(prob, relaxed=heat)
        mat = np.asarray(x, dtype=float).reshape(prob.n_jobs, prob.machines)
        assert np.allclose(mat.sum(axis=1), 1.0)  # each job on exactly one machine
        assert abs(energy - float(prob.energy(np.asarray(x, dtype=float)))) < 1e-9


def test_lpt_is_within_the_4_3_makespan_approximation() -> None:
    """LPT is the classical (4/3 - 1/3M) makespan approximation on every tiny instance."""
    for seed in range(8):
        prob = _random_schedule(7, 3, seed)
        assignment, _ = scheduling_classical(prob)
        mat = np.asarray(assignment, dtype=float).reshape(prob.n_jobs, prob.machines)
        lpt = [int(np.argmax(mat[j])) for j in range(prob.n_jobs)]
        x_opt, _ = scheduling_brute_force(prob)
        omat = np.asarray(x_opt, dtype=float).reshape(prob.n_jobs, prob.machines)
        opt = [int(np.argmax(omat[j])) for j in range(prob.n_jobs)]
        ratio = prob.makespan(lpt) / max(prob.makespan(opt), 1e-9)
        assert ratio <= 4.0 / 3.0 - 1.0 / (3.0 * prob.machines) + 1e-9


def test_brute_force_matches_decoder_lower_bound() -> None:
    """The brute-force optimum is <= any decoded schedule's load-variance energy."""
    for seed in range(5):
        prob = _random_schedule(8, 2, seed)
        _, e_opt = scheduling_brute_force(prob)
        heat = np.asarray(np.random.default_rng(seed).random((prob.n_jobs, prob.machines)))
        _, e_dec = scheduling_decode(prob, relaxed=heat)
        assert e_dec >= e_opt - 1e-9


def test_brute_force_is_guarded() -> None:
    prob = _random_schedule(30, 3, 0)  # 3^30 way over the guard
    with pytest.raises(ValueError, match="exponential"):
        scheduling_brute_force(prob)
