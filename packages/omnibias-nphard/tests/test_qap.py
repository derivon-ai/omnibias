# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
"""QAP as a QUBO-form DiscreteProblem: encoding exactness, decoder, oracle, baseline."""

from __future__ import annotations

import itertools

import numpy as np
import pytest
from omnibias.nphard import QAPProblem, placement_qap, qap
from omnibias.nphard._core.qap import (
    perm_to_x,
    qap_brute_force,
    qap_classical,
    qap_decode,
    qap_round,
)


def _connectivity(n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    m = rng.integers(0, 5, size=(n, n)).astype(float)
    m = m + m.T
    np.fill_diagonal(m, 0.0)
    return m


def _random_qap(dim: int, seed: int) -> QAPProblem:
    rng = np.random.default_rng(seed)
    flow = rng.integers(0, 9, size=(dim, dim)).astype(float)
    dist = rng.integers(0, 9, size=(dim, dim)).astype(float)
    flow = (flow + flow.T) / 2.0
    dist = (dist + dist.T) / 2.0
    np.fill_diagonal(flow, 0.0)
    np.fill_diagonal(dist, 0.0)
    return qap(flow, dist)


def test_energy_matches_to_polynomial_on_random_cube_points() -> None:
    """The DiscreteProblem energy and its SOS polynomial agree everywhere (real points)."""
    pytest.importorskip("omnibias.sos")
    rng = np.random.default_rng(0)
    prob = _random_qap(3, 1)
    poly = prob.to_polynomial()
    for _ in range(20):
        x = rng.standard_normal(prob.n)  # arbitrary real point (polynomial == energy)
        assert abs(poly.evaluate(list(x)) - float(prob.energy(x))) < 1e-9


def test_to_qubo_energy_matches_on_binary_points() -> None:
    """QAPProblem.energy equals its QUBOProblem energy on the whole cube."""
    rng = np.random.default_rng(2)
    prob = _random_qap(3, 3)
    qubo = prob.to_qubo()
    assert qubo.n == prob.n
    for _ in range(40):
        x = rng.integers(0, 2, size=prob.n).astype(float)
        assert abs(float(prob.energy(x)) - float(qubo.energy(x))) < 1e-9


def test_permutation_has_zero_penalty_and_objective_energy() -> None:
    """A valid permutation pays no penalty: its energy is the pure QAP objective."""
    prob = _random_qap(4, 4)
    for perm in itertools.permutations(range(prob.dim)):
        x = perm_to_x(perm, prob.dim)
        assert abs(float(prob.energy(x)) - float(prob.objective(x))) < 1e-9


def test_flip_deltas_match_bruteforce_recompute() -> None:
    """Closed-form single-bit flip deltas match explicit re-evaluation."""
    rng = np.random.default_rng(5)
    prob = _random_qap(3, 6)
    x = rng.integers(0, 2, size=prob.n).astype(float)
    deltas = np.asarray(prob.flip_deltas(x))
    base = float(prob.energy(x))
    for i in range(prob.n):
        y = x.copy()
        y[i] = 1.0 - y[i]
        assert abs(deltas[i] - (float(prob.energy(y)) - base)) < 1e-9


def test_decoder_returns_a_valid_permutation() -> None:
    """Hungarian + 2-opt always decodes to a feasible permutation indicator."""
    for seed in range(5):
        prob = _random_qap(4, seed)
        heat = np.random.default_rng(seed).random((prob.dim, prob.dim))
        x, energy = qap_decode(prob, relaxed=heat)
        mat = np.asarray(x, dtype=float).reshape(prob.dim, prob.dim)
        assert np.allclose(mat.sum(axis=0), 1.0) and np.allclose(mat.sum(axis=1), 1.0)
        assert abs(energy - float(prob.energy(np.asarray(x, dtype=float)))) < 1e-9


def test_qap_round_is_hungarian_only_and_feasible() -> None:
    """qap_round is the raw heatmap decision (no local search): still a permutation."""
    prob = _random_qap(4, 1)
    heat = np.random.default_rng(1).random((prob.dim, prob.dim))
    x = np.asarray(qap_round(heat, prob.dim), dtype=float).reshape(prob.dim, prob.dim)
    assert np.allclose(x.sum(axis=0), 1.0) and np.allclose(x.sum(axis=1), 1.0)


def test_default_penalty_keeps_the_minimiser_a_permutation() -> None:
    """The safe default penalty makes brute-force minimiser a zero-penalty permutation."""
    for seed in range(6):
        prob = _random_qap(4, seed)
        x, _ = qap_brute_force(prob)
        xv = np.asarray(x, dtype=float)
        assert abs(float(prob.energy(xv)) - float(prob.objective(xv))) < 1e-9  # no penalty paid


def test_decoder_and_classical_are_competitive_with_brute_force() -> None:
    """On tiny instances the decoded / FAQ solutions are valid upper bounds >= optimum."""
    for seed in range(5):
        prob = _random_qap(4, seed)
        _, e_opt = qap_brute_force(prob)
        heat = np.asarray(np.random.default_rng(seed).random((prob.dim, prob.dim)))
        _, e_dec = qap_decode(prob, relaxed=heat)
        _, e_cla = qap_classical(prob)
        assert e_dec >= e_opt - 1e-9  # decoded is an upper bound
        assert e_cla >= e_opt - 1e-9  # classical is an upper bound


def test_brute_force_refuses_large_dim() -> None:
    """The exponential oracle is guarded (dim!) and refuses large instances."""
    prob = _random_qap(9, 0)
    with pytest.raises(ValueError, match="exponential"):
        qap_brute_force(prob)


def test_frontend_default_penalty_is_positive() -> None:
    prob = _random_qap(4, 0)
    assert prob.penalty > 0.0


def test_placement_qap_builds_symmetric_integer_manhattan_distance() -> None:
    """placement_qap: N = rows*cols modules; D is the integer Manhattan slot distance."""
    prob = placement_qap(_connectivity(6, 0), (2, 3))
    assert prob.dim == 6 and prob.n == 36
    dist = prob.distance
    assert np.array_equal(dist, dist.T)  # symmetric
    assert np.all(dist == np.round(dist))  # integer Manhattan distances
    assert np.all(np.diagonal(dist) == 0.0)
    # slot 0 = (0,0), slot 1 = (0,1) -> 1; slot 5 = (1,2) -> |1-0| + |2-0| = 3
    assert dist[0, 1] == 1.0 and dist[0, 5] == 3.0


def test_placement_qap_uses_connectivity_as_the_flow() -> None:
    conn = _connectivity(6, 3)
    prob = placement_qap(conn, (2, 3))
    assert np.array_equal(prob.flow, conn)


def test_placement_qap_decodes_to_a_valid_placement() -> None:
    prob = placement_qap(_connectivity(6, 1), (2, 3))
    heat = np.random.default_rng(1).random((prob.dim, prob.dim))
    x, _ = qap_decode(prob, relaxed=heat)
    mat = np.asarray(x, dtype=float).reshape(prob.dim, prob.dim)
    assert np.allclose(mat.sum(axis=0), 1.0) and np.allclose(mat.sum(axis=1), 1.0)


def test_placement_qap_rejects_a_grid_that_does_not_match_the_connectivity() -> None:
    with pytest.raises(ValueError, match="slots"):
        placement_qap(_connectivity(5, 0), (2, 3))  # 5 modules, 6 slots
    with pytest.raises(ValueError, match="square"):
        placement_qap(np.zeros((2, 3)), (2, 3))  # non-square connectivity
