# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
"""The Go-like MCTS search track: oracle match, prior-beats-uniform, determinism, gap."""

from __future__ import annotations

import numpy as np
import pytest
from omnibias.nphard import brute_force_min, certify_gap, placement_qap, schedule
from omnibias.nphard.jax import relax as relax_j
from omnibias.nphard.search import (
    hungarian_rollout,
    mcts_search,
    mdp_for,
    random_rollout,
    relaxation_prior,
    uniform_prior,
)


def _placement(n: int, grid: tuple[int, int], seed: int) -> object:
    rng = np.random.default_rng(seed)
    m = rng.integers(0, 5, size=(n, n)).astype(float)
    m = m + m.T
    np.fill_diagonal(m, 0.0)
    return placement_qap(m, grid)


def _schedule(jobs: int, machines: int, seed: int) -> object:
    rng = np.random.default_rng(seed)
    hi = 10 if jobs <= 6 else 12
    return schedule(rng.integers(1, hi, size=jobs).astype(float), machines)


def _guided(problem: object, shape: tuple[int, int], iterations: int, seed: int) -> object:
    mdp = mdp_for(problem)
    heat = np.asarray(relax_j(problem)).reshape(shape)
    return mcts_search(
        mdp,
        prior_fn=relaxation_prior(heat, temperature=1.0),
        rollout_fn=random_rollout(mdp, np.random.default_rng(seed)),
        iterations=iterations,
        seed=seed,
    )


def _uniform(problem: object, iterations: int, seed: int) -> object:
    mdp = mdp_for(problem)
    return mcts_search(
        mdp,
        prior_fn=uniform_prior,
        rollout_fn=random_rollout(mdp, np.random.default_rng(seed)),
        iterations=iterations,
        seed=seed,
    )


def test_guided_mcts_matches_the_oracle_on_a_tiny_instance() -> None:
    """On a tiny scheduling instance the relaxation-guided search reaches the exact optimum."""
    pytest.importorskip("jax")
    prob = _schedule(6, 2, 0)  # verified: guided == brute-force optimum
    _, e_opt = brute_force_min(prob)
    result = _guided(prob, (6, 2), iterations=80, seed=0)
    assert result.energy == pytest.approx(e_opt)


def test_guided_prior_beats_the_uniform_prior() -> None:
    """Deterministic demonstrator: the differentiable prior reaches the optimum where the
    uniform-prior search does not (J=8, M=3, seed=3: guided=621=opt, uniform=627)."""
    pytest.importorskip("jax")
    prob = _schedule(8, 3, 3)
    _, e_opt = brute_force_min(prob)
    guided = _guided(prob, (8, 3), iterations=80, seed=3)
    uniform = _uniform(prob, iterations=80, seed=3)
    assert guided.energy <= uniform.energy  # the prior helps (or ties)
    assert guided.energy == pytest.approx(e_opt)  # guided reaches the optimum
    assert uniform.energy > e_opt + 1e-6  # uniform does not, here


def test_search_is_deterministic_given_the_seed() -> None:
    pytest.importorskip("jax")
    prob = _schedule(6, 2, 1)
    a = _guided(prob, (6, 2), iterations=60, seed=5)
    b = _guided(prob, (6, 2), iterations=60, seed=5)
    assert a.assignment == b.assignment and a.energy == b.energy


def test_mcts_solution_feeds_a_sound_certificate() -> None:
    """The heuristic MCTS solution is still handed to certify_gap for a sound gap."""
    pytest.importorskip("jax")
    prob = _schedule(6, 2, 0)
    _, e_opt = brute_force_min(prob)
    result = _guided(prob, (6, 2), iterations=80, seed=0)
    cert = certify_gap(prob, result.assignment, kind="spectral")
    assert cert.lower_bound <= e_opt + 1e-6  # sound lower bound
    assert cert.energy >= e_opt - 1e-6  # heuristic upper bound
    assert cert.is_sound


def test_mdp_rejects_a_family_without_a_construction_encoding() -> None:
    from omnibias.nphard import gap

    prob = gap(np.ones((2, 2)), np.ones((2, 2)), np.array([2.0, 2.0]))
    with pytest.raises(TypeError, match="construction MDP"):
        mdp_for(prob)


def test_relaxation_prior_is_a_valid_distribution_over_legal_actions() -> None:
    pytest.importorskip("jax")
    prob = _schedule(6, 2, 0)
    mdp = mdp_for(prob)
    heat = np.asarray(relax_j(prob)).reshape(6, 2)
    prior = relaxation_prior(heat, temperature=1.0)
    p = prior((), mdp.legal_actions(()))
    assert abs(sum(p) - 1.0) < 1e-9 and all(v >= 0.0 for v in p)


def test_hungarian_rollout_completes_valid_permutations_respecting_the_prefix() -> None:
    """The Hungarian completion of any partial placement is a full valid permutation."""
    pytest.importorskip("jax")
    prob = _placement(6, (2, 3), 0)
    mdp = mdp_for(prob)
    heat = np.asarray(relax_j(prob)).reshape(6, 6)
    rollout = hungarian_rollout(mdp, heat)
    for partial in ((), (2,), (2, 0, 5)):
        x, energy = rollout(partial)
        mat = np.asarray(x, dtype=float).reshape(6, 6)
        assert np.allclose(mat.sum(axis=0), 1.0) and np.allclose(mat.sum(axis=1), 1.0)
        assert energy == pytest.approx(float(prob.energy(np.asarray(x, dtype=float))))
        for step, loc in enumerate(partial):  # the fixed prefix is respected
            assert mat[step, loc] == 1.0


def test_placement_mcts_returns_a_certified_valid_placement() -> None:
    """Rescoped, honest: the relaxation-guided search's placement is valid and soundly
    certified. We deliberately do NOT assert it beats a uniform-prior search -- we measured
    (the omnibias_experiments project's omnibias-nphard/placement.py) that the QAP relaxation heatmap is a weak
    construction prior, so the sound certificate, not beating uninformed search, is the value.
    """
    pytest.importorskip("jax")
    prob = _placement(6, (2, 3), 0)
    mdp = mdp_for(prob)
    heat = np.asarray(relax_j(prob)).reshape(6, 6)
    result = mcts_search(
        mdp,
        prior_fn=relaxation_prior(heat, temperature=1.0),
        rollout_fn=hungarian_rollout(mdp, heat),
        iterations=120,
        seed=0,
    )
    mat = np.asarray(result.assignment, dtype=float).reshape(6, 6)
    assert np.allclose(mat.sum(axis=0), 1.0) and np.allclose(mat.sum(axis=1), 1.0)
    _, e_opt = brute_force_min(prob)
    cert = certify_gap(prob, result.assignment, kind="glb")
    assert cert.method == "gilmore_lawler" and cert.is_sound and cert.certified
    assert cert.lower_bound <= e_opt + 1e-9 <= cert.energy + 1e-9
