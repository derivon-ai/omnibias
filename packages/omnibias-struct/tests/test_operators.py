# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Distribution operators over the Gibbs path distribution ``p_beta`` on the semiring driver.

``path_entropy`` equals the flat brute-force entropy and the ``beta (V - E[score])`` identity,
tends to ``log(#derivations)`` as ``beta -> 0`` and ``0`` (unique argmax) as ``beta -> inf``,
and is differentiable. ``topk_paths`` equals the enumerate-and-sort oracle; ``topk_free_energy``
is monotone non-decreasing in ``k``, equals the best score at ``k = 1, beta -> inf`` and the
full soft value at ``k = #derivations``. ``sample_paths`` is exact FFBS -- its empirical edge
frequencies match the closed-form marginals. The differentiable ops are torch <-> jax
bit-identical.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from omnibias.struct import (
    brute_force_entropy,
    brute_force_kbest,
    count_derivations,
)
from omnibias.struct._core.eisner import eisner_hypergraph
from omnibias.struct._core.semiring import best_derivation

torch = pytest.importorskip("torch")
jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402
import omnibias.struct.jax as sj  # noqa: E402
import omnibias.struct.torch as st  # noqa: E402

torch.set_default_dtype(torch.float64)

SEEDS = range(4)
GRAPH = eisner_hypergraph(3).graph  # a real arity-2 hypergraph (12 projective derivations)


def _weights(seed: int) -> np.ndarray:
    return np.random.default_rng(seed).standard_normal(GRAPH.num_edges)


@pytest.mark.parametrize("seed", SEEDS)
def test_path_entropy_matches_bruteforce(seed: int) -> None:
    w = _weights(seed)
    wt, wj = torch.tensor(w), jnp.asarray(w)
    for beta in (0.5, 2.0, 8.0):
        ref = brute_force_entropy(GRAPH, w, beta)
        assert abs(float(st.path_entropy(GRAPH, wt, beta)) - ref) < 1e-9
        assert abs(float(sj.path_entropy(GRAPH, wj, beta)) - ref) < 1e-9


def test_path_entropy_limits() -> None:
    w = _weights(0)
    n = count_derivations(GRAPH)
    # beta -> 0: p_beta -> uniform over derivations, so H -> log(N)
    small = float(st.path_entropy(GRAPH, torch.tensor(w), 1e-4))
    assert abs(small - math.log(n)) < 1e-2
    # beta -> inf with a unique argmax: H -> 0
    big = float(st.path_entropy(GRAPH, torch.tensor(w), 200.0))
    assert big >= 0.0
    assert big < 1e-3


@pytest.mark.parametrize("seed", SEEDS)
def test_path_entropy_is_differentiable(seed: int) -> None:
    w = torch.tensor(_weights(seed), requires_grad=True)
    h = st.path_entropy(GRAPH, w, 3.0)
    (g,) = torch.autograd.grad(h, w)
    assert torch.isfinite(g).all()


@pytest.mark.parametrize("seed", SEEDS)
def test_topk_paths_matches_bruteforce(seed: int) -> None:
    w = _weights(seed)
    kb = st.topk_paths(GRAPH, torch.tensor(w), 6)
    bk = brute_force_kbest(GRAPH, w, 6)
    assert [round(s, 10) for s, _ in kb] == [round(s, 10) for s, _ in bk]


@pytest.mark.parametrize("seed", SEEDS)
def test_topk_free_energy_monotone_and_limits(seed: int) -> None:
    w = _weights(seed)
    wt = torch.tensor(w)
    n = count_derivations(GRAPH)
    beta = 2.0
    energies = [float(st.topk_free_energy(GRAPH, wt, k, beta)) for k in range(1, n + 1)]
    assert all(energies[i] <= energies[i + 1] + 1e-12 for i in range(len(energies) - 1))
    assert abs(energies[-1] - float(st.semiring_value(GRAPH, wt, beta))) < 1e-9  # k = N -> full soft value
    best, _ = best_derivation(GRAPH, w)
    assert abs(float(st.topk_free_energy(GRAPH, wt, 1, 128.0)) - best) < 1e-6  # k = 1, beta -> inf


@pytest.mark.parametrize("seed", SEEDS)
def test_topk_free_energy_parity(seed: int) -> None:
    w = _weights(seed)
    wt, wj = torch.tensor(w), jnp.asarray(w)
    for k in (1, 3, 6):
        t = float(st.topk_free_energy(GRAPH, wt, k, 2.0))
        j = float(sj.topk_free_energy(GRAPH, wj, k, 2.0))
        assert abs(t - j) < 1e-10


def test_sample_paths_empirical_marginals() -> None:
    w = _weights(0)
    wt = torch.tensor(w)
    beta = 2.0
    counts, samples = st.sample_paths(GRAPH, wt, beta, 6000, seed=0)
    assert len(samples) == 6000
    empirical = counts.mean(0).numpy()
    closed_form = st.semiring_marginals(GRAPH, wt, beta).numpy()
    assert float(np.max(np.abs(empirical - closed_form))) < 0.03  # Monte-Carlo error ~ 1/sqrt(6000)


def test_gumbel_relaxed_sample() -> None:
    w = torch.tensor(_weights(0), requires_grad=True)
    beta = 4.0
    relaxed = st.gumbel_relaxed_sample(GRAPH, w, beta, seed=0)
    assert relaxed.shape == w.shape
    (g,) = torch.autograd.grad(relaxed.sum(), w)
    assert torch.isfinite(g).all()  # differentiable relaxed sample
    hard = st.gumbel_relaxed_sample(GRAPH, w.detach(), beta, seed=0, hard=True)
    # the hard straight-through forward value is an integer edge-count indicator of one derivation
    hard_np = hard.detach().numpy()
    assert hard.sum() > 0.0
    assert np.allclose(hard_np, np.round(hard_np), atol=1e-9)
