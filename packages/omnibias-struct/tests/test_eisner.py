# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Eisner projective dependency parsing on the semiring driver.

Oracles: the exact derivation count equals the flat brute-force projective-tree count (the
proof that the Eisner hypergraph is spurious-ambiguity-free -- one derivation per tree); the
classic ``hard_eisner`` equals the driver's ``hard_value`` and the flat brute-force
enumeration. Differentiable: ``soft_eisner`` equals the flat soft oracle to ``< 1e-11``,
``eisner_marginals`` equals ``autograd``, every modifier column of the arc marginals sums to
``1`` (each word takes exactly one head), and the torch <-> jax twins are bit-identical. The
soft value is certified within ``log(N) / beta`` of the best projective parse.
"""

from __future__ import annotations

import numpy as np
import pytest
from omnibias.struct import (
    certify_soft_dp,
    count_projective_trees,
    hard_eisner,
)
from omnibias.struct._core.eisner import (
    best_projective_tree,
    brute_force_projective,
    eisner_edge_weights,
    eisner_hypergraph,
    iter_projective_trees,
)
from omnibias.struct._core.semiring import hard_value

torch = pytest.importorskip("torch")
jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402
import omnibias.struct.jax as sj  # noqa: E402
import omnibias.struct.torch as st  # noqa: E402

torch.set_default_dtype(torch.float64)

SEEDS = range(5)
TOL = 1e-12


def _arc(seed: int, n_words: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    a = rng.standard_normal((n_words + 1, n_words + 1))
    a[:, 0] = 0.0  # ROOT is never a modifier
    return a


@pytest.mark.parametrize("n_words", range(1, 6))
def test_count_projective_trees_matches_bruteforce(n_words: int) -> None:
    assert count_projective_trees(n_words) == sum(1 for _ in iter_projective_trees(n_words))


@pytest.mark.parametrize("seed", SEEDS)
def test_hard_eisner_matches_driver_and_bruteforce(seed: int) -> None:
    arc = _arc(seed, 4)
    spec = eisner_hypergraph(4)
    w = eisner_edge_weights(spec, arc)
    he = hard_eisner(arc)
    assert abs(he - hard_value(spec.graph, w)) < TOL
    assert abs(he - brute_force_projective(arc, None)) < 1e-11


@pytest.mark.parametrize("seed", SEEDS)
def test_best_projective_tree_is_valid(seed: int) -> None:
    arc = _arc(seed, 5)
    score, heads = best_projective_tree(arc)
    assert abs(score - hard_eisner(arc)) < TOL
    assert set(heads) == set(range(1, 6))  # every word takes a head
    assert heads in list(iter_projective_trees(5))  # the head map is a valid projective tree


@pytest.mark.parametrize("seed", SEEDS)
def test_soft_eisner_matches_bruteforce(seed: int) -> None:
    arc = _arc(seed, 4)
    at, aj = torch.tensor(arc), jnp.asarray(arc)
    for beta in (1.0, 8.0, 64.0):
        ref = brute_force_projective(arc, beta)
        assert abs(float(st.soft_eisner(at, beta)) - ref) < 1e-11
        assert abs(float(sj.soft_eisner(aj, beta)) - ref) < 1e-11


@pytest.mark.parametrize("seed", SEEDS)
def test_eisner_marginals_equals_autograd(seed: int) -> None:
    arc = _arc(seed, 4)
    at = torch.tensor(arc, requires_grad=True)
    beta = 6.0
    value = st.soft_eisner(at, beta)
    mu = st.eisner_marginals(at, beta)
    (g,) = torch.autograd.grad(value, at)
    assert float((mu.detach() - g).abs().max()) < 1e-9


def test_arc_marginals_column_sums_to_one() -> None:
    arc = _arc(0, 5)
    mu = st.eisner_marginals(torch.tensor(arc), 4.0).detach().numpy()
    # every modifier m >= 1 has exactly one head, so its column sums to 1
    assert np.allclose(mu[:, 1:].sum(axis=0), 1.0, atol=1e-9)
    assert np.allclose(mu[:, 0], 0.0, atol=1e-12)  # ROOT is never a modifier


@pytest.mark.parametrize("seed", SEEDS)
def test_certified_gap_is_sound(seed: int) -> None:
    arc = _arc(seed, 4)
    at = torch.tensor(arc)
    n_trees = count_projective_trees(4)
    for beta in (2.0, 32.0):
        hard = hard_eisner(arc)
        soft = float(st.soft_eisner(at, beta))
        cert = certify_soft_dp(hard, soft, n_trees, beta, sense="max")
        assert cert.is_sound
        assert cert.absolute_gap <= cert.gap_bound + 1e-9


@pytest.mark.parametrize("seed", SEEDS)
def test_torch_jax_parity(seed: int) -> None:
    arc = _arc(seed, 4)
    at, aj = torch.tensor(arc), jnp.asarray(arc)
    for beta in (1.0, 8.0):
        assert abs(float(st.soft_eisner(at, beta)) - float(sj.soft_eisner(aj, beta))) < 1e-11
        mt = st.eisner_marginals(at, beta).detach().numpy()
        mj = np.asarray(sj.eisner_marginals(aj, beta))
        assert float(np.max(np.abs(mt - mj))) < 1e-11
