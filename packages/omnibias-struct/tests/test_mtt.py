# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Matrix-tree non-projective dependency parsing (Kirchhoff determinant partition).

The honest distinction from the ``lse_beta`` DPs: the partition here is **exact and closed
form** -- ``matrix_tree_partition`` equals the flat brute-force sum over *all* spanning
arborescences to machine precision (the determinant is that sum, not an approximation of it).
Oracles: the exact count equals the brute-force count and Cayley's ``(n + 1)^(n - 1)``;
Chu-Liu/Edmonds ``max_arborescence`` equals the brute-force maximum. Differentiable:
``matrix_tree_marginals`` equals ``autograd`` and each modifier column sums to ``1``; the
torch <-> jax twins are bit-identical. The ``beta -> inf`` gap is taken against the maximum
arborescence, bounded by ``log(N) / beta``.
"""

from __future__ import annotations

import numpy as np
import pytest
from omnibias.struct import (
    brute_force_arborescence,
    certify_soft_dp,
    count_arborescences,
    hard_matrix_tree,
    iter_arborescences,
    matrix_tree_partition,
    max_arborescence,
)

torch = pytest.importorskip("torch")
jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402
import omnibias.struct.jax as sj  # noqa: E402
import omnibias.struct.torch as st  # noqa: E402

torch.set_default_dtype(torch.float64)

SEEDS = range(6)


def _arc(seed: int, n_words: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    a = rng.standard_normal((n_words + 1, n_words + 1))
    a[:, 0] = 0.0  # ROOT is never a modifier
    return a


@pytest.mark.parametrize("n_words", range(1, 6))
def test_count_arborescences_matches_bruteforce_and_cayley(n_words: int) -> None:
    n = count_arborescences(n_words)
    assert n == sum(1 for _ in iter_arborescences(n_words))
    assert n == (n_words + 1) ** (n_words - 1)


@pytest.mark.parametrize("seed", SEEDS)
def test_hard_matrix_tree_matches_bruteforce(seed: int) -> None:
    arc = _arc(seed, 4)
    assert abs(hard_matrix_tree(arc) - brute_force_arborescence(arc, None)) < 1e-11


@pytest.mark.parametrize("seed", SEEDS)
def test_max_arborescence_is_valid(seed: int) -> None:
    arc = _arc(seed, 5)
    score, heads = max_arborescence(arc)
    assert abs(score - hard_matrix_tree(arc)) < 1e-12
    assert set(heads) == set(range(1, 6))  # every word takes a head
    # following heads from any word reaches the ROOT (0) without a cycle
    for m in range(1, 6):
        seen: set[int] = set()
        cur = m
        while cur != 0:
            assert cur not in seen
            seen.add(cur)
            cur = heads[cur]


@pytest.mark.parametrize("seed", SEEDS)
def test_soft_matrix_tree_is_exact(seed: int) -> None:
    # The determinant partition is the *exact* sum over arborescences, not an lse bound.
    arc = _arc(seed, 4)
    at, aj = torch.tensor(arc), jnp.asarray(arc)
    for beta in (1.0, 4.0, 16.0):
        ref = brute_force_arborescence(arc, beta)
        assert abs(matrix_tree_partition(arc, beta) - ref) < 1e-9
        assert abs(float(st.soft_matrix_tree(at, beta)) - ref) < 1e-9
        assert abs(float(sj.soft_matrix_tree(aj, beta)) - ref) < 1e-9


@pytest.mark.parametrize("seed", SEEDS)
def test_matrix_tree_marginals_equals_autograd(seed: int) -> None:
    arc = _arc(seed, 4)
    at = torch.tensor(arc, requires_grad=True)
    beta = 5.0
    value = st.soft_matrix_tree(at, beta)
    mu = st.matrix_tree_marginals(at, beta)
    (g,) = torch.autograd.grad(value, at)
    assert float((mu.detach() - g).abs().max()) < 1e-9


def test_arc_marginals_column_sums_to_one() -> None:
    arc = _arc(0, 5)
    mu = st.matrix_tree_marginals(torch.tensor(arc), 4.0).detach().numpy()
    assert np.allclose(mu[:, 1:].sum(axis=0), 1.0, atol=1e-9)  # each word has exactly one head
    assert np.allclose(mu[:, 0], 0.0, atol=1e-12)  # ROOT is never a modifier


@pytest.mark.parametrize("seed", SEEDS)
def test_certified_gap_is_sound(seed: int) -> None:
    # gap is against the maximum arborescence, bounded by log(N)/beta
    arc = _arc(seed, 4)
    at = torch.tensor(arc)
    n_trees = count_arborescences(4)
    hard = hard_matrix_tree(arc)
    brute = brute_force_arborescence(arc, None)
    for beta in (2.0, 16.0):
        soft = float(st.soft_matrix_tree(at, beta))
        cert = certify_soft_dp(hard, soft, n_trees, beta, sense="max", brute_force_value=brute)
        assert cert.is_sound
        assert cert.agrees_with_bruteforce
        assert cert.absolute_gap <= cert.gap_bound + 1e-9


@pytest.mark.parametrize("seed", SEEDS)
def test_torch_jax_parity(seed: int) -> None:
    arc = _arc(seed, 4)
    at, aj = torch.tensor(arc), jnp.asarray(arc)
    for beta in (1.0, 8.0):
        assert abs(float(st.soft_matrix_tree(at, beta)) - float(sj.soft_matrix_tree(aj, beta))) < 1e-10
        mt = st.matrix_tree_marginals(at, beta).detach().numpy()
        mj = np.asarray(sj.matrix_tree_marginals(aj, beta))
        assert float(np.max(np.abs(mt - mj))) < 1e-10
