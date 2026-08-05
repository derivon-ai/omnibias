# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Spectral graph operators (jax backend) -- key oracles.

Cross-backend bit-parity is asserted separately in ``test_cross_backend``; this
module checks the analytic oracles hold on the jax path itself.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")
jax.config.update("jax_enable_x64", True)

import omnibias.graph.jax.ops as G


def ring_adjacency(n: int):
    a = np.zeros((n, n))
    for i in range(n):
        a[i, (i + 1) % n] = 1.0
        a[i, (i - 1) % n] = 1.0
    return jnp.asarray(a)


def two_block_sbm(n_per: int, p_in: float, p_out: float):
    n = 2 * n_per
    a = np.full((n, n), p_out)
    a[:n_per, :n_per] = p_in
    a[n_per:, n_per:] = p_in
    np.fill_diagonal(a, 0.0)
    return jnp.asarray(a)


@pytest.mark.parametrize("n", [4, 6, 8, 12])
def test_ring_eigenvalues(n: int) -> None:
    evals, _ = G.laplacian_spectrum(ring_adjacency(n))
    exact = np.sort([2 - 2 * math.cos(2 * math.pi * k / n) for k in range(n)])
    assert np.allclose(np.asarray(evals), exact, atol=1e-10)


def test_normalized_spectrum_in_0_2() -> None:
    ln = G.normalized_laplacian(ring_adjacency(10))
    ev = np.linalg.eigvalsh(np.asarray(ln))
    assert ev.min() > -1e-10 and ev.max() < 2.0 + 1e-10


def test_heat_kernel_semigroup_and_constant() -> None:
    a = two_block_sbm(3, 1.0, 0.3)
    h_half = np.asarray(G.graph_heat_kernel(a, 0.25))
    h_full = np.asarray(G.graph_heat_kernel(a, 0.5))
    assert np.allclose(h_half @ h_half, h_full, atol=1e-10)
    ones = np.ones(6)
    assert np.allclose(np.asarray(G.graph_heat_kernel(a, 1.0)) @ ones, ones, atol=1e-10)


def test_fiedler_separates_blocks() -> None:
    v = np.asarray(G.fiedler_vector(two_block_sbm(5, 1.0, 0.02)))
    b0, b1 = v[:5], v[5:]
    assert np.all(np.sign(b0) == np.sign(b0[0]))
    assert np.all(np.sign(b1) == np.sign(b1[0]))
    assert np.sign(b0[0]) != np.sign(b1[0])


def test_cut_relaxation_value() -> None:
    a = two_block_sbm(4, 1.0, 0.1)
    rel = G.spectral_clustering_relaxation(a, k=2, normalized=True)
    evals, _ = G.laplacian_spectrum(a, normalized=True)
    assert abs(float(rel.relaxed_cut_value) - float(evals[:2].sum())) < 1e-10


def test_differentiable_through_jax_grad() -> None:
    a = two_block_sbm(3, 1.0, 0.2)

    def cut(x):
        return G.spectral_clustering_relaxation(x, k=2, normalized=False).relaxed_cut_value

    grad = jax.grad(cut)(a)
    assert np.all(np.isfinite(np.asarray(grad)))


def test_non_square_raises() -> None:
    with pytest.raises(ValueError, match="square"):
        G.graph_laplacian(jnp.zeros((3, 4)))
