# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Spectral graph operators (torch backend).

Oracles:

* **Ring graph** ``C_n``: combinatorial-Laplacian spectrum
  ``lambda_k = 2 - 2 cos(2 pi k / n)`` (analytic).
* **Normalized Laplacian** spectrum ``subset [0, 2]``.
* **Heat kernel**: ``H(0) = I``, symmetric PD, semigroup ``H(s)H(t) = H(s+t)``,
  and ``H 1 = 1`` for the combinatorial Laplacian (``L 1 = 0``).
* **Two-block SBM**: the Fiedler vector separates the planted blocks by sign.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

torch = pytest.importorskip("torch")
torch.set_default_dtype(torch.float64)

import omnibias.graph.torch.ops as G


def ring_adjacency(n: int) -> torch.Tensor:
    a = torch.zeros(n, n, dtype=torch.float64)
    for i in range(n):
        a[i, (i + 1) % n] = 1.0
        a[i, (i - 1) % n] = 1.0
    return a


def two_block_sbm(n_per: int, p_in: float, p_out: float) -> torch.Tensor:
    n = 2 * n_per
    a = torch.full((n, n), p_out, dtype=torch.float64)
    a[:n_per, :n_per] = p_in
    a[n_per:, n_per:] = p_in
    a.fill_diagonal_(0.0)
    return a


class TestLaplacians:
    def test_graph_laplacian_definition(self) -> None:
        a = two_block_sbm(3, 1.0, 0.1)
        lap = G.graph_laplacian(a)
        d = a.sum(dim=-1)
        assert torch.allclose(torch.diag(lap), d)
        # off-diagonal is -A
        off = lap - torch.diag(torch.diag(lap))
        assert torch.allclose(off, -a + torch.diag(torch.diag(a)))
        # rows of the combinatorial Laplacian sum to zero
        assert torch.allclose(lap.sum(dim=-1), torch.zeros(6), atol=1e-12)

    def test_normalized_spectrum_in_0_2(self) -> None:
        a = ring_adjacency(10)
        ln = G.normalized_laplacian(a)
        ev = torch.linalg.eigvalsh(ln)
        assert float(ev.min()) > -1e-10
        assert float(ev.max()) < 2.0 + 1e-10
        # smallest eigenvalue is 0 (connected graph)
        assert abs(float(ev.min())) < 1e-10

    def test_random_walk_rows_of_transition_sum_to_one(self) -> None:
        a = two_block_sbm(3, 1.0, 0.2)
        lrw = G.random_walk_laplacian(a)
        # L_rw = I - P, so P = I - L_rw has rows summing to 1
        n = a.shape[0]
        p = torch.eye(n) - lrw
        assert torch.allclose(p.sum(dim=-1), torch.ones(n), atol=1e-12)

    def test_isolated_node_convention(self) -> None:
        # node 2 isolated -> zero degree -> zero normalization, no NaN/inf
        a = torch.tensor(
            [[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
        )
        ln = G.normalized_laplacian(a)
        lrw = G.random_walk_laplacian(a)
        assert torch.isfinite(ln).all()
        assert torch.isfinite(lrw).all()


class TestRingSpectrum:
    @pytest.mark.parametrize("n", [4, 6, 8, 12])
    def test_ring_eigenvalues(self, n: int) -> None:
        a = ring_adjacency(n)
        evals, _ = G.laplacian_spectrum(a)
        exact = np.sort([2 - 2 * math.cos(2 * math.pi * k / n) for k in range(n)])
        assert np.allclose(evals.numpy(), exact, atol=1e-10)


class TestHeatKernel:
    def test_identity_at_zero(self) -> None:
        a = ring_adjacency(6)
        h0 = G.graph_heat_kernel(a, 0.0)
        assert torch.allclose(h0, torch.eye(6), atol=1e-10)

    def test_symmetric_and_semigroup(self) -> None:
        a = two_block_sbm(3, 1.0, 0.3)
        h_half = G.graph_heat_kernel(a, 0.25)
        h_full = G.graph_heat_kernel(a, 0.5)
        assert torch.allclose(h_half, h_half.T, atol=1e-12)
        assert torch.allclose(h_half @ h_half, h_full, atol=1e-10)

    def test_conserves_constant_vector(self) -> None:
        a = ring_adjacency(7)
        h = G.graph_heat_kernel(a, 1.3)
        ones = torch.ones(7)
        assert torch.allclose(h @ ones, ones, atol=1e-10)


class TestSBMFiedler:
    def test_fiedler_separates_blocks(self) -> None:
        a = two_block_sbm(5, 1.0, 0.02)
        v = G.fiedler_vector(a).numpy()
        block0, block1 = v[:5], v[5:]
        # each block has a consistent sign; the two blocks have opposite sign
        assert np.all(np.sign(block0) == np.sign(block0[0]))
        assert np.all(np.sign(block1) == np.sign(block1[0]))
        assert np.sign(block0[0]) != np.sign(block1[0])

    def test_embedding_shape_and_drop_first(self) -> None:
        a = two_block_sbm(4, 1.0, 0.05)
        emb = G.spectral_embedding(a, n_components=2, normalized=True)
        assert emb.shape == (8, 2)
        # with drop_first the constant null vector is excluded: columns are
        # orthogonal to the all-ones vector up to normalization scale
        col0 = emb[:, 0]
        assert abs(float(col0.sum())) < 1e-6


class TestCutRelaxation:
    def test_relaxed_value_is_sum_of_smallest_eigenvalues(self) -> None:
        a = two_block_sbm(4, 1.0, 0.1)
        rel = G.spectral_clustering_relaxation(a, k=2, normalized=True)
        evals, _ = G.laplacian_spectrum(a, normalized=True)
        assert abs(float(rel.relaxed_cut_value) - float(evals[:2].sum())) < 1e-10
        assert rel.embedding.shape == (8, 2)

    def test_relaxed_value_lower_bounds_are_nonnegative(self) -> None:
        a = ring_adjacency(9)
        rel = G.spectral_clustering_relaxation(a, k=3, normalized=True)
        assert float(rel.relaxed_cut_value) >= -1e-12


class TestDifferentiable:
    def test_laplacian_spectrum_differentiable(self) -> None:
        a = two_block_sbm(3, 1.0, 0.2).requires_grad_(True)
        rel = G.spectral_clustering_relaxation(a, k=2, normalized=False)
        rel.relaxed_cut_value.backward()
        assert a.grad is not None
        assert torch.isfinite(a.grad).all()

    def test_heat_kernel_differentiable(self) -> None:
        a = ring_adjacency(5).requires_grad_(True)
        h = G.graph_heat_kernel(a, 0.4)
        h.sum().backward()
        assert a.grad is not None
        assert torch.isfinite(a.grad).all()


class TestValidation:
    def test_non_square_raises(self) -> None:
        with pytest.raises(ValueError, match="square"):
            G.graph_laplacian(torch.zeros(3, 4))

    def test_too_many_components_raises(self) -> None:
        with pytest.raises(ValueError, match="cannot take"):
            G.spectral_embedding(ring_adjacency(3), n_components=3, drop_first=True)

    def test_negative_time_raises(self) -> None:
        with pytest.raises(ValueError, match="t must be"):
            G.graph_heat_kernel(ring_adjacency(4), -0.1)

    def test_bad_k_raises(self) -> None:
        with pytest.raises(ValueError, match="exceeds"):
            G.spectral_clustering_relaxation(ring_adjacency(4), k=5)
