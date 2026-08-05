# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Cross-backend bit-parity: torch vs jax to float64 ULP (``rtol=1e-9``).

Raw eigenvectors of a *degenerate* spectrum are only defined up to a rotation
within each eigenspace, so parity is asserted on backend-invariant quantities:
the Laplacians, the eigenvalues, the heat kernel ``exp(-tL)``, and the invariant
subspace projector ``Y Y^T`` -- plus every (algebraic) relaxation operator.
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")
jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")
jax.config.update("jax_enable_x64", True)
torch.set_default_dtype(torch.float64)

import omnibias.graph.jax.ops as GJ
import omnibias.graph.torch.ops as GT

RTOL = 1e-9
ATOL = 1e-12


def _sym_adjacency(seed: int, n: int = 6) -> np.ndarray:
    rng = np.random.default_rng(seed)
    m = rng.uniform(0.1, 1.0, size=(n, n))
    m = 0.5 * (m + m.T)
    np.fill_diagonal(m, 0.0)
    return m


def _assert_close(x_t, x_j) -> None:
    a = x_t.detach().numpy() if hasattr(x_t, "detach") else np.asarray(x_t)
    b = np.asarray(x_j)
    assert np.allclose(a, b, rtol=RTOL, atol=ATOL), np.max(np.abs(a - b))


@pytest.mark.parametrize("seed", [0, 1, 7])
@pytest.mark.parametrize(
    "name", ["graph_laplacian", "normalized_laplacian", "random_walk_laplacian"]
)
def test_laplacian_parity(seed: int, name: str) -> None:
    m = _sym_adjacency(seed)
    _assert_close(getattr(GT, name)(torch.tensor(m)), getattr(GJ, name)(jnp.asarray(m)))


@pytest.mark.parametrize("normalized", [False, True])
def test_eigenvalue_parity(normalized: bool) -> None:
    m = _sym_adjacency(3)
    et, _ = GT.laplacian_spectrum(torch.tensor(m), normalized=normalized)
    ej, _ = GJ.laplacian_spectrum(jnp.asarray(m), normalized=normalized)
    _assert_close(et, ej)


@pytest.mark.parametrize("t", [0.1, 0.7, 2.0])
def test_heat_kernel_parity(t: float) -> None:
    m = _sym_adjacency(5)
    _assert_close(GT.graph_heat_kernel(torch.tensor(m), t), GJ.graph_heat_kernel(jnp.asarray(m), t))


def test_embedding_subspace_projector_parity() -> None:
    # A graph with a clear spectral gap so the smallest-k subspace is unique.
    n_per, k = 5, 2
    m = np.full((2 * n_per, 2 * n_per), 0.02)
    m[:n_per, :n_per] = 1.0
    m[n_per:, n_per:] = 1.0
    np.fill_diagonal(m, 0.0)
    yt = GT.spectral_embedding(torch.tensor(m), k, normalized=True, drop_first=False).detach().numpy()
    yj = np.asarray(GJ.spectral_embedding(jnp.asarray(m), k, normalized=True, drop_first=False))
    # Y Y^T (the projector onto the smallest-k eigenspace) is rotation-invariant.
    assert np.allclose(yt @ yt.T, yj @ yj.T, rtol=RTOL, atol=1e-9)


def test_sinkhorn_parity() -> None:
    rng = np.random.default_rng(2)
    la = rng.normal(size=(5, 5))
    _assert_close(
        GT.sinkhorn_normalize(torch.tensor(la), n_iters=100),
        GJ.sinkhorn_normalize(jnp.asarray(la), n_iters=100),
    )


@pytest.mark.parametrize("tau", [0.05, 0.5, 2.0])
def test_soft_sort_parity(tau: float) -> None:
    rng = np.random.default_rng(4)
    s = rng.normal(size=7)
    _assert_close(
        GT.soft_sort(torch.tensor(s), temperature=tau),
        GJ.soft_sort(jnp.asarray(s), temperature=tau),
    )


def test_soft_top_k_parity() -> None:
    rng = np.random.default_rng(11)
    s = rng.normal(size=8)
    _assert_close(
        GT.soft_top_k(torch.tensor(s), 3, temperature=0.2),
        GJ.soft_top_k(jnp.asarray(s), 3, temperature=0.2),
    )


def test_gumbel_sinkhorn_parity_with_shared_noise() -> None:
    rng = np.random.default_rng(9)
    la = rng.normal(size=(5, 5))
    noise = rng.gumbel(size=(5, 5))
    _assert_close(
        GT.gumbel_sinkhorn(torch.tensor(la), temperature=0.3, n_iters=100, noise=torch.tensor(noise)),
        GJ.gumbel_sinkhorn(jnp.asarray(la), temperature=0.3, n_iters=100, noise=jnp.asarray(noise)),
    )
