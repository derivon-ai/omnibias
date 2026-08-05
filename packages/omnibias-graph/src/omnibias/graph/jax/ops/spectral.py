# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Differentiable spectral graph operators (jax).

Bit-identical twin of :mod:`omnibias.graph.torch.ops.spectral`; see that module
for the mathematical definitions. All operators take a symmetric non-negative
adjacency ``A`` of shape ``(n, n)`` and are differentiable in ``A``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import jax.numpy as jnp

Array = Any


def _check_adjacency(a: Array) -> None:
    if a.ndim != 2 or a.shape[0] != a.shape[1]:
        raise ValueError(f"adjacency must be square (n, n); got {tuple(a.shape)}")


def degree(adjacency: Array) -> Array:
    r"""Weighted node degrees ``d_i = sum_j A_{ij}`` -> shape ``(n,)``."""
    _check_adjacency(adjacency)
    return adjacency.sum(axis=-1)


def graph_laplacian(adjacency: Array) -> Array:
    r"""Combinatorial Laplacian ``L = D - A``."""
    _check_adjacency(adjacency)
    d = adjacency.sum(axis=-1)
    return jnp.diag(d) - adjacency


def normalized_laplacian(adjacency: Array) -> Array:
    r"""Symmetric normalized Laplacian ``L_sym = I - D^{-1/2} A D^{-1/2}``."""
    _check_adjacency(adjacency)
    d = adjacency.sum(axis=-1)
    d_inv_sqrt = jnp.where(d > 0, 1.0 / jnp.sqrt(d), jnp.zeros_like(d))
    a_norm = d_inv_sqrt[:, None] * adjacency * d_inv_sqrt[None, :]
    n = adjacency.shape[0]
    return jnp.eye(n, dtype=adjacency.dtype) - a_norm


def random_walk_laplacian(adjacency: Array) -> Array:
    r"""Random-walk Laplacian ``L_rw = I - D^{-1} A``."""
    _check_adjacency(adjacency)
    d = adjacency.sum(axis=-1)
    d_inv = jnp.where(d > 0, 1.0 / d, jnp.zeros_like(d))
    n = adjacency.shape[0]
    return jnp.eye(n, dtype=adjacency.dtype) - d_inv[:, None] * adjacency


def _symmetric_laplacian(adjacency: Array, *, normalized: bool) -> Array:
    return (
        normalized_laplacian(adjacency) if normalized else graph_laplacian(adjacency)
    )


def laplacian_spectrum(
    adjacency: Array, *, normalized: bool = False
) -> tuple[Array, Array]:
    r"""Ascending eigenvalues / eigenvectors of the (symmetric) Laplacian."""
    lap = _symmetric_laplacian(adjacency, normalized=normalized)
    lap = 0.5 * (lap + lap.T)
    evals, evecs = jnp.linalg.eigh(lap)
    return evals, evecs


def spectral_embedding(
    adjacency: Array,
    n_components: int,
    *,
    normalized: bool = True,
    drop_first: bool = True,
) -> Array:
    r"""Laplacian-eigenmaps embedding: the smallest non-trivial eigenvectors."""
    if n_components < 1:
        raise ValueError(f"n_components must be >= 1; got {n_components}")
    n = adjacency.shape[0]
    start = 1 if drop_first else 0
    if start + n_components > n:
        raise ValueError(
            f"cannot take {n_components} components (drop_first={drop_first}) "
            f"from an {n}-node graph"
        )
    _, evecs = laplacian_spectrum(adjacency, normalized=normalized)
    return evecs[:, start : start + n_components]


def fiedler_vector(adjacency: Array, *, normalized: bool = False) -> Array:
    r"""The eigenvector of the second-smallest Laplacian eigenvalue -> ``(n,)``."""
    _, evecs = laplacian_spectrum(adjacency, normalized=normalized)
    return evecs[:, 1]


def graph_heat_kernel(adjacency: Array, t: float, *, normalized: bool = False) -> Array:
    r"""Heat kernel ``H = exp(-t L)`` via the symmetric eigendecomposition."""
    if t < 0:
        raise ValueError(f"diffusion time t must be >= 0; got {t}")
    evals, evecs = laplacian_spectrum(adjacency, normalized=normalized)
    damp = jnp.exp(-t * evals)
    return (evecs * damp[None, :]) @ evecs.T


@dataclass(frozen=True)
class CutRelaxation:
    """Rayleigh-Ritz relaxation of the ``k``-way ratio / normalized cut."""

    embedding: Array
    relaxed_cut_value: Array
    eigenvalues: Array


def spectral_clustering_relaxation(
    adjacency: Array, k: int, *, normalized: bool = True
) -> CutRelaxation:
    r"""Continuous eigenvector relaxation of the ``k``-way graph cut."""
    if k < 1:
        raise ValueError(f"k must be >= 1; got {k}")
    if k > adjacency.shape[0]:
        raise ValueError(f"k={k} exceeds the number of nodes {adjacency.shape[0]}")
    evals, evecs = laplacian_spectrum(adjacency, normalized=normalized)
    return CutRelaxation(
        embedding=evecs[:, :k],
        relaxed_cut_value=evals[:k].sum(),
        eigenvalues=evals[:k],
    )
