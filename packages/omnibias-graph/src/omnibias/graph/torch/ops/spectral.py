# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Differentiable spectral graph operators (torch).

All operators take a weighted adjacency matrix ``A`` of shape ``(n, n)`` (assumed
symmetric / undirected, non-negative, zero diagonal) and are differentiable in
``A``. The three Laplacians follow the standard definitions with the isolated-node
convention ``0`` for the inverse-degree factors:

.. math::

    L = D - A, \qquad
    L_{\mathrm{sym}} = I - D^{-1/2} A D^{-1/2}, \qquad
    L_{\mathrm{rw}} = I - D^{-1} A,

with :math:`D = \operatorname{diag}(A\mathbf 1)`. Spectral embedding, the heat
kernel :math:`e^{-tL}`, and the Rayleigh-Ritz cut relaxation are built from the
symmetric eigendecomposition (`torch.linalg.eigh`), which is differentiable for
matrices with simple spectrum.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


def _check_adjacency(a: Tensor) -> None:
    if a.ndim != 2 or a.shape[0] != a.shape[1]:
        raise ValueError(f"adjacency must be square (n, n); got {tuple(a.shape)}")


def degree(adjacency: Tensor) -> Tensor:
    r"""Weighted node degrees ``d_i = sum_j A_{ij}`` -> shape ``(n,)``."""
    _check_adjacency(adjacency)
    return adjacency.sum(dim=-1)


def graph_laplacian(adjacency: Tensor) -> Tensor:
    r"""Combinatorial Laplacian ``L = D - A``."""
    _check_adjacency(adjacency)
    d = adjacency.sum(dim=-1)
    return torch.diag(d) - adjacency


def normalized_laplacian(adjacency: Tensor) -> Tensor:
    r"""Symmetric normalized Laplacian ``L_sym = I - D^{-1/2} A D^{-1/2}``.

    Isolated nodes (``d_i = 0``) contribute ``0`` to the normalization. The
    spectrum lies in ``[0, 2]``.
    """
    _check_adjacency(adjacency)
    d = adjacency.sum(dim=-1)
    d_inv_sqrt = torch.where(
        d > 0, d.rsqrt(), torch.zeros_like(d)
    )
    a_norm = d_inv_sqrt.unsqueeze(-1) * adjacency * d_inv_sqrt.unsqueeze(-2)
    n = adjacency.shape[0]
    eye = torch.eye(n, dtype=adjacency.dtype, device=adjacency.device)
    return eye - a_norm


def random_walk_laplacian(adjacency: Tensor) -> Tensor:
    r"""Random-walk Laplacian ``L_rw = I - D^{-1} A`` (row-stochastic transition).

    Not symmetric in general; isolated nodes contribute ``0``.
    """
    _check_adjacency(adjacency)
    d = adjacency.sum(dim=-1)
    d_inv = torch.where(d > 0, d.reciprocal(), torch.zeros_like(d))
    n = adjacency.shape[0]
    eye = torch.eye(n, dtype=adjacency.dtype, device=adjacency.device)
    return eye - d_inv.unsqueeze(-1) * adjacency


def _symmetric_laplacian(adjacency: Tensor, *, normalized: bool) -> Tensor:
    return (
        normalized_laplacian(adjacency) if normalized else graph_laplacian(adjacency)
    )


def laplacian_spectrum(
    adjacency: Tensor, *, normalized: bool = False
) -> tuple[Tensor, Tensor]:
    r"""Ascending eigenvalues / eigenvectors of the (symmetric) Laplacian.

    Returns ``(eigenvalues (n,), eigenvectors (n, n))`` with columns as
    eigenvectors, sorted by ascending eigenvalue (`torch.linalg.eigh`).
    """
    lap = _symmetric_laplacian(adjacency, normalized=normalized)
    lap = 0.5 * (lap + lap.transpose(-1, -2))
    evals, evecs = torch.linalg.eigh(lap)
    return evals, evecs


def spectral_embedding(
    adjacency: Tensor,
    n_components: int,
    *,
    normalized: bool = True,
    drop_first: bool = True,
) -> Tensor:
    r"""Laplacian-eigenmaps embedding: the smallest non-trivial eigenvectors.

    Returns an ``(n, n_components)`` matrix whose columns are the eigenvectors of
    the Laplacian for the smallest eigenvalues. With ``drop_first=True`` the
    (constant, eigenvalue-``0``) null vector of a connected graph is skipped so
    the first returned column is the **Fiedler vector**.
    """
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


def fiedler_vector(adjacency: Tensor, *, normalized: bool = False) -> Tensor:
    r"""The eigenvector of the second-smallest Laplacian eigenvalue -> ``(n,)``."""
    _, evecs = laplacian_spectrum(adjacency, normalized=normalized)
    return evecs[:, 1]


def graph_heat_kernel(
    adjacency: Tensor, t: float, *, normalized: bool = False
) -> Tensor:
    r"""Heat kernel ``H = exp(-t L)`` via the symmetric eigendecomposition.

    ``H_{ij}`` is the amount of heat diffused from node ``j`` to node ``i`` after
    time ``t``. For the combinatorial Laplacian each column sums to ``1`` only in
    the ``t -> inf`` equilibrium; ``H`` is symmetric positive-definite for all
    ``t >= 0``.
    """
    if t < 0:
        raise ValueError(f"diffusion time t must be >= 0; got {t}")
    evals, evecs = laplacian_spectrum(adjacency, normalized=normalized)
    damp = torch.exp(-t * evals)
    return (evecs * damp.unsqueeze(-2)) @ evecs.transpose(-1, -2)


@dataclass(frozen=True)
class CutRelaxation:
    """Rayleigh-Ritz relaxation of the ``k``-way ratio / normalized cut.

    Attributes
    ----------
    embedding
        ``(n, k)`` matrix of the ``k`` smallest Laplacian eigenvectors -- the
        continuous relaxation of the cluster-indicator matrix.
    relaxed_cut_value
        ``sum`` of the ``k`` smallest eigenvalues: a lower bound on the discrete
        minimum cut (the relaxation only *loosens* the combinatorial problem).
    eigenvalues
        The ``k`` smallest eigenvalues (ascending).
    """

    embedding: Tensor
    relaxed_cut_value: Tensor
    eigenvalues: Tensor


def spectral_clustering_relaxation(
    adjacency: Tensor, k: int, *, normalized: bool = True
) -> CutRelaxation:
    r"""Continuous eigenvector relaxation of the ``k``-way graph cut.

    The discrete balanced ``k``-cut is NP-hard; its standard convex *relaxation*
    ``min_Y tr(Y^T L Y)`` s.t. ``Y^T Y = I`` is solved exactly by the ``k``
    smallest Laplacian eigenvectors (Rayleigh-Ritz). This returns that relaxation
    only -- the discrete rounding step (k-means on the embedding) is **not**
    performed and exact balanced partitioning is out of scope (see the
    graph-limitation cookbook).
    """
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
