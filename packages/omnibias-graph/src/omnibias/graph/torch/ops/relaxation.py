# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Differentiable relaxations of discrete combinatorial objects (torch).

Each operator carries a temperature ``tau > 0`` that recovers the hard object as
``tau -> 0``:

* :func:`sinkhorn_normalize` -- projection onto the Birkhoff polytope
  (doubly-stochastic matrices) by log-domain matrix scaling, mirroring the
  Sinkhorn iteration in :func:`omnibias.torch.information.sinkhorn_distance`.
* :func:`gumbel_sinkhorn` -- a differentiable relaxation of the assignment /
  matching problem (Mena et al., 2018): Sinkhorn-normalise ``(log_alpha + noise)
  / tau``.
* :func:`soft_sort` / :func:`soft_sort_permutation` -- the SoftSort operator
  (Prillo & Eisenschlos, 2020): a row-stochastic soft permutation that becomes
  the exact argsort permutation as ``tau -> 0``.
* :func:`soft_top_k` -- membership weights in ``[0, 1]`` summing to exactly
  ``k``, built from the SoftSort permutation.
"""

from __future__ import annotations

import torch
from torch import Tensor


def sinkhorn_normalize(log_alpha: Tensor, *, n_iters: int = 20) -> Tensor:
    r"""Project ``exp(log_alpha)`` onto the doubly-stochastic matrices.

    ``log_alpha`` is a square ``(n, n)`` log-domain matrix. Alternating row / column
    log-normalisation converges to the unique doubly-stochastic matrix
    ``diag(u) exp(log_alpha) diag(v)`` (Sinkhorn's theorem). Returns the
    normalised matrix in the probability domain.
    """
    if log_alpha.ndim != 2 or log_alpha.shape[-1] != log_alpha.shape[-2]:
        raise ValueError(
            f"log_alpha must be square (n, n); got {tuple(log_alpha.shape)}"
        )
    if n_iters < 1:
        raise ValueError(f"n_iters must be >= 1; got {n_iters}")
    log_p = log_alpha
    for _ in range(n_iters):
        log_p = log_p - torch.logsumexp(log_p, dim=-1, keepdim=True)
        log_p = log_p - torch.logsumexp(log_p, dim=-2, keepdim=True)
    return torch.exp(log_p)


def gumbel_sinkhorn(
    log_alpha: Tensor,
    *,
    temperature: float = 1.0,
    n_iters: int = 20,
    noise: Tensor | None = None,
) -> Tensor:
    r"""Gumbel-Sinkhorn relaxation of a permutation / assignment matrix.

    Returns ``sinkhorn_normalize((log_alpha + noise) / temperature)``. Pass a
    pre-sampled Gumbel ``noise`` tensor (same shape as ``log_alpha``) for the
    stochastic operator; with ``noise=None`` the map is deterministic. As
    ``temperature -> 0`` the result approaches a hard permutation matrix.
    """
    if temperature <= 0.0:
        raise ValueError(f"temperature must be > 0; got {temperature}")
    perturbed = log_alpha if noise is None else log_alpha + noise
    return sinkhorn_normalize(perturbed / temperature, n_iters=n_iters)


def sample_gumbel(
    shape: tuple[int, ...],
    *,
    dtype: torch.dtype | None = None,
    device: torch.device | None = None,
    generator: torch.Generator | None = None,
    eps: float = 1e-20,
) -> Tensor:
    r"""I.i.d. standard Gumbel noise ``-log(-log(U))``, ``U ~ Uniform(0, 1)``."""
    u = torch.rand(shape, dtype=dtype, device=device, generator=generator)
    return -torch.log(-torch.log(u + eps) + eps)


def soft_sort_permutation(
    scores: Tensor, *, temperature: float = 1.0, descending: bool = True
) -> Tensor:
    r"""SoftSort row-stochastic soft permutation for a score vector ``(n,)``.

    ``P[i, j] = softmax_j(-|sorted(scores)[i] - scores[j]| / tau)``; row ``i`` is a
    soft one-hot selecting the input that is the ``i``-th in sorted order. As
    ``tau -> 0`` (distinct scores) ``P`` becomes the exact argsort permutation.
    """
    if temperature <= 0.0:
        raise ValueError(f"temperature must be > 0; got {temperature}")
    s = scores.reshape(-1)
    sorted_s = torch.sort(s, descending=descending).values
    dist = torch.abs(sorted_s.unsqueeze(-1) - s.unsqueeze(-2))
    return torch.softmax(-dist / temperature, dim=-1)


def soft_sort(
    scores: Tensor, *, temperature: float = 1.0, descending: bool = True
) -> Tensor:
    r"""Differentiable sorted values ``P @ scores`` -> ``(n,)``.

    Approaches ``torch.sort(scores)`` as ``temperature -> 0``.
    """
    p = soft_sort_permutation(scores, temperature=temperature, descending=descending)
    return p @ scores.reshape(-1)


def soft_top_k(
    scores: Tensor, k: int, *, temperature: float = 1.0
) -> Tensor:
    r"""Soft top-``k`` membership weights in ``[0, 1]`` summing to exactly ``k``.

    ``m_j = sum_{i < k} P[i, j]`` with ``P`` the descending SoftSort permutation:
    the (relaxed) probability that input ``j`` is among the ``k`` largest scores.
    As ``temperature -> 0`` this is the hard top-``k`` indicator.
    """
    n = scores.reshape(-1).shape[0]
    if k < 1 or k > n:
        raise ValueError(f"k must be in [1, {n}]; got {k}")
    p = soft_sort_permutation(scores, temperature=temperature, descending=True)
    return p[:k, :].sum(dim=0)
