# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Differentiable relaxations of discrete combinatorial objects (jax).

Bit-identical twin of :mod:`omnibias.graph.torch.ops.relaxation`; see that module
for the definitions and references. Each operator carries a temperature ``tau > 0``
that recovers the hard object as ``tau -> 0``.
"""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
from jax.scipy.special import logsumexp

Array = Any


def sinkhorn_normalize(log_alpha: Array, *, n_iters: int = 20) -> Array:
    r"""Project ``exp(log_alpha)`` onto the doubly-stochastic matrices."""
    if log_alpha.ndim != 2 or log_alpha.shape[-1] != log_alpha.shape[-2]:
        raise ValueError(
            f"log_alpha must be square (n, n); got {tuple(log_alpha.shape)}"
        )
    if n_iters < 1:
        raise ValueError(f"n_iters must be >= 1; got {n_iters}")
    log_p = log_alpha
    for _ in range(n_iters):
        log_p = log_p - logsumexp(log_p, axis=-1, keepdims=True)
        log_p = log_p - logsumexp(log_p, axis=-2, keepdims=True)
    return jnp.exp(log_p)


def gumbel_sinkhorn(
    log_alpha: Array,
    *,
    temperature: float = 1.0,
    n_iters: int = 20,
    noise: Array | None = None,
) -> Array:
    r"""Gumbel-Sinkhorn relaxation of a permutation / assignment matrix."""
    if temperature <= 0.0:
        raise ValueError(f"temperature must be > 0; got {temperature}")
    perturbed = log_alpha if noise is None else log_alpha + noise
    return sinkhorn_normalize(perturbed / temperature, n_iters=n_iters)


def sample_gumbel(
    shape: tuple[int, ...],
    key: Any,
    *,
    dtype: Any = None,
    eps: float = 1e-20,
) -> Array:
    r"""I.i.d. standard Gumbel noise ``-log(-log(U))``, ``U ~ Uniform(0, 1)``."""
    u = jax.random.uniform(key, shape, dtype=dtype or jnp.float64)
    return -jnp.log(-jnp.log(u + eps) + eps)


def soft_sort_permutation(
    scores: Array, *, temperature: float = 1.0, descending: bool = True
) -> Array:
    r"""SoftSort row-stochastic soft permutation for a score vector ``(n,)``."""
    if temperature <= 0.0:
        raise ValueError(f"temperature must be > 0; got {temperature}")
    s = scores.reshape(-1)
    sorted_s = jnp.sort(s)
    if descending:
        sorted_s = sorted_s[::-1]
    dist = jnp.abs(sorted_s[:, None] - s[None, :])
    return jax.nn.softmax(-dist / temperature, axis=-1)


def soft_sort(
    scores: Array, *, temperature: float = 1.0, descending: bool = True
) -> Array:
    r"""Differentiable sorted values ``P @ scores`` -> ``(n,)``."""
    p = soft_sort_permutation(scores, temperature=temperature, descending=descending)
    return p @ scores.reshape(-1)


def soft_top_k(scores: Array, k: int, *, temperature: float = 1.0) -> Array:
    r"""Soft top-``k`` membership weights in ``[0, 1]`` summing to exactly ``k``."""
    n = scores.reshape(-1).shape[0]
    if k < 1 or k > n:
        raise ValueError(f"k must be in [1, {n}]; got {k}")
    p = soft_sort_permutation(scores, temperature=temperature, descending=True)
    return p[:k, :].sum(axis=0)
