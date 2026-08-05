# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Differentiable soft Dynamic Time Warping (jax).

Bit-identical twin of :mod:`omnibias.struct.torch.dtw` (float64; needs ``jax_enable_x64``).
Soft-DTW replaces the hard ``min`` of the DTW recursion with a **softmin** ``-lse_beta(-.)``
built from the shared :func:`omnibias.struct.jax._logsumexp.logsumexp_beta`, unrolled so
``jax.grad`` flows through it. As ``beta -> inf`` (the temperature axis) it anneals to hard
DTW from below; the ``delta -> 0`` tower differentiates it exactly
(:func:`soft_dtw_marginals` is the closed-form soft-alignment matrix, pinned equal to
``jax.grad``). Do not conflate the two axes.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import Array
from omnibias.struct.jax._logsumexp import logsumexp_beta


def _softmin(values: list[Array], beta: float) -> Array:
    """Soft ``min`` combine ``-lse_beta(-values)`` over a small list of scalars."""
    return -logsumexp_beta(-jnp.stack(values), beta, axis=-1)


def _dtw_forward(cost: Array, beta: float) -> list[list[Array]]:
    n, m = cost.shape
    r: list[list[Array]] = [[cost[0, 0] for _ in range(m)] for _ in range(n)]
    for i in range(n):
        for j in range(m):
            if i == 0 and j == 0:
                r[i][j] = cost[0, 0]
                continue
            preds: list[Array] = []
            if i > 0:
                preds.append(r[i - 1][j])
            if j > 0:
                preds.append(r[i][j - 1])
            if i > 0 and j > 0:
                preds.append(r[i - 1][j - 1])
            r[i][j] = cost[i, j] + _softmin(preds, beta)
    return r


def _dtw_backward(cost: Array, beta: float) -> list[list[Array]]:
    n, m = cost.shape
    e: list[list[Array]] = [[cost[n - 1, m - 1] for _ in range(m)] for _ in range(n)]
    for i in range(n - 1, -1, -1):
        for j in range(m - 1, -1, -1):
            if i == n - 1 and j == m - 1:
                e[i][j] = cost[n - 1, m - 1]
                continue
            succs: list[Array] = []
            if i < n - 1:
                succs.append(e[i + 1][j])
            if j < m - 1:
                succs.append(e[i][j + 1])
            if i < n - 1 and j < m - 1:
                succs.append(e[i + 1][j + 1])
            e[i][j] = cost[i, j] + _softmin(succs, beta)
    return e


def soft_dtw(cost: Array, beta: float = 1.0) -> Array:
    r"""Soft-DTW value ``-beta^-1 log sum_paths exp(-beta cost(path))`` of an ``(n, m)`` cost matrix.

    The recursive softmin of the DTW lattice; equals the flat softmin over all monotonic
    paths and ``-> hard DTW`` as ``beta -> inf``. Differentiable in ``cost``.
    """
    return _dtw_forward(cost, beta)[-1][-1]


def soft_dtw_marginals(cost: Array, beta: float = 1.0) -> Array:
    r"""Closed-form soft-alignment matrix ``E[i, j] = P_beta(cell (i, j) on the warping path)``.

    Forward-backward over the DTW lattice with the tower softmax; equals
    ``d soft_dtw / d cost`` (each cell's expected usage) and concentrates on the hard DTW
    path as ``beta -> inf``. The source and sink cells always have marginal ``1``.
    """
    n, m = cost.shape
    fwd = _dtw_forward(cost, beta)
    bwd = _dtw_backward(cost, beta)
    value = fwd[n - 1][m - 1]
    rows = [
        jnp.stack([jnp.exp(-beta * (fwd[i][j] + bwd[i][j] - cost[i, j] - value)) for j in range(m)])
        for i in range(n)
    ]
    return jnp.stack(rows, axis=0)


def soft_dtw_batched(cost: Array, beta: float = 1.0) -> Array:
    r"""Batched :func:`soft_dtw` -> ``(B,)`` for ``cost`` ``(B, n, m)`` (via ``jax.vmap``)."""
    out: Array = jax.vmap(lambda c: soft_dtw(c, beta))(cost)
    return out


__all__ = [
    "soft_dtw",
    "soft_dtw_batched",
    "soft_dtw_marginals",
]
