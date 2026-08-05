# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Differentiable Monotonic Alignment Search (jax).

Bit-identical twin of :mod:`omnibias.struct.torch.monotonic` (float64; needs
``jax_enable_x64``). Soft-MAS replaces the ``max`` of ``Q[i, j] = S[i, j] + max(Q[i-1, j-1],
Q[i, j-1])`` with the shared ``lse_beta``, unrolled so ``jax.grad`` flows through it. As
``beta -> inf`` (the temperature axis) it anneals to hard MAS; the ``delta -> 0`` tower gives
the closed-form alignment marginals (:func:`soft_mas_marginals`), pinned equal to
``jax.grad``. Do not conflate the two axes.
"""

from __future__ import annotations

import jax.numpy as jnp
from jax import Array
from omnibias.struct.jax._logsumexp import logsumexp_beta

_NEG = -1.0e30  # finite sentinel for unreachable (i > j) lattice cells


def _forward(score: Array, beta: float) -> list[list[Array]]:
    n_tokens, n_frames = score.shape
    neg = jnp.full((), _NEG, dtype=score.dtype)
    q: list[list[Array]] = [[neg for _ in range(n_frames)] for _ in range(n_tokens)]
    q[0][0] = score[0, 0]
    for j in range(1, n_frames):
        for i in range(min(j, n_tokens - 1) + 1):
            preds = [q[i][j - 1]]
            if i > 0:
                preds.append(q[i - 1][j - 1])
            q[i][j] = score[i, j] + logsumexp_beta(jnp.stack(preds), beta, axis=-1)
    return q


def _backward(score: Array, beta: float) -> list[list[Array]]:
    n_tokens, n_frames = score.shape
    neg = jnp.full((), _NEG, dtype=score.dtype)
    q: list[list[Array]] = [[neg for _ in range(n_frames)] for _ in range(n_tokens)]
    q[n_tokens - 1][n_frames - 1] = score[n_tokens - 1, n_frames - 1]
    for j in range(n_frames - 2, -1, -1):
        lo = max(0, n_tokens - 1 - (n_frames - 1 - j))
        for i in range(lo, min(j, n_tokens - 1) + 1):
            succs = [q[i][j + 1]]
            if i + 1 < n_tokens:
                succs.append(q[i + 1][j + 1])
            q[i][j] = score[i, j] + logsumexp_beta(jnp.stack(succs), beta, axis=-1)
    return q


def soft_mas(score: Array, beta: float = 1.0) -> Array:
    r"""Soft-MAS value ``beta^-1 log sum_alignments exp(beta score)`` of an ``(L, T)`` matrix.

    The recursive softmax over monotonic-surjective alignments; ``-> hard MAS`` as
    ``beta -> inf``. Differentiable in ``score``.
    """
    n_tokens, n_frames = score.shape
    return _forward(score, beta)[n_tokens - 1][n_frames - 1]


def soft_mas_marginals(score: Array, beta: float = 1.0) -> Array:
    r"""Closed-form soft-alignment matrix ``A[i, j] = P_beta(frame j assigned to token i)``.

    Forward-backward over the MAS lattice with the tower softmax; equals
    ``d soft_mas / d score`` and each frame column sums to ``1``. Concentrates on the hard
    MAS alignment as ``beta -> inf``.
    """
    n_tokens, n_frames = score.shape
    fwd = _forward(score, beta)
    bwd = _backward(score, beta)
    value = fwd[n_tokens - 1][n_frames - 1]
    rows = [
        jnp.stack([jnp.exp(beta * (fwd[i][j] + bwd[i][j] - score[i, j] - value)) for j in range(n_frames)])
        for i in range(n_tokens)
    ]
    return jnp.stack(rows, axis=0)


def soft_mas_batched(score: Array, beta: float = 1.0) -> Array:
    r"""Batched :func:`soft_mas` -> ``(B,)`` for ``score`` ``(B, L, T)`` (via ``jax.vmap``)."""
    import jax

    out: Array = jax.vmap(lambda s: soft_mas(s, beta))(score)
    return out


__all__ = [
    "soft_mas",
    "soft_mas_batched",
    "soft_mas_marginals",
]
