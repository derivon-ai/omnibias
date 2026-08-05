# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Differentiable Monotonic Alignment Search (torch).

Bit-identical twin of :mod:`omnibias.struct.jax.monotonic` (float64). Soft-MAS replaces the
``max`` of the MAS recursion ``Q[i, j] = S[i, j] + max(Q[i-1, j-1], Q[i, j-1])`` with the
shared ``lse_beta``, unrolled so ``autograd`` flows through it. As ``beta -> inf`` (the
temperature axis) it anneals to hard MAS; the ``delta -> 0`` tower gives the closed-form
alignment marginals (:func:`soft_mas_marginals`), pinned equal to ``autograd``. Do not
conflate the two axes.
"""

from __future__ import annotations

import torch
from omnibias.struct.torch._logsumexp import logsumexp_beta
from torch import Tensor

_NEG = -1.0e30  # finite sentinel for unreachable (i > j) lattice cells


def _forward(score: Tensor, beta: float) -> list[list[Tensor]]:
    n_tokens, n_frames = score.shape
    neg = torch.full((), _NEG, dtype=score.dtype, device=score.device)
    q: list[list[Tensor]] = [[neg for _ in range(n_frames)] for _ in range(n_tokens)]
    q[0][0] = score[0, 0]
    for j in range(1, n_frames):
        for i in range(min(j, n_tokens - 1) + 1):
            preds = [q[i][j - 1]]
            if i > 0:
                preds.append(q[i - 1][j - 1])
            q[i][j] = score[i, j] + logsumexp_beta(torch.stack(preds), beta, axis=-1)
    return q


def _backward(score: Tensor, beta: float) -> list[list[Tensor]]:
    n_tokens, n_frames = score.shape
    neg = torch.full((), _NEG, dtype=score.dtype, device=score.device)
    q: list[list[Tensor]] = [[neg for _ in range(n_frames)] for _ in range(n_tokens)]
    q[n_tokens - 1][n_frames - 1] = score[n_tokens - 1, n_frames - 1]
    for j in range(n_frames - 2, -1, -1):
        lo = max(0, n_tokens - 1 - (n_frames - 1 - j))
        for i in range(lo, min(j, n_tokens - 1) + 1):
            succs = [q[i][j + 1]]
            if i + 1 < n_tokens:
                succs.append(q[i + 1][j + 1])
            q[i][j] = score[i, j] + logsumexp_beta(torch.stack(succs), beta, axis=-1)
    return q


def soft_mas(score: Tensor, beta: float = 1.0) -> Tensor:
    r"""Soft-MAS value ``beta^-1 log sum_alignments exp(beta score)`` of an ``(L, T)`` matrix.

    The recursive softmax over monotonic-surjective alignments; ``-> hard MAS`` as
    ``beta -> inf``. Differentiable in ``score``.
    """
    n_tokens, n_frames = score.shape
    return _forward(score, beta)[n_tokens - 1][n_frames - 1]


def soft_mas_marginals(score: Tensor, beta: float = 1.0) -> Tensor:
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
        torch.stack([torch.exp(beta * (fwd[i][j] + bwd[i][j] - score[i, j] - value)) for j in range(n_frames)])
        for i in range(n_tokens)
    ]
    return torch.stack(rows, dim=0)


def soft_mas_batched(score: Tensor, beta: float = 1.0) -> Tensor:
    r"""Batched :func:`soft_mas` -> ``(B,)`` for ``score`` ``(B, L, T)`` (via ``torch.func.vmap``)."""
    from torch.func import vmap

    def fwd(s: Tensor) -> Tensor:
        return soft_mas(s, beta)

    out: Tensor = vmap(fwd)(score)
    return out


__all__ = [
    "soft_mas",
    "soft_mas_batched",
    "soft_mas_marginals",
]
