# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Structured attention as linear-chain soft-DP marginals (torch).

Bit-identical twin of :mod:`omnibias.struct.jax.attention` (float64). Ordinary attention is
an independent per-row ``softmax``; **structured attention** couples the rows with a
transition score and reads the attention weights off the linear-chain *marginals*
(:func:`omnibias.struct.torch.soft_viterbi_marginals`) -- the closed-form forward-backward
softmax of the ``delta -> 0`` tower. With zero transitions it collapses back to plain
row-wise ``softmax(beta * scores)``; as ``beta -> inf`` it concentrates on the Viterbi path.
"""

from __future__ import annotations

from omnibias.struct.torch.soft_dp import soft_viterbi_marginals
from torch import Tensor


def structured_attention(
    scores: Tensor,
    transitions: Tensor,
    beta: float = 1.0,
    *,
    start: Tensor | None = None,
) -> Tensor:
    r"""Structured attention weights ``(T, S)`` -- linear-chain marginals of ``scores``.

    ``scores`` are per-position, per-key logits ``(T, S)`` and ``transitions`` ``(S, S)``
    couple adjacent positions; the returned weights are ``P_beta(key s at position t)``, each
    row summing to ``1``. Zero ``transitions`` recovers independent ``softmax(beta * scores)``
    attention; a non-zero transition biases attention towards structurally coherent paths.
    """
    return soft_viterbi_marginals(scores, transitions, beta, start=start)


def structured_attention_batched(
    scores: Tensor,
    transitions: Tensor,
    beta: float = 1.0,
    *,
    start: Tensor | None = None,
) -> Tensor:
    r"""Batched :func:`structured_attention` -> ``(B, T, S)`` for ``scores`` ``(B, T, S)``.

    ``transitions`` may be shared ``(S, S)`` or per-example ``(B, S, S)``; ``start`` may be
    ``None``, shared ``(S,)``, or per-example ``(B, S)``. Maps the linear-chain marginals over
    the leading batch axis with ``torch.func.vmap`` -- bit-identical to looping.
    """
    from torch.func import vmap

    t_dim = 0 if transitions.dim() == 3 else None
    if start is None:

        def fwd(sc: Tensor, tr: Tensor) -> Tensor:
            return structured_attention(sc, tr, beta)

        out: Tensor = vmap(fwd, in_dims=(0, t_dim))(scores, transitions)
        return out
    s_dim = 0 if start.dim() == 2 else None

    def fwd_start(sc: Tensor, tr: Tensor, st: Tensor) -> Tensor:
        return structured_attention(sc, tr, beta, start=st)

    out_start: Tensor = vmap(fwd_start, in_dims=(0, t_dim, s_dim))(scores, transitions, start)
    return out_start


__all__ = ["structured_attention", "structured_attention_batched"]
