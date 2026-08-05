# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Structured attention as linear-chain soft-DP marginals (jax).

Bit-identical twin of :mod:`omnibias.struct.torch.attention` (float64; needs
``jax_enable_x64``). Ordinary attention is an independent per-row ``softmax``; **structured
attention** couples the rows with a transition score and reads the attention weights off the
linear-chain *marginals* (:func:`omnibias.struct.jax.soft_viterbi_marginals`). With zero
transitions it collapses back to plain row-wise ``softmax(beta * scores)``; as ``beta -> inf``
it concentrates on the Viterbi path.
"""

from __future__ import annotations

from jax import Array
from omnibias.struct.jax.soft_dp import soft_viterbi_marginals


def structured_attention(
    scores: Array,
    transitions: Array,
    beta: float = 1.0,
    *,
    start: Array | None = None,
) -> Array:
    r"""Structured attention weights ``(T, S)`` -- linear-chain marginals of ``scores``.

    ``scores`` are per-position, per-key logits ``(T, S)`` and ``transitions`` ``(S, S)``
    couple adjacent positions; the returned weights are ``P_beta(key s at position t)``, each
    row summing to ``1``. Zero ``transitions`` recovers independent ``softmax(beta * scores)``
    attention; a non-zero transition biases attention towards structurally coherent paths.
    """
    return soft_viterbi_marginals(scores, transitions, beta, start=start)


def structured_attention_batched(
    scores: Array,
    transitions: Array,
    beta: float = 1.0,
    *,
    start: Array | None = None,
) -> Array:
    r"""Batched :func:`structured_attention` -> ``(B, T, S)`` for ``scores`` ``(B, T, S)``.

    ``transitions`` may be shared ``(S, S)`` or per-example ``(B, S, S)``; ``start`` may be
    ``None``, shared ``(S,)``, or per-example ``(B, S)``. Maps the marginals over the leading
    batch axis with ``jax.vmap`` -- bit-identical to looping.
    """
    import jax

    t_dim = 0 if transitions.ndim == 3 else None
    if start is None:
        out: Array = jax.vmap(
            lambda sc, tr: structured_attention(sc, tr, beta), in_axes=(0, t_dim)
        )(scores, transitions)
        return out
    s_dim = 0 if start.ndim == 2 else None
    out_start: Array = jax.vmap(
        lambda sc, tr, st: structured_attention(sc, tr, beta, start=st), in_axes=(0, t_dim, s_dim)
    )(scores, transitions, start)
    return out_start


__all__ = ["structured_attention", "structured_attention_batched"]
