# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Differentiable Eisner projective dependency parsing on the semiring driver (jax).

Bit-identical twin of :mod:`omnibias.struct.torch.eisner` (float64 -- enable
``jax_enable_x64``). :func:`soft_eisner` is the ``lse_beta`` partition over projective trees
and :func:`eisner_marginals` reads off the closed-form arc marginals ``P_beta(arc h -> m)``
(the exact gradient of ``soft_eisner`` w.r.t. the arc-score matrix, pinned equal to
``autograd``).
"""

from __future__ import annotations

import jax.numpy as jnp
from jax import Array
from omnibias.struct._core.eisner import EisnerSpec, eisner_hypergraph
from omnibias.struct.jax.semiring import semiring_marginals, semiring_value


def _edge_weights(spec: EisnerSpec, arc: Array) -> Array:
    zero = jnp.zeros((), dtype=arc.dtype)
    rows: list[Array] = []
    for s in spec.edge_specs:
        rows.append(arc[int(s[1]), int(s[2])] if s[0] == "arc" else zero)
    return jnp.stack(rows)


def soft_eisner(arc: Array, beta: float = 1.0) -> Array:
    r"""Soft projective-parse partition ``beta^-1 log sum_trees exp(beta score)`` of ``arc``.

    ``arc`` is the ``(n + 1, n + 1)`` arc-score matrix (``arc[h, m]`` = head ``h`` -> modifier
    ``m``; index ``0`` is the ``ROOT``). Differentiable in ``arc``; ``-> best projective tree
    score`` as ``beta -> inf``.
    """
    spec = eisner_hypergraph(int(arc.shape[0]) - 1)
    return semiring_value(spec.graph, _edge_weights(spec, arc), beta)


def eisner_marginals(arc: Array, beta: float = 1.0) -> Array:
    r"""Closed-form arc marginals ``P_beta(arc h -> m)`` as an ``(n + 1, n + 1)`` matrix.

    Equals ``d soft_eisner / d arc`` (the exact gradient); each modifier column ``m >= 1``
    sums to ``1``. Inside-outside via the tower softmax; equal to ``autograd``.
    """
    spec = eisner_hypergraph(int(arc.shape[0]) - 1)
    mu = semiring_marginals(spec.graph, _edge_weights(spec, arc), beta)
    out = jnp.zeros_like(arc)
    for e, s in enumerate(spec.edge_specs):
        if s[0] == "arc":
            out = out.at[int(s[1]), int(s[2])].add(mu[e])
    return out


def soft_eisner_batched(arc: Array, beta: float = 1.0) -> Array:
    r"""Batched :func:`soft_eisner` -> ``(B,)`` for ``arc`` ``(B, n + 1, n + 1)`` (via ``jax.vmap``)."""
    import jax

    out: Array = jax.vmap(lambda a: soft_eisner(a, beta))(arc)
    return out


__all__ = ["eisner_marginals", "soft_eisner", "soft_eisner_batched"]
