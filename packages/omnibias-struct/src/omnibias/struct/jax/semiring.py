# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Differentiable semiring / hypergraph DP driver (jax).

Bit-identical twin of :mod:`omnibias.struct.torch.semiring` (float64 -- enable
``jax_enable_x64``). Runs the :class:`~omnibias.struct._core.semiring.LogSemiring`
(``lse_beta``) reduction of a :class:`~omnibias.struct._core.semiring.Hypergraph` over a
backend ``edge_weights`` vector (:func:`semiring_value`) and reads off the closed-form edge
marginals by an inside-outside sweep and the tower softmax (:func:`semiring_marginals`); the
marginal of edge ``e`` equals ``d value / d edge_weights[e]``. Reproduces the hand-written
soft-DP layers to ``< 1e-12`` -- the additive-safety cross-check that the driver subsumes
them without changing their numerics.
"""

from __future__ import annotations

import jax.numpy as jnp
from jax import Array
from omnibias.struct._core.semiring import Hypergraph
from omnibias.struct.jax._logsumexp import logsumexp_beta

_NEG = -1.0e30  # finite sentinel for unreachable nodes (avoids -inf * beta grads)


def _inside(graph: Hypergraph, edge_weights: Array, beta: float) -> list[Array]:
    r"""Forward soft node values ``inside[v] = lse_beta`` over derivations rooted at ``v``."""
    neg = jnp.asarray(_NEG, dtype=edge_weights.dtype)
    inside: list[Array] = [neg] * graph.num_nodes
    for v in range(graph.num_nodes):
        inc = graph.incoming(v)
        if not inc:
            continue
        rows: list[Array] = []
        for ei in inc:
            val = edge_weights[ei]
            for t in graph.edges[ei].tails:
                val = val + inside[t]
            rows.append(val)
        inside[v] = logsumexp_beta(jnp.stack(rows), beta, axis=-1)
    return inside


def semiring_value(graph: Hypergraph, edge_weights: Array, beta: float = 1.0) -> Array:
    r"""Soft (``lse_beta``) value of ``graph`` at its ``root`` for the ``edge_weights`` vector.

    ``edge_weights`` is ``(graph.num_edges,)``; differentiable in it. Generalises
    :func:`omnibias.struct.jax.soft_shortest_path`'s topological reduce to arbitrary
    arity-1/2 hyperedges. ``-> max derivation score`` as ``beta -> inf``.
    """
    if edge_weights.shape[0] != graph.num_edges:
        raise ValueError(
            f"edge_weights must have length num_edges={graph.num_edges}, got {edge_weights.shape[0]}"
        )
    value: Array = _inside(graph, edge_weights, beta)[graph.root]
    return value


def semiring_marginals(graph: Hypergraph, edge_weights: Array, beta: float = 1.0) -> Array:
    r"""Closed-form edge marginals ``mu[e] = P_beta(edge e in the derivation)`` (``(num_edges,)``).

    Inside-outside with the tower softmax; ``mu[e] = d semiring_value / d edge_weights[e]``
    (the exact gradient), and the marginals of the edges into ``root`` sum to ``1``.
    """
    if edge_weights.shape[0] != graph.num_edges:
        raise ValueError(
            f"edge_weights must have length num_edges={graph.num_edges}, got {edge_weights.shape[0]}"
        )
    inside = _inside(graph, edge_weights, beta)
    value = inside[graph.root]
    neg = jnp.asarray(_NEG, dtype=edge_weights.dtype)
    outside: list[Array] = [neg] * graph.num_nodes
    outside[graph.root] = jnp.zeros((), dtype=edge_weights.dtype)
    as_tail: list[list[tuple[int, int]]] = [[] for _ in range(graph.num_nodes)]
    for ei, e in enumerate(graph.edges):
        for pos, t in enumerate(e.tails):
            as_tail[t].append((ei, pos))
    for v in range(graph.num_nodes - 1, -1, -1):
        if v == graph.root:
            continue
        rows: list[Array] = []
        for ei, pos in as_tail[v]:
            e = graph.edges[ei]
            term = edge_weights[ei] + outside[e.head]
            for j, t in enumerate(e.tails):
                if j != pos:
                    term = term + inside[t]
            rows.append(term)
        if rows:
            outside[v] = logsumexp_beta(jnp.stack(rows), beta, axis=-1)
    marg: list[Array] = []
    for ei, e in enumerate(graph.edges):
        term = edge_weights[ei] + outside[e.head]
        for t in e.tails:
            term = term + inside[t]
        marg.append(jnp.exp(beta * (term - value)))
    return jnp.stack(marg)


__all__ = ["semiring_marginals", "semiring_value"]
