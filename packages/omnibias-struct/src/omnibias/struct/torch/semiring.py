# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Differentiable semiring / hypergraph DP driver (torch).

Bit-identical twin of :mod:`omnibias.struct.jax.semiring` (float64). Runs the
:class:`~omnibias.struct._core.semiring.LogSemiring` (``lse_beta``) reduction of a
:class:`~omnibias.struct._core.semiring.Hypergraph` over a backend ``edge_weights`` vector,
unrolled so ``autograd`` flows through it (:func:`semiring_value`), and reads off the
closed-form **edge marginals** by one extra backward (outside) sweep and the tower softmax
``exp(beta (inside_tails + edge_weight + outside_head - value))`` (:func:`semiring_marginals`).
The marginal of edge ``e`` equals ``d value / d edge_weights[e]`` -- the inside-outside
generalization of the forward-backward path marginal -- which the tests pin equal to
``autograd``.

This is the general engine the CKY / Eisner families lift to; it reproduces the
hand-written :func:`~omnibias.struct.torch.soft_viterbi` / ``soft_shortest_path`` /
``soft_dtw`` / ``soft_align`` layers to ``< 1e-12`` (the additive-safety cross-check),
without modifying them. The ``beta -> inf`` reduction is the relaxation axis; the
``delta -> 0`` tower differentiates it exactly -- do not conflate the two.
"""

from __future__ import annotations

import torch
from omnibias.struct._core.semiring import Hypergraph
from omnibias.struct.torch._logsumexp import logsumexp_beta
from torch import Tensor

_NEG = -1.0e30  # finite sentinel for unreachable nodes (avoids -inf * beta grads)


def _inside(graph: Hypergraph, edge_weights: Tensor, beta: float) -> list[Tensor]:
    r"""Forward soft node values ``inside[v] = lse_beta`` over derivations rooted at ``v``."""
    neg = torch.full((), _NEG, dtype=edge_weights.dtype, device=edge_weights.device)
    inside: list[Tensor] = [neg] * graph.num_nodes
    for v in range(graph.num_nodes):
        inc = graph.incoming(v)
        if not inc:
            continue
        rows: list[Tensor] = []
        for ei in inc:
            val = edge_weights[ei]
            for t in graph.edges[ei].tails:
                val = val + inside[t]
            rows.append(val)
        inside[v] = logsumexp_beta(torch.stack(rows), beta, axis=-1)
    return inside


def semiring_value(graph: Hypergraph, edge_weights: Tensor, beta: float = 1.0) -> Tensor:
    r"""Soft (``lse_beta``) value of ``graph`` at its ``root`` for the ``edge_weights`` vector.

    ``edge_weights`` is ``(graph.num_edges,)``; differentiable in it. Generalises
    :func:`omnibias.struct.torch.soft_shortest_path`'s topological reduce to arbitrary
    arity-1/2 hyperedges (parse spans, dependency arcs). ``-> max derivation score`` as
    ``beta -> inf``.
    """
    if edge_weights.shape[0] != graph.num_edges:
        raise ValueError(
            f"edge_weights must have length num_edges={graph.num_edges}, got {edge_weights.shape[0]}"
        )
    value: Tensor = _inside(graph, edge_weights, beta)[graph.root]
    return value


def semiring_marginals(graph: Hypergraph, edge_weights: Tensor, beta: float = 1.0) -> Tensor:
    r"""Closed-form edge marginals ``mu[e] = P_beta(edge e in the derivation)`` (``(num_edges,)``).

    Inside-outside with the tower softmax; ``mu[e] = d semiring_value / d edge_weights[e]``
    (the exact gradient), and the marginals of the edges into ``root`` sum to ``1``. As
    ``beta -> inf`` the mass concentrates on the best derivation
    (:func:`omnibias.struct.best_derivation`).
    """
    if edge_weights.shape[0] != graph.num_edges:
        raise ValueError(
            f"edge_weights must have length num_edges={graph.num_edges}, got {edge_weights.shape[0]}"
        )
    inside = _inside(graph, edge_weights, beta)
    value = inside[graph.root]
    neg = torch.full((), _NEG, dtype=edge_weights.dtype, device=edge_weights.device)
    outside: list[Tensor] = [neg] * graph.num_nodes
    outside[graph.root] = torch.zeros((), dtype=edge_weights.dtype, device=edge_weights.device)
    # For each node, the incoming edges in which it appears as a tail (with sibling tails).
    as_tail: list[list[tuple[int, int]]] = [[] for _ in range(graph.num_nodes)]
    for ei, e in enumerate(graph.edges):
        for pos, t in enumerate(e.tails):
            as_tail[t].append((ei, pos))
    for v in range(graph.num_nodes - 1, -1, -1):
        if v == graph.root:
            continue
        rows: list[Tensor] = []
        for ei, pos in as_tail[v]:
            e = graph.edges[ei]
            term = edge_weights[ei] + outside[e.head]
            for j, t in enumerate(e.tails):
                if j != pos:
                    term = term + inside[t]
            rows.append(term)
        if rows:
            outside[v] = logsumexp_beta(torch.stack(rows), beta, axis=-1)
    marg: list[Tensor] = []
    for ei, e in enumerate(graph.edges):
        term = edge_weights[ei] + outside[e.head]
        for t in e.tails:
            term = term + inside[t]
        marg.append(torch.exp(beta * (term - value)))
    return torch.stack(marg)


__all__ = ["semiring_marginals", "semiring_value"]
