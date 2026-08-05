# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Distribution operators over the semiring hypergraph -- entropy, sampling, k-best (oracles).

Given the Gibbs distribution ``p_beta(D) proportional to exp(beta * score(D))`` over the
derivations of a :class:`~omnibias.struct._core.semiring.Hypergraph`, this module carries the
pure-numpy ground truth the differentiable backends (:mod:`omnibias.struct.torch.distributions`
/ :mod:`omnibias.struct.jax.distributions`) are checked against:

* :func:`brute_force_entropy` -- the Shannon entropy ``H(p_beta)`` by flat enumeration (the
  oracle for the closed-form ``H = beta * (V_beta - E_p[score])`` identity);
* :func:`sample_derivations` -- **exact** forward-filtering backward-sampling from ``p_beta``
  (its empirical edge frequencies converge to the closed-form marginals);
* :func:`kbest_derivations` -- **exact** k-best derivations by the topological k-best DP
  (Huang & Chiang 2005), with :func:`brute_force_kbest` the enumerate-and-sort oracle.

Pure numpy / Python (no backend import). Tiny graphs only for the brute-force oracles.
"""

from __future__ import annotations

import math
from collections.abc import Iterator, Sequence

import numpy as np
from numpy.typing import NDArray
from omnibias.struct._core.semiring import (
    Hypergraph,
    derivation_weight,
    enumerate_derivations,
)

FloatArray = NDArray[np.float64]

_NEG = -1.0e30


def _inside(graph: Hypergraph, weights: FloatArray, beta: float) -> FloatArray:
    r"""Per-node soft (``lse_beta``) inside values over sub-derivations rooted at each node."""
    inside = np.full(graph.num_nodes, _NEG)
    for v in range(graph.num_nodes):
        inc = graph.incoming(v)
        if not inc:
            continue
        rows = np.array(
            [weights[ei] + sum(inside[t] for t in graph.edges[ei].tails) for ei in inc]
        )
        m = float(np.max(rows))
        inside[v] = m + math.log(float(np.sum(np.exp(beta * (rows - m))))) / beta
    return inside


def brute_force_entropy(graph: Hypergraph, weights: Sequence[float], beta: float) -> float:
    r"""Shannon entropy (nats) of ``p_beta`` by flat enumeration -- the oracle for the identity.

    ``H = -sum_D p_beta(D) log p_beta(D)``; equals the closed-form ``beta (V_beta - E[score])``.
    Ranges from ``log(#derivations)`` as ``beta -> 0`` to ``log(#argmax)`` as ``beta -> inf``.
    """
    if beta <= 0.0:
        raise ValueError(f"beta must be > 0, got {beta}")
    scores = np.array(
        [derivation_weight(weights, d) for d in enumerate_derivations(graph)], dtype=float
    )
    if scores.size == 0:
        raise ValueError("graph has no complete derivations (root unreachable)")
    m = float(np.max(scores))
    probs = np.exp(beta * (scores - m))
    probs /= float(np.sum(probs))
    nz = probs > 0.0
    return float(-np.sum(probs[nz] * np.log(probs[nz])))


def sample_derivations(
    graph: Hypergraph,
    weights: Sequence[float],
    beta: float,
    num_samples: int,
    seed: int | None = None,
) -> tuple[FloatArray, list[tuple[int, ...]]]:
    r"""Exact forward-filtering backward-sampling of ``num_samples`` derivations from ``p_beta``.

    Returns ``(counts, samples)``: ``counts`` is ``(num_samples, num_edges)`` per-edge usage
    counts and ``samples`` the chosen edge-index tuples. Each derivation is drawn *exactly*
    from ``p_beta`` (top-down, choosing each node's incoming edge with the tower softmax of
    ``edge_weight + inside(tails)``), so ``counts.mean(0) -> `` the closed-form marginals.
    """
    if beta <= 0.0:
        raise ValueError(f"beta must be > 0, got {beta}")
    if num_samples < 1:
        raise ValueError(f"num_samples must be >= 1, got {num_samples}")
    w = np.asarray(weights, dtype=float)
    inside = _inside(graph, w, beta)
    rng = np.random.default_rng(seed)
    counts = np.zeros((num_samples, graph.num_edges))
    samples: list[tuple[int, ...]] = []
    for s in range(num_samples):
        chosen: list[int] = []
        stack = [graph.root]
        while stack:
            v = stack.pop()
            inc = graph.incoming(v)
            if not inc:
                continue
            rows = np.array(
                [w[ei] + sum(inside[t] for t in graph.edges[ei].tails) for ei in inc]
            )
            m = float(np.max(rows))
            probs = np.exp(beta * (rows - m))
            probs /= float(np.sum(probs))
            ei = int(inc[int(rng.choice(len(inc), p=probs))])
            chosen.append(ei)
            counts[s, ei] += 1.0
            stack.extend(graph.edges[ei].tails)
        samples.append(tuple(chosen))
    return counts, samples


def _combine(
    tail_lists: list[list[tuple[float, tuple[int, ...]]]], k: int
) -> Iterator[tuple[tuple[float, tuple[int, ...]], ...]]:
    r"""Cartesian product of the per-tail k-best lists (each ``<= k`` long, arity ``<= 2``)."""
    if not tail_lists:
        yield ()
        return
    head, rest = tail_lists[0], tail_lists[1:]
    for item in head:
        for tail in _combine(rest, k):
            yield (item, *tail)


def kbest_derivations(
    graph: Hypergraph, weights: Sequence[float], k: int
) -> list[tuple[float, tuple[int, ...]]]:
    r"""Exact ``k``-best derivations by score (descending) -- the topological k-best DP.

    Huang & Chiang (2005): the head's k-best are generated from the k-best of the tails, so a
    single topological sweep keeping the top ``k`` per node is exact. Returns ``(score,
    edge-tuple)`` pairs, at most ``k`` (fewer if the graph has fewer derivations).
    """
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")
    w = np.asarray(weights, dtype=float)
    kbest: list[list[tuple[float, tuple[int, ...]]]] = [[] for _ in range(graph.num_nodes)]
    for v in range(graph.num_nodes):
        inc = graph.incoming(v)
        if not inc:
            continue
        cands: list[tuple[float, tuple[int, ...]]] = []
        for ei in inc:
            e = graph.edges[ei]
            tail_lists = [kbest[t] for t in e.tails]
            if any(len(tl) == 0 for tl in tail_lists):
                continue  # a tail has no sub-derivation yet (unreachable)
            for combo in _combine(tail_lists, k):
                score = float(w[ei]) + float(sum(c[0] for c in combo))
                deriv = (ei, *[x for c in combo for x in c[1]])
                cands.append((score, deriv))
        cands.sort(key=lambda x: -x[0])
        kbest[v] = cands[:k]
    return kbest[graph.root]


def brute_force_kbest(
    graph: Hypergraph, weights: Sequence[float], k: int
) -> list[tuple[float, tuple[int, ...]]]:
    r"""Enumerate-and-sort oracle for :func:`kbest_derivations` (exponential; tiny graphs)."""
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")
    scored = [(derivation_weight(weights, d), d) for d in enumerate_derivations(graph)]
    scored.sort(key=lambda x: -x[0])
    return scored[:k]


__all__ = [
    "brute_force_entropy",
    "brute_force_kbest",
    "kbest_derivations",
    "sample_derivations",
]
