# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""A semiring / hypergraph generalized-DP driver -- the foundation the other DPs lift to.

Every dynamic program in this package is a reduction over a set of *derivations* of a
weighted hypergraph: nodes are DP items (chain cells, DAG vertices, parse spans), and a
**hyperedge** ``h <- (t_1, ..., t_k)`` (arity ``k in {0, 1, 2}``) builds item ``h`` from
already-built tail items with an additive edge weight. A *derivation* of the ``root`` is a
tree of hyperedges; its score is the sum of its edge weights. The DP value is a semiring
reduction over all derivations:

* :class:`MaxPlusSemiring` -- ``max`` of derivation scores (the hard optimum, ``beta -> inf``);
* :class:`LogSemiring` -- ``lse_beta`` of derivation scores (the soft relaxation);
* :class:`CountingSemiring` -- the exact number of derivations (the ``N`` in ``log N / beta``).

Because ``lse_beta`` and ``max`` distribute over the additive edge weights, the flat
reduction over (exponentially many) derivations equals the topological node recursion --
this module computes the recursion and validates it against the brute-force flat oracle
(:func:`enumerate_derivations` / :func:`brute_force_value`) on tiny graphs. The
differentiable twins in :mod:`omnibias.struct.torch.semiring` /
:mod:`omnibias.struct.jax.semiring` run the same ``LogSemiring`` recursion on backend
tensors and read off the closed-form edge marginals; both are pinned bit-for-bit to the
hand-written :func:`~omnibias.struct.torch.soft_viterbi` / ``soft_shortest_path`` /
``soft_dtw`` / ``soft_align`` layers, which is the *additive-safety* proof that the driver
subsumes them without changing their numerics.

Pure numpy / Python (no backend import); :func:`from_dag` lifts a :class:`DAG` into a
hypergraph so any shortest-/longest-path DP inherits the driver for free.
"""

from __future__ import annotations

import math
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from typing import Protocol, TypeVar

import numpy as np
from numpy.typing import NDArray
from omnibias.struct._core.trellis import DAG

FloatArray = NDArray[np.float64]

# Edge-weight vectors may arrive as a plain Python sequence or a numpy array (the
# family oracles build them with numpy); both are indexed and ``float()``-cast internally.
WeightVector = Sequence[float] | FloatArray

T = TypeVar("T")


def _logsumexp(scores: FloatArray, beta: float) -> float:
    r"""Stable ``beta^-1 log sum_i exp(beta scores_i)`` (returns ``-inf`` if all ``-inf``)."""
    if scores.size == 0:
        return -math.inf
    m = float(np.max(scores))
    if not math.isfinite(m):
        return m
    return m + math.log(float(np.sum(np.exp(beta * (scores - m))))) / beta


# ---------------------------------------------------------------------------
# Hypergraph structure
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class HyperEdge:
    r"""A weighted hyperedge ``head <- tails`` of arity ``len(tails) in {0, 1, 2}``.

    ``tails`` are node indices, each strictly less than ``head`` (topological order); an
    arity-0 edge is an **axiom** (a leaf / base case). The edge weight is supplied
    separately as ``weights[edge_index]`` so it can be an arbitrary differentiable function
    of the backend parameters -- the structure here is pure data.
    """

    head: int
    tails: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        if len(self.tails) > 2:
            raise ValueError(f"hyperedge arity must be <= 2, got {len(self.tails)} tails")
        for t in self.tails:
            if t >= self.head:
                raise ValueError(
                    f"hyperedge {self.head} <- {self.tails} violates topological order "
                    f"(tail {t} >= head {self.head})"
                )


@dataclass(frozen=True)
class Hypergraph:
    r"""A topologically-ordered weighted hypergraph with a distinguished ``root`` goal node.

    Nodes are ``0 .. num_nodes - 1``; every :class:`HyperEdge`'s tails are ``< head`` so a
    single forward sweep computes every node value. :meth:`incoming` lists the edges into a
    node; :attr:`num_edges` is the length of the parallel ``weights`` vector consumed by the
    reductions and the backend driver.
    """

    num_nodes: int
    edges: tuple[HyperEdge, ...]
    root: int
    _incoming: tuple[tuple[int, ...], ...] = ()

    def __post_init__(self) -> None:
        if self.num_nodes < 1:
            raise ValueError(f"num_nodes must be >= 1, got {self.num_nodes}")
        if not 0 <= self.root < self.num_nodes:
            raise ValueError(f"root {self.root} out of range [0, {self.num_nodes})")
        inc: list[list[int]] = [[] for _ in range(self.num_nodes)]
        for ei, e in enumerate(self.edges):
            if not 0 <= e.head < self.num_nodes:
                raise ValueError(f"edge {ei} head {e.head} out of range [0, {self.num_nodes})")
            for t in e.tails:
                if not 0 <= t < self.num_nodes:
                    raise ValueError(f"edge {ei} tail {t} out of range [0, {self.num_nodes})")
            inc[e.head].append(ei)
        object.__setattr__(self, "_incoming", tuple(tuple(x) for x in inc))

    @property
    def num_edges(self) -> int:
        """Number of hyperedges (the length of the ``weights`` vector)."""
        return len(self.edges)

    def incoming(self, node: int) -> tuple[int, ...]:
        """Edge indices whose head is ``node`` (the alternatives reduced at ``node``)."""
        return self._incoming[node]


# ---------------------------------------------------------------------------
# Semiring protocol + the three canonical reductions
# ---------------------------------------------------------------------------


class Semiring(Protocol[T]):
    r"""A DP semiring: ``edge`` combines a weight with tail values, ``reduce`` over edges.

    ``zero`` is the reduction identity (the value of a node with no incoming edges);
    ``one`` the combination identity (the empty tail product). ``edge(weight, tails)`` is
    the value contributed by one hyperedge; ``reduce(values)`` folds the alternatives.
    """

    @property
    def zero(self) -> T: ...
    @property
    def one(self) -> T: ...
    def edge(self, weight: float, tails: Sequence[T]) -> T: ...
    def reduce(self, values: Sequence[T]) -> T: ...


class MaxPlusSemiring:
    r"""The hard ``(max, +)`` tropical semiring -- the ``beta -> inf`` optimum."""

    zero: float = -math.inf
    one: float = 0.0

    def edge(self, weight: float, tails: Sequence[float]) -> float:
        return weight + sum(tails)

    def reduce(self, values: Sequence[float]) -> float:
        return max(values) if values else self.zero


class LogSemiring:
    r"""The soft ``(lse_beta, +)`` log semiring -- the differentiable relaxation."""

    one: float = 0.0

    def __init__(self, beta: float = 1.0) -> None:
        if beta <= 0.0:
            raise ValueError(f"beta must be > 0, got {beta}")
        self.beta = float(beta)

    @property
    def zero(self) -> float:
        return -math.inf

    def edge(self, weight: float, tails: Sequence[float]) -> float:
        return weight + sum(tails)

    def reduce(self, values: Sequence[float]) -> float:
        return _logsumexp(np.asarray(values, dtype=float), self.beta)


# Alias matching the plan's naming (the log semiring *is* the lse_beta reduction).
LSEBeta = LogSemiring


class CountingSemiring:
    r"""The ``(+, x)`` counting semiring -- exact number of derivations (weights ignored)."""

    zero: int = 0
    one: int = 1

    def edge(self, weight: float, tails: Sequence[int]) -> int:
        prod = 1
        for t in tails:
            prod *= t
        return prod

    def reduce(self, values: Sequence[int]) -> int:
        return sum(values)


# ---------------------------------------------------------------------------
# The generic topological driver + convenience reductions
# ---------------------------------------------------------------------------


def semiring_value(
    graph: Hypergraph,
    weights: WeightVector | None,
    semiring: Semiring[T],
) -> T:
    r"""Reduce ``graph`` to its ``root`` value under ``semiring`` (one topological sweep).

    ``weights`` is a length-``graph.num_edges`` vector of edge weights (``None`` for the
    weight-free :class:`CountingSemiring`). This is the pure-numpy oracle the backend
    drivers reproduce; it is exact for any DAG-shaped derivation forest.
    """
    if weights is not None and len(weights) != graph.num_edges:
        raise ValueError(f"weights must have length num_edges={graph.num_edges}, got {len(weights)}")
    vals: list[T] = [semiring.zero] * graph.num_nodes
    for v in range(graph.num_nodes):
        inc = graph.incoming(v)
        if not inc:
            continue
        edge_vals: list[T] = []
        for ei in inc:
            e = graph.edges[ei]
            w = 0.0 if weights is None else float(weights[ei])
            edge_vals.append(semiring.edge(w, [vals[t] for t in e.tails]))
        vals[v] = semiring.reduce(edge_vals)
    return vals[graph.root]


def hard_value(graph: Hypergraph, weights: WeightVector) -> float:
    r"""Best (max-plus) derivation score of ``graph`` -- the ``beta -> inf`` optimum."""
    return semiring_value(graph, weights, MaxPlusSemiring())


def soft_value(graph: Hypergraph, weights: WeightVector, beta: float) -> float:
    r"""Soft ``lse_beta`` derivation value (numpy oracle for the backend ``semiring_value``)."""
    return semiring_value(graph, weights, LogSemiring(beta))


def count_derivations(graph: Hypergraph) -> int:
    r"""Exact number of complete derivations rooted at ``graph.root`` -- the ``N`` in the gap."""
    return semiring_value(graph, None, CountingSemiring())


def best_derivation(graph: Hypergraph, weights: WeightVector) -> tuple[float, tuple[int, ...]]:
    r"""Max-plus optimal derivation: its score and the tuple of edge indices (pre-order)."""
    w = [float(x) for x in weights]
    if len(w) != graph.num_edges:
        raise ValueError(f"weights must have length num_edges={graph.num_edges}, got {len(w)}")
    value = [-math.inf] * graph.num_nodes
    choice = [-1] * graph.num_nodes
    for v in range(graph.num_nodes):
        for ei in graph.incoming(v):
            e = graph.edges[ei]
            score = w[ei] + sum(value[t] for t in e.tails)
            if score > value[v]:
                value[v], choice[v] = score, ei

    def build(node: int) -> tuple[int, ...]:
        ei = choice[node]
        if ei < 0:
            raise ValueError(f"node {node} has no derivation (unreachable)")
        out = [ei]
        for t in graph.edges[ei].tails:
            out.extend(build(t))
        return tuple(out)

    return value[graph.root], build(graph.root)


def enumerate_derivations(graph: Hypergraph) -> Iterator[tuple[int, ...]]:
    r"""Yield every complete derivation as an edge-index tuple (brute-force; tiny graphs)."""
    memo: dict[int, list[tuple[int, ...]]] = {}

    def at(node: int) -> list[tuple[int, ...]]:
        if node in memo:
            return memo[node]
        out: list[tuple[int, ...]] = []
        for ei in graph.incoming(node):
            e = graph.edges[ei]
            tail_derivs = [at(t) for t in e.tails]
            for combo in _product(tail_derivs):
                merged = (ei, *[x for sub in combo for x in sub])
                out.append(merged)
        memo[node] = out
        return out

    yield from at(graph.root)


def _product(lists: list[list[tuple[int, ...]]]) -> Iterator[tuple[tuple[int, ...], ...]]:
    if not lists:
        yield ()
        return
    head, rest = lists[0], lists[1:]
    for item in head:
        for tail in _product(rest):
            yield (item, *tail)


def derivation_weight(weights: WeightVector, derivation: Sequence[int]) -> float:
    r"""Additive score of a derivation: ``sum(weights[e] for e in derivation)``."""
    return float(sum(float(weights[e]) for e in derivation))


def brute_force_value(
    graph: Hypergraph,
    weights: WeightVector,
    beta: float | None = None,
) -> float:
    r"""Flat oracle: ``max`` (``beta is None``) or ``lse_beta`` over *enumerated* derivations.

    The ground truth for :func:`hard_value` / :func:`soft_value` -- enumerates every
    derivation (exponential, tiny graphs only) and reduces their scores directly.
    """
    scores = np.array(
        [derivation_weight(weights, d) for d in enumerate_derivations(graph)], dtype=np.float64
    )
    if scores.size == 0:
        raise ValueError("graph has no complete derivations (root unreachable)")
    if beta is None:
        return float(np.max(scores))
    if beta <= 0.0:
        raise ValueError(f"beta must be > 0, got {beta}")
    return _logsumexp(scores, beta)


# ---------------------------------------------------------------------------
# Lifting an existing DAG into the driver (reused by shortest-path / alignment)
# ---------------------------------------------------------------------------


def from_dag(dag: DAG) -> tuple[Hypergraph, dict[tuple[int, int], int]]:
    r"""Lift a :class:`DAG` into a :class:`Hypergraph` (arity-1 edges + a source axiom).

    Returns ``(graph, edge_index)`` where ``edge_index[(u, v)]`` is the hyperedge position
    of DAG edge ``u -> v`` (so a backend builds ``weights`` by scattering the per-edge
    *scores* there; the source axiom edge keeps weight ``0``). ``graph.root`` is
    ``dag.sink``. The driver value equals the soft path *score* (max convention); softmin
    shortest-path cost is its negation, matching
    :func:`omnibias.struct.torch.soft_shortest_path`.
    """
    edges: list[HyperEdge] = [HyperEdge(head=dag.source, tails=())]
    edge_index: dict[tuple[int, int], int] = {}
    for u, v in sorted(dag.edges):
        edge_index[(u, v)] = len(edges)
        edges.append(HyperEdge(head=v, tails=(u,)))
    graph = Hypergraph(num_nodes=dag.num_nodes, edges=tuple(edges), root=dag.sink)
    return graph, edge_index


__all__ = [
    "CountingSemiring",
    "HyperEdge",
    "Hypergraph",
    "LSEBeta",
    "LogSemiring",
    "MaxPlusSemiring",
    "Semiring",
    "best_derivation",
    "brute_force_value",
    "count_derivations",
    "derivation_weight",
    "enumerate_derivations",
    "from_dag",
    "hard_value",
    "semiring_value",
    "soft_value",
]
