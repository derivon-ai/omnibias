# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""CKY parsing as a hypergraph DP: chart construction + hard / brute / counting oracles.

A binarized (Chomsky-normal-form) grammar over ``R`` nonterminals parses a length-``L``
sentence by filling a chart of items ``(A, i, j)`` -- "nonterminal ``A`` spans tokens
``[i, j)``". An item of width 1 is *lexical* (an axiom weighted by the emission score
``emit[i, A]``); a wider item is built by a binary rule ``A -> B C`` splitting the span at
some ``k``, weighted by the rule score. That is exactly an arity-2
:class:`~omnibias.struct._core.semiring.Hypergraph`, so :func:`build_chart` lifts the
grammar onto the shared driver and the soft inside partition / inside-outside marginals come
for free from :mod:`omnibias.struct.torch.parse` / :mod:`omnibias.struct.jax.parse`.

This module is the backend-agnostic register: :func:`hard_cky` / :func:`best_parse_tree`
are an independent classic CKY DP (the ground truth the driver's ``hard_value`` is pinned
to); :func:`count_parse_trees` is the exact derivation count -- the ``N`` in the certified
``log(N) / beta`` gap (for the fully-ambiguous single-nonterminal grammar it is the Catalan
number ``C_{L-1}``); :func:`brute_force_cky` enumerates every tree via the driver oracle.
Pure numpy / Python.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray
from omnibias.struct._core.semiring import (
    HyperEdge,
    Hypergraph,
    brute_force_value,
    count_derivations,
    soft_value,
)

FloatArray = NDArray[np.float64]

# A parse tree: a leaf ``(A, i)`` or an internal node ``(A, i, j, left, right)``.
Tree = tuple[object, ...]


@dataclass(frozen=True)
class BinaryGrammar:
    r"""A Chomsky-normal-form grammar: ``num_nonterminals`` symbols and binary ``A -> B C`` rules.

    ``rules`` is a tuple of ``(A, B, C)`` triples (head, left child, right child); ``start``
    is the goal nonterminal spanning the whole sentence. Every nonterminal may also be
    lexical (emit a token) -- the emission scores are supplied per parse, not fixed here.
    """

    num_nonterminals: int
    rules: tuple[tuple[int, int, int], ...]
    start: int = 0

    def __post_init__(self) -> None:
        if self.num_nonterminals < 1:
            raise ValueError(f"num_nonterminals must be >= 1, got {self.num_nonterminals}")
        if not 0 <= self.start < self.num_nonterminals:
            raise ValueError(f"start {self.start} out of range [0, {self.num_nonterminals})")
        for a, b, c in self.rules:
            for x in (a, b, c):
                if not 0 <= x < self.num_nonterminals:
                    raise ValueError(f"rule ({a},{b},{c}) has a symbol out of range")

    @property
    def num_rules(self) -> int:
        """Number of binary rules."""
        return len(self.rules)


@dataclass(frozen=True)
class ChartSpec:
    r"""The lifted CKY chart: a :class:`Hypergraph` plus the per-edge weight sourcing.

    ``edge_specs[e]`` is ``("emit", i, A)`` (a lexical axiom drawing ``emit[i, A]``) or
    ``("rule", r, 0)`` (a binary edge drawing rule score ``r``; the trailing ``0`` is unused).
    ``node_of[(A, i, j)]`` maps a chart item to its node index; :attr:`root` is
    ``(start, 0, length)``.
    """

    graph: Hypergraph
    edge_specs: tuple[tuple[str, int, int], ...]
    length: int
    num_nonterminals: int
    num_rules: int
    node_of: dict[tuple[int, int, int], int]
    root: int = field(default=0)


def build_chart(grammar: BinaryGrammar, length: int) -> ChartSpec:
    r"""Lift ``grammar`` parsing a length-``length`` sentence into a chart :class:`Hypergraph`.

    Items are ordered by span width so tails always precede heads (topological); lexical
    axioms carry ``("emit", i, A)`` and binary edges ``("rule", r)``.
    """
    if length < 1:
        raise ValueError(f"length must be >= 1, got {length}")
    r_nt, length_l = grammar.num_nonterminals, length
    node_of: dict[tuple[int, int, int], int] = {}
    nid = 0
    for w in range(1, length_l + 1):
        for i in range(0, length_l - w + 1):
            j = i + w
            for a in range(r_nt):
                node_of[(a, i, j)] = nid
                nid += 1
    edges: list[HyperEdge] = []
    specs: list[tuple[str, int, int]] = []
    for i in range(length_l):
        for a in range(r_nt):
            edges.append(HyperEdge(head=node_of[(a, i, i + 1)], tails=()))
            specs.append(("emit", i, a))
    for w in range(2, length_l + 1):
        for i in range(0, length_l - w + 1):
            j = i + w
            for r, (a, b, c) in enumerate(grammar.rules):
                for k in range(i + 1, j):
                    edges.append(
                        HyperEdge(head=node_of[(a, i, j)], tails=(node_of[(b, i, k)], node_of[(c, k, j)]))
                    )
                    specs.append(("rule", r, 0))
    root = node_of[(grammar.start, 0, length_l)]
    graph = Hypergraph(num_nodes=nid, edges=tuple(edges), root=root)
    return ChartSpec(
        graph=graph,
        edge_specs=tuple(specs),
        length=length_l,
        num_nonterminals=r_nt,
        num_rules=grammar.num_rules,
        node_of=node_of,
        root=root,
    )


def chart_edge_weights(spec: ChartSpec, emit: FloatArray, rule: FloatArray) -> FloatArray:
    r"""Numpy edge-weight vector for ``spec`` from emission ``(L, R)`` and rule ``(num_rules,)`` scores."""
    e = np.asarray(emit, dtype=float)
    r = np.asarray(rule, dtype=float)
    out = np.zeros(spec.graph.num_edges)
    for idx, s in enumerate(spec.edge_specs):
        if s[0] == "emit":
            out[idx] = e[s[1], s[2]]
        else:
            out[idx] = r[s[1]]
    return out


def hard_cky(grammar: BinaryGrammar, emit: FloatArray, rule: FloatArray) -> float:
    r"""Classic CKY best-parse score (max-plus), independent of the hypergraph driver."""
    e = np.asarray(emit, dtype=float)
    r = np.asarray(rule, dtype=float)
    length_l, r_nt = e.shape[0], grammar.num_nonterminals
    neg = -math.inf
    chart = np.full((r_nt, length_l + 1, length_l + 1), neg)
    for i in range(length_l):
        for a in range(r_nt):
            chart[a, i, i + 1] = e[i, a]
    for w in range(2, length_l + 1):
        for i in range(0, length_l - w + 1):
            j = i + w
            for ridx, (a, b, c) in enumerate(grammar.rules):
                for k in range(i + 1, j):
                    val = r[ridx] + chart[b, i, k] + chart[c, k, j]
                    if val > chart[a, i, j]:
                        chart[a, i, j] = val
    return float(chart[grammar.start, 0, length_l])


def best_parse_tree(grammar: BinaryGrammar, emit: FloatArray, rule: FloatArray) -> tuple[float, Tree]:
    r"""Classic CKY Viterbi parse: the best score and its parse tree (nested tuples)."""
    e = np.asarray(emit, dtype=float)
    r = np.asarray(rule, dtype=float)
    length_l, r_nt = e.shape[0], grammar.num_nonterminals
    neg = -math.inf
    chart = np.full((r_nt, length_l + 1, length_l + 1), neg)
    back: dict[tuple[int, int, int], tuple[int, int, int, int]] = {}
    for i in range(length_l):
        for a in range(r_nt):
            chart[a, i, i + 1] = e[i, a]
    for w in range(2, length_l + 1):
        for i in range(0, length_l - w + 1):
            j = i + w
            for ridx, (a, b, c) in enumerate(grammar.rules):
                for k in range(i + 1, j):
                    val = r[ridx] + chart[b, i, k] + chart[c, k, j]
                    if val > chart[a, i, j]:
                        chart[a, i, j] = val
                        back[(a, i, j)] = (ridx, b, c, k)

    def build(a: int, i: int, j: int) -> Tree:
        if j - i == 1:
            return (a, i)
        ridx, b, c, k = back[(a, i, j)]
        return (a, i, j, build(b, i, k), build(c, k, j))

    return float(chart[grammar.start, 0, length_l]), build(grammar.start, 0, length_l)


def count_parse_trees(grammar: BinaryGrammar, length: int) -> int:
    r"""Exact number of parse trees of a length-``length`` sentence -- the ``N`` in the gap."""
    return count_derivations(build_chart(grammar, length).graph)


def soft_cky(grammar: BinaryGrammar, emit: FloatArray, rule: FloatArray, beta: float) -> float:
    r"""Soft ``lse_beta`` inside partition (numpy oracle for the backend ``soft_inside``)."""
    spec = build_chart(grammar, int(np.asarray(emit).shape[0]))
    return soft_value(spec.graph, chart_edge_weights(spec, emit, rule), beta)


def brute_force_cky(
    grammar: BinaryGrammar, emit: FloatArray, rule: FloatArray, beta: float | None = None
) -> float:
    r"""Flat oracle over *enumerated* parse trees: ``max`` (``beta is None``) or ``lse_beta``."""
    spec = build_chart(grammar, int(np.asarray(emit).shape[0]))
    return brute_force_value(spec.graph, chart_edge_weights(spec, emit, rule), beta)


__all__ = [
    "BinaryGrammar",
    "ChartSpec",
    "Tree",
    "best_parse_tree",
    "brute_force_cky",
    "build_chart",
    "chart_edge_weights",
    "count_parse_trees",
    "hard_cky",
    "soft_cky",
]
