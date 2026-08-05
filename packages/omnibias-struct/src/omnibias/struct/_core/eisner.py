# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Eisner projective dependency parsing as a hypergraph DP + hard / brute-force oracles.

A projective dependency tree of ``n`` words (positions ``1 .. n`` plus a ``ROOT`` at ``0``)
assigns each word a single head so the arcs form a tree rooted at ``0`` and *do not cross*.
Eisner's ``O(n^3)`` algorithm builds it from **complete** and **incomplete** half-spans
(``C[s][t][dir]`` / ``I[s][t][dir]``): an incomplete right span ``I[s][t][->]`` *is* the arc
``s -> t`` glued from two complete children, and complete spans collect a word's descendants.
Those items are nodes and the glue steps are arity-2 hyperedges, so :func:`eisner_hypergraph`
lifts the whole thing onto the shared semiring driver -- the soft partition over projective
trees and the closed-form arc marginals then come from
:mod:`omnibias.struct.torch.eisner` / :mod:`omnibias.struct.jax.eisner` for free.

Backend-agnostic register: :func:`hard_eisner` / :func:`best_projective_tree` are an
independent classic Eisner DP; :func:`brute_force_projective` enumerates *every* head
assignment and keeps the valid projective trees (the ground truth); :func:`count_projective_trees`
is the exact derivation count -- and the fact that it equals the brute-force tree count is
the proof that the Eisner hypergraph is spurious-ambiguity-free (one derivation per tree).
Arc scores live in a dense ``(n + 1, n + 1)`` matrix ``arc[h, m]`` (head ``h``, modifier
``m >= 1``). Pure numpy / Python.
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Iterator
from dataclasses import dataclass

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


@dataclass(frozen=True)
class EisnerSpec:
    r"""The lifted Eisner chart: a :class:`Hypergraph` plus per-edge arc sourcing.

    ``edge_specs[e]`` is ``("arc", h, m)`` (draws ``arc[h, m]``) or ``("base", -1, -1)``
    (weight ``0`` -- an axiom or a complete-span glue). ``root`` is the complete right span
    ``C[0][n][->]`` spanning ``ROOT`` to the last word.
    """

    graph: Hypergraph
    edge_specs: tuple[tuple[str, int, int], ...]
    n_words: int
    node_of: dict[tuple[str, int, int], int]
    root: int


def eisner_hypergraph(n_words: int) -> EisnerSpec:
    r"""Lift Eisner's projective-parse DP for ``n_words`` onto a chart :class:`Hypergraph`.

    Items are ordered by span width (and incomplete before complete within a width) so tails
    always precede heads. ``CR/CL`` are complete right/left spans, ``IR/IL`` incomplete.
    """
    if n_words < 1:
        raise ValueError(f"n_words must be >= 1, got {n_words}")
    positions = n_words + 1
    node_of: dict[tuple[str, int, int], int] = {}
    nid = 0
    for w in range(0, positions):
        if w >= 1:
            for s in range(0, positions - w):
                t = s + w
                node_of[("IR", s, t)] = nid
                nid += 1
                if s >= 1:
                    node_of[("IL", s, t)] = nid
                    nid += 1
        for s in range(0, positions - w):
            t = s + w
            node_of[("CR", s, t)] = nid
            nid += 1
            node_of[("CL", s, t)] = nid
            nid += 1
    edges: list[HyperEdge] = []
    specs: list[tuple[str, int, int]] = []
    for s in range(positions):
        edges.append(HyperEdge(head=node_of[("CR", s, s)], tails=()))
        specs.append(("base", -1, -1))
        edges.append(HyperEdge(head=node_of[("CL", s, s)], tails=()))
        specs.append(("base", -1, -1))
    for w in range(1, positions):
        for s in range(0, positions - w):
            t = s + w
            for r in range(s, t):
                edges.append(
                    HyperEdge(head=node_of[("IR", s, t)], tails=(node_of[("CR", s, r)], node_of[("CL", r + 1, t)]))
                )
                specs.append(("arc", s, t))
            if s >= 1:
                for r in range(s, t):
                    edges.append(
                        HyperEdge(head=node_of[("IL", s, t)], tails=(node_of[("CR", s, r)], node_of[("CL", r + 1, t)]))
                    )
                    specs.append(("arc", t, s))
            for r in range(s + 1, t + 1):
                edges.append(
                    HyperEdge(head=node_of[("CR", s, t)], tails=(node_of[("IR", s, r)], node_of[("CR", r, t)]))
                )
                specs.append(("base", -1, -1))
            for r in range(s, t):
                if r >= 1:
                    edges.append(
                        HyperEdge(head=node_of[("CL", s, t)], tails=(node_of[("CL", s, r)], node_of[("IL", r, t)]))
                    )
                    specs.append(("base", -1, -1))
    root = node_of[("CR", 0, n_words)]
    graph = Hypergraph(num_nodes=nid, edges=tuple(edges), root=root)
    return EisnerSpec(graph=graph, edge_specs=tuple(specs), n_words=n_words, node_of=node_of, root=root)


def eisner_edge_weights(spec: EisnerSpec, arc: FloatArray) -> FloatArray:
    r"""Numpy edge-weight vector for ``spec`` from the arc-score matrix ``arc[h, m]``."""
    a = np.asarray(arc, dtype=float)
    out = np.zeros(spec.graph.num_edges)
    for idx, s in enumerate(spec.edge_specs):
        if s[0] == "arc":
            out[idx] = a[s[1], s[2]]
    return out


def hard_eisner(arc: FloatArray) -> float:
    r"""Classic Eisner best projective-parse score (max-plus), independent of the driver."""
    a = np.asarray(arc, dtype=float)
    positions = a.shape[0]
    n = positions - 1
    neg = -math.inf
    cr = np.full((positions, positions), neg)
    cl = np.full((positions, positions), neg)
    ir = np.full((positions, positions), neg)
    il = np.full((positions, positions), neg)
    for s in range(positions):
        cr[s, s] = 0.0
        cl[s, s] = 0.0
    for w in range(1, positions):
        for s in range(0, positions - w):
            t = s + w
            glue = max(cr[s, r] + cl[r + 1, t] for r in range(s, t))
            ir[s, t] = glue + a[s, t]
            if s >= 1:
                il[s, t] = glue + a[t, s]
            cr[s, t] = max(ir[s, r] + cr[r, t] for r in range(s + 1, t + 1))
            cl_best = neg
            for r in range(s, t):
                if r >= 1:
                    cl_best = max(cl_best, cl[s, r] + il[r, t])
            cl[s, t] = cl_best
    return float(cr[0, n])


def best_projective_tree(arc: FloatArray) -> tuple[float, dict[int, int]]:
    r"""Classic Eisner Viterbi parse: the best score and the head map ``{modifier: head}``."""
    a = np.asarray(arc, dtype=float)
    positions = a.shape[0]
    n = positions - 1
    neg = -math.inf
    cr = np.full((positions, positions), neg)
    cl = np.full((positions, positions), neg)
    ir = np.full((positions, positions), neg)
    il = np.full((positions, positions), neg)
    bcr: dict[tuple[int, int], int] = {}
    bcl: dict[tuple[int, int], int] = {}
    bir: dict[tuple[int, int], int] = {}
    bil: dict[tuple[int, int], int] = {}
    for s in range(positions):
        cr[s, s] = 0.0
        cl[s, s] = 0.0
    for w in range(1, positions):
        for s in range(0, positions - w):
            t = s + w
            best, arg = neg, s
            for r in range(s, t):
                v = cr[s, r] + cl[r + 1, t]
                if v > best:
                    best, arg = v, r
            ir[s, t], bir[(s, t)] = best + a[s, t], arg
            if s >= 1:
                il[s, t], bil[(s, t)] = best + a[t, s], arg
            best, arg = neg, s + 1
            for r in range(s + 1, t + 1):
                v = ir[s, r] + cr[r, t]
                if v > best:
                    best, arg = v, r
            cr[s, t], bcr[(s, t)] = best, arg
            best, arg = neg, s
            for r in range(s, t):
                if r >= 1:
                    v = cl[s, r] + il[r, t]
                    if v > best:
                        best, arg = v, r
            cl[s, t], bcl[(s, t)] = best, arg

    heads: dict[int, int] = {}

    def rec(kind: str, s: int, t: int) -> None:
        if s == t:
            return
        if kind == "IR":
            heads[t] = s
            r = bir[(s, t)]
            rec("CR", s, r)
            rec("CL", r + 1, t)
        elif kind == "IL":
            heads[s] = t
            r = bil[(s, t)]
            rec("CR", s, r)
            rec("CL", r + 1, t)
        elif kind == "CR":
            r = bcr[(s, t)]
            rec("IR", s, r)
            rec("CR", r, t)
        else:  # CL
            r = bcl[(s, t)]
            rec("CL", s, r)
            rec("IL", r, t)

    rec("CR", 0, n)
    return float(cr[0, n]), heads


def count_projective_trees(n_words: int) -> int:
    r"""Exact number of projective dependency trees of ``n_words`` -- the ``N`` in the gap."""
    return count_derivations(eisner_hypergraph(n_words).graph)


def soft_eisner(arc: FloatArray, beta: float) -> float:
    r"""Soft ``lse_beta`` projective-parse partition (numpy oracle for the backend layer)."""
    spec = eisner_hypergraph(int(np.asarray(arc).shape[0]) - 1)
    return soft_value(spec.graph, eisner_edge_weights(spec, arc), beta)


# ---------------------------------------------------------------------------
# Brute-force ground truth: enumerate head assignments, keep valid projective trees
# ---------------------------------------------------------------------------


def _is_tree(heads: dict[int, int], n: int) -> bool:
    for m in range(1, n + 1):
        seen: set[int] = set()
        cur = m
        while cur != 0:
            if cur in seen:
                return False
            seen.add(cur)
            cur = heads[cur]
    return True


def _descendants(heads: dict[int, int], n: int, head: int) -> set[int]:
    children: dict[int, list[int]] = {i: [] for i in range(n + 1)}
    for m in range(1, n + 1):
        children[heads[m]].append(m)
    out: set[int] = set()
    stack = [head]
    while stack:
        x = stack.pop()
        for c in children[x]:
            if c not in out:
                out.add(c)
                stack.append(c)
    return out


def _is_projective(heads: dict[int, int], n: int) -> bool:
    for m in range(1, n + 1):
        h = heads[m]
        lo, hi = min(h, m), max(h, m)
        desc = _descendants(heads, n, h)
        for k in range(lo + 1, hi):
            if k not in desc:
                return False
    return True


def iter_projective_trees(n: int) -> Iterator[dict[int, int]]:
    r"""Yield every valid projective head map ``{modifier: head}`` of ``n`` words (brute force)."""
    for assignment in itertools.product(range(n + 1), repeat=n):
        heads = {m: assignment[m - 1] for m in range(1, n + 1)}
        if any(heads[m] == m for m in range(1, n + 1)):
            continue
        if _is_tree(heads, n) and _is_projective(heads, n):
            yield heads


def brute_force_projective(arc: FloatArray, beta: float | None = None) -> float:
    r"""Flat oracle over *enumerated* projective trees: ``max`` (``beta is None``) or ``lse_beta``."""
    a = np.asarray(arc, dtype=float)
    n = a.shape[0] - 1
    scores = np.array(
        [float(sum(a[heads[m], m] for m in range(1, n + 1))) for heads in iter_projective_trees(n)]
    )
    if scores.size == 0:
        raise ValueError("no projective trees (n must be >= 1)")
    if beta is None:
        return float(np.max(scores))
    if beta <= 0.0:
        raise ValueError(f"beta must be > 0, got {beta}")
    m = float(np.max(scores))
    return m + math.log(float(np.sum(np.exp(beta * (scores - m))))) / beta


def brute_force_eisner(arc: FloatArray, beta: float | None = None) -> float:
    r"""Flat oracle via the driver's derivation enumeration over the Eisner hypergraph."""
    spec = eisner_hypergraph(int(np.asarray(arc).shape[0]) - 1)
    return brute_force_value(spec.graph, eisner_edge_weights(spec, arc), beta)


__all__ = [
    "EisnerSpec",
    "best_projective_tree",
    "brute_force_eisner",
    "brute_force_projective",
    "count_projective_trees",
    "eisner_edge_weights",
    "eisner_hypergraph",
    "hard_eisner",
    "iter_projective_trees",
    "soft_eisner",
]
