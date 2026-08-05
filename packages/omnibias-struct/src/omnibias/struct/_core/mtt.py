# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Matrix-tree non-projective dependency marginals + Chu-Liu/Edmonds hard oracle.

Unlike the ``lse_beta`` DPs elsewhere in this package, the non-projective partition is
**exact and closed form**: by Tutte's directed Matrix-Tree Theorem the weighted sum over the
exponentially many spanning arborescences of a dense arc graph is a single determinant

    ``Z(beta) = det L(beta)``,  ``L`` the Kirchhoff Laplacian of ``exp(beta * arc)``,

so ``matrix_tree_partition = log Z / beta`` is an *exact* soft value (no relaxation error at
finite ``beta``), and the closed-form arc marginals come from ``L^{-1}`` (Koo et al. 2007).
This is the honest distinction to keep: ``beta -> inf`` is still the temperature axis
(``log Z / beta -> `` the maximum-weight arborescence score), but there is **no** ``lse_beta``
approximation of a ``max`` here -- the determinant sums the trees exactly. The ``beta -> inf``
gap is therefore taken against the Chu-Liu/Edmonds maximum arborescence, bounded by
``log(#arborescences) / beta`` with ``#arborescences = (n + 1)^(n - 1)`` (Cayley).

Register:

* :func:`matrix_tree_partition` / :func:`matrix_tree_marginals` -- numpy oracles (the
  determinant value and the ``L^{-1}`` marginals), column-max stabilised so high ``beta`` is
  safe.
* :func:`max_arborescence` -- Chu-Liu/Edmonds maximum spanning arborescence rooted at ``0``
  (multi-root: the wall may govern several words), with :func:`hard_matrix_tree` its score.
* :func:`iter_arborescences` / :func:`count_arborescences` / :func:`brute_force_arborescence`
  -- the flat ground truth (enumerate every valid head map, keep the arborescences).

Arc scores live in a dense ``(n + 1, n + 1)`` matrix ``arc[h, m]`` (head ``h``, modifier
``m >= 1``; row/column ``0`` is the ROOT wall, which is never a modifier). Pure numpy.
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Iterator

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


# ---------------------------------------------------------------------------
# Kirchhoff Laplacian (column-max stabilised) + exact determinant partition
# ---------------------------------------------------------------------------


def _stabilised_laplacian(arc: FloatArray, beta: float) -> tuple[FloatArray, FloatArray]:
    r"""Return ``(L_tilde, c)``: the column-scaled Kirchhoff Laplacian and its log offsets.

    ``L = L_tilde @ diag(exp(c))`` is the true Laplacian of ``exp(beta * arc)`` over the ``n``
    words; scaling each modifier column by its max log-weight ``c[m]`` keeps ``L_tilde``
    entries in ``[0, 1]`` so ``det`` / ``inv`` stay finite at large ``beta``. Column ``m`` of
    ``L`` gathers every candidate head of word ``m`` (root and words, ``h != m``).
    """
    a = np.asarray(arc, dtype=float)
    if a.ndim != 2 or a.shape[0] != a.shape[1]:
        raise ValueError(f"arc must be square (n+1, n+1), got {a.shape}")
    n = a.shape[0] - 1
    if n < 1:
        raise ValueError(f"need at least one word (arc >= 2x2), got n={n}")
    lw = beta * a
    heads = np.arange(n + 1)
    ltilde = np.zeros((n, n))
    c = np.zeros(n)
    for m in range(1, n + 1):
        valid = heads[heads != m]
        c[m - 1] = float(np.max(lw[valid, m]))
    for m in range(1, n + 1):
        for h in range(n + 1):
            if h == m:
                continue
            wt = math.exp(lw[h, m] - c[m - 1])
            ltilde[m - 1, m - 1] += wt
            if h >= 1:
                ltilde[h - 1, m - 1] -= wt
    return ltilde, c


def matrix_tree_partition(arc: FloatArray, beta: float) -> float:
    r"""Exact non-projective soft value ``log det L(beta) / beta`` (numpy oracle).

    The determinant sums ``exp(beta * score)`` over *all* spanning arborescences exactly, so
    this is not an ``lse_beta`` relaxation -- it equals the flat
    :func:`brute_force_arborescence` partition to machine precision. As ``beta -> inf`` it
    decreases to the maximum-arborescence score (:func:`hard_matrix_tree`).
    """
    if beta <= 0.0:
        raise ValueError(f"beta must be > 0, got {beta}")
    ltilde, c = _stabilised_laplacian(arc, beta)
    sign, logabsdet = np.linalg.slogdet(ltilde)
    if sign <= 0.0:
        raise ValueError("Laplacian determinant is non-positive; arc weights are degenerate")
    return float((logabsdet + float(np.sum(c))) / beta)


def matrix_tree_marginals(arc: FloatArray, beta: float) -> FloatArray:
    r"""Closed-form arc marginals ``P_beta(h -> m)`` via ``L^{-1}`` (Koo et al. 2007).

    Returns an ``(n + 1, n + 1)`` matrix; ``[h, m]`` is the probability that arc ``h -> m`` is
    in a sampled arborescence. Equals ``d matrix_tree_partition / d arc`` exactly, and every
    modifier column ``m >= 1`` sums to ``1`` (each word takes exactly one head). Column-max
    stabilised so it matches the differentiable backends at any ``beta``.
    """
    if beta <= 0.0:
        raise ValueError(f"beta must be > 0, got {beta}")
    a = np.asarray(arc, dtype=float)
    n = a.shape[0] - 1
    ltilde, c = _stabilised_laplacian(a, beta)
    binv = np.linalg.inv(ltilde)
    diag_b = np.diag(binv)
    out = np.zeros((n + 1, n + 1))
    for m in range(1, n + 1):
        root_tilde = math.exp(beta * a[0, m] - c[m - 1])
        out[0, m] = root_tilde * diag_b[m - 1]
        for h in range(1, n + 1):
            if h == m:
                continue
            a_tilde = math.exp(beta * a[h, m] - c[m - 1])
            out[h, m] = a_tilde * (diag_b[m - 1] - binv[m - 1, h - 1])
    return out


def count_arborescences(n_words: int) -> int:
    r"""Number of spanning arborescences rooted at ``0`` over ``n_words`` -- ``(n + 1)^(n - 1)``.

    Cayley's formula (spanning trees of the complete graph ``K_{n+1}``, each oriented away
    from the fixed root ``0``); this is the ``N`` in the ``log(N) / beta`` gap bound.
    """
    if n_words < 1:
        raise ValueError(f"n_words must be >= 1, got {n_words}")
    return int((n_words + 1) ** (n_words - 1))


# ---------------------------------------------------------------------------
# Chu-Liu/Edmonds maximum spanning arborescence (rooted at 0, multi-root)
# ---------------------------------------------------------------------------


def _find_cycle(parent: dict[int, int], nodes: set[int], root: int) -> list[int] | None:
    r"""Return one directed cycle in the ``parent`` (chosen-in-edge) map, or ``None``."""
    colour: dict[int, int] = {v: 0 for v in nodes}  # 0 white, 1 grey, 2 black
    for start in nodes:
        if colour[start] != 0:
            continue
        path: list[int] = []
        x = start
        while x != root and colour[x] == 0:
            colour[x] = 1
            path.append(x)
            x = parent[x]
        if x != root and colour[x] == 1:
            return path[path.index(x):]
        for y in path:
            colour[y] = 2
    return None


def _edmonds(nodes: set[int], root: int, weight: dict[tuple[int, int], float]) -> dict[int, int]:
    r"""Recursive Chu-Liu/Edmonds: max arborescence rooted at ``root`` as ``{node: parent}``."""
    parent: dict[int, int] = {}
    for v in nodes:
        if v == root:
            continue
        best_u, best_w = None, -math.inf
        for u in nodes:
            if u == v:
                continue
            w = weight.get((u, v))
            if w is not None and w > best_w:
                best_w, best_u = w, u
        if best_u is None:
            raise ValueError(f"node {v} has no incoming edge")
        parent[v] = best_u
    cycle = _find_cycle(parent, nodes, root)
    if cycle is None:
        return parent
    cyc = set(cycle)
    supernode = max(nodes) + 1
    in_cost = {v: weight[(parent[v], v)] for v in cycle}
    new_nodes = (nodes - cyc) | {supernode}
    new_weight: dict[tuple[int, int], float] = {}
    origin: dict[tuple[int, int], tuple[int, int]] = {}
    for (u, v), w in weight.items():
        if u in cyc and v in cyc:
            continue
        if v in cyc:  # edge entering the cycle: discount the arc it would replace
            key = (u, supernode)
            adj = w - in_cost[v]
            if key not in new_weight or adj > new_weight[key]:
                new_weight[key] = adj
                origin[key] = (u, v)
        elif u in cyc:  # edge leaving the cycle
            key = (supernode, v)
            if key not in new_weight or w > new_weight[key]:
                new_weight[key] = w
                origin[key] = (u, v)
        else:
            key = (u, v)
            if key not in new_weight or w > new_weight[key]:
                new_weight[key] = w
                origin[key] = (u, v)
    sub = _edmonds(new_nodes, root, new_weight)
    result: dict[int, int] = {}
    entered_at: int | None = None
    for v, u in sub.items():
        if v == supernode:
            ou, ov = origin[(u, supernode)]
            result[ov] = ou
            entered_at = ov
        elif u == supernode:
            ou, ov = origin[(supernode, v)]
            result[v] = ou
        else:
            result[v] = u
    for v in cycle:
        if v != entered_at:
            result[v] = parent[v]
    return result


def max_arborescence(arc: FloatArray) -> tuple[float, dict[int, int]]:
    r"""Chu-Liu/Edmonds maximum spanning arborescence rooted at ``0`` (multi-root).

    Returns ``(score, heads)`` with ``heads[m]`` the chosen head of word ``m`` and ``score``
    the summed arc scores -- the ``beta -> inf`` limit of :func:`matrix_tree_partition`.
    """
    a = np.asarray(arc, dtype=float)
    n = a.shape[0] - 1
    if n < 1:
        raise ValueError(f"need at least one word (arc >= 2x2), got n={n}")
    nodes = set(range(n + 1))
    weight = {(h, m): float(a[h, m]) for m in range(1, n + 1) for h in range(n + 1) if h != m}
    heads = _edmonds(nodes, 0, weight)
    score = float(sum(a[heads[m], m] for m in range(1, n + 1)))
    return score, {m: heads[m] for m in range(1, n + 1)}


def hard_matrix_tree(arc: FloatArray) -> float:
    r"""Score of the maximum spanning arborescence (Chu-Liu/Edmonds), independent of the det."""
    return max_arborescence(arc)[0]


# ---------------------------------------------------------------------------
# Brute-force ground truth: enumerate head maps, keep valid arborescences
# ---------------------------------------------------------------------------


def _reaches_root(heads: dict[int, int], n: int) -> bool:
    for m in range(1, n + 1):
        seen: set[int] = set()
        cur = m
        while cur != 0:
            if cur in seen:
                return False
            seen.add(cur)
            cur = heads[cur]
    return True


def iter_arborescences(n_words: int) -> Iterator[dict[int, int]]:
    r"""Yield every valid spanning arborescence head map ``{modifier: head}`` (brute force)."""
    if n_words < 1:
        raise ValueError(f"n_words must be >= 1, got {n_words}")
    for assignment in itertools.product(range(n_words + 1), repeat=n_words):
        heads = {m: assignment[m - 1] for m in range(1, n_words + 1)}
        if any(heads[m] == m for m in range(1, n_words + 1)):
            continue
        if _reaches_root(heads, n_words):
            yield heads


def brute_force_arborescence(arc: FloatArray, beta: float | None = None) -> float:
    r"""Flat oracle over *enumerated* arborescences: ``max`` (``beta is None``) or ``lse_beta``.

    With ``beta`` set this equals :func:`matrix_tree_partition` to machine precision (the
    determinant is the exact same sum); with ``beta is None`` it equals
    :func:`hard_matrix_tree`.
    """
    a = np.asarray(arc, dtype=float)
    n = a.shape[0] - 1
    scores = np.array(
        [float(sum(a[heads[m], m] for m in range(1, n + 1))) for heads in iter_arborescences(n)]
    )
    if scores.size == 0:
        raise ValueError("no arborescences (n must be >= 1)")
    if beta is None:
        return float(np.max(scores))
    if beta <= 0.0:
        raise ValueError(f"beta must be > 0, got {beta}")
    hi = float(np.max(scores))
    return hi + math.log(float(np.sum(np.exp(beta * (scores - hi))))) / beta


__all__ = [
    "brute_force_arborescence",
    "count_arborescences",
    "hard_matrix_tree",
    "iter_arborescences",
    "matrix_tree_marginals",
    "matrix_tree_partition",
    "max_arborescence",
]
