# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
r"""Backend-agnostic (numpy) tour decoder, local search, and exact oracle.

The decoder rounds a (predicted or relaxed) directed cost / heat matrix into a
**valid tour** -- nearest-neighbour construction from several starts, refined by
2-opt segment reversal and or-opt single-city relocation (both re-evaluate the
full asymmetric tour cost, so they are correct for directed / asymmetric TSP).

:func:`held_karp_dp` is the exact ``O(2^n n^2)`` Held-Karp bitmask dynamic program
-- the ground-truth optimum used to self-check the certificate sandwich and to
score decision-focused regret on small instances (keep ``n <= ~16``).
"""

from __future__ import annotations

import numpy as np

_MAX_EXACT_N = 18


def is_valid_tour(tour: tuple[int, ...] | list[int], n: int) -> bool:
    """Whether ``tour`` is a permutation of ``range(n)`` (a valid Hamiltonian cycle)."""
    return len(tour) == n and sorted(tour) == list(range(n))


def tour_cost(tour: tuple[int, ...] | list[int], cost: np.ndarray) -> float:
    """Cost of the cyclic tour under the directed cost matrix ``cost``."""
    c = np.asarray(cost, dtype=float)
    n = len(tour)
    return float(sum(c[tour[i], tour[(i + 1) % n]] for i in range(n)))


def nearest_neighbor(cost: np.ndarray, start: int = 0) -> list[int]:
    """Greedy nearest-neighbour tour construction from ``start``."""
    c = np.asarray(cost, dtype=float)
    n = c.shape[0]
    unvisited = set(range(n))
    unvisited.discard(start)
    tour = [start]
    cur = start
    while unvisited:
        nxt = min(unvisited, key=lambda k: c[cur, k])
        tour.append(nxt)
        unvisited.discard(nxt)
        cur = nxt
    return tour


def two_opt(tour: list[int], cost: np.ndarray) -> tuple[list[int], float]:
    """2-opt + or-opt local search to a local optimum (asymmetric-safe recompute)."""
    c = np.asarray(cost, dtype=float)
    best = list(tour)
    best_c = tour_cost(best, c)
    improved = True
    while improved:
        improved = False
        n = len(best)
        for i in range(n - 1):  # 2-opt: reverse segment [i, k]
            for k in range(i + 1, n):
                cand = best[:i] + best[i : k + 1][::-1] + best[k + 1 :]
                cand_c = tour_cost(cand, c)
                if cand_c < best_c - 1e-12:
                    best, best_c, improved = cand, cand_c, True
        for i in range(n):  # or-opt: relocate a single city
            for j in range(n):
                if i == j:
                    continue
                cand = best[:]
                city = cand.pop(i)
                cand.insert(j, city)
                cand_c = tour_cost(cand, c)
                if cand_c < best_c - 1e-12:
                    best, best_c, improved = cand, cand_c, True
    return best, best_c


def decode_tour(
    cost: np.ndarray, *, heat: np.ndarray | None = None, n_starts: int = 5
) -> tuple[tuple[int, ...], float]:
    r"""Round a directed cost matrix to a valid tour (NN + local search).

    If a fractional ``heat`` matrix (e.g. a relaxation's arc-use) is given, the
    greedy construction follows ``-heat`` (prefer high-use arcs); otherwise it
    follows ``cost`` directly. The best of ``n_starts`` starts is refined by
    :func:`two_opt`. Returns ``(tour, tour_cost)``.
    """
    c = np.asarray(cost, dtype=float)
    n = c.shape[0]
    guide = -np.asarray(heat, dtype=float) if heat is not None else c
    best_tour: list[int] | None = None
    best_cost = np.inf
    for s in range(min(n_starts, n)):
        refined, _ = two_opt(nearest_neighbor(guide, s), c)
        rc = tour_cost(refined, c)
        if rc < best_cost:
            best_tour, best_cost = refined, rc
    assert best_tour is not None
    return tuple(best_tour), float(best_cost)


def held_karp_dp(cost: np.ndarray) -> tuple[tuple[int, ...], float]:
    r"""Exact optimal directed tour + cost via Held-Karp bitmask DP (``n <= 18``)."""
    c = np.asarray(cost, dtype=float)
    n = c.shape[0]
    if n > _MAX_EXACT_N:
        raise ValueError(
            f"held_karp_dp is exponential (O(2^n n^2)); n={n} exceeds the {_MAX_EXACT_N} cap. "
            "Use decode_tour + certify_tour_gap for a certified heuristic instead."
        )
    size = 1 << n
    inf = 1e18
    dp = np.full((size, n), inf)
    par = np.full((size, n), -1, dtype=int)
    dp[1, 0] = 0.0
    all_k = np.arange(n)
    for mask in range(size):
        if not (mask & 1):
            continue
        row = dp[mask]
        js = np.where(row < inf)[0]
        if js.size == 0:
            continue
        free = all_k[((mask >> all_k) & 1) == 0]
        if free.size == 0:
            continue
        new_masks = mask | (1 << free)
        for j in js:
            nc = row[j] + c[j, free]
            better = nc < dp[new_masks, free]
            if bool(np.any(better)):
                fk = free[better]
                dp[new_masks[better], fk] = nc[better]
                par[new_masks[better], fk] = j
    full = size - 1
    best, best_j = inf, -1
    for j in range(1, n):
        total = dp[full, j] + c[j, 0]
        if total < best:
            best, best_j = total, j
    tour: list[int] = []
    mask, j = full, best_j
    while j != -1:
        tour.append(j)
        prev = int(par[mask, j])
        mask ^= 1 << j
        j = prev
    tour.reverse()
    return tuple(tour), float(best)


__all__ = [
    "decode_tour",
    "held_karp_dp",
    "is_valid_tour",
    "nearest_neighbor",
    "tour_cost",
    "two_opt",
]
