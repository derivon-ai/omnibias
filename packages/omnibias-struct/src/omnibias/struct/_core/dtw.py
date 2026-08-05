# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Dynamic Time Warping lattice: hard DTW + brute-force oracles (pure numpy).

DTW aligns two sequences by a monotonic warping path through an ``(n, m)`` grid of local
costs ``D``. From cell ``(i, j)`` a path steps to ``(i+1, j)`` (insertion), ``(i, j+1)``
(deletion), or ``(i+1, j+1)`` (match); the path cost is the sum of ``D`` over the cells it
visits, and hard DTW is the minimum-cost path from ``(0, 0)`` to ``(n-1, m-1)``.

This is the ``beta -> inf`` limit that the differentiable soft-DTW anneals towards. The
soft value is a **softmin** over paths (``-lse_beta`` of the negated path costs), so it
sandwiches hard DTW from below by the closed-form gap ``log(N) / beta`` where ``N`` is the
number of monotonic paths (a Delannoy number) -- the ``min``-sense
:func:`omnibias.struct.certify_soft_dp`. :func:`brute_force_soft_dtw` enumerates every path
(the ground truth the recursive soft-DTW is pinned against on tiny grids).
"""

from __future__ import annotations

import math
from collections.abc import Iterator, Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class DTWLattice:
    r"""The ``(n_rows, n_cols)`` monotonic warping grid (steps: down, right, diagonal).

    Pure structure -- the differentiable soft DTW consumes a backend cost tensor alongside
    the lattice shape, exactly as :class:`~omnibias.struct.ChainTrellis` pairs with backend
    emissions. Intended for tiny grids (the brute-force oracle is exponential).
    """

    n_rows: int
    n_cols: int

    def __post_init__(self) -> None:
        if self.n_rows < 1 or self.n_cols < 1:
            raise ValueError(f"lattice must be at least 1x1, got {self.n_rows}x{self.n_cols}")

    def enumerate_paths(self) -> Iterator[tuple[tuple[int, int], ...]]:
        """Yield every monotonic ``(0,0) -> (n-1,m-1)`` path as a tuple of ``(i, j)`` cells."""
        target = (self.n_rows - 1, self.n_cols - 1)

        def dfs(cell: tuple[int, int], acc: tuple[tuple[int, int], ...]) -> Iterator[tuple[tuple[int, int], ...]]:
            if cell == target:
                yield acc
                return
            i, j = cell
            for ni, nj in ((i + 1, j), (i, j + 1), (i + 1, j + 1)):
                if ni < self.n_rows and nj < self.n_cols:
                    yield from dfs((ni, nj), (*acc, (ni, nj)))

        yield from dfs((0, 0), ((0, 0),))

    def path_cost(self, path: Sequence[tuple[int, int]], cost: FloatArray) -> float:
        """Sum of local costs ``D`` over the cells a path visits."""
        return float(sum(float(cost[int(i), int(j)]) for i, j in path))

    def count_paths(self) -> int:
        """Exact number of monotonic paths -- the Delannoy number ``D(n-1, m-1)``."""
        counts = np.zeros((self.n_rows, self.n_cols), dtype=object)
        counts[0, 0] = 1
        for i in range(self.n_rows):
            for j in range(self.n_cols):
                if i == 0 and j == 0:
                    continue
                total = 0
                if i > 0:
                    total += counts[i - 1, j]
                if j > 0:
                    total += counts[i, j - 1]
                if i > 0 and j > 0:
                    total += counts[i - 1, j - 1]
                counts[i, j] = total
        return int(counts[self.n_rows - 1, self.n_cols - 1])


def hard_dtw(cost: FloatArray) -> float:
    r"""Minimum-cost monotonic warping (classic DTW) of an ``(n, m)`` local-cost matrix."""
    d = np.asarray(cost, dtype=float)
    n, m = d.shape
    r = np.full((n, m), np.inf)
    r[0, 0] = d[0, 0]
    for i in range(n):
        for j in range(m):
            if i == 0 and j == 0:
                continue
            best = np.inf
            if i > 0:
                best = min(best, r[i - 1, j])
            if j > 0:
                best = min(best, r[i, j - 1])
            if i > 0 and j > 0:
                best = min(best, r[i - 1, j - 1])
            r[i, j] = d[i, j] + best
    return float(r[n - 1, m - 1])


def brute_force_dtw(cost: FloatArray) -> float:
    r"""Exact hard DTW by enumerating every monotonic path (oracle for tiny grids)."""
    d = np.asarray(cost, dtype=float)
    lattice = DTWLattice(*d.shape)
    return min(lattice.path_cost(p, d) for p in lattice.enumerate_paths())


def brute_force_soft_dtw(cost: FloatArray, beta: float) -> float:
    r"""Exact soft DTW ``-beta^-1 log sum_paths exp(-beta cost(path))`` (global softmin).

    The ground truth for the recursive soft-DTW: because ``lse_beta`` distributes over the
    additive path costs, the recursion equals this flat softmin over all monotonic paths.
    """
    if beta <= 0.0:
        raise ValueError(f"beta must be > 0, got {beta}")
    d = np.asarray(cost, dtype=float)
    lattice = DTWLattice(*d.shape)
    neg_costs = np.array([-lattice.path_cost(p, d) for p in lattice.enumerate_paths()])
    mx = float(np.max(neg_costs))
    lse = mx + math.log(float(np.sum(np.exp(beta * (neg_costs - mx))))) / beta
    return -lse


__all__ = [
    "DTWLattice",
    "brute_force_dtw",
    "brute_force_soft_dtw",
    "hard_dtw",
]
