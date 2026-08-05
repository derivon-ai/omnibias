# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
r"""The classical greedy baseline, a feasibility-preserving polish, and the exact oracle.

* :func:`greedy_maximize` -- the textbook greedy algorithm (repeatedly add the feasible
  element of largest marginal gain). For a **cardinality** constraint greedy already
  carries the ``(1 - 1/e)`` guarantee; for a **general matroid** it only guarantees
  ``1/2`` -- which is exactly where continuous greedy (``1 - 1/e`` for every matroid)
  wins. It is the best-in-class baseline the differentiable method must beat or match.
* :func:`local_search` -- a feasibility-preserving add / swap polish for a rounded set.
* :func:`brute_force_max` -- the exact matroid-constrained optimum by enumerating all
  ``2^n`` subsets. **Exponential** (``O(2^n)``); the small-``n`` oracle that self-checks
  the certificate sandwich, never the solver.
"""

from __future__ import annotations

import heapq
from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray
from omnibias.submodular.functions import SubmodularFunction
from omnibias.submodular.matroid import Matroid, MatroidIntersection

FloatArray = NDArray[np.float64]

_MAX_EXACT_N = 20


def greedy_maximize(
    function: SubmodularFunction, matroid: Matroid
) -> tuple[tuple[int, ...], float]:
    r"""Greedy: repeatedly add the feasible element of largest positive marginal gain."""
    n = function.n
    x = np.zeros(n, dtype=float)
    while True:
        gains = function.marginal_gains(x)
        best_i, best_gain = -1, 1e-12
        for i in range(n):
            if x[i] == 1.0 or gains[i] <= best_gain:
                continue
            x[i] = 1.0
            feasible = matroid.is_independent(x)
            x[i] = 0.0
            if feasible:
                best_i, best_gain = i, float(gains[i])
        if best_i < 0:
            break
        x[best_i] = 1.0
    return tuple(int(v) for v in x), float(function.value(x))


def lazy_greedy(
    function: SubmodularFunction, matroid: Matroid
) -> tuple[tuple[int, ...], float]:
    r"""Minoux's accelerated ("lazy" / CELF) greedy -- same output as :func:`greedy_maximize`.

    Submodularity means a marginal gain can only *shrink* as the set grows, so a gain
    computed at an earlier (smaller) set is a valid upper bound now. A max-heap keyed by
    ``(-gain, index)`` lets us skip re-evaluating every element each round: pop the top,
    refresh its gain once, and accept it the moment its fresh gain dominates the runner-up.

    To stay *bit-identical* to the naive greedy we track, per heap entry, the accepted-set
    size at which its gain was last computed. A decision is only trusted when the runner-up
    is **fresh** (computed at the current set): if the runner-up is stale and within
    floating-point range, it is refreshed first, so the winner is chosen fresh-vs-fresh by
    exact gain and then lowest index -- exactly the naive greedy's scan order. (Rounding can
    make a stale bound a hair below the true current gain, so a bare stale comparison is not
    FP-robust; refreshing the runner-up removes that hazard.) Matroid independence is
    downward-closed, so an element that becomes infeasible stays infeasible and is dropped.
    """
    n = function.n
    x = np.zeros(n, dtype=float)
    cur = float(function.value(x))
    gains = function.marginal_gains(x)
    # (-gain, index, checked_size): checked_size is the accepted-set size at which the gain
    # was computed; entries with checked_size < size are stale and refreshed on pop.
    heap: list[tuple[float, int, int]] = [(-float(gains[i]), i, 0) for i in range(n)]
    heapq.heapify(heap)
    size = 0
    while heap:
        neg_g, i, checked = heapq.heappop(heap)
        x[i] = 1.0
        if not matroid.is_independent(x):
            x[i] = 0.0
            continue
        if checked != size:  # stale: refresh i's gain at the current set, then reinsert
            gi = float(function.value(x)) - cur
            x[i] = 0.0
            heapq.heappush(heap, (-gi, i, size))
            continue
        x[i] = 0.0
        gi = -neg_g
        if heap:
            neg_top, j, checked_top = heap[0]
            gtop = -neg_top
            if checked_top != size and gi <= gtop + 1e-9 * (1.0 + abs(gi)):
                # Runner-up j is stale and within FP range: refresh it and requeue i, so the
                # fresh-vs-fresh heap order (exact gain, then lowest index) makes the call.
                _, jj, _ = heapq.heapreplace(heap, (-gi, i, size))
                x[jj] = 1.0
                gj = float(function.value(x)) - cur
                x[jj] = 0.0
                heapq.heappush(heap, (-gj, jj, size))
                continue
        if gi <= 1e-12:
            break
        x[i] = 1.0
        cur += gi
        size += 1
    return tuple(int(v) for v in x), cur


def stochastic_greedy(
    function: SubmodularFunction,
    matroid: Matroid,
    *,
    epsilon: float = 0.05,
    seed: int = 0,
) -> tuple[tuple[int, ...], float]:
    r"""Mirzasoleiman et al. stochastic greedy -- ``(1 - 1/e - epsilon)`` in expectation.

    Each round samples ``s = ceil((n / k) * log(1 / epsilon))`` still-feasible elements and
    adds the sampled element of largest marginal gain (``k`` is the matroid rank; the RNG is
    seeded for determinism). Sampling from the feasible remainder guarantees progress and
    trades a controllable ``epsilon`` slack in the ratio for far fewer oracle calls than a
    full greedy sweep. The guarantee is in expectation over the sampling.
    """
    if not 0.0 < epsilon < 1.0:
        raise ValueError("epsilon must lie in the open interval (0, 1)")
    n = function.n
    k = matroid.rank()
    x = np.zeros(n, dtype=float)
    cur = float(function.value(x))
    if k <= 0:
        return tuple(int(v) for v in x), cur
    rng = np.random.default_rng(seed)
    sample_size = int(np.ceil((n / k) * np.log(1.0 / epsilon)))
    for _ in range(k):
        feasible: list[int] = []
        for i in range(n):
            if x[i] == 1.0:
                continue
            x[i] = 1.0
            if matroid.is_independent(x):
                feasible.append(i)
            x[i] = 0.0
        if not feasible:
            break
        s = min(sample_size, len(feasible))
        sample = rng.choice(np.asarray(feasible, dtype=np.int64), size=s, replace=False)
        best_i, best_gain = -1, 1e-12
        for raw in sample:
            i = int(raw)
            x[i] = 1.0
            gain = float(function.value(x)) - cur
            x[i] = 0.0
            if gain > best_gain:
                best_i, best_gain = i, gain
        if best_i < 0:
            break
        x[best_i] = 1.0
        cur += best_gain
    return tuple(int(v) for v in x), cur


def local_search(
    function: SubmodularFunction, matroid: Matroid, x0: object, *, tol: float = 1e-12
) -> tuple[tuple[int, ...], float]:
    r"""Polish a feasible ``0/1`` point by feasibility-preserving adds and 1-swaps."""
    x = np.asarray(x0, dtype=float).reshape(-1).copy()
    base = float(function.value(x))
    improved = True
    while improved:
        improved = False
        for i in range(function.n):  # feasible add
            if x[i] == 1.0:
                continue
            x[i] = 1.0
            if matroid.is_independent(x):
                val = float(function.value(x))
                if val > base + tol:
                    base = val
                    improved = True
                    break
            x[i] = 0.0
        if improved:
            continue
        inside = [i for i in range(function.n) if x[i] == 1.0]
        outside = [i for i in range(function.n) if x[i] == 0.0]
        for j in inside:  # feasible 1-swap
            for i in outside:
                x[j], x[i] = 0.0, 1.0
                if matroid.is_independent(x):
                    val = float(function.value(x))
                    if val > base + tol:
                        base = val
                        improved = True
                        break
                x[j], x[i] = 1.0, 0.0
            if improved:
                break
    return tuple(int(v) for v in x), base


def brute_force_max(
    function: SubmodularFunction, matroid: Matroid, *, max_n: int = _MAX_EXACT_N
) -> tuple[tuple[int, ...], float]:
    r"""Exact matroid-constrained maximum by enumerating all ``2^n`` subsets.

    Exponential (``O(2^n)`` value evaluations); intended as the small-``n`` oracle that
    self-checks the certificate sandwich. Raises :class:`ValueError` for ``n > max_n``.
    """
    n = function.n
    if n > max_n:
        raise ValueError(
            f"brute_force_max is exponential (O(2^n)); n={n} exceeds the {max_n} cap. "
            "Use maximize + certify_submodular_gap for a certified heuristic instead."
        )
    idx = np.arange(1 << n, dtype=np.int64)
    bits = ((idx[:, None] >> np.arange(n)[None, :]) & 1).astype(float)
    try:  # fast vectorized feasibility for partition-structured matroids
        groups = matroid.groups()
        caps = matroid.caps()
        feasible = np.ones(1 << n, dtype=bool)
        for g, c in zip(groups, caps, strict=True):
            feasible &= bits[:, g].sum(axis=1) <= float(c) + 1e-9
    except NotImplementedError:  # general matroid: fall back to the independence oracle
        feasible = np.fromiter(
            (matroid.is_independent(bits[m]) for m in range(1 << n)), dtype=bool, count=1 << n
        )
    vals = np.asarray(function.value(bits), dtype=float)
    vals = np.where(feasible, vals, -np.inf)
    best = int(np.argmax(vals))
    return tuple(int(v) for v in bits[best]), float(vals[best])


def p_matroid_greedy(
    function: SubmodularFunction, matroids: MatroidIntersection | Sequence[Matroid]
) -> tuple[tuple[int, ...], float]:
    r"""Greedy over a ``p``-matroid intersection -- the a-priori ``1/(p+1)`` guarantee.

    For monotone submodular ``f`` maximized over the common independent sets of ``p``
    matroids, the greedy that repeatedly adds the largest-gain element keeping the set
    independent in **all** ``p`` matroids achieves ``1/(p+1) OPT``. ``matroids`` is either a
    :class:`~omnibias.submodular.matroid.MatroidIntersection` or a sequence of matroids (wrapped
    into one). Reduces to :func:`greedy_maximize` (``1/2`` for a single matroid, ``p = 1``).
    """
    intersection = (
        matroids
        if isinstance(matroids, MatroidIntersection)
        else MatroidIntersection(list(matroids))
    )
    return greedy_maximize(function, intersection)


__all__ = [
    "brute_force_max",
    "greedy_maximize",
    "lazy_greedy",
    "local_search",
    "p_matroid_greedy",
    "stochastic_greedy",
]
