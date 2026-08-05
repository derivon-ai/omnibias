# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
r"""Backend-free scaffolding for the differentiable multilinear-extension twins.

The torch / jax twins of :class:`~omnibias.submodular.functions.BudgetAdditive` share one
piece of *constant* structure: the exact Poisson-binomial convolution over the distribution
of the modular sum ``T = sum_i a_i x_i``. Its **support** (the achievable subset sums) and
per-element transition indices depend only on the ground weights ``a`` -- not on the
selection probabilities ``p`` -- so they are enumerated once here in pure numpy and then
driven differentiably in each backend.

:func:`budget_multilinear_schedule` returns ``(support, src_for_dst, zero_index)`` where

* ``support[k]`` is the ``k``-th distinct achievable subset sum (forward-added, so the
  values are reproducible and match by exact float equality);
* ``src_for_dst[i, dst]`` is the support index ``src`` with ``support[src] + a_i ==
  support[dst]`` (the "element ``i`` chosen" parent) or ``-1`` when there is none;
* ``zero_index`` is the index of the empty sum ``0`` (the DP's initial mass).

The differentiable DP is then, per element ``i``: ``prob <- prob (1 - p_i) + gather(prob,
src_for_dst[i]) p_i`` (invalid parents contribute ``0``), and
``F(p) = sum_k prob[k] * min(support[k], budget)`` -- exact, and identical in both backends
because the schedule is shared. This is *constant-support*: differentiate through ``p`` (and
``budget`` via the ``min``); the support structure itself is fixed data.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


def budget_multilinear_schedule(ground: object) -> tuple[FloatArray, IntArray, int]:
    r"""Enumerate the budget-additive convolution support and transition indices from ``a``.

    Returns ``(support, src_for_dst, zero_index)`` (see the module docstring). Runs the
    exact subset-sum DP in ``O(n * |support|)`` with ``|support| <= 2^n`` distinct sums;
    every child sum ``s + a_i`` is built by forward addition, so later lookups match by
    exact float equality (no fragile subtraction).
    """
    a = np.asarray(ground, dtype=float).reshape(-1)
    n = int(a.shape[0])
    support: list[float] = [0.0]
    index: dict[float, int] = {0.0: 0}
    current: list[float] = [0.0]  # subset sums of {0..i-1}
    edges: list[list[tuple[int, int]]] = []
    for i in range(n):
        ai = float(a[i])
        edges_i: list[tuple[int, int]] = []
        next_current = list(current)
        for s in current:
            s2 = s + ai
            if s2 not in index:
                index[s2] = len(support)
                support.append(s2)
                next_current.append(s2)
            edges_i.append((index[s2], index[s]))
        edges.append(edges_i)
        current = next_current
    length = len(support)
    src_for_dst = np.full((n, length), -1, dtype=np.int64)
    for i in range(n):
        for dst, src in edges[i]:
            src_for_dst[i, dst] = src
    return np.asarray(support, dtype=float), src_for_dst, 0


__all__ = ["budget_multilinear_schedule"]
