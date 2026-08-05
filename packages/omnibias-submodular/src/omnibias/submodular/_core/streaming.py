# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
r"""One-pass streaming submodular maximization under a cardinality constraint.

:func:`sieve_streaming` is the Badanidiyuru-Mirzasoleiman-Karbasi-Krause (2014) sieve: a
single sweep over the ground set that keeps ``O(log k / epsilon)`` candidate solutions, one
per geometric guess of the optimum ``OPT in [m, k m]`` (``m`` the best singleton gain). For
each guess ``v`` an element is accepted only if its current marginal gain clears the
*uniform-share* threshold ``(v/2 - f(S_v)) / (k - |S_v|)`` -- so every accepted element pays
its way toward ``v/2``, and the best candidate is a ``(1/2 - epsilon)``-approximation to the
cardinality-constrained optimum. This is honest: like the rest of the package it is an
*approximation with an a-priori ratio*, not an exact NP solver, and it applies only to the
cardinality (uniform-matroid) constraint.

Pure numpy; the objective is consulted purely through :meth:`value` / :meth:`marginal_gains`,
so it works for greedy-path functions (:class:`~omnibias.submodular.functions.LogDeterminant`,
:class:`~omnibias.submodular.functions.Saturated`) with no multilinear extension.
"""

from __future__ import annotations

from collections.abc import Sequence
from math import ceil, floor, log

import numpy as np
from numpy.typing import NDArray
from omnibias.submodular.functions import SubmodularFunction

FloatArray = NDArray[np.float64]

_TOL = 1e-12

# The sieve's a-priori guarantee is 1/2 - epsilon for a cardinality constraint.
SIEVE_BASE_RATIO = 0.5


def _thresholds(m: float, k: int, epsilon: float) -> list[float]:
    r"""Geometric guesses ``(1 + epsilon)^j`` covering ``OPT in [m, k m]``.

    ``OPT >= m`` (best singleton) and ``OPT <= k m`` (submodularity: every marginal ``<= m``),
    so a ``(1 + epsilon)``-spaced net over ``[m, k m]`` contains a guess within ``(1 + epsilon)``
    of ``OPT``, which is what turns the per-guess ``v/2`` accounting into ``1/2 - epsilon``.
    """
    if m <= _TOL:
        return []
    base = log(1.0 + epsilon)
    lo = floor(log(m) / base)
    hi = ceil(log(k * m) / base)
    return [(1.0 + epsilon) ** j for j in range(lo, hi + 1)]


def sieve_streaming(
    function: SubmodularFunction,
    k: int,
    *,
    epsilon: float = 0.1,
    order: Sequence[int] | None = None,
) -> tuple[tuple[int, ...], float]:
    r"""Sieve-streaming: a one-pass ``(1/2 - epsilon)`` maximizer for a cardinality budget ``k``.

    Parameters
    ----------
    function:
        The monotone submodular objective (consulted only through :meth:`value` /
        :meth:`marginal_gains`, so greedy-path functions are fine).
    k:
        The cardinality budget ``|S| <= k`` (uniform-matroid constraint).
    epsilon:
        The accuracy slack in the ratio ``1/2 - epsilon`` (smaller ``epsilon`` -> more
        threshold guesses). Must lie in ``(0, 1)``.
    order:
        The stream order (a permutation of ``range(n)``); defaults to the natural order.
        The guarantee is order-independent, but the exact returned set can differ by order.

    Returns
    -------
    ``(selection, value)`` -- the best candidate set (a ``0/1`` tuple) and its ``f`` value.
    """
    if k <= 0:
        raise ValueError("k must be a positive cardinality budget")
    if not 0.0 < epsilon < 1.0:
        raise ValueError("epsilon must lie in the open interval (0, 1)")
    n = function.n
    if order is None:
        stream = list(range(n))
    else:
        stream = [int(i) for i in order]
        if sorted(stream) != list(range(n)):
            raise ValueError("order must be a permutation of range(n)")

    f_empty = float(function.value(np.zeros(n, dtype=float)))
    singles = np.asarray(function.marginal_gains(np.zeros(n, dtype=float)), dtype=float)
    m = float(np.max(singles)) if n else 0.0
    if m <= _TOL:  # every singleton gain is (nearly) zero -> nothing worth adding
        return tuple(0 for _ in range(n)), f_empty

    thresholds = _thresholds(m, k, epsilon)
    xs = [np.zeros(n, dtype=float) for _ in thresholds]
    curs = [f_empty for _ in thresholds]
    sizes = [0 for _ in thresholds]

    for e in stream:
        for t, v in enumerate(thresholds):
            size = sizes[t]
            if size >= k:
                continue
            x = xs[t]
            x[e] = 1.0
            new_val = float(function.value(x))
            x[e] = 0.0
            gain = new_val - curs[t]
            threshold = (0.5 * v - (curs[t] - f_empty)) / (k - size)
            if gain >= threshold - _TOL and gain > _TOL:
                x[e] = 1.0
                curs[t] = new_val
                sizes[t] = size + 1

    best = int(np.argmax(curs))
    return tuple(int(val) for val in xs[best]), float(curs[best])


__all__ = [
    "SIEVE_BASE_RATIO",
    "sieve_streaming",
]
