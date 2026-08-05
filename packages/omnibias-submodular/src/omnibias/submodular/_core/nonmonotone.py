# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
r"""Non-monotone submodular maximization: double greedy + measured continuous greedy.

The monotone maximizers (greedy, continuous greedy) can be arbitrarily bad on a
*non-monotone* submodular ``f`` -- a graph cut is the canonical example, where the full set
scores ``0``. Two classical algorithms carry an a-priori guarantee **without** monotonicity:

* :func:`double_greedy` (Buchbinder-Feldman-Naor-Schwartz) -- **unconstrained**. One pass
  maintains a growing set ``X`` (from ``empty``) and a shrinking set ``Y`` (from ``V``); for
  each element it compares the add-to-``X`` marginal ``a`` and the drop-from-``Y`` marginal
  ``b`` (submodularity gives ``a + b >= 0``). The *deterministic* rule (take the larger)
  is ``1/3``; the *randomized* rule (add with probability ``a_+ / (a_+ + b_+)``) is ``1/2``.
* :func:`measured_continuous_greedy` (Feldman-Naor-Schwartz) -- **matroid-constrained**,
  ``1/e``. Frank-Wolfe on the multilinear extension with the *measured* update
  ``p += (1/T) y (1 - p)`` (``y`` the positive-gradient matroid basis -- negative-gradient
  coordinates are never added), whose ``(1 - p)`` damping stops ``p`` from saturating and is
  exactly what buys ``1/e`` on non-monotone ``f``. The fractional point is pipage-rounded.

Plus :func:`nonmonotone_upper_bound`, the sound singleton-marginal bound
``f(O) <= f(empty) + sum_i [f({i}) - f(empty)]_+`` (submodularity alone, no monotonicity),
which :func:`~omnibias.submodular.certify_nonmonotone_gap` turns into the certified sandwich.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from omnibias.submodular._core.rounding import pipage_round
from omnibias.submodular.functions import SubmodularFunction
from omnibias.submodular.matroid import Matroid
from omnibias.submodular.problem import ContinuousGreedySchedule

FloatArray = NDArray[np.float64]


def double_greedy(
    function: SubmodularFunction, *, randomized: bool = True, seed: int = 0
) -> tuple[tuple[int, ...], float]:
    r"""Buchbinder et al. double greedy for **unconstrained** non-monotone maximization.

    ``randomized=True`` is the ``1/2``-approximation (in expectation over the seeded RNG);
    ``randomized=False`` is the deterministic ``1/3``-approximation. One pass over the ground
    set; two value evaluations per element.
    """
    n = function.n
    x = np.zeros(n, dtype=float)  # grows from the empty set
    y = np.ones(n, dtype=float)  # shrinks from the full set
    f_x = float(function.value(x))
    f_y = float(function.value(y))
    rng = np.random.default_rng(seed)
    for i in range(n):
        x[i] = 1.0
        f_x_plus = float(function.value(x))
        x[i] = 0.0
        a = f_x_plus - f_x  # marginal of adding i to X
        y[i] = 0.0
        f_y_minus = float(function.value(y))
        y[i] = 1.0
        b = f_y_minus - f_y  # marginal of dropping i from Y
        if randomized:
            a_pos, b_pos = max(a, 0.0), max(b, 0.0)
            total = a_pos + b_pos
            take = True if total <= 0.0 else bool(rng.random() < a_pos / total)
        else:
            take = a >= b
        if take:  # keep i: X gains it, Y already has it
            x[i] = 1.0
            f_x = f_x_plus
        else:  # drop i: Y loses it, X never had it
            y[i] = 0.0
            f_y = f_y_minus
    return tuple(int(v) for v in x), float(function.value(x))


def measured_continuous_greedy(
    function: SubmodularFunction,
    matroid: Matroid,
    *,
    schedule: ContinuousGreedySchedule | None = None,
) -> tuple[tuple[int, ...], float]:
    r"""Feldman-Naor-Schwartz measured continuous greedy (matroid, non-monotone ``1/e``).

    Runs the measured Frank-Wolfe flow ``p += (1/T) y (1 - p)`` with the positive-gradient
    matroid basis ``y`` (so a coordinate with negative gradient is never added -- essential
    for non-monotone ``f``), then deterministically pipage-rounds ``p`` to a feasible integral
    set. Returns ``(selection, f(selection))``.
    """
    sched = schedule or ContinuousGreedySchedule()
    n = function.n
    p = np.zeros(n, dtype=float)
    inv = 1.0 / float(sched.steps)
    for _ in range(sched.steps):
        grad = function.multilinear_grad(p)
        y = matroid.max_weight_basis(grad)  # positive-gradient argmax (non-monotone-safe)
        p = p + inv * y * (1.0 - p)  # measured update: (1 - p) damping keeps p unsaturated
    rounded: tuple[tuple[int, ...], float] = pipage_round(function, matroid, p)
    return rounded


def nonmonotone_upper_bound(function: SubmodularFunction) -> float:
    r"""The sound singleton bound ``f(O) <= f(empty) + sum_i [f({i}) - f(empty)]_+``.

    By submodularity ``f(O) - f(empty) <= sum_{i in O} [f({i}) - f(empty)]``, and dropping
    the constraint to *all* positive singleton marginals only widens it -- a rigorous upper
    bound on the (constrained or unconstrained) optimum that needs no monotonicity.
    """
    empty = np.zeros(function.n, dtype=float)
    f_empty = float(function.value(empty))
    singleton_gains = np.asarray(function.marginal_gains(empty), dtype=float)
    return f_empty + float(np.sum(np.maximum(singleton_gains, 0.0)))


__all__ = [
    "double_greedy",
    "measured_continuous_greedy",
    "nonmonotone_upper_bound",
]
