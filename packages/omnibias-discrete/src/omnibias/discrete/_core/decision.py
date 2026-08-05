# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
r"""Oracle-agnostic decision-focused ("predict-then-optimize") primitives.

Predict-then-optimize learning scores a cost-prediction model by the quality of the *decisions*
its predictions induce, not by prediction error. Two backend-neutral (numpy) primitives are
oracle-agnostic -- given an exact linear-optimization oracle ``x^*(c)`` for a specific
combinatorial family they reduce to the same arithmetic regardless of the family -- so they live
here once and every consumer binds its own exact oracle / decoder:

* :func:`spo_plus_subgradient` -- the SPO+ (Elmachtoub & Grigas, 2022) subgradient
  ``2 (x^*(c_true) - x^*(2 c_pred - c_true))`` of the smart predict-then-optimize surrogate;
* :func:`mean_normalized_regret` -- the decision-quality metric: the mean true-cost excess of the
  decision decoded from the *predicted* cost, normalized by the mean true optimum.

``omnibias-nphard`` (generalized / quadratic assignment) and ``omnibias-routing`` (TSP) are the
consumers; each keeps its own public wrapper, oracle and decoder and calls these.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


def spo_plus_subgradient(
    cost_pred: FloatArray,
    cost_true: FloatArray,
    oracle: Callable[[FloatArray], FloatArray],
) -> FloatArray:
    r"""SPO+ subgradient over a ``(B, ...)`` batch given an exact linear oracle ``x^*(c)``.

    ``cost_pred`` and ``cost_true`` are equal-shaped ``(B, ...)`` cost tensors; ``oracle`` maps a
    single instance's cost ``(...)`` to its optimal-decision indicator of the same shape. Returns
    the ``(B, ...)`` subgradient ``2 (x^*(c_true) - x^*(2 c_pred - c_true))``. The oracle's
    optimization direction (min- or max-cost) is the caller's choice; the SPO+ identity is
    unchanged either way.
    """
    grad = np.zeros_like(cost_pred)
    for b in range(cost_pred.shape[0]):
        x_true = oracle(cost_true[b])
        x_spo = oracle(2.0 * cost_pred[b] - cost_true[b])
        grad[b] = 2.0 * (x_true - x_spo)
    return grad


def mean_normalized_regret(
    cost_pred: FloatArray,
    cost_true: FloatArray,
    opt: FloatArray,
    decision_cost: Callable[[FloatArray, FloatArray], float],
) -> float:
    r"""Mean decision regret normalized by the mean true optimum.

    For each instance ``b`` of a ``(B, ...)`` batch, ``decision_cost(cost_pred[b], cost_true[b])``
    is the *true* cost of the decision decoded from the predicted cost; subtracting the true
    optimum ``opt[b]`` gives the per-instance regret. Returns ``mean(regret) / max(mean(opt), eps)``
    (``eps = 1e-12``), which is ``0`` exactly at oracle-optimal decisions.
    """
    regrets = [
        decision_cost(cost_pred[b], cost_true[b]) - opt[b] for b in range(cost_pred.shape[0])
    ]
    return float(np.mean(regrets) / max(float(np.mean(opt)), 1e-12))


__all__ = ["mean_normalized_regret", "spo_plus_subgradient"]
