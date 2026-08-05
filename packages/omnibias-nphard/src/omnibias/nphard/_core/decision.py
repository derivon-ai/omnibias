# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
r"""Backend-agnostic (numpy) decision-focused ("predict-then-optimize") helpers.

Scored on the **generalized assignment** family, whose objective ``sum_{a,t} c_at x_at``
is *linear* in the assignment (so the SPO+ subgradient is well defined). A model predicts
the cost matrix ``c`` from features; the decision quality is the **normalized regret** --
the true cost of the assignment decoded from *predicted* costs, minus the true optimum,
normalized by the optimum. Both helpers are backend-neutral (they consume plain cost /
resource / capacity arrays) and are re-exported alongside the differentiable
``decision_cost`` in the ``torch`` / ``jax`` twins so the metric and the ``ours`` loss
share one implementation.

* :func:`normalized_regret` -- the decision-quality metric (uses the exact oracle).
* :func:`spo_plus_gradient` -- the SPO+ (Elmachtoub & Grigas, 2022) subgradient of the
  smart predict-then-optimize surrogate, using the exact oracle (a decision baseline).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from omnibias.discrete import mean_normalized_regret, spo_plus_subgradient
from omnibias.nphard._core.gap import gap, gap_brute_force, gap_decode

FloatArray = NDArray[np.float64]


def _as_batch(cost: object) -> FloatArray:
    arr = np.asarray(cost, dtype=float)
    return arr[None] if arr.ndim == 2 else arr


def _oracle_indicator(cost: FloatArray, resource: FloatArray, capacity: FloatArray) -> FloatArray:
    """``(A, T)`` indicator of the min-cost capacity-feasible assignment (exact oracle)."""
    problem = gap(cost, resource, capacity)
    x, _ = gap_brute_force(problem)
    a_t = problem.n_agents * problem.n_tasks
    return np.asarray(x[:a_t], dtype=float).reshape(problem.n_agents, problem.n_tasks)


def _decoded_assignment(cost: FloatArray, resource: FloatArray, capacity: FloatArray) -> list[int]:
    """Task->agent assignment decoded from a predicted cost (greedy + capacity repair)."""
    problem = gap(cost, resource, capacity)
    x, _ = gap_decode(problem)
    a_t = problem.n_agents * problem.n_tasks
    heat = np.asarray(x[:a_t], dtype=float).reshape(problem.n_agents, problem.n_tasks)
    return [int(np.argmax(heat[:, t])) for t in range(problem.n_tasks)]


def normalized_regret(
    cost_pred: object,
    cost_true: object,
    resource: object,
    capacity: object,
    opt: object | None = None,
) -> float:
    r"""Mean normalized GAP decision regret over a batch of predicted cost matrices.

    For each instance, decode an assignment from the **predicted** cost, evaluate it under
    the **true** cost, and subtract the true optimum; normalize the mean excess by the mean
    optimum. ``0`` is oracle-optimal decisions. ``cost_pred`` / ``cost_true`` are ``(A, T)``
    or ``(B, A, T)``; ``resource`` / ``capacity`` are the shared (known) GAP data; ``opt``
    (the true optima, ``(B,)``) is computed exactly by :func:`gap_brute_force` if omitted.
    """
    pred = _as_batch(cost_pred)
    true = _as_batch(cost_true)
    if pred.shape != true.shape:
        raise ValueError("cost_pred and cost_true must have the same shape")
    res = np.asarray(resource, dtype=float)
    cap = np.asarray(capacity, dtype=float)
    opt_arr = (
        np.asarray(
            [gap_brute_force(gap(true[b], res, cap))[1] for b in range(true.shape[0])],
            dtype=float,
        )
        if opt is None
        else np.asarray(opt, dtype=float)
    )

    def decision_cost(pred_b: FloatArray, true_b: FloatArray) -> float:
        assignment = _decoded_assignment(pred_b, res, cap)
        return float(gap(true_b, res, cap).assignment_cost(assignment))

    return float(mean_normalized_regret(pred, true, opt_arr, decision_cost))


def spo_plus_gradient(
    cost_pred: object, cost_true: object, resource: object, capacity: object
) -> FloatArray:
    r"""SPO+ subgradient w.r.t. predicted GAP costs (uses the exact oracle).

    The SPO+ surrogate (Elmachtoub & Grigas, 2022) has subgradient
    ``2 (x^*(c_true) - x^*(2 c_pred - c_true))`` in the assignment variables, where ``x^*``
    is an optimal (capacity-feasible) assignment indicator. Returns a ``(B, A, T)`` (or
    ``(A, T)``) gradient -- a decision-focused baseline for the *linear-objective* GAP.
    """
    pred = _as_batch(cost_pred)
    true = _as_batch(cost_true)
    if pred.shape != true.shape:
        raise ValueError("cost_pred and cost_true must have the same shape")
    res = np.asarray(resource, dtype=float)
    cap = np.asarray(capacity, dtype=float)
    grad = spo_plus_subgradient(pred, true, lambda c: _oracle_indicator(c, res, cap))
    squeezed: FloatArray = grad[0] if np.asarray(cost_pred).ndim == 2 else grad
    return squeezed


__all__ = ["normalized_regret", "spo_plus_gradient"]
