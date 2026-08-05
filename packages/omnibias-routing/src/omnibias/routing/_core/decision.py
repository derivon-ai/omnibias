# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
r"""Backend-agnostic (numpy) decision-focused routing helpers.

Predict-then-optimize routing has unknown per-arc costs predicted from features;
the decision quality is scored by **normalized regret** -- the true cost of the
tour decoded from *predicted* costs, minus the true optimum, normalized by the
optimum. These helpers are backend-neutral (they consume plain cost matrices) and
are re-exported by both :mod:`omnibias.routing.jax.decision_focused` and
:mod:`omnibias.routing.torch.decision_focused` so the differentiable ``ours`` loss
and the metrics share one implementation.

* :func:`edge_matrix` -- arc-incidence matrix of a tour.
* :func:`optimal_tour_costs` -- exact per-instance optima (Held-Karp DP, small ``n``).
* :func:`normalized_regret` -- the decision-quality metric.
* :func:`spo_plus_gradient` -- the SPO+ (Elmachtoub & Grigas, 2022) subgradient of
  the smart "predict-then-optimize" surrogate, using the exact oracle (a baseline).
"""

from __future__ import annotations

import numpy as np
from omnibias.discrete import mean_normalized_regret, spo_plus_subgradient
from omnibias.routing._core.decode import decode_tour, held_karp_dp, tour_cost


def edge_matrix(tour: tuple[int, ...] | list[int], n: int) -> np.ndarray:
    """``(n, n)`` arc-incidence matrix: 1 on each traversed directed arc of ``tour``."""
    mat = np.zeros((n, n))
    m = len(tour)
    for i in range(m):
        mat[tour[i], tour[(i + 1) % m]] = 1.0
    return mat


def optimal_tour_costs(costs: np.ndarray) -> np.ndarray:
    """Exact optimal tour cost of every instance in a ``(B, n, n)`` batch (small ``n``)."""
    arr = np.asarray(costs, dtype=float)
    return np.array([held_karp_dp(arr[b])[1] for b in range(arr.shape[0])])


def normalized_regret(
    pred_costs: np.ndarray,
    true_costs: np.ndarray,
    opt: np.ndarray | None = None,
    *,
    n_starts: int = 5,
) -> float:
    r"""Mean normalized decision regret over a batch of instances.

    For each instance, decode a tour from the **predicted** cost matrix, evaluate it
    under the **true** cost matrix, and subtract the true optimum; normalize the mean
    excess by the mean optimum. ``0`` is oracle-optimal decisions. Both arguments are
    ``(B, n, n)``; ``opt`` (the true optima, ``(B,)``) is computed exactly if omitted.
    """
    pred = np.asarray(pred_costs, dtype=float)
    true = np.asarray(true_costs, dtype=float)
    if pred.ndim != 3 or true.shape != pred.shape:
        raise ValueError("pred_costs and true_costs must both be (B, n, n) of equal shape")
    opt_arr = optimal_tour_costs(true) if opt is None else np.asarray(opt, dtype=float)

    def decision_cost(pred_b: np.ndarray, true_b: np.ndarray) -> float:
        tour, _ = decode_tour(pred_b, n_starts=n_starts)
        return float(tour_cost(tour, true_b))

    return float(mean_normalized_regret(pred, true, opt_arr, decision_cost))


def spo_plus_gradient(pred_costs: np.ndarray, true_costs: np.ndarray) -> np.ndarray:
    r"""SPO+ subgradient w.r.t. predicted arc costs (uses the exact oracle).

    The SPO+ surrogate (Elmachtoub & Grigas, 2022) has subgradient
    ``2 (x^*(c_true) - x^*(2 c_pred - c_true))`` in the arc-use variables, where
    ``x^*`` is an optimal tour's incidence matrix. Returns a ``(B, n, n)`` gradient
    for a ``(B, n, n)`` predicted / true cost batch -- a decision-focused baseline.
    """
    pred = np.asarray(pred_costs, dtype=float)
    true = np.asarray(true_costs, dtype=float)
    if pred.ndim != 3 or true.shape != pred.shape:
        raise ValueError("pred_costs and true_costs must both be (B, n, n) of equal shape")
    n = pred.shape[1]
    grad: np.ndarray = spo_plus_subgradient(pred, true, lambda c: edge_matrix(held_karp_dp(c)[0], n))
    return grad


__all__ = [
    "edge_matrix",
    "normalized_regret",
    "optimal_tour_costs",
    "spo_plus_gradient",
]
