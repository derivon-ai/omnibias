# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
r"""Decision-focused routing (JAX): backprop the true decision cost through the relaxation.

:func:`decision_cost` is the differentiable "smart predict-then-optimize" objective:
the mean **true** cost of the decision the relaxation makes from *predicted* arc
costs, ``mean_b <c_true^b, relax(c_pred^b)>``. Because the relaxation layer
(:mod:`omnibias.routing.jax.relaxation`) is differentiable, minimising this trains a
cost model *through* the optimizer -- lower regret than a two-stage MSE fit when the
model is misspecified. The exact-oracle metrics (:func:`normalized_regret`,
:func:`spo_plus_gradient`, :func:`optimal_tour_costs`) are the shared numpy helpers.
"""

from __future__ import annotations

import jax.numpy as jnp
from jax import Array
from omnibias.routing._core.decision import (
    edge_matrix,
    normalized_regret,
    optimal_tour_costs,
    spo_plus_gradient,
)
from omnibias.routing.jax.relaxation import (
    assignment_relaxation,
    flow_relaxation,
    held_karp_layer,
)
from omnibias.routing.problem import RelaxationSchedule

_LAYERS = {
    "assignment": assignment_relaxation,
    "flow": flow_relaxation,
    "held_karp": held_karp_layer,
}


def decision_cost(
    cost_pred: Array,
    cost_true: Array,
    *,
    kind: str = "assignment",
    schedule: RelaxationSchedule | None = None,
) -> Array:
    r"""Differentiable mean true cost of the relaxed decision (the ``ours`` loss).

    ``cost_pred`` / ``cost_true`` are ``(B, n, n)`` (or ``(n, n)``) arc-cost matrices.
    The relaxation of ``cost_pred`` is scored under ``cost_true``; gradients flow into
    ``cost_pred``. ``kind`` picks the relaxation strength (``"assignment"`` is the
    best-conditioned training layer; ``"flow"`` / ``"held_karp"`` are tighter).
    """
    if kind not in _LAYERS:
        raise ValueError(f"unknown relaxation kind {kind!r}; choose from {tuple(_LAYERS)}")
    x = _LAYERS[kind](cost_pred, schedule)
    ct = jnp.asarray(cost_true, dtype=jnp.float64)
    prod = jnp.sum(x * ct, axis=(-2, -1))
    return jnp.mean(prod)


__all__ = [
    "decision_cost",
    "edge_matrix",
    "normalized_regret",
    "optimal_tour_costs",
    "spo_plus_gradient",
]
