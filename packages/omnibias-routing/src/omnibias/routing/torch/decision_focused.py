# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
r"""Decision-focused routing (torch): bit-identical twin of the JAX module.

See :mod:`omnibias.routing.jax.decision_focused` for the math. :func:`decision_cost`
is the differentiable mean true cost of the relaxed decision (backprop through the
relaxation via ``autograd``); the exact-oracle metrics are the shared numpy helpers.
"""

from __future__ import annotations

import torch
from omnibias.routing._core.decision import (
    edge_matrix,
    normalized_regret,
    optimal_tour_costs,
    spo_plus_gradient,
)
from omnibias.routing.problem import RelaxationSchedule
from omnibias.routing.torch.relaxation import (
    assignment_relaxation,
    flow_relaxation,
    held_karp_layer,
)
from torch import Tensor

_LAYERS = {
    "assignment": assignment_relaxation,
    "flow": flow_relaxation,
    "held_karp": held_karp_layer,
}


def decision_cost(
    cost_pred: Tensor,
    cost_true: Tensor,
    *,
    kind: str = "assignment",
    schedule: RelaxationSchedule | None = None,
) -> Tensor:
    r"""Differentiable mean true cost of the relaxed decision (the ``ours`` loss).

    ``cost_pred`` / ``cost_true`` are ``(B, n, n)`` (or ``(n, n)``) arc-cost matrices.
    The relaxation of ``cost_pred`` is scored under ``cost_true``; gradients flow into
    ``cost_pred``. ``kind`` picks the relaxation strength (``"assignment"`` is the
    best-conditioned training layer; ``"flow"`` / ``"held_karp"`` are tighter).
    """
    if kind not in _LAYERS:
        raise ValueError(f"unknown relaxation kind {kind!r}; choose from {tuple(_LAYERS)}")
    x = _LAYERS[kind](cost_pred, schedule)
    ct = torch.as_tensor(cost_true, dtype=torch.float64)
    prod = torch.sum(x * ct, dim=(-2, -1))
    return torch.mean(prod)


__all__ = [
    "decision_cost",
    "edge_matrix",
    "normalized_regret",
    "optimal_tour_costs",
    "spo_plus_gradient",
]
