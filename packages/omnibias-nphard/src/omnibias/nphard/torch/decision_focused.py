# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
r"""Decision-focused QAP (torch): backprop the true QAP cost through the relaxation.

:func:`qap_decision_cost` is the differentiable "smart predict-then-optimize" objective
for QAP: it builds the QUBO from a *predicted* flow ``F_pred`` (interaction
``kron(F_pred, D)`` + permutation penalty), relaxes it with
:func:`omnibias.qubo.torch.qubo_relaxation`, and scores the resulting soft assignment
under the **true** flow ``F_true`` (``x^T kron(F_true, D) x``). Because the relaxation is
unrolled and differentiable, minimising this trains a flow model *through* the QAP
solver. The exact-oracle metrics (:func:`normalized_regret`, :func:`spo_plus_gradient`)
are the shared numpy helpers, scored on the linear-objective GAP family.
"""

from __future__ import annotations

import torch
from omnibias.nphard._core.decision import normalized_regret, spo_plus_gradient
from omnibias.nphard._core.qap import permutation_penalty_arrays
from omnibias.qubo.problem import AnnealSchedule
from omnibias.qubo.torch import qubo_relaxation
from torch import Tensor


def qap_decision_cost(
    flow_pred: object,
    distance: object,
    flow_true: object,
    *,
    penalty: float | None = None,
    schedule: AnnealSchedule | None = None,
) -> Tensor:
    r"""Differentiable true QAP cost of the decision relaxed from a *predicted* flow.

    ``flow_pred`` (differentiated into), ``distance`` and ``flow_true`` are ``(dim, dim)``.
    Builds ``Q(F_pred) = kron(F_pred, D) + lambda * P_onehot``, relaxes to a soft
    assignment, and returns its cost under ``F_true`` -- the ``ours`` training loss.
    """
    flow = torch.as_tensor(flow_pred, dtype=torch.float64)
    dist = torch.as_tensor(distance, dtype=torch.float64)
    flow_t = torch.as_tensor(flow_true, dtype=torch.float64)
    dim = int(flow.shape[0])
    if penalty is None:
        # detached (like the jax twin's stop_gradient): we never differentiate through the
        # penalty *magnitude*, only through the interaction, and float() on a grad-tracking
        # tensor would both warn and silently drop the graph.
        fd, dd = flow.detach(), dist.detach()
        penalty = float(fd.abs().sum() * dd.abs().max() + fd.abs().max() * dd.abs().sum()) + 1.0
    q_pen_np, c_pen_np, _ = permutation_penalty_arrays(dim)
    q_pen = torch.as_tensor(q_pen_np, dtype=torch.float64)
    c_pen = torch.as_tensor(c_pen_np, dtype=torch.float64)
    q = torch.kron(flow, dist) + penalty * q_pen
    c = penalty * c_pen
    x = qubo_relaxation(q, c, schedule=schedule)
    interaction_true = torch.kron(flow_t, dist)
    cost: Tensor = x @ (interaction_true @ x)
    return cost


__all__ = ["normalized_regret", "qap_decision_cost", "spo_plus_gradient"]
