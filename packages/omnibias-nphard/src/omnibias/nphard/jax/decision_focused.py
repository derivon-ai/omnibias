# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
r"""Decision-focused QAP (JAX): backprop the true QAP cost through the relaxation.

Bit-identical twin of :mod:`omnibias.nphard.torch.decision_focused`.
:func:`qap_decision_cost` builds the QUBO from a *predicted* flow ``F_pred`` (interaction
``kron(F_pred, D)`` + permutation penalty), relaxes it with
:func:`omnibias.qubo.jax.qubo_relaxation`, and scores the resulting soft assignment under
the **true** flow ``F_true`` (``x^T kron(F_true, D) x``). Because the relaxation is
unrolled and differentiable (``jax.grad`` / ``jit`` friendly), minimising this trains a
flow model *through* the QAP solver. The exact-oracle metrics
(:func:`normalized_regret`, :func:`spo_plus_gradient`) are the shared numpy helpers.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import Array
from omnibias.nphard._core.decision import normalized_regret, spo_plus_gradient
from omnibias.nphard._core.qap import permutation_penalty_arrays
from omnibias.qubo.jax import qubo_relaxation
from omnibias.qubo.problem import AnnealSchedule


def qap_decision_cost(
    flow_pred: object,
    distance: object,
    flow_true: object,
    *,
    penalty: float | None = None,
    schedule: AnnealSchedule | None = None,
) -> Array:
    r"""Differentiable true QAP cost of the decision relaxed from a *predicted* flow.

    ``flow_pred`` (differentiated into), ``distance`` and ``flow_true`` are ``(dim, dim)``.
    Builds ``Q(F_pred) = kron(F_pred, D) + lambda * P_onehot``, relaxes to a soft
    assignment, and returns its cost under ``F_true`` -- the ``ours`` training loss.
    """
    flow = jnp.asarray(flow_pred, dtype=jnp.float64)
    dist = jnp.asarray(distance, dtype=jnp.float64)
    flow_t = jnp.asarray(flow_true, dtype=jnp.float64)
    dim = int(flow.shape[0])
    penalty_val: float | Array
    if penalty is None:
        # jit-safe: keep as a traced scalar with the gradient stopped (we never
        # differentiate through the penalty *magnitude*, only through the interaction).
        penalty_val = (
            jax.lax.stop_gradient(
                jnp.sum(jnp.abs(flow)) * jnp.max(jnp.abs(dist))
                + jnp.max(jnp.abs(flow)) * jnp.sum(jnp.abs(dist))
            )
            + 1.0
        )
    else:
        penalty_val = penalty
    q_pen_np, c_pen_np, _ = permutation_penalty_arrays(dim)
    q_pen = jnp.asarray(q_pen_np, dtype=jnp.float64)
    c_pen = jnp.asarray(c_pen_np, dtype=jnp.float64)
    q = jnp.kron(flow, dist) + penalty_val * q_pen
    c = penalty_val * c_pen
    x = qubo_relaxation(q, c, schedule=schedule)
    interaction_true = jnp.kron(flow_t, dist)
    cost: Array = x @ (interaction_true @ x)
    return cost


__all__ = ["normalized_regret", "qap_decision_cost", "spo_plus_gradient"]
