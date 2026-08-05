# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
"""Decision-focused learning: GAP regret / SPO+, and QAP train-through improvement."""

from __future__ import annotations

import numpy as np
import pytest
from omnibias.nphard import normalized_regret, spo_plus_gradient


def _gap_data(seed: int, agents: int = 3, tasks: int = 4, cap: int = 8) -> tuple:
    rng = np.random.default_rng(seed)
    cost_true = rng.integers(1, 9, size=(agents, tasks)).astype(float)
    resource = rng.integers(1, 4, size=(agents, tasks)).astype(float)
    capacity = np.full(agents, float(cap))
    return cost_true, resource, capacity


def test_normalized_regret_is_small_at_truth_and_larger_for_noise() -> None:
    """Predicting the true cost -> near-oracle decisions; noisy costs -> larger regret."""
    cost_true, resource, capacity = _gap_data(0)
    r_true = normalized_regret(cost_true, cost_true, resource, capacity)
    rng = np.random.default_rng(9)
    r_noise = np.mean([
        normalized_regret(
            cost_true + 4.0 * rng.standard_normal(cost_true.shape), cost_true, resource, capacity
        )
        for _ in range(6)
    ])
    assert r_true >= 0.0
    assert r_true <= 0.15  # decoding from the true cost is near-optimal
    assert r_true <= r_noise + 1e-9  # truth is at least as good as noisy predictions


def test_spo_plus_gradient_shape_and_vanishes_at_truth() -> None:
    """The SPO+ subgradient is (A, T), finite, and zero when pred == true."""
    cost_true, resource, capacity = _gap_data(1)
    g = spo_plus_gradient(cost_true, cost_true, resource, capacity)
    assert g.shape == cost_true.shape
    assert np.all(np.isfinite(g))
    assert np.allclose(g, 0.0)  # pred == true -> the two oracle assignments coincide


def test_spo_plus_gradient_batches() -> None:
    cost_true, resource, capacity = _gap_data(2)
    batch = np.stack([cost_true, cost_true + 1.0])
    g = spo_plus_gradient(batch, batch, resource, capacity)
    assert g.shape == batch.shape


def test_regret_rejects_mismatched_shapes() -> None:
    cost_true, resource, capacity = _gap_data(0)
    with pytest.raises(ValueError, match="same shape"):
        normalized_regret(cost_true, cost_true[:, :-1], resource, capacity)


def test_train_through_the_qap_relaxation_lowers_the_decoded_decision() -> None:
    """Backprop through the unrolled QAP relaxation strictly lowers the decoded decision.

    Deterministic demonstrator (seed 0): an uninformed predicted flow (theta = 0) decodes
    a poor permutation; normalized-gradient descent on the differentiable decision cost
    -- gradients flowing *through* the annealed relaxation -- recovers the optimal one.
    Uses the Hungarian-only ``qap_round`` (no local search, so the raw heatmap decision's
    improvement is visible; a 2-opt decoder would mask it)."""
    pytest.importorskip("jax")
    import jax
    import jax.numpy as jnp

    jax.config.update("jax_enable_x64", True)
    from omnibias.nphard import brute_force_min, qap
    from omnibias.nphard._core.qap import qap_round
    from omnibias.nphard.jax import qap_decision_cost
    from omnibias.nphard.jax import relax as relax_j
    from omnibias.qubo.problem import AnnealSchedule

    rng = np.random.default_rng(0)
    dim = 4
    dist = rng.integers(0, 9, size=(dim, dim)).astype(float)
    dist = (dist + dist.T) / 2.0
    np.fill_diagonal(dist, 0.0)
    flow_true = rng.integers(0, 9, size=(dim, dim)).astype(float)
    flow_true = (flow_true + flow_true.T) / 2.0
    np.fill_diagonal(flow_true, 0.0)

    train_sched = AnnealSchedule(beta0=0.4, beta_growth=1.3, stages=6, steps=40)
    eval_sched = AnnealSchedule()  # decisive schedule for the eval-time decision

    def decoded_true_cost(flow_pred: np.ndarray) -> float:
        heat = np.asarray(relax_j(qap(flow_pred, dist), schedule=eval_sched)).reshape(dim, dim)
        x = qap_round(heat, dim)
        return float(qap(flow_true, dist).objective(x))

    def loss(theta: jnp.ndarray) -> jnp.ndarray:
        return qap_decision_cost(theta, dist, flow_true, schedule=train_sched)

    value_and_grad = jax.jit(jax.value_and_grad(loss))
    theta = jnp.zeros((dim, dim))  # uninformed predicted flow
    loss0, _ = value_and_grad(theta)
    for _ in range(120):
        _, grad = value_and_grad(theta)
        theta = theta - 0.5 * grad / (jnp.linalg.norm(grad) + 1e-12)  # normalized step
    loss1, grad1 = value_and_grad(theta)

    e_untrained = decoded_true_cost(np.zeros((dim, dim)))
    e_trained = decoded_true_cost(np.asarray(theta))
    _, e_opt = brute_force_min(qap(flow_true, dist))

    assert bool(jnp.all(jnp.isfinite(grad1)))
    assert float(loss1) < float(loss0) - 1e-9  # training lowers the differentiable loss
    assert e_trained < e_untrained - 1e-6  # and strictly improves the decoded decision
    # training closes almost all of the optimality gap (here ~99%: 230.5 -> 185.5, opt 185)
    assert (e_trained - e_opt) < 0.1 * (e_untrained - e_opt)
