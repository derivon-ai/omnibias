# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
"""Policy trainers (BPTT / truncated-BPTT / zero-order / actor-adjoint / actor-adjoint-jet)."""

from __future__ import annotations

import numpy as np


def _layers_jax(key):
    import jax
    import jax.numpy as jnp

    k1, k2 = jax.random.split(key)
    w1 = 0.1 * jax.random.normal(k1, (6, 2))
    b1 = jnp.zeros(6)
    w2 = 0.1 * jax.random.normal(k2, (2, 6))
    b2 = jnp.zeros(2)
    return [(w1, b1, "tanh"), (w2, b2, None)]


def _layers_torch(seed=0):
    import torch

    g = torch.Generator().manual_seed(seed)
    w1 = 0.1 * torch.randn(6, 2, generator=g)
    b1 = torch.zeros(6)
    w2 = 0.1 * torch.randn(2, 6, generator=g)
    b2 = torch.zeros(2)
    return [(w1, b1, "tanh"), (w2, b2, None)]


def test_bptt_and_actor_adjoint_agree_at_full_horizon_jax():
    """Gate G1 at the trainer level: full-horizon BPTT step == our exact adjoint step."""
    import jax

    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp
    from omnibias.control.jax.envs import DoubleGyrePointMass
    from omnibias.control.jax.policy import actor_adjoint_step, bptt_step

    env = DoubleGyrePointMass()
    layers = _layers_jax(jax.random.PRNGKey(0))
    y0 = jnp.array([0.3, 0.4])
    r_bptt = bptt_step(env, layers, y0, horizon=4, lr=0.01)
    r_adj = actor_adjoint_step(env, layers, y0, horizon=4, lr=0.01)
    assert np.allclose(r_bptt.diagnostics["grad_theta"], r_adj.diagnostics["grad_theta"], atol=1e-8)
    assert r_bptt.total_cost == r_adj.total_cost


def test_bptt_and_actor_adjoint_agree_at_full_horizon_torch():
    import torch
    from omnibias.control.torch.envs import DoubleGyrePointMass
    from omnibias.control.torch.policy import actor_adjoint_step, bptt_step

    torch.set_default_dtype(torch.float64)
    env = DoubleGyrePointMass()
    layers = _layers_torch(0)
    y0 = torch.tensor([0.3, 0.4])
    r_bptt = bptt_step(env, layers, y0, horizon=4, lr=0.01)
    r_adj = actor_adjoint_step(env, layers, y0, horizon=4, lr=0.01)
    assert np.allclose(r_bptt.diagnostics["grad_theta"], r_adj.diagnostics["grad_theta"], atol=1e-8)


def test_truncated_bptt_differs_from_full_horizon():
    import jax

    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp
    from omnibias.control.jax.envs import DoubleGyrePointMass
    from omnibias.control.jax.policy import bptt_step, truncated_bptt_step

    env = DoubleGyrePointMass()
    layers = _layers_jax(jax.random.PRNGKey(2))
    y0 = jnp.array([0.3, 0.4])
    full = bptt_step(env, layers, y0, horizon=6, lr=0.01)
    trunc = truncated_bptt_step(env, layers, y0, horizon=6, window=2, lr=0.01)
    assert not np.allclose(full.diagnostics["grad_theta"], trunc.diagnostics["grad_theta"])
    assert trunc.grad_norm > 0.0


def test_truncated_bptt_rejects_bad_window():
    import jax
    import jax.numpy as jnp
    import pytest
    from omnibias.control.jax.envs import DoubleGyrePointMass
    from omnibias.control.jax.policy import truncated_bptt_step

    env = DoubleGyrePointMass()
    layers = _layers_jax(jax.random.PRNGKey(3))
    y0 = jnp.array([0.3, 0.4])
    with pytest.raises(ValueError):
        truncated_bptt_step(env, layers, y0, horizon=4, window=5, lr=0.01)


def test_zero_order_step_produces_finite_gradient():
    import jax

    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp
    from omnibias.control.jax.envs import DoubleGyrePointMass
    from omnibias.control.jax.policy import zero_order_step

    env = DoubleGyrePointMass()
    layers = _layers_jax(jax.random.PRNGKey(4))
    y0 = jnp.array([0.3, 0.4])
    result = zero_order_step(
        env, layers, y0, horizon=4, lr=0.01, n_samples=32, key=jax.random.PRNGKey(5)
    )
    assert np.all(np.isfinite(result.diagnostics["grad_theta"]))
    assert result.grad_norm >= 0.0


def test_actor_adjoint_jet_step_matches_full_adjoint_with_true_terminal_head_jax():
    """If the head predicts the *true* terminal cost gradient exactly, the short-window
    jet step matches the full adjoint step over that same window (sanity: the
    substitution machinery introduces zero extra bias when the substitution is exact)."""
    import jax

    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp
    from omnibias.control.jax.envs import DoubleGyrePointMass
    from omnibias.control.jax.policy import PSDTerminalHead, actor_adjoint_jet_step, actor_adjoint_step

    env = DoubleGyrePointMass()
    layers = _layers_jax(jax.random.PRNGKey(6))
    y0 = jnp.array([0.3, 0.4])
    window = 3
    r_full = actor_adjoint_step(env, layers, y0, horizon=window, lr=0.01)

    # An affine terminal_cost (quadratic tracking) has an EXACT PSD-quadratic
    # value whose gradient the head can represent exactly: grad Phi(y) = 2*w*(y-target).
    target = jnp.asarray(env.target)
    l_matrix = jnp.sqrt(2.0 * env.terminal_cost_weight) * jnp.eye(2)
    head = PSDTerminalHead(l_matrix=l_matrix, target=target)
    r_jet = actor_adjoint_jet_step(env, layers, head, y0, window=window, lr=0.01)
    assert r_jet.diagnostics["head_error"] < 1e-8
    assert np.allclose(r_full.diagnostics["grad_theta"], r_jet.diagnostics["grad_theta"], atol=1e-6)


def test_fit_terminal_head_reduces_regression_error_jax():
    import jax

    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp
    from omnibias.control.jax.policy import PSDTerminalHead, fit_terminal_head

    target = jnp.zeros(2)
    true_l = jnp.array([[1.0, 0.0], [0.3, 0.8]])
    states = [jnp.array([0.5, -0.3]), jnp.array([-0.2, 0.4]), jnp.array([1.0, 1.0])]
    true_grads = [(true_l @ true_l.T) @ (s - target) for s in states]

    head0 = PSDTerminalHead(l_matrix=0.01 * jnp.eye(2), target=target)
    err0 = sum(
        float(jnp.sum((head0.value_gradient(s) - g) ** 2)) for s, g in zip(states, true_grads)
    )
    fitted = fit_terminal_head(head0, states, true_grads, lr=0.05, steps=200)
    err1 = sum(
        float(jnp.sum((fitted.value_gradient(s) - g) ** 2)) for s, g in zip(states, true_grads)
    )
    assert err1 < err0 * 0.1


def test_actor_adjoint_jet_step_matches_full_adjoint_with_true_terminal_head_torch():
    import torch
    from omnibias.control.torch.envs import DoubleGyrePointMass
    from omnibias.control.torch.policy import PSDTerminalHead, actor_adjoint_jet_step, actor_adjoint_step

    torch.set_default_dtype(torch.float64)
    env = DoubleGyrePointMass()
    layers = _layers_torch(7)
    y0 = torch.tensor([0.3, 0.4])
    window = 3
    r_full = actor_adjoint_step(env, layers, y0, horizon=window, lr=0.01)

    target = torch.tensor(env.target)
    l_matrix = (2.0 * env.terminal_cost_weight) ** 0.5 * torch.eye(2)
    head = PSDTerminalHead(l_matrix=l_matrix, target=target)
    r_jet = actor_adjoint_jet_step(env, layers, head, y0, window=window, lr=0.01)
    assert r_jet.diagnostics["head_error"] < 1e-8
    assert np.allclose(r_full.diagnostics["grad_theta"], r_jet.diagnostics["grad_theta"], atol=1e-6)
