# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
"""jax/torch adjoint.py: closed-form dpi/dy exactness (G1) + full-pipeline parity (G9)."""

from __future__ import annotations

import numpy as np


def _small_layers_jax(key):
    import jax
    import jax.numpy as jnp

    k1, k2 = jax.random.split(key)
    w1 = 0.1 * jax.random.normal(k1, (6, 2))
    b1 = jnp.zeros(6)
    w2 = 0.1 * jax.random.normal(k2, (2, 6))
    b2 = jnp.zeros(2)
    return [(w1, b1, "tanh"), (w2, b2, None)]


def _small_layers_torch(seed=0):
    import torch

    g = torch.Generator().manual_seed(seed)
    w1 = 0.1 * torch.randn(6, 2, generator=g)
    b1 = torch.zeros(6)
    w2 = 0.1 * torch.randn(2, 6, generator=g)
    b2 = torch.zeros(2)
    return [(w1, b1, "tanh"), (w2, b2, None)]


def test_policy_jacobian_dy_matches_autodiff_jax():
    import jax

    jax.config.update("jax_enable_x64", True)
    from omnibias.control.jax.adjoint import policy_jacobian_dtheta, policy_jacobian_dy, policy_forward

    layers = _small_layers_jax(jax.random.PRNGKey(0))
    y = jax.numpy.array([0.3, -0.2])
    dy_closed = np.asarray(policy_jacobian_dy(layers, y))
    dy_autodiff = np.asarray(jax.jacfwd(lambda yy: policy_forward(layers, yy))(y))
    assert np.allclose(dy_closed, dy_autodiff, atol=1e-10)
    # sanity: dtheta jacobian has the right shape (action_dim, n_params)
    n_params = sum(w.size + b.size for w, b, _ in layers)
    dtheta = np.asarray(policy_jacobian_dtheta(layers, y))
    assert dtheta.shape == (2, n_params)


def test_policy_jacobian_dy_matches_autodiff_torch():
    import torch
    from omnibias.control.torch.adjoint import policy_forward, policy_jacobian_dtheta, policy_jacobian_dy

    torch.set_default_dtype(torch.float64)
    layers = _small_layers_torch(0)
    y = torch.tensor([0.3, -0.2])
    dy_closed = policy_jacobian_dy(layers, y).detach().numpy()
    dy_autodiff = torch.func.jacfwd(lambda yy: policy_forward(layers, yy))(y).detach().numpy()
    assert np.allclose(dy_closed, dy_autodiff, atol=1e-10)
    n_params = sum(w.numel() + b.numel() for w, b, _ in layers)
    dtheta = policy_jacobian_dtheta(layers, y).detach().numpy()
    assert dtheta.shape == (2, n_params)


def test_actor_adjoint_gradient_matches_full_bptt_jax():
    """G1: full-horizon exact adjoint == plain reverse-mode AD through the rollout."""
    import jax

    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp
    from omnibias.control.jax.adjoint import actor_adjoint_gradient, flatten_layers, policy_forward
    from omnibias.control.jax.envs import DoubleGyrePointMass

    env = DoubleGyrePointMass()
    layers = _small_layers_jax(jax.random.PRNGKey(1))
    y0 = jnp.array([0.3, 0.4])
    horizon = 4
    result = actor_adjoint_gradient(env, layers, y0, horizon)

    theta, unravel = flatten_layers(layers)

    def rollout_cost(th):
        ll = unravel(th)
        y = y0
        total = 0.0
        for _ in range(horizon):
            u = policy_forward(ll, y)
            total = total + env.cost(y, u)
            y = env.step(y, u)
        return total + env.terminal_cost(y)

    bptt_grad = np.asarray(jax.grad(rollout_cost)(theta))
    assert np.allclose(result.grad_theta, bptt_grad, atol=1e-8)


def test_actor_adjoint_gradient_matches_full_bptt_torch():
    import torch
    from omnibias.control.torch.adjoint import actor_adjoint_gradient, flatten_layers, policy_forward
    from omnibias.control.torch.envs import DoubleGyrePointMass

    torch.set_default_dtype(torch.float64)
    env = DoubleGyrePointMass()
    layers = _small_layers_torch(1)
    y0 = torch.tensor([0.3, 0.4])
    horizon = 4
    result = actor_adjoint_gradient(env, layers, y0, horizon)

    theta, structure = flatten_layers(layers)
    theta_req = theta.detach().clone().requires_grad_(True)
    ll = structure.unflatten(theta_req)
    y = y0
    total = torch.zeros(())
    for _ in range(horizon):
        u = policy_forward(ll, y)
        total = total + env.cost(y, u)
        y = env.step(y, u)
    total = total + env.terminal_cost(y)
    (bptt_grad,) = torch.autograd.grad(total, theta_req)
    assert np.allclose(result.grad_theta, bptt_grad.detach().numpy(), atol=1e-8)


def test_adjoint_recursion_is_the_shared_core_module():
    """G9 (structural): both backends call the same backend-free recursion."""
    from omnibias.control.jax import adjoint as jax_adjoint
    from omnibias.control.torch import adjoint as torch_adjoint

    assert jax_adjoint.adjoint_recursion is torch_adjoint.adjoint_recursion
    assert jax_adjoint.closed_loop_jacobian is torch_adjoint.closed_loop_jacobian
    assert jax_adjoint.policy_gradient is torch_adjoint.policy_gradient
