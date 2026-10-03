# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""A fourth-order PINN residual stays correct and trainable through spatial jets.

Nested autodiff is an independent small-model oracle, not the training path.
The clamped manufactured problem is d^4u/dx^4 = 24 on [0, 1].
"""
from __future__ import annotations

import numpy as np
import pytest

jax = pytest.importorskip("jax")
torch = pytest.importorskip("torch")
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from omnibias.jax.jet import jet_to_tower as jax_jet_to_tower
from omnibias.jax.jet import mlp_jet as jax_mlp_jet
from omnibias.torch.jet import jet_to_tower as torch_jet_to_tower
from omnibias.torch.jet import mlp_jet as torch_mlp_jet

_INITIAL = (0.7, -0.4, 0.1, 0.2, 0.6, -0.3, 0.05)
_POINTS = (0.1, 0.35, 0.7, 0.9)


def _torch_value(theta, x):
    return (theta[4:6] * torch.tanh(theta[:2] * x + theta[2:4])).sum() + theta[6]


def _torch_jet(theta, x, order):
    layers = [
        (theta[:2].reshape(2, 1), theta[2:4], "tanh"),
        (theta[4:6].reshape(1, 2), theta[6:].reshape(1), None),
    ]
    point = x.reshape(1)
    return torch_jet_to_tower(
        torch_mlp_jet(point, torch.ones_like(point), layers, order=order)
    )[:, 0]


def _torch_oracle(theta, x, order):
    coordinate = x.detach().clone().requires_grad_(True)
    derivatives = [_torch_value(theta, coordinate)]
    for _ in range(order):
        derivatives.append(torch.autograd.grad(
            derivatives[-1], coordinate, create_graph=True,
        )[0])
    return torch.stack(derivatives)


def _torch_loss(theta, derivative_fn):
    fourth = torch.stack([
        derivative_fn(theta, theta.new_tensor(x), 4)[4] for x in _POINTS
    ])
    left = derivative_fn(theta, theta.new_tensor(0.0), 1)
    right = derivative_fn(theta, theta.new_tensor(1.0), 1)
    return (fourth - 24.0).square().mean() + 10.0 * (
        left.square().sum() + right.square().sum()
    )


def _jax_value(theta, x):
    return jnp.sum(theta[4:6] * jnp.tanh(theta[:2] * x + theta[2:4])) + theta[6]


def _jax_jet(theta, x, order):
    layers = [
        (theta[:2].reshape(2, 1), theta[2:4], "tanh"),
        (theta[4:6].reshape(1, 2), theta[6:].reshape(1), None),
    ]
    point = jnp.reshape(x, (1,))
    return jax_jet_to_tower(
        jax_mlp_jet(point, jnp.ones_like(point), layers, order=order)
    )[:, 0]


def _jax_oracle(theta, x, order):
    differentiated = _jax_value
    derivatives = [differentiated(theta, x)]
    for _ in range(order):
        differentiated = jax.grad(differentiated, argnums=1)
        derivatives.append(differentiated(theta, x))
    return jnp.stack(derivatives)


def _jax_loss(theta, derivative_fn):
    fourth = jnp.stack([
        derivative_fn(theta, jnp.asarray(x), 4)[4] for x in _POINTS
    ])
    left = derivative_fn(theta, jnp.asarray(0.0), 1)
    right = derivative_fn(theta, jnp.asarray(1.0), 1)
    return jnp.mean((fourth - 24.0) ** 2) + 10.0 * (
        jnp.sum(left ** 2) + jnp.sum(right ** 2)
    )


def test_torch_fourth_order_residual_and_parameter_gradient_match_oracle():
    theta = torch.tensor(_INITIAL, dtype=torch.float64, requires_grad=True)
    for x in _POINTS:
        coordinate = theta.new_tensor(x)
        actual = _torch_jet(theta, coordinate, 4)
        expected = _torch_oracle(theta, coordinate, 4)
        torch.testing.assert_close(actual, expected, rtol=1e-11, atol=1e-12)
    actual_loss = _torch_loss(theta, _torch_jet)
    oracle_loss = _torch_loss(theta, _torch_oracle)
    actual_grad = torch.autograd.grad(actual_loss, theta)[0]
    oracle_grad = torch.autograd.grad(oracle_loss, theta)[0]
    torch.testing.assert_close(actual_loss, oracle_loss, rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(actual_grad, oracle_grad, rtol=1e-10, atol=1e-11)
    assert torch.isfinite(actual_grad).all()
    assert torch.count_nonzero(actual_grad) == theta.numel()


def test_jax_fourth_order_residual_and_parameter_gradient_match_oracle():
    theta = jnp.asarray(_INITIAL, dtype=jnp.float64)
    for x in _POINTS:
        actual = _jax_jet(theta, jnp.asarray(x), 4)
        expected = _jax_oracle(theta, jnp.asarray(x), 4)
        np.testing.assert_allclose(actual, expected, rtol=1e-11, atol=1e-12)
    actual_loss, actual_grad = jax.value_and_grad(_jax_loss)(theta, _jax_jet)
    oracle_loss, oracle_grad = jax.value_and_grad(_jax_loss)(theta, _jax_oracle)
    np.testing.assert_allclose(actual_loss, oracle_loss, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(actual_grad, oracle_grad, rtol=1e-10, atol=1e-11)
    assert bool(jnp.all(jnp.isfinite(actual_grad)))
    assert int(jnp.count_nonzero(actual_grad)) == theta.size


def test_torch_optimizer_step_reduces_pinn_loss():
    theta = torch.nn.Parameter(torch.tensor(_INITIAL, dtype=torch.float64))
    optimizer = torch.optim.Adam([theta], lr=1e-3)
    before = _torch_loss(theta, _torch_jet)
    optimizer.zero_grad()
    before.backward()
    assert theta.grad is not None and torch.isfinite(theta.grad).all()
    optimizer.step()
    after = _torch_loss(theta, _torch_jet)
    assert torch.isfinite(after)
    assert after.item() < before.item()


def test_jax_gradient_step_reduces_pinn_loss():
    theta = jnp.asarray(_INITIAL, dtype=jnp.float64)
    before, gradient = jax.value_and_grad(_jax_loss)(theta, _jax_jet)
    assert bool(jnp.all(jnp.isfinite(gradient)))
    updated = theta - 1e-4 * gradient
    after = _jax_loss(updated, _jax_jet)
    assert bool(jnp.isfinite(after))
    assert float(after) < float(before)
