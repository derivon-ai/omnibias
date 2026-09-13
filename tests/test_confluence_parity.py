# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Independent analytic/AD oracles for the attained collision chart."""

import math

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import torch
from omnibias.jax.confluence import centered_pair as jpair
from omnibias.torch.confluence import centered_pair as tpair


@pytest.mark.parametrize("activation", ["sigmoid", "tanh"])
@pytest.mark.parametrize("rho", [0.0, 1e-24, 1e-8, 0.03, 2.0])
def test_values_and_live_derivatives(activation, rho):
    z = torch.tensor([-2.1, -0.3, 0.4, 2.0], dtype=torch.float64, requires_grad=True)
    p = torch.tensor(rho, dtype=torch.float64, requires_grad=True)
    eta = torch.tensor([-1.4, -0.2, 0.7, 1.2], dtype=torch.float64, requires_grad=True)
    a, b = tpair(z, p, eta, activation=activation)
    ja, jb = jax.jit(lambda x, r, e: jpair(x, r, e, activation=activation))(
        jnp.array(z.detach()), jnp.array(rho), jnp.array(eta.detach())
    )
    np.testing.assert_allclose(a.detach(), ja, rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(b.detach(), jb, rtol=1e-10, atol=1e-12)
    act = torch.sigmoid if activation == "sigmoid" else torch.tanh
    if rho >= 0.03:
        h = math.sqrt(rho)
        expected_a = (act(z + h * eta) + act(z - h * eta)) / 2
        expected_b = (act(z + h * eta) - act(z - h * eta)) / (2 * h)
    else:
        # Independent differentiable activation Taylor expansion, with AD
        # derivatives rather than the shared integer coefficient evaluator.
        fn = act
        derivs = [fn(z)]
        for _ in range(11):
            derivs.append(torch.autograd.grad(derivs[-1].sum(), z, create_graph=True)[0])
        expected_a = sum(
            derivs[2 * k] * rho**k * eta ** (2 * k) / math.factorial(2 * k) for k in range(6)
        )
        expected_b = sum(
            derivs[2 * k + 1] * rho**k * eta ** (2 * k + 1) / math.factorial(2 * k + 1)
            for k in range(6)
        )
    torch.testing.assert_close(a, expected_a, rtol=1e-10, atol=1e-12)
    torch.testing.assert_close(b, expected_b, rtol=1e-10, atol=1e-12)
    actual = a + b
    expected = expected_a + expected_b
    for _ in range(4):
        actual = torch.autograd.grad(actual.sum(), z, create_graph=True)[0]
        expected = torch.autograd.grad(expected.sum(), z, create_graph=True)[0]
        torch.testing.assert_close(actual, expected, rtol=1e-10, atol=1e-12)
    gp = torch.autograd.grad((a + b).sum(), p)[0]
    jp = jax.grad(
        lambda r: sum(
            jpair(jnp.array(z.detach()), r, jnp.array(eta.detach()), activation=activation)
        ).sum()
    )(jnp.array(rho))
    np.testing.assert_allclose(gp.detach(), jp, rtol=1e-10, atol=1e-12)


def test_fourth_spread_derivative_at_boundary():
    z = torch.tensor(0.2, dtype=torch.float64)
    p = torch.tensor(0.0, dtype=torch.float64, requires_grad=True)
    eta = torch.tensor(0.7, dtype=torch.float64)
    a, b = tpair(z, p, eta)
    value = a + b
    for _ in range(4):
        value = torch.autograd.grad(value, p, create_graph=True)[0]
    zz = z.clone().requires_grad_()
    ds = [torch.sigmoid(zz)]
    for _ in range(9):
        ds.append(torch.autograd.grad(ds[-1], zz, create_graph=True)[0])
    expected = math.factorial(4) * (
        ds[8] * eta**8 / math.factorial(8) + ds[9] * eta**9 / math.factorial(9)
    )
    torch.testing.assert_close(value, expected, rtol=1e-10, atol=1e-12)


def test_tails_and_safe_inactive_branches():
    z = torch.tensor([-80.0, -20.0, 20.0, 80.0], dtype=torch.float64, requires_grad=True)
    rho = torch.tensor([0.0, 1e-30, 1.0, 1e20], dtype=torch.float64, requires_grad=True)
    a, b = tpair(z, rho, torch.ones_like(z))
    assert torch.isfinite(a).all() and torch.isfinite(b).all()
    for derivative in torch.autograd.grad((a + b).sum(), (z, rho)):
        assert torch.isfinite(derivative).all()
