# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Smoke tests for the three reference architectures."""

from __future__ import annotations

import pytest
import torch
from omnibias.torch.architectures import CmbNet, CvxLasso, CvxLogistic, PINNHeat

# --- PINN ----------------------------------------------------------------


def test_pinn_heat_forward_and_loss_decreases() -> None:
    torch.manual_seed(0)
    model = PINNHeat(hidden=16, base="softplus", alpha=0.1)
    optim = torch.optim.Adam(model.parameters(), lr=1e-2)
    x = torch.rand(64)
    t = torch.rand(64)

    losses: list[float] = []
    for _ in range(10):
        optim.zero_grad()
        u, res = model(x, t)
        loss = (res**2).mean() + (u[0] ** 2)
        loss.backward()
        optim.step()
        losses.append(loss.item())
    assert losses[-1] < losses[0], f"loss did not decrease: {losses[0]:.4e} -> {losses[-1]:.4e}"


def test_pinn_heat_chain_rule_matches_autograd() -> None:
    """The closed-form derivatives implemented by :class:`PINNOMBU` must
    match autograd to float epsilon for the trained network."""
    torch.manual_seed(0)
    model = PINNHeat(hidden=12, base="softplus", alpha=0.05)

    x = torch.linspace(0.1, 0.9, 5, requires_grad=True)
    t = torch.linspace(0.1, 0.9, 5, requires_grad=True)
    u, _ = model(x, t)
    ut_ag = torch.autograd.grad(u.sum(), t, create_graph=True)[0]
    ux_ag = torch.autograd.grad(u.sum(), x, create_graph=True)[0]
    uxx_ag = torch.autograd.grad(ux_ag.sum(), x)[0]

    inp = torch.stack([x.detach(), t.detach()], dim=-1)
    with torch.no_grad():
        _, z = model.base_forward(inp)
        ut_pinn = model.first_derivative(z, axis=1).squeeze(-1)
        uxx_pinn = model.second_derivative(z, axis=0).squeeze(-1)

    assert torch.allclose(ut_ag, ut_pinn, atol=1e-5)
    assert torch.allclose(uxx_ag, uxx_pinn, atol=1e-5)


# --- CmbNet --------------------------------------------------------------


def test_cmbnet_forward_shape() -> None:
    net = CmbNet(in_channels=1, num_classes=10, width=(8, 16, 32))
    out = net(torch.randn(2, 1, 28, 28))
    assert out.shape == (2, 10)


def test_cmbnet_one_step_loss_decreases() -> None:
    torch.manual_seed(0)
    net = CmbNet(in_channels=1, num_classes=10, width=(8, 16, 32))
    optim = torch.optim.Adam(net.parameters(), lr=1e-3)
    X = torch.randn(16, 1, 28, 28)
    y = torch.randint(0, 10, (16,))

    losses: list[float] = []
    for _ in range(10):
        optim.zero_grad()
        loss = torch.nn.functional.cross_entropy(net(X), y)
        loss.backward()
        optim.step()
        losses.append(loss.item())
    assert losses[-1] < losses[0], f"loss did not decrease: {losses[0]:.4e} -> {losses[-1]:.4e}"


# --- CvxLasso ------------------------------------------------------------


def test_cvxlasso_forward_shape() -> None:
    model = CvxLasso(n_features=20, n_obs=10, T=5, tau=0.1)
    y = torch.randn(3, 10)
    x_hat = model(y)
    assert x_hat.shape == (3, 20)


def test_cvxlasso_one_step_loss_decreases() -> None:
    torch.manual_seed(0)
    model = CvxLasso(n_features=20, n_obs=10, T=10, tau=0.05, init_step=1.0)
    # Synthetic ground truth.
    x_true = torch.zeros(8, 20)
    for i in range(8):
        idx = torch.randperm(20)[:3]
        x_true[i, idx] = torch.randn(3)
    with torch.no_grad():
        model.A.copy_(torch.randn(10, 20) / 10**0.5)
        y = x_true @ model.A.T + 0.01 * torch.randn(8, 10)

    optim = torch.optim.Adam(model.parameters(), lr=1e-2)
    losses: list[float] = []
    for _ in range(10):
        optim.zero_grad()
        x_hat = model(y)
        loss = (x_hat - x_true).pow(2).mean()
        loss.backward()
        optim.step()
        losses.append(loss.item())
    assert losses[-1] < losses[0], f"loss did not decrease: {losses[0]:.4e} -> {losses[-1]:.4e}"


# --- CvxLogistic --------------------------------------------------------


def test_cvxlogistic_forward_shape() -> None:
    model = CvxLogistic(n_features=8, T=5, init_step=0.5)
    X = torch.randn(20, 8)
    y = (torch.randn(20) > 0).float()
    w = model(X, y)
    assert w.shape == (8,)


def test_cvxlogistic_separable_problem_recovers_sign() -> None:
    """On a perfectly linearly separable problem, unrolled GD should
    push weights in the right direction within a few iterations."""
    torch.manual_seed(0)
    n = 8
    model = CvxLogistic(n_features=n, T=20, init_step=1.0)
    # ground truth: y = 1 iff first feature is positive
    X = torch.randn(64, n)
    y = (X[:, 0] > 0).float()
    w = model(X, y)
    # First weight should be positive (the discriminative direction)
    assert w[0] > 0
