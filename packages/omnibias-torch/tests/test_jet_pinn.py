# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Deep, arbitrary-order omnibias PINN (`JetMLP` / `DeepPINNHeat`).

These exercise the *closed-form* multivariate-jet derivative path against
``torch.func`` autograd as ground truth. Everything runs in float64 so the
closed-form tower is checked to ~machine precision (the whole point: the spatial
differential operator is exact, not an autograd/finite-difference approximation).
"""

from __future__ import annotations

import dataclasses

import pytest
import torch
from omnibias.torch.activations.registry import get_activation
from omnibias.torch.architectures import DeepPINNHeat, JetMLP

torch.manual_seed(0)


def _val_fn(net: JetMLP):
    """Scalar-output value function of a single point, for ``torch.func``."""
    def val(xi: torch.Tensor) -> torch.Tensor:
        return net.value(xi).squeeze(-1)

    return val


def test_jetmlp_value_equals_jet_row0() -> None:
    net = JetMLP(2, 16, 1, depth=3, base="tanh").double()
    x = torch.randn(7, 2, dtype=torch.float64)
    jet = net.jet(x, 2)
    assert torch.allclose(jet[:, 0, :], net.value(x), atol=1e-14)


def test_jetmlp_gradient_matches_autograd() -> None:
    net = JetMLP(2, 16, 1, depth=3, base="tanh").double()
    x = torch.randn(7, 2, dtype=torch.float64)
    g = net.gradient(x).squeeze(-1)  # (B, 2)
    g_ad = torch.func.vmap(torch.func.jacrev(_val_fn(net)))(x)
    assert torch.allclose(g, g_ad, atol=1e-11)


def test_jetmlp_hessian_matches_autograd() -> None:
    net = JetMLP(2, 16, 1, depth=3, base="tanh").double()
    x = torch.randn(7, 2, dtype=torch.float64)
    h = net.hessian(x).squeeze(-1)  # (B, 2, 2)
    h_ad = torch.func.vmap(torch.func.hessian(_val_fn(net)))(x)
    assert torch.allclose(h, h_ad, atol=1e-10)
    # Hessian must be symmetric.
    assert torch.allclose(h, h.transpose(1, 2), atol=1e-12)


def test_jetmlp_third_order_matches_autograd() -> None:
    """The arbitrary-order claim: every 3rd-order partial matches nested autograd."""
    net = JetMLP(2, 12, 1, depth=2, base="tanh").double()
    x = torch.randn(5, 2, dtype=torch.float64)
    parts = net.partials(x, 3)
    val = _val_fn(net)
    t3 = torch.func.vmap(
        torch.func.jacfwd(torch.func.jacfwd(torch.func.jacfwd(val)))
    )(x)  # (B, 2, 2, 2): t3[...,i,j,k] = d^3 u / dx_i dx_j dx_k
    assert torch.allclose(parts[(3, 0)].squeeze(-1), t3[:, 0, 0, 0], atol=1e-9)
    assert torch.allclose(parts[(2, 1)].squeeze(-1), t3[:, 0, 0, 1], atol=1e-9)
    assert torch.allclose(parts[(1, 2)].squeeze(-1), t3[:, 0, 1, 1], atol=1e-9)
    assert torch.allclose(parts[(0, 3)].squeeze(-1), t3[:, 1, 1, 1], atol=1e-9)


def test_jetmlp_fourth_order_matches_autograd() -> None:
    """Order independent of the chain-rule depth: a 4th-order partial still matches."""
    net = JetMLP(2, 8, 1, depth=2, base="tanh").double()
    x = torch.randn(3, 2, dtype=torch.float64)
    parts = net.partials(x, 4)
    val = _val_fn(net)
    jf = torch.func.jacfwd
    t4 = torch.func.vmap(jf(jf(jf(jf(val)))))(x)  # (B, 2, 2, 2, 2)
    assert torch.allclose(parts[(4, 0)].squeeze(-1), t4[:, 0, 0, 0, 0], atol=1e-8)
    assert torch.allclose(parts[(2, 2)].squeeze(-1), t4[:, 0, 0, 1, 1], atol=1e-8)


def test_jetmlp_value_grad_hessian_consistent() -> None:
    net = JetMLP(2, 10, 1, depth=3, base="tanh").double()
    x = torch.randn(6, 2, dtype=torch.float64)
    v, g, h = net.value_grad_hessian(x)
    assert torch.allclose(v, net.value(x), atol=1e-14)
    assert torch.allclose(g, net.gradient(x), atol=1e-13)
    assert torch.allclose(h, net.hessian(x), atol=1e-13)


@pytest.mark.parametrize("depth", [1, 2, 3, 4])
def test_jetmlp_depth_sweep_matches_autograd(depth: int) -> None:
    net = JetMLP(2, 12, 1, depth=depth, base="tanh").double()
    x = torch.randn(5, 2, dtype=torch.float64)
    g = net.gradient(x).squeeze(-1)
    g_ad = torch.func.vmap(torch.func.jacrev(_val_fn(net)))(x)
    assert torch.allclose(g, g_ad, atol=1e-10)


@pytest.mark.parametrize("base", ["tanh", "sigmoid", "softplus"])
def test_jetmlp_activation_sweep_matches_autograd(base: str) -> None:
    net = JetMLP(2, 12, 1, depth=2, base=base).double()
    x = torch.randn(5, 2, dtype=torch.float64)
    h = net.hessian(x).squeeze(-1)
    h_ad = torch.func.vmap(torch.func.hessian(_val_fn(net)))(x)
    assert torch.allclose(h, h_ad, atol=1e-9)


def test_jetmlp_multioutput_gradient_shape_and_parity() -> None:
    net = JetMLP(3, 8, out_dim=2, depth=2, base="tanh").double()
    x = torch.randn(4, 3, dtype=torch.float64)
    g = net.gradient(x)  # (B, in_dim=3, out_dim=2)
    assert g.shape == (4, 3, 2)
    j_ad = torch.func.vmap(torch.func.jacrev(net.value))(x)  # (B, out=2, in=3)
    assert torch.allclose(g, j_ad.transpose(1, 2), atol=1e-11)


def test_jetmlp_partials_order2_equal_hessian() -> None:
    net = JetMLP(2, 10, 1, depth=2, base="tanh").double()
    x = torch.randn(5, 2, dtype=torch.float64)
    parts = net.partials(x, 2)
    h = net.hessian(x).squeeze(-1)
    assert torch.allclose(parts[(2, 0)].squeeze(-1), h[:, 0, 0], atol=1e-12)
    assert torch.allclose(parts[(1, 1)].squeeze(-1), h[:, 0, 1], atol=1e-12)
    assert torch.allclose(parts[(0, 2)].squeeze(-1), h[:, 1, 1], atol=1e-12)


def test_jetmlp_fastpath_none_raises() -> None:
    """An activation without a closed-form derivative kernel is rejected."""
    nofp = dataclasses.replace(get_activation("tanh"), name="tanh_nofp", fastpath=None)
    net = JetMLP(2, 8, 1, depth=2, base=nofp)
    x = torch.randn(3, 2)
    with pytest.raises(ValueError, match="closed-form derivative"):
        net.gradient(x)


def test_jetmlp_invalid_dims_raise() -> None:
    with pytest.raises(ValueError):
        JetMLP(0, 8, 1, depth=2)
    with pytest.raises(ValueError):
        JetMLP(2, 0, 1, depth=2)
    with pytest.raises(ValueError):
        JetMLP(2, 8, 0, depth=2)
    with pytest.raises(ValueError):
        JetMLP(2, 8, 1, depth=0)


# --- DeepPINNHeat --------------------------------------------------------


def test_deeppinnheat_residual_matches_autograd() -> None:
    model = DeepPINNHeat(hidden=12, depth=3, base="tanh", alpha=0.05).double()
    x = torch.linspace(0.1, 0.9, 6, dtype=torch.float64, requires_grad=True)
    t = torch.linspace(0.1, 0.9, 6, dtype=torch.float64, requires_grad=True)
    u, res = model(x, t)
    ut = torch.autograd.grad(u.sum(), t, create_graph=True)[0]
    ux = torch.autograd.grad(u.sum(), x, create_graph=True)[0]
    uxx = torch.autograd.grad(ux.sum(), x)[0]
    res_ad = ut - 0.05 * uxx
    assert torch.allclose(res, res_ad, atol=1e-10)


def test_deeppinnheat_training_decreases() -> None:
    torch.manual_seed(0)
    model = DeepPINNHeat(hidden=16, depth=2, base="tanh", alpha=0.1)
    optim = torch.optim.Adam(model.parameters(), lr=1e-2)
    losses: list[float] = []
    for _ in range(12):
        optim.zero_grad()
        x = torch.rand(64)
        t = torch.rand(64)
        u, res = model(x, t)
        loss = (res**2).mean() + (u[0] ** 2)
        loss.backward()
        optim.step()
        losses.append(loss.item())
    assert losses[-1] < losses[0]


def test_deeppinnheat_shape_mismatch_raises() -> None:
    model = DeepPINNHeat(hidden=8, depth=2)
    with pytest.raises(ValueError, match="matching shape"):
        model(torch.rand(5), torch.rand(6))
