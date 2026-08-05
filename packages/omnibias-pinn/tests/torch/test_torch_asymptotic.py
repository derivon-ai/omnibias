# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Asymptotic / removable boundary-condition losses (torch)."""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")
torch.set_default_dtype(torch.float64)

from omnibias.pinn.torch.losses import (  # noqa: E402
    asymptotic_bc_loss,
    asymptotic_ratio,
    far_field_decay_loss,
    network_ray_jet,
)
from omnibias.torch.jet import jet_to_tower, mlp_jet  # noqa: E402


def _build_mlp(seed: int = 0, dims=(2, 5, 4, 1), act: str = "tanh", grad: bool = False):
    rng = np.random.default_rng(seed)
    layers = []
    for i in range(len(dims) - 1):
        din, dout = dims[i], dims[i + 1]
        W = torch.tensor(rng.normal(scale=0.6, size=(dout, din)), requires_grad=grad)
        b = torch.tensor(rng.normal(scale=0.4, size=(dout,)), requires_grad=grad)
        spec = None if i == len(dims) - 2 else act
        layers.append((W, b, spec))
    x0 = torch.tensor(rng.normal(size=(dims[0],)))
    v = torch.tensor(rng.normal(size=(dims[0],)))
    return layers, x0, v


def test_network_ray_jet_matches_mlp_jet() -> None:
    layers, x0, v = _build_mlp(seed=1)
    got = network_ray_jet(layers, x0, v, order=3, out_index=0)
    full = mlp_jet(x0, v, layers, 3)
    assert tuple(got.shape) == (4,)
    assert torch.allclose(got, full[:, 0], rtol=1e-13, atol=1e-13)


def test_asymptotic_ratio_equals_leading_coefficients() -> None:
    layers, x0, v = _build_mlp(seed=2)
    net_jet = network_ray_jet(layers, x0, v, order=2)
    for rate in (0, 1, 2):
        val = asymptotic_ratio(layers, x0, v, rate=rate, order=2)
        assert float(val) == pytest.approx(float(net_jet[rate]), abs=1e-13)


def test_rate0_is_value_and_rate1_is_directional_derivative() -> None:
    layers, x0, v = _build_mlp(seed=3)
    tower = jet_to_tower(mlp_jet(x0, v, layers, 1))
    value = asymptotic_ratio(layers, x0, v, rate=0)
    slope = asymptotic_ratio(layers, x0, v, rate=1)
    assert float(value) == pytest.approx(float(tower[0, 0]), abs=1e-12)
    assert float(slope) == pytest.approx(float(tower[1, 0]), abs=1e-12)


def test_asymptotic_bc_loss_backpropagates() -> None:
    layers, x0, v = _build_mlp(seed=5, grad=True)
    loss = asymptotic_bc_loss(layers, x0, v, target=0.3, rate=1)
    loss.backward()
    W2 = layers[-1][0]
    assert W2.grad is not None
    assert float(torch.max(torch.abs(W2.grad))) > 1e-8


def test_asymptotic_bc_loss_zero_at_target() -> None:
    layers, x0, v = _build_mlp(seed=4)
    target = asymptotic_ratio(layers, x0, v, rate=1).detach()
    loss = asymptotic_bc_loss(layers, x0, v, target=target, rate=1)
    assert float(loss) == pytest.approx(0.0, abs=1e-20)


def test_far_field_decay_loss_value_and_backprop() -> None:
    layers, x0, v = _build_mlp(seed=6, grad=True)
    order = 3
    net_jet = network_ray_jet(layers, x0, v, order=order)
    loss = far_field_decay_loss(layers, x0, v, order=order)
    expected = (net_jet * net_jet).mean().detach()
    assert float(loss.detach()) == pytest.approx(float(expected), abs=1e-13)
    assert float(loss.detach()) > 0.0
    loss.backward()
    W2 = layers[-1][0]
    assert W2.grad is not None
    assert float(torch.max(torch.abs(W2.grad))) > 1e-8


def test_invalid_arguments_raise() -> None:
    layers, x0, v = _build_mlp(seed=7)
    with pytest.raises(ValueError, match="rate must be >= 0"):
        asymptotic_ratio(layers, x0, v, rate=-1)
    with pytest.raises(ValueError, match="must be >= rate"):
        asymptotic_ratio(layers, x0, v, rate=2, order=1)
    with pytest.raises(ValueError, match="order must be >= 0"):
        far_field_decay_loss(layers, x0, v, order=-1)
