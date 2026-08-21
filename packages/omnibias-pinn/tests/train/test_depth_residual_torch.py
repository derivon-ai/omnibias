# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Depth-causal residual sweep, PyTorch (theory 08-05)."""

from __future__ import annotations

import math

import pytest
import torch
from omnibias.pinn.train import DepthResidualConfig, DepthResidualForbidden
from omnibias.pinn.train.torch.depth_residual import (
    depth_residual_sweep,
    field_tower,
    poisson_1d_residual,
)


def test_manufactured_constant_field_is_exact() -> None:
    torch.set_default_dtype(torch.float64)
    xs = torch.linspace(0.1, 0.9, 9)
    layers = [(torch.zeros(1, 1), torch.ones(1), None)]
    decode = (torch.ones(1, 1), None)
    cfg = DepthResidualConfig(n_directions=1, jet_order=2, damping=0.0)
    tower = field_tower(layers, decode, xs, config=cfg)
    residual = poisson_1d_residual(torch.full_like(xs, 2.0))(tower)
    assert float(residual.abs().max()) <= 1e-12
    assert torch.allclose(tower[2], torch.full_like(xs, -2.0))


def test_g2_honesty_keys_sealed() -> None:
    torch.set_default_dtype(torch.float64)
    xs = torch.linspace(0.1, 0.9, 5)
    layers = [(0.2 * torch.ones(2, 1), torch.zeros(2), "tanh")]
    decode = (0.1 * torch.ones(1, 2), torch.zeros(1))
    force = (math.pi**2) * torch.sin(math.pi * xs)
    cfg = DepthResidualConfig(n_directions=1, jet_order=2, damping=1e-3)
    _layers, _dec, report = depth_residual_sweep(
        layers, decode, poisson_1d_residual(force), xs, config=cfg
    )
    assert report.navier_stokes_proof_claim is False
    assert report.stretch_cleared is False
    assert report.hilbert_not_in_scope is True
    assert report.greedy_only_claimed_optimal is False
    assert report.last_layer_only is False
    assert report.n_gn_steps == 1


def test_flood_forbid() -> None:
    torch.set_default_dtype(torch.float64)
    layers = [(torch.ones(1, 1), None, None)]
    decode = (torch.ones(1, 1), None)
    xs = torch.tensor([0.3, 0.7])
    cfg = DepthResidualConfig(n_directions=2, jet_order=2, damping=0.0)
    with pytest.raises(DepthResidualForbidden, match="allow_full"):
        depth_residual_sweep(
            layers,
            decode,
            poisson_1d_residual(torch.ones_like(xs)),
            xs,
            config=cfg,
        )


def test_unlock_ratio_skips_later_layer() -> None:
    torch.set_default_dtype(torch.float64)
    g = torch.Generator().manual_seed(0)
    w1 = 0.4 * torch.randn(3, 1, generator=g)
    b1 = 0.4 * torch.randn(3, generator=g)
    w2 = 0.4 * torch.randn(3, 3, generator=g)
    b2 = 0.4 * torch.randn(3, generator=g)
    layers = [(w1, b1, "tanh"), (w2, b2, "tanh")]
    d1 = (0.4 * torch.randn(1, 3, generator=g), torch.zeros(1))
    d2 = (0.4 * torch.randn(1, 3, generator=g), torch.zeros(1))
    xs = torch.linspace(0.1, 0.9, 7)
    force = (math.pi**2) * torch.sin(math.pi * xs)
    cfg = DepthResidualConfig(
        n_directions=1, jet_order=2, damping=1e-3, unlock_ratio=0.0
    )
    new_layers, new_dec, report = depth_residual_sweep(
        layers, [d1, d2], poisson_1d_residual(force), xs, config=cfg
    )
    assert report.unlocked == (True, False)
    assert report.n_gn_steps == 1
    assert torch.equal(new_layers[1][0], w2)
    assert torch.equal(new_dec[1][0], d2[0])


def test_last_layer_only_matched_budget() -> None:
    torch.set_default_dtype(torch.float64)
    xs = torch.linspace(0.1, 0.9, 5)
    layers = [
        (0.2 * torch.ones(2, 1), torch.zeros(2), "tanh"),
        (0.2 * torch.ones(2, 2), torch.zeros(2), "tanh"),
    ]
    decodes = [
        (0.1 * torch.ones(1, 2), torch.zeros(1)),
        (0.1 * torch.ones(1, 2), torch.zeros(1)),
    ]
    force = (math.pi**2) * torch.sin(math.pi * xs)
    cfg = DepthResidualConfig(
        n_directions=1,
        jet_order=2,
        damping=1e-3,
        steps_per_layer=2,
        last_layer_only=True,
    )
    new_layers, _dec, report = depth_residual_sweep(
        layers, decodes, poisson_1d_residual(force), xs, config=cfg
    )
    assert report.last_layer_only is True
    assert report.n_gn_steps == 4
    assert report.unlocked == (False, True)
    assert torch.equal(new_layers[0][0], layers[0][0])


def test_one_sweep_reduces_manufactured_residual() -> None:
    torch.set_default_dtype(torch.float64)
    xs = torch.linspace(0.1, 0.9, 9)
    # Exact field is N=1, u=x(1-x), f=2. Hidden state is the constant 0.4;
    # only the decode scale is free, so one undamped GN recovers N=1.
    layers = [(torch.zeros(1, 1), torch.tensor([0.4]), None)]
    decode = (torch.ones(1, 1), None)
    cfg = DepthResidualConfig(
        n_directions=1,
        jet_order=2,
        damping=0.0,
        last_layer_only=True,
        last_layer_scope="decode",
    )
    pde = poisson_1d_residual(torch.full_like(xs, 2.0))
    before = float(torch.mean(pde(field_tower(layers, decode, xs, config=cfg)) ** 2))
    new_layers, new_dec, report = depth_residual_sweep(
        layers, decode, pde, xs, config=cfg
    )
    after = float(torch.mean(pde(field_tower(new_layers, new_dec, xs, config=cfg)) ** 2))
    assert after < before
    assert after <= 1e-12
    assert report.n_gn_steps == 1
    assert torch.equal(new_layers[0][1], torch.tensor([0.4]))
