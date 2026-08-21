# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""G1 last-layer comparison for the named 1-D Poisson (theory 08-05)."""

from __future__ import annotations

import math
import statistics

import torch
from omnibias.pinn.train import DepthResidualConfig
from omnibias.pinn.train.torch.depth_residual import (
    depth_residual_sweep,
    field_tower,
    poisson_1d_residual,
)

SEEDS = (0, 1, 2, 3, 4)
WIDTHS = (8, 2)
N_COLLOC = 21
STEPS = 10
DAMPING = 1e-3
PDE_MSE_MAX = 50.0


def _collocation() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    xs = torch.linspace(0.0, 1.0, N_COLLOC + 2)[1:-1]
    u_star = torch.sin(math.pi * xs)
    force = (math.pi**2) * u_star
    return xs, u_star, force


def _init(seed: int) -> tuple[list[object], list[object]]:
    g = torch.Generator().manual_seed(seed)
    w1, w2 = WIDTHS
    layers = [
        (1.0 * torch.randn(w1, 1, generator=g), 0.3 * torch.randn(w1, generator=g), "tanh"),
        (0.3 * torch.randn(w2, w1, generator=g), 0.3 * torch.randn(w2, generator=g), "tanh"),
    ]
    decodes = [
        (1.0 * torch.randn(1, w1, generator=g), torch.zeros(1)),
        (0.5 * torch.randn(1, w2, generator=g), torch.zeros(1)),
    ]
    return layers, decodes


def _metrics(
    layers: list[object],
    decodes: list[object],
    xs: torch.Tensor,
    u_star: torch.Tensor,
    force: torch.Tensor,
    cfg: DepthResidualConfig,
) -> tuple[float, float]:
    tower = field_tower(layers, decodes, xs, config=cfg)  # type: ignore[arg-type]
    residual = poisson_1d_residual(force)(tower)
    field = tower[0]
    mse = float(torch.mean(residual**2))
    skill = 1.0 - float(torch.mean((field - u_star) ** 2)) / float(
        torch.mean(u_star**2)
    )
    return mse, skill


def test_g1_named_poisson_beats_last_layer_mean() -> None:
    torch.set_default_dtype(torch.float64)
    xs, u_star, force = _collocation()
    zero_mse = float(torch.mean(force**2))
    assert zero_mse > PDE_MSE_MAX
    pde = poisson_1d_residual(force)
    cfg_d = DepthResidualConfig(
        n_directions=1, jet_order=2, damping=DAMPING, steps_per_layer=STEPS
    )
    cfg_l = DepthResidualConfig(
        n_directions=1,
        jet_order=2,
        damping=DAMPING,
        steps_per_layer=STEPS,
        last_layer_only=True,
    )
    depth_mse: list[float] = []
    last_mse: list[float] = []
    n_leq = 0
    for seed in SEEDS:
        layers, decodes = _init(seed)
        ld, dd, report_d = depth_residual_sweep(layers, decodes, pde, xs, config=cfg_d)
        ll, dl, report_l = depth_residual_sweep(layers, decodes, pde, xs, config=cfg_l)
        md, sd = _metrics(ld, dd, xs, u_star, force, cfg_d)
        ml, _sl = _metrics(ll, dl, xs, u_star, force, cfg_l)
        depth_mse.append(md)
        last_mse.append(ml)
        n_leq += int(md <= ml)
        assert sd > 0.0
        assert md < PDE_MSE_MAX
        assert report_d.n_gn_steps == report_l.n_gn_steps == 2 * STEPS
        assert report_d.navier_stokes_proof_claim is False
        assert report_d.stretch_cleared is False
        assert report_d.hilbert_not_in_scope is True
    assert n_leq >= 2
    assert statistics.mean(depth_mse) <= statistics.mean(last_mse)
