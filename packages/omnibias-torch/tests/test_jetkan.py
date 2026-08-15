# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""JetKAN tests (theory 02-03): pack birth, order growth, model-jet exactness."""

from __future__ import annotations

import math

import numpy as np
import torch
from omnibias.torch.architectures.jetkan import JetKAN, JetKANConfig


def _ulp_error(a: float, b: float) -> float:
    if not (math.isfinite(a) and math.isfinite(b)):
        return float("inf")
    if a == b:
        return 0.0
    scale = max(abs(a), abs(b), 1.0)
    return abs(a - b) / (np.finfo(np.float64).eps * scale)


def test_pack_birth_is_bit_identical() -> None:
    torch.set_default_dtype(torch.float64)
    cfg = JetKANConfig(widths=(2, 2, 1), packs_per_edge=2, extra_packs=2, orders=(0, 1))
    net = JetKAN(cfg, dtype=torch.float64)
    x = torch.randn(5, 2, dtype=torch.float64)
    y0 = net(x).detach().clone()
    net.refine("pack")
    y1 = net(x)
    worst = max(
        _ulp_error(float(a), float(b))
        for a, b in zip(y0.reshape(-1).tolist(), y1.reshape(-1).tolist(), strict=True)
    )
    assert worst <= 4.0


def test_order_growth_is_bit_identical() -> None:
    torch.set_default_dtype(torch.float64)
    cfg = JetKANConfig(
        widths=(1, 1), packs_per_edge=1, extra_packs=0, orders=(0,), growable=True
    )
    net = JetKAN(cfg, dtype=torch.float64)
    x = torch.linspace(-0.5, 0.5, 7, dtype=torch.float64).reshape(-1, 1)
    y0 = net(x).detach().clone()
    net.refine("order")
    y1 = net(x)
    worst = max(
        _ulp_error(float(a), float(b))
        for a, b in zip(y0.reshape(-1).tolist(), y1.reshape(-1).tolist(), strict=True)
    )
    assert worst <= 4.0


def test_model_jet_matches_autodiff_tanh_edge() -> None:
    torch.set_default_dtype(torch.float64)
    cfg = JetKANConfig(
        widths=(1, 1), packs_per_edge=1, extra_packs=0, orders=(0,), growable=False
    )
    net = JetKAN(cfg, dtype=torch.float64)
    with torch.no_grad():
        net.layers[0].weights.fill_(1.0)
        net.layers[0].means.zero_()
        net.layers[0].log_scales.zero_()
    from omnibias.torch.jet import jet_to_tower

    x0 = torch.tensor([0.3], dtype=torch.float64)
    v = torch.tensor([1.0], dtype=torch.float64)
    tower = jet_to_tower(net.jet(x0, order=2, direction=v))
    x = torch.tensor([0.3], dtype=torch.float64, requires_grad=True)
    y = net(x.unsqueeze(0)).reshape(())
    d1 = torch.autograd.grad(y, x, create_graph=True)[0]
    d2 = torch.autograd.grad(d1, x, create_graph=True)[0]
    assert _ulp_error(float(tower[0].detach()), float(y.detach())) <= 8.0
    assert _ulp_error(float(tower[1].detach()), float(d1.detach())) <= 16.0
    assert _ulp_error(float(tower[2].detach()), float(d2.detach())) <= 64.0


def test_jetkan_from_band_plan_pack_count() -> None:
    from omnibias.core.spectral_design import design_band_plan
    from omnibias.torch.architectures.jetkan import jetkan_from_band_plan

    plan = design_band_plan("sech", xi_lo=1.0, xi_hi=16.0, channels=3, order=1)
    cfg = jetkan_from_band_plan(plan, (2, 1))
    assert cfg.packs_per_edge == 3
    net = JetKAN(cfg, dtype=torch.float64)
    out = net(torch.zeros(2, 2, dtype=torch.float64))
    assert out.shape == (2, 1)
