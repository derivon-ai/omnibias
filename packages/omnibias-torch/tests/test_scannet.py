# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""ScanNet tests (theory 02-01): interior shift, config, band-plan wiring."""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch
from omnibias.core.spectral_design import design_band_plan
from omnibias.torch.architectures.scannet import ScanNet, ScanNetConfig, scannet_from_band_plan


def _ulp_error(a: float, b: float) -> float:
    if not (math.isfinite(a) and math.isfinite(b)):
        return float("inf")
    if a == b:
        return 0.0
    scale = max(abs(a), abs(b), 1.0)
    return abs(a - b) / (np.finfo(np.float64).eps * scale)


def test_g1_layer_interior_shift() -> None:
    torch.set_default_dtype(torch.float64)
    cfg = ScanNetConfig(
        dim_in=1,
        channels=(2,),
        bank_sizes=(5,),
        bank_extents=(1.0,),
        template="grad",
        readout="pooled",
    )
    net = ScanNet(cfg, dtype=torch.float64)
    layer = net.layers[0]
    spacing = float(layer.scan.offsets[1] - layer.scan.offsets[0])
    z = torch.tensor([[0.1, -0.05]], dtype=torch.float64)
    r0 = net.layer_scan_response(z, 0)
    r_shift = net.layer_scan_response(z + spacing, 0)
    left = r_shift[..., :-1].reshape(-1).tolist()
    right = r0[..., 1:].reshape(-1).tolist()
    worst = max(_ulp_error(a, b) for a, b in zip(left, right, strict=True))
    assert worst <= 4.0


def test_scannet_from_band_plan_extents() -> None:
    plan = design_band_plan("sech", xi_lo=1.0, xi_hi=32.0, channels=4, order=2)
    cfg = scannet_from_band_plan(plan, dim_in=2, width=3, bank_size=7, readout="argmax")
    assert cfg.channels == (3, 3, 3, 3)
    assert cfg.bank_extents == pytest.approx(plan.scales)
    net = ScanNet(cfg, dtype=torch.float64)
    x = torch.zeros((4, 2), dtype=torch.float64)
    out = net(x)
    assert out.shape[-1] == 3


def test_concat_u_and_pooled_shape() -> None:
    torch.set_default_dtype(torch.float64)
    cfg = ScanNetConfig(
        dim_in=3,
        channels=(4, 2),
        bank_sizes=(5, 5),
        bank_extents=(1.0, 2.0),
        readout="pooled",
    )
    net = ScanNet(cfg, dtype=torch.float64)
    x = torch.randn(6, 2, dtype=torch.float64)
    u = torch.randn(6, 1, dtype=torch.float64)
    out = net(x, u)
    assert out.shape == (6, 2)
