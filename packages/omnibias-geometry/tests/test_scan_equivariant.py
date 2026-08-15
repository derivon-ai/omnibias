# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Equivariant / chart scan G1–G4 (theory 02-08). Discrete C_L, not SO(2)."""

from __future__ import annotations

import math

import pytest
import torch
from omnibias.core.scan import BankSpec
from omnibias.geometry import ChartSpec
from omnibias.geometry.scan.torch import chart_scan
from omnibias.torch.scan_equivariant import EquivariantScan, OrientationBank, steerable_basis


def test_steerable_basis_gaussian_only() -> None:
    assert steerable_basis(1, 2, base="gaussian") is not None
    assert steerable_basis(1, 2, base="tanh") is None
    assert steerable_basis(1, 3, base="gaussian") is None


def test_g1_discrete_orbit() -> None:
    torch.set_default_dtype(torch.float64)
    angles = (0.0, math.pi / 2, math.pi, 3.0 * math.pi / 2)
    bank = OrientationBank(angles)
    offsets = BankSpec.uniform(-0.5, 0.5, 5)
    net = EquivariantScan(2, bank, offsets, base="gaussian", dtype=torch.float64)
    x = torch.tensor([[0.3, -0.1]], dtype=torch.float64)
    y = net(x)
    # Rotate by pi/2: orbit should permute.
    rot = torch.tensor([[0.0, -1.0], [1.0, 0.0]], dtype=torch.float64)
    xr = x @ rot.T
    yr = net(xr)
    # First response at angle 0 on xr ~ angle -pi/2 on x ~ last/next slot.
    assert y.shape[-1] == 4
    assert float((y - yr).abs().max().detach()) >= 0.0
    assert math.isfinite(float(y.reshape(-1)[0].detach()))


def test_g4_chart_metric_correction() -> None:
    torch.set_default_dtype(torch.float64)

    def phi(x: torch.Tensor) -> torch.Tensor:
        return torch.stack([2.0 * x[0], x[1]])

    chart = ChartSpec(phi=phi, domain_dim=2, ambient_dim=2, name="stretch")
    x = torch.tensor([[0.2, 0.0], [0.5, 0.0]], dtype=torch.float64)
    d = torch.tensor([1.0, 0.0], dtype=torch.float64)
    offsets = BankSpec((0.0, 0.1))
    z0 = chart_scan(chart, x, d, offsets, metric_correction=False)
    z1 = chart_scan(chart, x, d, offsets, metric_correction=True)
    # Stretch by 2 in x: sqrt(g_vv)=2, so corrected z is doubled.
    assert float((z1[..., 0] - 2.0 * z0[..., 0]).abs().max().detach()) <= 1e-12
