# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""PirateNet identity skip (torch)."""

from __future__ import annotations

import torch
from omnibias.torch.architectures.piratenet import (
    PirateNet,
    PirateNetConfig,
    init_pirate_params,
    pirate_features,
)


def test_alpha_zero_is_embedding() -> None:
    cfg = PirateNetConfig(in_dim=2, hidden=6, n_layers=3, seed=1)
    params = init_pirate_params(cfg)
    assert float(torch.max(torch.abs(params["alpha"]))) == 0.0
    x = torch.linspace(-1.0, 1.0, 10, dtype=torch.float64)
    coords = torch.stack([x, x * x], dim=-1)
    feat = pirate_features(params, coords)
    emb = torch.tanh(coords @ params["We"].T + params["be"])
    assert float(torch.max(torch.abs(feat - emb))) < 1e-14


def test_module_forward_shape() -> None:
    net = PirateNet(PirateNetConfig(in_dim=1, hidden=5, n_layers=1, seed=0))
    y = net(torch.linspace(-1.0, 1.0, 7, dtype=torch.float64)[:, None])
    assert y.shape == (7,)
