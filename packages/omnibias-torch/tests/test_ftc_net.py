# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Torch FTC-Net twin (theory 09-03 / 09-17)."""

from __future__ import annotations

import torch
from omnibias.core.ftc import worked_example
from omnibias.torch.architectures.ftc_net import FTCNet, FTCNetConfig, dual_ftc_loss, ftc_block


def test_g1_and_g4_matches_core() -> None:
    torch.set_default_dtype(torch.float64)
    x = torch.tensor(0.0)
    w = torch.tensor(1.0)
    lo = torch.tensor(-0.1)
    hi = torch.tensor(0.1)
    integral, deriv, collapse = ftc_block(x, w, lo, hi)
    ex = worked_example()
    assert float(integral) == ex["I"]
    assert float(deriv) == ex["dI"]
    assert float(collapse) == ex["collapse"]
    net = FTCNet(FTCNetConfig())
    assert net.config.width == 12
    loss = dual_ftc_loss([float(integral)], [float(deriv)], [float(deriv)], [0.0], I_a=float(integral))
    assert loss.max_r_D < 1e-12
