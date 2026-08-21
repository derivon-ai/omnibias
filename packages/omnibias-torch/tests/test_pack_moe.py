# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Torch Pack-MoE twin (09-07)."""

from __future__ import annotations

import torch
from omnibias.core.pack_moe import ExpertWindow, PackMoEConfig
from omnibias.torch.architectures.pack_moe import pack_moe_forward, worked_example


def test_g1_g4_worked() -> None:
    torch.set_default_dtype(torch.float64)
    y = pack_moe_forward(
        torch.tensor(0.0),
        (1.0, 3.0),
        (ExpertWindow(-0.2, 0.0), ExpertWindow(0.0, 0.2)),
        config=PackMoEConfig(router="band"),
    )
    assert abs(float(y) - 2.0) < 1e-12
    assert worked_example()["g_a_err"] < 1e-12
