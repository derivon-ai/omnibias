# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Torch Collapse-Net twin (09-11)."""

from __future__ import annotations

import torch
from omnibias.core.collapse_net import CollapseNetConfig
from omnibias.torch.architectures.collapse_net import (
    collapse_net_forward,
    collapse_remainder,
    worked_example,
)


def test_g1() -> None:
    torch.set_default_dtype(torch.float64)
    ex = worked_example()
    assert ex["rel_err"] < 1e-6
    cfg = CollapseNetConfig(order=1, delta=0.1, mode="collapsed")
    y = collapse_net_forward(torch.tensor(0.0), config=cfg)
    assert abs(float(y) - 0.25) < 1e-12
    rem = collapse_remainder(torch.tensor(0.0), config=cfg)
    assert abs(float(rem) - ex["remainder"]) < 1e-15
