# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Torch integral-kernel twin (09-14)."""

from __future__ import annotations

import torch
from omnibias.core.integral_kernel import integral_cell as core_cell
from omnibias.torch.architectures.integral_kernel import integral_cell, worked_example
from omnibias.torch.blocks.operator import OperatorBlock


def test_g1_matches_operator_block() -> None:
    torch.set_default_dtype(torch.float64)
    ex = worked_example()
    assert ex["g1_err"] < 1e-12
    block = OperatorBlock(op="integral", base="sigmoid", channels=1, init_delta=1.0).double()
    z = torch.tensor([[0.5]])
    out = float(block(z).detach().reshape(-1)[0])
    cell = float(integral_cell(z.reshape(-1), -0.5, 0.5).detach()[0])
    assert abs(out - cell) < 1e-12
    assert abs(out - core_cell(0.5, -0.5, 0.5)) < 1e-12
