# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Torch inverse-design twin (09-22)."""

from __future__ import annotations

import torch
from omnibias.torch.optim_inverse import invert_input, worked_example


def test_g1() -> None:
    torch.set_default_dtype(torch.float64)
    ex = worked_example()
    assert ex["residual"] < 1e-12
    report = invert_input(None, torch.tensor(0.5), torch.tensor(0.0))
    assert abs(report.residual) < 1e-12
    assert abs(report.x - ex["x"]) < 1e-12
