# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Torch remainder-training twin (09-18)."""

from __future__ import annotations

import torch
from omnibias.torch.optim_remainder import remainder_loss, worked_example


def test_g1_g4() -> None:
    torch.set_default_dtype(torch.float64)
    ex = worked_example()
    assert ex["R2_err"] < 1e-12
    report = remainder_loss(
        torch.tensor([ex["exp"]]),
        torch.tensor([1.0, 1.0, 1.0]),
        torch.tensor([0.2]),
    )
    assert abs(float(report["max_abs"]) - ex["R2"]) < 1e-12
