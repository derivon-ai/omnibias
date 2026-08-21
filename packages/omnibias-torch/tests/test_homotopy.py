# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Torch homotopy twin (09-20)."""

from __future__ import annotations

import torch
from omnibias.torch.optim_homotopy import homotopy_step, worked_example


def test_g1() -> None:
    torch.set_default_dtype(torch.float64)
    ex = worked_example()
    assert ex["accepted"] is True
    assert float(ex["abs_h"]) < 1e-10
    trial, residual, decision = homotopy_step(torch.tensor(1.0), 0.1)
    assert decision.accepted is True
    assert residual < 1e-10
    assert abs(trial - float(ex["theta"])) < 1e-12
