# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Torch Riccati-flow twin (09-10)."""

from __future__ import annotations

import torch
from omnibias.core.riccati_flow import RiccatiFlowConfig
from omnibias.torch.architectures.riccati_flow import riccati_flow, worked_example


def test_g1() -> None:
    torch.set_default_dtype(torch.float64)
    ex = worked_example()
    assert ex["err"] < 1e-12
    y = riccati_flow(torch.tensor(0.25), config=RiccatiFlowConfig(t=1.0))
    assert abs(float(y) - ex["s"]) < 1e-12
