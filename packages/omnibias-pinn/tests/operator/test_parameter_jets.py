# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Operator twins for theory 09-27."""

from __future__ import annotations

import torch
from omnibias.pinn.operator.torch.parameter_jets import mixed_jet, worked_example


def test_g1_torch() -> None:
    torch.set_default_dtype(torch.float64)
    ex = worked_example()
    assert ex["abs_sum"] < 1e-8
    du = mixed_jet(None, torch.tensor([ex["x"], ex["t"]]), torch.tensor([ex["mu"]]))
    assert abs(du + ex["u"]) < 1e-8
