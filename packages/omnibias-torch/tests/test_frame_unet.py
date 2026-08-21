# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Torch Frame-UNet twin (09-04)."""

from __future__ import annotations

import torch
from omnibias.torch.architectures.frame_unet import frame_unet_forward, worked_example


def test_g1_g4() -> None:
    torch.set_default_dtype(torch.float64)
    ex = worked_example()
    assert ex["band_err"] < 1e-12
    assert ex["collapse_err"] < 1e-12
    assert ex["skip_gap"] > 1e-3
    y, skips = frame_unet_forward(torch.tensor(0.0))
    assert y.ndim == 0
    assert skips["kinds"] == ("band", "collapse")
