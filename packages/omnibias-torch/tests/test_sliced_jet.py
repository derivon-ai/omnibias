# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Torch twin for theory 09-28."""

from __future__ import annotations

import torch
from omnibias.torch.architectures.sliced_jet import SlicedJetEncoder, worked_example


def test_g1_torch() -> None:
    torch.set_default_dtype(torch.float64)
    ex = worked_example()
    assert ex["encoder_mae"] == 0.0
    image = torch.tensor([[1.0, 0.0], [0.0, 0.0]])
    recon = SlicedJetEncoder().reconstruct(image)
    assert float((recon - image).abs().mean().detach()) == 0.0
