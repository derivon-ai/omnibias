# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Torch sharpness-loss twin (09-23)."""

from __future__ import annotations

import torch
from omnibias.torch.optim_sharp_loss import sharpness_augmented_loss, worked_example


def test_g1() -> None:
    torch.set_default_dtype(torch.float64)
    ex = worked_example()
    assert ex["hvp"] == 10.0
    assert ex["aug"] == 1.0
    assert sharpness_augmented_loss(None, torch.tensor(0.0)) == 1.0

    def loss_fn(params: torch.Tensor) -> torch.Tensor:
        return 5.0 * params.reshape(-1)[0] ** 2

    aug = sharpness_augmented_loss(loss_fn, torch.tensor([0.0]))
    assert abs(aug - 1.0) < 1e-12
