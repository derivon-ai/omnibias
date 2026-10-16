# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Every scalar-limit branch retains its input device and promoted dtype."""
from __future__ import annotations

import math

import pytest
import torch
from omnibias.torch.jet import limit_of_ratio


@pytest.mark.parametrize("device", [
    "cpu", pytest.param("cuda", marks=pytest.mark.skipif(
        not torch.cuda.is_available(), reason="a usable CUDA runtime is unavailable")),
])
@pytest.mark.parametrize("num,den,expected", [
    ([0, 0, 0], [0, 1, 0], 0.0),
    ([0, 0, 1], [0, 1, 0], 0.0),
    ([0, 1, 0], [0, 0, 1], math.inf),
    ([0, -1, 0], [0, 0, 1], -math.inf),
    ([0, 2, 0], [0, 1, 0], 2.0),
])
def test_limit_device_and_dtype(device, num, den, expected):
    numerator = torch.tensor(num, dtype=torch.float32, device=device)
    denominator = torch.tensor(den, dtype=torch.float64, device=device)
    result = limit_of_ratio(numerator, denominator)
    assert result.device == numerator.device
    assert result.dtype == torch.float64
    assert result.item() == expected
