# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""08-01 recommended stack, PyTorch twin.

Host-side conversion into :mod:`omnibias.core.train_stack`. Bit-identical
to the jax twin on the same floats. Do not wrap the driver in
``torch.compile``.
"""

from __future__ import annotations

from omnibias.core import train_stack as core

import torch
from torch import Tensor

TrainStackConfig = core.TrainStackConfig
TrainStackReport = core.TrainStackReport
honesty_payload = core.honesty_payload

__all__ = [
    "TrainStackConfig",
    "TrainStackReport",
    "honesty_payload",
    "recommended_stack_step",
    "stack_minimize",
]


def _rows(matrix: Tensor) -> list[list[float]]:
    data = torch.as_tensor(matrix).detach().reshape(-1, matrix.shape[-1])
    return [[float(v) for v in row] for row in data]


def _vec(vector: Tensor) -> list[float]:
    return [float(v) for v in torch.as_tensor(vector).detach().reshape(-1)]


def recommended_stack_step(
    features: Tensor,
    targets: Tensor,
    params: Tensor,
    hidden: int,
    dim: int,
    *,
    config: core.TrainStackConfig | None = None,
) -> tuple[list[float], core.TrainStackReport]:
    return core.recommended_stack_step(
        _rows(features),
        _vec(targets),
        _vec(params),
        hidden,
        dim,
        config=config,
    )


def stack_minimize(
    features: Tensor,
    targets: Tensor,
    params: Tensor,
    hidden: int,
    dim: int,
    *,
    steps: int = 8,
    config: core.TrainStackConfig | None = None,
) -> tuple[list[float], list[core.TrainStackReport]]:
    return core.stack_minimize(
        _rows(features),
        _vec(targets),
        _vec(params),
        hidden,
        dim,
        steps=steps,
        config=config,
    )
