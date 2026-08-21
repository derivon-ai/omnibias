# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Remainder training (torch; theory 09-18).

The loss is ``R_N``, not spec 03-10 or 03-13. Jets use founding bias
collapse (``delta -> 0``). Temperature collapse (``beta -> inf``,
feasibility) does not appear. Do not conflate the two.
"""

from __future__ import annotations

from collections.abc import Sequence

from omnibias.core import remainder_train as core

import torch
from torch import Tensor

DISCLAIMER = core.DISCLAIMER
RemainderTrainConfig = core.RemainderTrainConfig
honesty_payload = core.honesty_payload


def remainder_loss(
    values: Sequence[float] | Tensor,
    jet: Sequence[float] | Tensor,
    xs: Sequence[float] | Tensor,
    x0: float = 0.0,
    *,
    config: RemainderTrainConfig | None = None,
) -> dict[str, float | bool | int]:
    vs = [float(v) for v in torch.as_tensor(values).reshape(-1).tolist()]
    js = [float(v) for v in torch.as_tensor(jet).reshape(-1).tolist()]
    pts = [float(v) for v in torch.as_tensor(xs).reshape(-1).tolist()]
    return core.remainder_loss(vs, js, pts, x0, config=config)


def worked_example() -> dict[str, float]:
    return core.worked_example()


__all__ = [
    "DISCLAIMER",
    "RemainderTrainConfig",
    "honesty_payload",
    "remainder_loss",
    "worked_example",
]
