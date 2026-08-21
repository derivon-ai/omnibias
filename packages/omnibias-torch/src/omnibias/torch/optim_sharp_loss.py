# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Sharpness-augmented loss (torch; theory 09-23).

``L + mu * ritz``. Exact HVPs from founding bias collapse
(``delta -> 0``). Temperature collapse (``beta -> inf``,
feasibility) does not appear. do not conflate the two. Not 08-06.
"""

from __future__ import annotations

from collections.abc import Callable

from omnibias.core import sharp_loss as core

import torch
from torch import Tensor

DISCLAIMER = core.DISCLAIMER
SharpnessLossConfig = core.SharpnessLossConfig
honesty_payload = core.honesty_payload

ScalarFn = Callable[[Tensor], Tensor]


def exact_trace(loss_fn: ScalarFn, params: Tensor) -> float:
    """Exact ``Tr H`` from unit HVPs, not Hutchinson."""
    from omnibias.torch.optim import hvp

    p = params.reshape(-1)
    total = 0.0
    for i in range(int(p.numel())):
        eye = torch.zeros_like(p)
        eye[i] = 1.0
        total += float(hvp(loss_fn, p, eye).reshape(-1)[i])
    return total


def sharpness_augmented_loss(
    loss_fn: ScalarFn | None,
    theta: Tensor | float,
    *,
    config: SharpnessLossConfig | None = None,
) -> float:
    """``L + mu * ritz``. Artifact ``schedule_only`` is false."""
    cfg = core.DEFAULT_CONFIG if config is None else config
    if loss_fn is None:
        start = float(theta.reshape(-1)[0].detach()) if isinstance(theta, Tensor) else float(theta)
        return core.sharpness_augmented_loss(None, start, config=cfg)
    p = theta if isinstance(theta, Tensor) else torch.as_tensor([float(theta)])
    p = p.reshape(-1)
    raw = float(loss_fn(p).detach())
    if cfg.kind == "trace":
        ritz = exact_trace(loss_fn, p)
    else:
        from omnibias.torch.optim_sharpness import sharpness_lambda_max

        ritz = sharpness_lambda_max(loss_fn, p, n_lanczos=cfg.lanczos_k)
    return raw + float(cfg.mu) * ritz


def worked_example() -> dict[str, float]:
    return core.worked_example()


__all__ = [
    "DISCLAIMER",
    "SharpnessLossConfig",
    "exact_trace",
    "honesty_payload",
    "sharpness_augmented_loss",
    "worked_example",
]
