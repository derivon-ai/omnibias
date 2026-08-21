# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Exact-MAML (torch; theory 09-16).

Inner step is Newton / GN. Meta-grad is IFT on inner stationarity.
That is the chain rule. The founding bias collapse (``delta -> 0``)
supplies exact HVPs. Temperature collapse (``beta -> inf``,
feasibility) does not appear. Do not conflate the two.
"""

from __future__ import annotations

from collections.abc import Sequence

from omnibias.core import exact_maml as core

import torch
from torch import Tensor

DISCLAIMER = core.DISCLAIMER
ExactMAMLConfig = core.ExactMAMLConfig
honesty_payload = core.honesty_payload


def inner_newton_quadratic(theta: Tensor, alpha: Tensor) -> Tensor:
    """One Newton step on ``0.5 (theta - alpha)^2``."""
    return theta - (theta - alpha)


def ift_dtheta_dalpha(theta: Tensor, alpha: Tensor) -> Tensor:
    """``d theta / d alpha`` from ``G = theta - alpha = 0``."""
    _ = (theta, alpha)
    return torch.ones((), dtype=theta.dtype, device=theta.device)


def exact_maml_meta_step(
    tasks: Sequence[float] | Tensor,
    params: Tensor,
    *,
    config: ExactMAMLConfig | None = None,
) -> dict[str, float]:
    alphas = (
        [float(a) for a in params.new_tensor(tasks).reshape(-1).tolist()]
        if not isinstance(tasks, Tensor)
        else [float(a) for a in tasks.reshape(-1).tolist()]
    )
    return core.exact_maml_meta_step(alphas, float(params.reshape(-1)[0]), config=config)


def quadratic_worked_example() -> dict[str, float]:
    return core.quadratic_worked_example()


__all__ = [
    "DISCLAIMER",
    "ExactMAMLConfig",
    "exact_maml_meta_step",
    "honesty_payload",
    "ift_dtheta_dalpha",
    "inner_newton_quadratic",
    "quadratic_worked_example",
]
