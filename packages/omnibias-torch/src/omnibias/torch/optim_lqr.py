# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Finite-horizon jet-LQR optimizer (torch; theory 08-11).

Exact directional jets from founding bias collapse (``delta -> 0``).
Temperature collapse (``beta -> inf``, feasibility) does not
appear. do not conflate the two. Not a plant LQR. Not DARE.
Not activation Riccati. Not a global min.
"""

from __future__ import annotations

from collections.abc import Callable

from omnibias.core import control_lqr as core
from omnibias.torch.line_search import directional_derivatives

import torch
from torch import Tensor

DISCLAIMER = core.DISCLAIMER
JetLQRConfig = core.JetLQRConfig
JetLQRReport = core.JetLQRReport
honesty_payload = core.honesty_payload
lqr_step_from_derivatives = core.lqr_step_from_derivatives
scalar_finite_horizon_lqr = core.scalar_finite_horizon_lqr


def jet_lqr_step(
    loss_fn: Callable[[Tensor], Tensor],
    params: Tensor,
    direction: Tensor,
    *,
    config: JetLQRConfig | None = None,
    order: int = 2,
) -> JetLQRReport:
    """Parameter-direction jet-LQR step. Eager; do not ``torch.compile``."""
    if order < 2:
        raise ValueError(f"order must be >= 2, got {order}")
    derivs = directional_derivatives(loss_fn, params, direction, order)
    return core.lqr_step_from_derivatives(derivs, config=config)


def apply_lqr_step(
    params: Tensor, direction: Tensor, report: JetLQRReport
) -> Tensor:
    """``params + report.step * direction`` (same dtype / device)."""
    step = torch.as_tensor(report.step, dtype=params.dtype, device=params.device)
    return params + step * direction


def worked_example() -> dict[str, float | bool]:
    return core.worked_example()


__all__ = [
    "DISCLAIMER",
    "JetLQRConfig",
    "JetLQRReport",
    "apply_lqr_step",
    "honesty_payload",
    "jet_lqr_step",
    "lqr_step_from_derivatives",
    "scalar_finite_horizon_lqr",
    "worked_example",
]
