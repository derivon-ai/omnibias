# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Jet-PID optimizer (torch; theory 08-10).

Exact directional jets from founding bias collapse (``delta -> 0``).
Temperature collapse (``beta -> inf``, feasibility) does not
appear. do not conflate the two. Not a plant PID. Not a global min.
"""

from __future__ import annotations

from collections.abc import Callable

from omnibias.core import control_pid as core
from omnibias.torch.line_search import directional_derivatives

import torch
from torch import Tensor

DISCLAIMER = core.DISCLAIMER
JetPIDConfig = core.JetPIDConfig
JetPIDReport = core.JetPIDReport
honesty_payload = core.honesty_payload
pid_step_from_derivatives = core.pid_step_from_derivatives


def jet_pid_step(
    loss_fn: Callable[[Tensor], Tensor],
    params: Tensor,
    direction: Tensor,
    *,
    config: JetPIDConfig | None = None,
    running_integral: float = 0.0,
    order: int | None = None,
) -> JetPIDReport:
    """Parameter-direction jet-PID step. Eager; do not ``torch.compile``."""
    cfg = core.DEFAULT_CONFIG if config is None else config
    need = 2 if cfg.kd != 0.0 else 1
    jet_order = need if order is None else order
    if jet_order < need:
        raise ValueError(f"order must be >= {need} for these gains, got {jet_order}")
    derivs = directional_derivatives(loss_fn, params, direction, jet_order)
    return core.pid_step_from_derivatives(
        derivs, config=cfg, running_integral=running_integral
    )


def apply_pid_step(
    params: Tensor, direction: Tensor, report: JetPIDReport
) -> Tensor:
    """``params + report.step * direction`` (same dtype / device)."""
    step = torch.as_tensor(report.step, dtype=params.dtype, device=params.device)
    return params + step * direction


def worked_example() -> dict[str, float | bool]:
    return core.worked_example()


__all__ = [
    "DISCLAIMER",
    "JetPIDConfig",
    "JetPIDReport",
    "apply_pid_step",
    "honesty_payload",
    "jet_pid_step",
    "pid_step_from_derivatives",
    "worked_example",
]
