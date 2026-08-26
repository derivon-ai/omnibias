# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Receding-horizon jet-MPC optimizer (torch; theory 08-12).

Exact directional jets from founding bias collapse (``delta -> 0``).
Temperature collapse (``beta -> inf``, feasibility) does not
appear. do not conflate the two. Not plant MPC. Not a general QP.
Not a global min.
"""

from __future__ import annotations

from collections.abc import Callable

from omnibias.core import control_mpc as core
from omnibias.torch.line_search import directional_derivatives

import torch
from torch import Tensor

DISCLAIMER = core.DISCLAIMER
JetMPCConfig = core.JetMPCConfig
JetMPCReport = core.JetMPCReport
honesty_payload = core.honesty_payload
mpc_step_from_derivatives = core.mpc_step_from_derivatives


def jet_mpc_step(
    loss_fn: Callable[[Tensor], Tensor],
    params: Tensor,
    direction: Tensor,
    *,
    config: JetMPCConfig | None = None,
    order: int = 2,
) -> JetMPCReport:
    """Parameter-direction jet-MPC step. Eager; do not ``torch.compile``."""
    if order < 2:
        raise ValueError(f"order must be >= 2, got {order}")
    derivs = directional_derivatives(loss_fn, params, direction, order)
    return core.mpc_step_from_derivatives(derivs, config=config)


def apply_mpc_step(
    params: Tensor, direction: Tensor, report: JetMPCReport
) -> Tensor:
    """``params + report.step * direction`` (same dtype / device)."""
    step = torch.as_tensor(report.step, dtype=params.dtype, device=params.device)
    return params + step * direction


def worked_example() -> dict[str, float | bool]:
    return core.worked_example()


__all__ = [
    "DISCLAIMER",
    "JetMPCConfig",
    "JetMPCReport",
    "apply_mpc_step",
    "honesty_payload",
    "jet_mpc_step",
    "mpc_step_from_derivatives",
    "worked_example",
]
