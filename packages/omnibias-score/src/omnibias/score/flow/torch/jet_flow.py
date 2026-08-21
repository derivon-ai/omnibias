# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Coupling jet-flow (torch; theory 09-06).

Closed-form ``log|det|`` from ``sigma'`` (founding bias collapse
``delta -> 0``). Temperature collapse (``beta -> inf``, feasibility)
does not appear. Do not conflate the two. Not ``integrate_cnf``.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from omnibias.core import coupling_flow as core
from torch import Tensor

DISCLAIMER = core.DISCLAIMER
JetFlowConfig = core.JetFlowConfig
honesty_payload = core.honesty_payload


def jet_flow_forward(x: Tensor, scale: Tensor, shift: Tensor | None = None) -> tuple[Tensor, Tensor]:
    """Elementwise ``tanh(s x + t)`` and closed-form log-det."""
    t = torch.zeros((), dtype=x.dtype, device=x.device) if shift is None else shift
    pre = scale * x + t
    y = torch.tanh(pre)
    log_det = torch.log(scale.abs() * (1.0 - y * y))
    return y, log_det


def jet_flow_inverse(
    y: Tensor,
    scale: Tensor,
    shift: Tensor | None = None,
    *,
    config: JetFlowConfig | None = None,
    tol: float = 1e-10,
) -> Tensor:
    cfg = core.DEFAULT_CONFIG if config is None else config
    t = 0.0 if shift is None else float(shift.reshape(-1)[0])
    s = float(scale.reshape(-1)[0])
    vals = [float(v) for v in y.reshape(-1).tolist()]
    inv = [core.jet_flow_inverse_1d(v, s, t, config=cfg) for v in vals]
    out = y.new_tensor(inv)
    _ = tol
    return out.reshape(y.shape)


def worked_example() -> dict[str, float]:
    return core.worked_example()


def jet_flow_forward_params(
    x: Sequence[float],
    params: Sequence[tuple[float, float]],
    *,
    config: JetFlowConfig | None = None,
) -> tuple[list[float], float]:
    return core.jet_flow_forward(x, params, config=config)


__all__ = [
    "DISCLAIMER",
    "JetFlowConfig",
    "honesty_payload",
    "jet_flow_forward",
    "jet_flow_forward_params",
    "jet_flow_inverse",
    "worked_example",
]
