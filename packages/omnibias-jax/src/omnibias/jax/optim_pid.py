# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Jet-PID optimizer (jax; theory 08-10).

Bit-identical partner of :mod:`omnibias.torch.optim_pid` on the same
directional derivatives. Exact jets from founding bias collapse
(``delta -> 0``). Temperature collapse (``beta -> inf``, feasibility)
does not appear. do not conflate the two. Not a plant PID. Not a
global min.

The driver is eager (host-side PID). Do not ``jit`` it. Enable 64-bit
JAX before the first array if you need torch parity
(:mod:`omnibias.jax.precision`).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from omnibias.core import control_pid as core
from omnibias.jax.line_search import directional_derivatives

import jax.numpy as jnp
from jax import Array
from jax.flatten_util import ravel_pytree

DISCLAIMER = core.DISCLAIMER
JetPIDConfig = core.JetPIDConfig
JetPIDReport = core.JetPIDReport
honesty_payload = core.honesty_payload
pid_step_from_derivatives = core.pid_step_from_derivatives


def jet_pid_step(
    loss_fn: Callable[[Any], Array],
    params: Any,
    direction: Any,
    *,
    config: JetPIDConfig | None = None,
    running_integral: float = 0.0,
    order: int | None = None,
) -> JetPIDReport:
    """Parameter-direction jet-PID step. Eager; do not ``jit``."""
    cfg = core.DEFAULT_CONFIG if config is None else config
    need = 2 if cfg.kd != 0.0 else 1
    jet_order = need if order is None else order
    if jet_order < need:
        raise ValueError(f"order must be >= {need} for these gains, got {jet_order}")
    derivs = directional_derivatives(loss_fn, params, direction, jet_order)
    return core.pid_step_from_derivatives(
        derivs, config=cfg, running_integral=running_integral
    )


def apply_pid_step(params: Any, direction: Any, report: JetPIDReport) -> Any:
    """``params + report.step * direction`` on the ravelled pair."""
    theta, unravel = ravel_pytree(params)
    direction_flat, _ = ravel_pytree(direction)
    if theta.shape != direction_flat.shape:
        raise ValueError(
            f"params and direction must ravel to the same shape, got {tuple(theta.shape)} "
            f"vs {tuple(direction_flat.shape)}"
        )
    stepped = theta + jnp.asarray(report.step, dtype=theta.dtype) * direction_flat
    return unravel(stepped)


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
