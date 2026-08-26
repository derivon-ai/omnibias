# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Receding-horizon jet-MPC optimizer (jax; theory 08-12).

Bit-identical partner of :mod:`omnibias.torch.optim_mpc` on the same
directional derivatives. Exact jets from founding bias collapse
(``delta -> 0``). Temperature collapse (``beta -> inf``, feasibility)
does not appear. do not conflate the two. Not plant MPC. Not a
general QP. Not a global min.

The driver is eager. Do not ``jit`` it. Enable 64-bit JAX before the
first array if you need torch parity (:mod:`omnibias.jax.precision`).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from omnibias.core import control_mpc as core
from omnibias.jax.line_search import directional_derivatives

import jax.numpy as jnp
from jax import Array
from jax.flatten_util import ravel_pytree

DISCLAIMER = core.DISCLAIMER
JetMPCConfig = core.JetMPCConfig
JetMPCReport = core.JetMPCReport
honesty_payload = core.honesty_payload
mpc_step_from_derivatives = core.mpc_step_from_derivatives


def jet_mpc_step(
    loss_fn: Callable[[Any], Array],
    params: Any,
    direction: Any,
    *,
    config: JetMPCConfig | None = None,
    order: int = 2,
) -> JetMPCReport:
    """Parameter-direction jet-MPC step. Eager; do not ``jit``."""
    if order < 2:
        raise ValueError(f"order must be >= 2, got {order}")
    derivs = directional_derivatives(loss_fn, params, direction, order)
    return core.mpc_step_from_derivatives(derivs, config=config)


def apply_mpc_step(params: Any, direction: Any, report: JetMPCReport) -> Any:
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
    "JetMPCConfig",
    "JetMPCReport",
    "apply_mpc_step",
    "honesty_payload",
    "jet_mpc_step",
    "mpc_step_from_derivatives",
    "worked_example",
]
