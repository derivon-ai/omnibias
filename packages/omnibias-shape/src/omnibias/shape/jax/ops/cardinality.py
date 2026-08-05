# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Cardinality / count surrogates and gate lifecycle helpers (jax).

Bit-identical algorithm to :mod:`omnibias.shape.torch.ops.cardinality`.
"""

from __future__ import annotations

import math

import jax.numpy as jnp
from jax import Array, nn

__all__ = ["anneal_lambda", "l0_surrogate", "prune_inactive"]

_SCHEDULES = ("linear", "exp", "cosine")


def l0_surrogate(gates: Array, *, kind: str = "sum", eps: float = 1e-3) -> Array:
    r"""Differentiable "number of active shapes" surrogate over ``gates`` in ``[0, 1]``."""
    if kind == "sum":
        return jnp.sum(gates)
    if kind == "concave":
        return jnp.sum(gates / (gates + eps))
    raise ValueError(f"kind must be 'sum' or 'concave', got {kind!r}")


def anneal_lambda(
    step: int,
    *,
    lam_start: float,
    lam_end: float,
    num_steps: int,
    schedule: str = "linear",
) -> float:
    r"""Count-penalty schedule (companion to the ``beta`` sharpness anneal)."""
    if num_steps < 1:
        raise ValueError(f"num_steps must be >= 1, got {num_steps}")
    if schedule not in _SCHEDULES:
        raise ValueError(f"unknown schedule {schedule!r}; choose from {_SCHEDULES}")
    frac = min(max(step / num_steps, 0.0), 1.0)
    if schedule == "linear":
        return lam_start + frac * (lam_end - lam_start)
    if schedule == "exp":
        if lam_start <= 0.0 or lam_end <= 0.0:
            raise ValueError("exp schedule needs lam_start, lam_end > 0")
        return math.exp(math.log(lam_start) + frac * (math.log(lam_end) - math.log(lam_start)))
    ease = 0.5 * (1.0 - math.cos(math.pi * frac))
    return lam_start + ease * (lam_end - lam_start)


def prune_inactive(
    centers: Array, gate_logits: Array, *, threshold: float = 0.5
) -> tuple[Array, Array]:
    r"""Drop shapes whose gate ``sigmoid(logit)`` has collapsed below ``threshold``."""
    keep = nn.sigmoid(gate_logits) >= threshold
    if not bool(jnp.any(keep)):
        keep = jnp.zeros_like(keep).at[int(jnp.argmax(gate_logits))].set(True)
    return centers[keep], gate_logits[keep]
