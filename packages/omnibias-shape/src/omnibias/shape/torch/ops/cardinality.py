# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Cardinality / count surrogates and gate lifecycle helpers (torch).

The "number of active shapes" is relaxed to a differentiable surrogate over the gates,
annealed alongside the sharpness ``beta`` (see ``omnibias.binary.BetaAnnealScheduler``)
so shapes are removed only once the coverage constraint is (softly) satisfied.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor

__all__ = ["anneal_lambda", "l0_surrogate", "prune_inactive"]

_SCHEDULES = ("linear", "exp", "cosine")


def l0_surrogate(gates: Tensor, *, kind: str = "sum", eps: float = 1e-3) -> Tensor:
    r"""Differentiable "number of active shapes" surrogate over ``gates`` in ``[0, 1]``.

    ``"sum"`` is ``sum(gates)`` (the convex L1-style relaxation); ``"concave"`` is the
    sharper ``sum(gates / (gates + eps))`` (a smooth L0 surrogate that saturates to 1 for
    any active gate).
    """
    if kind == "sum":
        return gates.sum()
    if kind == "concave":
        return (gates / (gates + eps)).sum()
    raise ValueError(f"kind must be 'sum' or 'concave', got {kind!r}")


def anneal_lambda(
    step: int,
    *,
    lam_start: float,
    lam_end: float,
    num_steps: int,
    schedule: str = "linear",
) -> float:
    r"""Count-penalty schedule (companion to the ``beta`` sharpness anneal).

    Grows (or decays) ``lambda`` from ``lam_start`` to ``lam_end`` over ``num_steps``,
    clamped to the endpoints outside ``[0, num_steps]``.
    """
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
    centers: Tensor, gate_logits: Tensor, *, threshold: float = 0.5
) -> tuple[Tensor, Tensor]:
    r"""Drop shapes whose gate ``sigmoid(logit)`` has collapsed below ``threshold``.

    Returns the pruned ``(centers, gate_logits)`` for a smaller subsequent solve. Keeps at
    least one shape so downstream solves are never degenerate.
    """
    keep = torch.sigmoid(gate_logits) >= threshold
    if not bool(keep.any()):
        keep = torch.zeros_like(keep)
        keep[int(torch.argmax(gate_logits))] = True
    return centers[keep], gate_logits[keep]
