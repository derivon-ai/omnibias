# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
r"""Differentiable safe closed-loop rollout (torch). Bit-identical twin."""

from __future__ import annotations

from collections.abc import Callable
from typing import cast

import torch
from omnibias.control.problem import FilterSchedule
from omnibias.control.torch.filter import cbf_filter, cbf_residual
from torch import Tensor
from torch.func import vmap


def safe_rollout(
    policy: Callable[[Tensor], Tensor],
    step: Callable[[Tensor, Tensor], Tensor],
    rows_fn: Callable[[Tensor], tuple[Tensor, Tensor]],
    x0: Tensor,
    *,
    horizon: int,
    schedule: FilterSchedule | None = None,
) -> tuple[Tensor, Tensor, Tensor]:
    r"""Roll the safe closed loop ``horizon`` steps (differentiable).

    See the JAX twin :func:`omnibias.control.jax.rollout.safe_rollout`. Returns
    ``(X (T,B,n), A (T,B,d), residual (T,B))``.
    """
    x = x0
    states: list[Tensor] = []
    actions: list[Tensor] = []
    resids: list[Tensor] = []
    for _ in range(horizon):
        a_nom = policy(x)
        G, h = rows_fn(x)
        a = cbf_filter(a_nom, G, h, schedule)
        x = step(x, a)
        states.append(x)
        actions.append(a)
        resids.append(cbf_residual(G, h, a))
    return torch.stack(states), torch.stack(actions), torch.stack(resids)


def barrier_trace(barrier: Callable[[Tensor], Tensor], X: Tensor) -> Tensor:
    r"""Barrier value ``h(x)`` at every ``(t, sample)`` of a rollout, shape ``(T, B)``."""
    return cast(Tensor, vmap(vmap(barrier))(X))


def min_barrier(barrier: Callable[[Tensor], Tensor], X: Tensor) -> Tensor:
    r"""Worst (minimum-over-time) barrier per sample, shape ``(B,)`` (safe iff ``>= 0``)."""
    return barrier_trace(barrier, X).min(dim=0).values


__all__ = ["barrier_trace", "min_barrier", "safe_rollout"]
