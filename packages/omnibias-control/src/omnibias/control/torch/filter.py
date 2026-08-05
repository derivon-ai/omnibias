# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
r"""Differentiable, batched, per-sample CBF-QP safety filter (torch).

Bit-identical twin of :mod:`omnibias.control.jax.filter`; see it for the derivation.

Terminology: the hinge belongs to **temperature collapse** -- the ``beta -> inf``
axis that sharpens one gate into a 0/1 feasibility indicator. That is a different
limit from the **founding bias collapse** (multi-bias ``delta -> 0`` limit to
``sigma^(K-1)``, a derivative; see :mod:`omnibias.torch.unit`). Note this filter
takes the hard clamp directly rather than annealing, so it sits at the
``beta = inf`` endpoint of that axis.
"""

from __future__ import annotations

import torch
from omnibias.control.problem import FilterSchedule, SafeAction
from torch import Tensor


def cbf_residual(G: Tensor, h: Tensor, a: Tensor) -> Tensor:
    r"""Per-sample worst constraint residual ``max_i (G_i a - h_i)`` (``(B,)``)."""
    return torch.max(torch.einsum("bmd,bd->bm", G, a) - h, dim=1).values


def cbf_filter(
    a_nom: Tensor, G: Tensor, h: Tensor, schedule: FilterSchedule | None = None
) -> Tensor:
    r"""Project ``a_nom`` onto the per-sample polytope ``{a : G_i a <= h_i}``.

    Hard-hinge exterior penalty minimised by accelerated (Nesterov) gradient descent
    over a short ``mu`` homotopy; differentiable, batched, ``vmap``-free. See the JAX
    twin :func:`omnibias.control.jax.filter.cbf_filter` for the full description.
    """
    sched = schedule if schedule is not None else FilterSchedule()
    a2 = torch.sum(G * G, dim=(1, 2))                      # (B,) per-sample ||G||_F^2

    def descent(x0: Tensor, mu: float, steps: int) -> Tensor:
        eta = sched.safety / (1.0 + mu * a2 + 1e-30)       # (B,)
        x = x0
        y = x0
        t = torch.ones((), dtype=a_nom.dtype, device=a_nom.device)
        for _ in range(steps):
            u = torch.einsum("bmd,bd->bm", G, y) - h        # (B, m)
            gate = torch.clamp(u, min=0.0)                  # hard temperature-collapse unit
            grad = (y - a_nom) + mu * torch.einsum("bmd,bm->bd", G, gate)
            x_next = y - eta[:, None] * grad
            t_next = 0.5 * (1.0 + torch.sqrt(1.0 + 4.0 * t * t))
            y = x_next + ((t - 1.0) / t_next) * (x_next - x)
            x = x_next
            t = t_next
        return x

    x = a_nom
    mu = sched.mu0
    for _ in range(sched.stages):
        x = descent(x, mu, sched.steps)
        mu *= sched.mu_growth
    return x


def filter_action(
    a_nom: Tensor, G: Tensor, h: Tensor, schedule: FilterSchedule | None = None
) -> SafeAction[Tensor]:
    """:func:`cbf_filter` wrapped in a :class:`SafeAction` with the residual diagnostic."""
    a = cbf_filter(a_nom, G, h, schedule)
    return SafeAction(action=a, nominal=a_nom, residual=cbf_residual(G, h, a))


__all__ = ["cbf_filter", "cbf_residual", "filter_action"]
