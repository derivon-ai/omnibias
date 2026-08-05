# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
r"""Differentiable TSP relaxation layers (torch) via the unrolled temperature-collapse penalty.

Bit-identical twin of :mod:`omnibias.routing.jax.relaxation` (float64); see that
module for the full math. One batched call; differentiable through ``autograd`` so
a cost model can be trained *through* the relaxation.

Terminology: "temperature-collapse penalty" here is the feasibility sense of
"collapse" (a hard-hinge constraint force), distinct from the
**founding bias collapse** (multi-bias ``delta -> 0`` limit to
``sigma^(K-1)``, a derivative; see :mod:`omnibias.torch.unit`).
"""

from __future__ import annotations

import math

import numpy as np
import torch
from omnibias.routing._core.relax_systems import RelaxSystem, build_system
from omnibias.routing.problem import RelaxationSchedule
from torch import Tensor


def _descent(
    cvec: Tensor,
    A_eq: Tensor,
    b_eq: Tensor,
    A_ineq: Tensor,
    b_ineq: Tensor,
    aeq2: float,
    ain2: float,
    sched: RelaxationSchedule,
) -> Tensor:
    reg = sched.reg
    steps = sched.steps

    def stage(x0: Tensor, mu: float) -> Tensor:
        eta = sched.step_safety / (reg + mu * (aeq2 + ain2) + 1e-30)
        x = x0
        y = x0
        t = 1.0
        for _ in range(steps):
            req = y @ A_eq.t() - b_eq
            rin = torch.clamp(y @ A_ineq.t() - b_ineq, min=0.0)
            grad = cvec + reg * y + mu * (req @ A_eq) + mu * (rin @ A_ineq)
            x_next = y - eta * grad
            t_next = 0.5 * (1.0 + math.sqrt(1.0 + 4.0 * t * t))
            y = x_next + ((t - 1.0) / t_next) * (x_next - x)
            x = x_next
            t = t_next
        return x

    x = torch.zeros_like(cvec)
    for mu in sched.mus():
        x = stage(x, mu)
    return x


def _relaxation(cost: object, kind: str, schedule: RelaxationSchedule | None) -> Tensor:
    sched = schedule or RelaxationSchedule()
    cost_t = torch.as_tensor(cost, dtype=torch.float64)
    single = cost_t.dim() == 2
    if single:
        cost_t = cost_t.unsqueeze(0)
    batch, n, _ = cost_t.shape
    sys: RelaxSystem = build_system(n, kind)
    arc_src = torch.as_tensor(np.array([i * n + j for (i, j) in sys.arcs]), dtype=torch.long)
    A_eq = torch.as_tensor(sys.A_eq, dtype=torch.float64)
    b_eq = torch.as_tensor(sys.b_eq, dtype=torch.float64)
    A_ineq = torch.as_tensor(sys.A_ineq, dtype=torch.float64)
    b_ineq = torch.as_tensor(sys.b_ineq, dtype=torch.float64)
    aeq2 = float(np.sum(sys.A_eq * sys.A_eq))
    ain2 = float(np.sum(sys.A_ineq * sys.A_ineq))

    flat = cost_t.reshape(batch, n * n)
    cx = flat[:, arc_src]
    cx = cx / (torch.mean(cx, dim=1, keepdim=True) + 1e-12)
    pad = sys.n_vars - sys.n_arcs
    cvec = cx if pad == 0 else torch.cat([cx, torch.zeros((batch, pad), dtype=torch.float64)], dim=1)

    x = _descent(cvec, A_eq, b_eq, A_ineq, b_ineq, aeq2, ain2, sched)
    arc = x[:, : sys.n_arcs]
    mat = torch.zeros((batch, n * n), dtype=torch.float64)
    mat = mat.index_copy(1, arc_src, arc)
    mat = mat.reshape(batch, n, n)
    return mat.squeeze(0) if single else mat


def assignment_relaxation(cost: object, schedule: RelaxationSchedule | None = None) -> Tensor:
    """Differentiable degree-constrained (assignment) relaxation -> arc-use ``(n, n)``."""
    return _relaxation(cost, "assignment", schedule)


def flow_relaxation(cost: object, schedule: RelaxationSchedule | None = None) -> Tensor:
    """Differentiable single-commodity-flow (subtour-free) relaxation -> arc-use."""
    return _relaxation(cost, "flow", schedule)


def held_karp_layer(cost: object, schedule: RelaxationSchedule | None = None) -> Tensor:
    r"""Differentiable multicommodity-flow (Held-Karp) relaxation -> arc-use (small ``n``)."""
    return _relaxation(cost, "held_karp", schedule)


__all__ = ["assignment_relaxation", "flow_relaxation", "held_karp_layer"]
