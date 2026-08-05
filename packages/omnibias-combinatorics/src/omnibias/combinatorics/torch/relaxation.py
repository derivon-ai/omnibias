# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
r"""Differentiable entropic / Sinkhorn relaxation layers (torch), annealed ``beta -> inf``.

Bit-identical twin of :mod:`omnibias.combinatorics.jax.relaxation` (float64). Each layer
relaxes a combinatorial problem onto its polytope with an entropic regulariser and anneals
the inverse temperature ``beta`` up a geometric :class:`~omnibias.discrete.AnnealSchedule`
homotopy; as ``beta -> inf`` the soft point collapses onto a **polytope vertex**. The
maps are differentiable through ``autograd``, so a model predicting the costs / weights
trains *through* the layer. The Sinkhorn (assignment) and soft-top-k (uniform / partition
matroid) kernels are **reused** from :mod:`omnibias.graph.torch.ops.relaxation` (not
re-implemented); transport uses a marginal Sinkhorn and flow / graphic matroid a shared
entropic mirror-descent engine.

Terminology: the ``beta -> inf`` hardening here is the feasibility / temperature sense
of "collapse" (a soft point becoming a 0/1 vertex), distinct from the
**founding bias collapse** (the multi-bias ``delta -> 0`` limit to the closed-form
derivative ``sigma^(K-1)``, a derivative; see ``docs/theory.md`` and
:mod:`omnibias.torch.unit`).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import torch
from omnibias.combinatorics._core.matroids import (
    GraphicMatroid,
    PartitionMatroid,
    UniformMatroid,
)
from omnibias.discrete import AnnealSchedule
from omnibias.graph.torch.ops.relaxation import sinkhorn_normalize, soft_top_k
from torch import Tensor

if TYPE_CHECKING:  # pragma: no cover - typing only
    from omnibias.combinatorics._core.matroids import Matroid
    from omnibias.combinatorics.problem import MinCostFlowProblem

_EPS = 1e-30


def _normalize(c: Tensor) -> Tensor:
    """Scale-invariant conditioning: divide by the mean magnitude."""
    return c / (torch.mean(torch.abs(c)) + 1e-12)


def _sinkhorn_marginals(log_alpha: Tensor, log_a: Tensor, log_b: Tensor, n_iters: int) -> Tensor:
    r"""Log-domain Sinkhorn onto the transportation polytope (row -> a, column -> b).

    The general-marginal Sinkhorn (graph's :func:`sinkhorn_normalize` is the uniform,
    square special case): alternately match the row marginals to ``exp(log_a)`` and the
    column marginals to ``exp(log_b)``.
    """
    log_p = log_alpha
    for _ in range(n_iters):
        log_p = log_p - torch.logsumexp(log_p, dim=1, keepdim=True) + log_a[:, None]
        log_p = log_p - torch.logsumexp(log_p, dim=0, keepdim=True) + log_b[None, :]
    return torch.exp(log_p)


def _entropic_descent(
    cost: Tensor,
    A_eq: Tensor,
    b_eq: Tensor,
    A_ineq: Tensor,
    b_ineq: Tensor,
    upper: Tensor,
    aeq2: float,
    ain2: float,
    schedule: AnnealSchedule,
) -> Tensor:
    r"""Entropic (KL-mirror) descent onto ``{A_eq x = b_eq, A_ineq x <= b_ineq, 0<=x<=u}``.

    Multiplicative (exponentiated-gradient) steps keep ``x > 0`` while a quadratic penalty
    (weight ``beta``) drives feasibility; annealing ``beta -> inf`` along the schedule
    collapses ``x`` onto a polytope vertex. The gradient is closed form:
    ``grad = c + beta A_eq^T (A_eq x - b_eq) + beta A_ineq^T relu(A_ineq x - b_ineq)``.
    ``aeq2`` / ``ain2`` are the (static, backend-shared) squared Frobenius norms of the
    constraint matrices, so the step size is bit-identical across backends.
    """
    c = _normalize(cost)
    x = 0.5 * upper
    for beta in schedule.betas():
        eta = schedule.step_safety / (1.0 + beta * (aeq2 + ain2) + _EPS)
        for _ in range(schedule.steps):
            req = A_eq @ x - b_eq
            rin = torch.clamp(A_ineq @ x - b_ineq, min=0.0)
            grad = c + beta * (req @ A_eq) + beta * (rin @ A_ineq)
            x = x * torch.exp(-eta * grad)
            x = torch.minimum(torch.clamp(x, min=0.0), upper)  # project onto [0, upper]
    return x


def assignment_relaxation(cost: Any, schedule: AnnealSchedule | None = None) -> Tensor:
    r"""Differentiable assignment relaxation -> doubly-stochastic ``(n, n)`` (Birkhoff)."""
    sched = schedule or AnnealSchedule()
    c = torch.as_tensor(cost, dtype=torch.float64)
    if c.ndim != 2 or c.shape[0] != c.shape[1]:
        raise ValueError(f"assignment cost must be square (n, n); got {tuple(c.shape)}")
    beta = sched.betas()[-1]
    scores = -_normalize(c)
    result: Tensor = sinkhorn_normalize(beta * scores, n_iters=sched.steps)
    return result


def transport_relaxation(
    cost: Any, supply: Any, demand: Any, schedule: AnnealSchedule | None = None
) -> Tensor:
    r"""Differentiable transportation relaxation -> coupling ``(m, n)`` (row->supply, col->demand)."""
    sched = schedule or AnnealSchedule()
    c = torch.as_tensor(cost, dtype=torch.float64)
    a = torch.as_tensor(supply, dtype=torch.float64)
    b = torch.as_tensor(demand, dtype=torch.float64)
    if c.ndim != 2:
        raise ValueError(f"transport cost must be a matrix (m, n); got {tuple(c.shape)}")
    beta = sched.betas()[-1]
    scores = -_normalize(c)
    log_a = torch.log(torch.clamp(a, min=_EPS))
    log_b = torch.log(torch.clamp(b, min=_EPS))
    return _sinkhorn_marginals(beta * scores, log_a, log_b, sched.steps)


def min_cost_flow_relaxation(
    cost: Any, problem: MinCostFlowProblem, schedule: AnnealSchedule | None = None
) -> Tensor:
    r"""Differentiable min-cost-flow relaxation -> arc flows ``(n_arcs,)`` on the flow polytope.

    ``cost`` is the (differentiable) per-arc cost tensor; ``problem`` supplies the fixed
    structure (node conservation, capacities).
    """
    sched = schedule or AnnealSchedule()
    system = problem.system()
    c = torch.as_tensor(cost, dtype=torch.float64).reshape(-1)
    A_eq = torch.as_tensor(system.A_eq, dtype=torch.float64)
    b_eq = torch.as_tensor(system.b_eq, dtype=torch.float64)
    A_ineq = torch.as_tensor(system.A_ineq, dtype=torch.float64)
    b_ineq = torch.as_tensor(system.b_ineq, dtype=torch.float64)
    upper = torch.as_tensor(system.x_upper, dtype=torch.float64)
    aeq2 = float(np.sum(system.A_eq * system.A_eq))
    ain2 = float(np.sum(system.A_ineq * system.A_ineq))
    return _entropic_descent(c, A_eq, b_eq, A_ineq, b_ineq, upper, aeq2, ain2, sched)


def matroid_relaxation(
    weights: Any, matroid: Matroid, schedule: AnnealSchedule | None = None
) -> Tensor:
    r"""Differentiable matroid relaxation -> soft membership ``(ground_size,)`` on the polytope.

    Uniform / partition matroids reuse graph's ``soft_top_k`` (temperature ``1 / beta``);
    the graphic matroid uses the entropic mirror-descent engine on its forest polytope.
    """
    sched = schedule or AnnealSchedule()
    w = torch.as_tensor(weights, dtype=torch.float64).reshape(-1)
    beta = sched.betas()[-1]
    temperature = 1.0 / beta
    if isinstance(matroid, UniformMatroid):
        top: Tensor = soft_top_k(w, matroid.k, temperature=temperature)
        return top
    if isinstance(matroid, PartitionMatroid):
        out = torch.zeros_like(w)
        for group, cap in zip(matroid.groups, matroid.caps, strict=True):
            idx = torch.as_tensor(list(group), dtype=torch.long)
            out[idx] = soft_top_k(w[idx], int(cap), temperature=temperature)
        return out
    if isinstance(matroid, GraphicMatroid):
        a_ineq, b_ineq = matroid.polytope_constraints()
        A_ineq = torch.as_tensor(a_ineq, dtype=torch.float64)
        b_ineq_t = torch.as_tensor(b_ineq, dtype=torch.float64)
        empty_eq = torch.zeros((0, w.shape[0]), dtype=torch.float64)
        empty_beq = torch.zeros((0,), dtype=torch.float64)
        upper = torch.ones_like(w)
        ain2 = float(np.sum(a_ineq * a_ineq))
        return _entropic_descent(-w, empty_eq, empty_beq, A_ineq, b_ineq_t, upper, 0.0, ain2, sched)
    raise TypeError(f"unsupported matroid type {type(matroid).__name__}")


__all__ = [
    "assignment_relaxation",
    "matroid_relaxation",
    "min_cost_flow_relaxation",
    "transport_relaxation",
]
