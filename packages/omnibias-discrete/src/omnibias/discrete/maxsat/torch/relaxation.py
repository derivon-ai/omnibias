# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
r"""Differentiable MaxSAT relaxation (torch) over the shared temperature-collapse core.

Bit-identical twin of :mod:`omnibias.discrete.maxsat.jax.relaxation` (float64); see that
module for the full math. Builds the closed-form weighted-violation-energy gradient and
hands it to :func:`omnibias.discrete.torch.anneal_descent`.

Terminology: the ``beta -> inf`` hardening of ``sigmoid`` here is the feasibility /
temperature sense of "collapse" (a soft indicator becoming a 0/1 step), distinct from
the **founding bias collapse** (the multi-bias ``delta -> 0`` limit to ``sigma^(K-1)``,
a derivative; see :mod:`omnibias.torch.unit`).
"""

from __future__ import annotations

import torch
from omnibias.discrete._core.schedule import AnnealSchedule
from omnibias.discrete.maxsat.problem import MaxSATProblem
from omnibias.discrete.torch import anneal_descent
from torch import Tensor


def maxsat_relaxation(problem: MaxSATProblem, schedule: AnnealSchedule | None = None) -> Tensor:
    r"""Differentiable annealed relaxation of a MaxSAT instance -> ``x in (0, 1)^n``."""
    sched = schedule or AnnealSchedule()
    n = problem.n
    clauses = problem.cnf.clauses
    scale = problem.grad_scale()

    def grad_x_fn(x: Tensor) -> Tensor:
        comps = [x.new_zeros(()) for _ in range(n)]
        for clause in clauses:
            weight = clause.weight
            factors = [
                (1.0 - x[abs(literal) - 1]) if literal > 0 else x[abs(literal) - 1]
                for literal in clause.literals
            ]
            for k, literal in enumerate(clause.literals):
                i = abs(literal) - 1
                dsign = -1.0 if literal > 0 else 1.0
                loo = x.new_ones(())
                for j, factor in enumerate(factors):
                    if j != k:
                        loo = loo * factor
                comps[i] = comps[i] + weight * dsign * loo
        return torch.stack(comps)

    result: Tensor = anneal_descent(grad_x_fn, scale, n, sched)
    return result


__all__ = ["maxsat_relaxation"]
