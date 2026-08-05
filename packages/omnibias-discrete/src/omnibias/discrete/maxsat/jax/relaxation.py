# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
r"""Differentiable MaxSAT relaxation (JAX) over the shared temperature-collapse core.

Bit-identical twin of :mod:`omnibias.discrete.maxsat.torch.relaxation` (float64). Builds
the **closed-form** gradient of the weighted-violation energy -- for each clause and each
of its literals, the leave-one-out product of the other violation factors times the
literal's sign -- and hands it to :func:`omnibias.discrete.jax.anneal_descent`, which
anneals ``x = sigmoid(beta theta)`` onto a binary vertex.

Terminology: the ``beta -> inf`` hardening of ``sigmoid`` here is the feasibility /
temperature sense of "collapse" (a soft indicator becoming a 0/1 step), distinct from
the **founding bias collapse** (the multi-bias ``delta -> 0`` limit to ``sigma^(K-1)``,
a derivative; see ``docs/theory.md`` and :mod:`omnibias.torch.unit`).
"""

from __future__ import annotations

import jax.numpy as jnp
from jax import Array
from omnibias.discrete._core.schedule import AnnealSchedule
from omnibias.discrete.jax import anneal_descent
from omnibias.discrete.maxsat.problem import MaxSATProblem


def maxsat_relaxation(problem: MaxSATProblem, schedule: AnnealSchedule | None = None) -> Array:
    r"""Differentiable annealed relaxation of a MaxSAT instance -> ``x in (0, 1)^n``."""
    sched = schedule or AnnealSchedule()
    n = problem.n
    clauses = problem.cnf.clauses
    scale = problem.grad_scale()

    def grad_x_fn(x: Array) -> Array:
        comps = [jnp.zeros((), dtype=x.dtype) for _ in range(n)]
        for clause in clauses:
            weight = clause.weight
            factors = [
                (1.0 - x[abs(literal) - 1]) if literal > 0 else x[abs(literal) - 1]
                for literal in clause.literals
            ]
            for k, literal in enumerate(clause.literals):
                i = abs(literal) - 1
                dsign = -1.0 if literal > 0 else 1.0
                loo = jnp.ones((), dtype=x.dtype)
                for j, factor in enumerate(factors):
                    if j != k:
                        loo = loo * factor
                comps[i] = comps[i] + weight * dsign * loo
        stacked: Array = jnp.stack(comps)
        return stacked

    result: Array = anneal_descent(grad_x_fn, scale, n, sched)
    return result


__all__ = ["maxsat_relaxation"]
