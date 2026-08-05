# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
r"""Differentiable QUBO relaxation layer (torch) via the annealed temperature-collapse penalty.

Bit-identical twin of :mod:`omnibias.qubo.jax.relaxation` (float64); a thin QUBO
front-end over the shared substrate core :func:`omnibias.discrete.torch.anneal_descent`.
Differentiable through ``autograd`` so a model that predicts ``Q`` / ``c`` can be trained
*through* the relaxation.

Terminology: the ``beta -> inf`` hardening of ``sigmoid`` here is the feasibility /
temperature sense of "collapse" (a soft indicator becoming a 0/1 step), distinct from
the **founding bias collapse** (the multi-bias ``delta -> 0`` limit to ``sigma^(K-1)``,
a derivative; see :mod:`omnibias.torch.unit`).
"""

from __future__ import annotations

from typing import Any

import torch
from omnibias.discrete.torch import anneal_descent
from omnibias.qubo.problem import AnnealSchedule
from torch import Tensor


def _unpack(problem_or_Q: Any, c: Any) -> tuple[Any, Any]:
    if hasattr(problem_or_Q, "Q") and hasattr(problem_or_Q, "c"):
        return problem_or_Q.Q, problem_or_Q.c
    return problem_or_Q, c


def qubo_relaxation(
    problem_or_Q: Any,
    c: Any = None,
    schedule: AnnealSchedule | None = None,
) -> Tensor:
    r"""Differentiable annealed relaxation of a QUBO -> soft assignment ``x in (0, 1)^n``.

    Accepts a :class:`~omnibias.qubo.problem.QUBOProblem` or an array-like ``Q`` (with an
    optional linear ``c``); pass tensors to differentiate through ``Q`` / ``c``.
    """
    sched = schedule or AnnealSchedule()
    Q_in, c_in = _unpack(problem_or_Q, c)
    Q = torch.as_tensor(Q_in, dtype=torch.float64)
    Q = 0.5 * (Q + Q.t())
    n = int(Q.shape[0])
    cvec = (
        torch.zeros(n, dtype=torch.float64)
        if c_in is None
        else torch.as_tensor(c_in, dtype=torch.float64)
    )

    # Frobenius norm upper-bounds the spectral norm -> a safe (smaller) descent step;
    # anneal_descent detaches it so the step size does not perturb the grad w.r.t. Q / c.
    scale = 2.0 * torch.sqrt(torch.sum(Q * Q)) + torch.max(torch.abs(cvec))

    def grad_x_fn(x: Tensor) -> Tensor:
        return 2.0 * (Q @ x) + cvec

    result: Tensor = anneal_descent(grad_x_fn, scale, n, sched)
    return result


__all__ = ["qubo_relaxation"]
