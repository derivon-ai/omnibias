# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
r"""Differentiable sparse ``l_p -> l_0`` relaxation (torch) over the shared temperature-collapse core.

Bit-identical twin of :mod:`omnibias.discrete.sparse.jax.relaxation` (float64); see that
module for the full math. Builds the closed-form ``l_p``-reweighted least-squares gradient
``A^T A x - A^T b + lambda p (x + eps)^{p-1}`` and hands it to
:func:`omnibias.discrete.torch.anneal_descent`.

Terminology (two distinct senses of "collapse"): both the ``beta -> inf`` sigmoid
hardening and the ``l_p -> l_0`` penalty-exponent homotopy here are the **feasibility** /
temperature sense (a soft object becoming a hard ``0/1`` selection). Neither is the
**founding bias collapse** -- the multi-bias ``delta -> 0`` limit to the closed-form
derivative ``sigma^(K-1)`` (see ``docs/theory.md`` and :mod:`omnibias.torch.unit`).
"""

from __future__ import annotations

import torch
from omnibias.discrete._core.schedule import AnnealSchedule
from omnibias.discrete.sparse.problem import SupportSelectionProblem
from omnibias.discrete.torch import anneal_descent
from torch import Tensor


def sparse_relaxation(
    problem: SupportSelectionProblem,
    *,
    p: float = 1.0,
    eps: float = 1e-3,
    schedule: AnnealSchedule | None = None,
) -> Tensor:
    r"""Differentiable ``l_p`` annealed relaxation of a support-selection instance.

    See :func:`omnibias.discrete.sparse.jax.relaxation.sparse_relaxation` for the full
    description; this is the bit-identical torch twin.
    """
    if not (0.0 < p <= 1.0):
        raise ValueError(f"p must satisfy 0 < p <= 1, got {p}")
    if eps <= 0.0:
        raise ValueError(f"eps must be positive, got {eps}")
    sched = schedule or AnnealSchedule()
    n = problem.n
    gram = torch.as_tensor(problem.gram_matrix, dtype=torch.float64)
    corr = torch.as_tensor(problem.correlation, dtype=torch.float64)
    lam = float(problem.lam)
    gram_norm = float(torch.linalg.norm(gram, 2)) if n else 1.0
    scale = gram_norm + lam * p * (eps ** (p - 1.0)) + 1.0

    def grad_x_fn(x: Tensor) -> Tensor:
        data = gram @ x - corr
        penalty = lam * p * torch.pow(x + eps, p - 1.0)
        out: Tensor = data + penalty
        return out

    result: Tensor = anneal_descent(grad_x_fn, scale, n, sched)
    return result


__all__ = ["sparse_relaxation"]
