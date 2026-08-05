# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
r"""Differentiable annealed relaxation core (torch) via the temperature-collapse penalty.

Bit-identical twin of :mod:`omnibias.discrete.jax.relaxation` (float64); see that module
for the full math. Differentiable through ``autograd`` so a model that predicts the
problem coefficients can be trained *through* the relaxation.

Terminology: the ``beta -> inf`` hardening of ``sigmoid`` here is the feasibility /
temperature sense of "collapse" (a soft indicator becoming a 0/1 step), distinct from
the **founding bias collapse** (the multi-bias ``delta -> 0`` limit to ``sigma^(K-1)``,
a derivative; see :mod:`omnibias.torch.unit`).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch
from omnibias.discrete._core.relax import initial_theta
from omnibias.discrete._core.schedule import AnnealSchedule
from torch import Tensor


def anneal_descent(
    grad_x_fn: Callable[[Tensor], Tensor],
    scale: Any,
    n: int,
    schedule: AnnealSchedule | None = None,
) -> Tensor:
    r"""Anneal ``x = sigmoid(beta theta)`` to a vertex by descending ``grad_x_fn``.

    See :func:`omnibias.discrete.jax.relaxation.anneal_descent` for the full description;
    this is the bit-identical torch twin.
    """
    sched = schedule or AnnealSchedule()
    theta = torch.as_tensor(initial_theta(n), dtype=torch.float64)
    # Detached so the step size never perturbs the gradient w.r.t. the coefficients.
    scale_c = torch.as_tensor(scale, dtype=torch.float64).detach()

    betas = sched.betas()
    for beta in betas:
        eta = sched.step_safety / (beta * 0.25 * scale_c + 1e-30)
        for _ in range(sched.steps):
            x = torch.sigmoid(beta * theta)
            grad_x = grad_x_fn(x)
            grad_theta = grad_x * (beta * x * (1.0 - x))
            theta = theta - eta * grad_theta
    return torch.sigmoid(betas[-1] * theta)


__all__ = ["anneal_descent"]
