# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Sharpness-scheduled cubic / learning-rate step (theory 08-06), PyTorch twin.

Exact Hessian-vector products (:func:`omnibias.torch.optim.hvp`) and
Lanczos give a Ritz ``lambda_max``. That value sets cubic ``sigma`` or
a gradient learning rate. Sharpness is a step-size signal, not a
generalization claim and not CCF stretch.

Do not wrap the driver in ``torch.compile``.
"""

from __future__ import annotations

import math
from collections.abc import Callable

from omnibias.core.sharpness import (
    SharpnessReport,
    SharpnessSchedule,
    make_report,
    scheduled_value,
)

import torch
from torch import Tensor
from torch.func import grad

ScalarFn = Callable[[Tensor], Tensor]


def sharpness_lambda_max(
    loss_fn: ScalarFn,
    params: Tensor,
    *,
    n_lanczos: int = 4,
    probe: Tensor | None = None,
) -> float:
    """Largest Ritz value of ``Hess L`` from exact HVPs (Lanczos)."""
    from omnibias.torch.optim import hvp, lanczos_tridiag

    if int(n_lanczos) < 1:
        raise ValueError(f"n_lanczos must be >= 1, got {n_lanczos}")
    p = params.reshape(-1)
    start = probe.reshape(-1) if probe is not None else grad(loss_fn)(p)
    if float(torch.linalg.vector_norm(start)) == 0.0:
        start = torch.zeros_like(p)
        start[0] = 1.0

    def matvec(v: Tensor) -> Tensor:
        return hvp(loss_fn, p, v)

    _q, tri = lanczos_tridiag(matvec, start, int(n_lanczos))
    return float(torch.max(torch.linalg.eigvalsh(tri)))


def sharpness_scheduled_step(
    loss_fn: ScalarFn,
    params: Tensor,
    *,
    schedule: SharpnessSchedule | None = None,
    krylov_dim: int | None = None,
) -> tuple[Tensor, SharpnessReport]:
    """One cubic (or gradient) step with ``sigma`` / ``lr`` from measured sharpness."""
    sched = SharpnessSchedule() if schedule is None else schedule
    p = params.reshape(-1)
    ell = sharpness_lambda_max(loss_fn, p, n_lanczos=sched.n_lanczos)
    report = make_report(ell, sched)
    if not math.isfinite(report.scheduled) or report.scheduled <= 0.0:
        raise ValueError(
            f"sharpness schedule produced a non-finite or non-positive value "
            f"{report.scheduled!r} from ell_k={ell!r}; refusing the step"
        )
    if sched.target == "lr":
        g = grad(loss_fn)(p)
        return p - report.scheduled * g, report
    from omnibias.torch.optim import cubic_regularized_newton_step

    k = int(sched.n_lanczos) if krylov_dim is None else int(krylov_dim)
    delta = cubic_regularized_newton_step(loss_fn, p, report.scheduled, krylov_dim=k)
    return p + delta, report


def sharpness_scheduled_minimize(
    loss_fn: ScalarFn,
    params: Tensor,
    *,
    schedule: SharpnessSchedule | None = None,
    steps: int = 20,
    krylov_dim: int | None = None,
) -> tuple[Tensor, list[float], list[float]]:
    """Repeat :func:`sharpness_scheduled_step`; returns params, losses, ``ell_k``."""
    if int(steps) < 1:
        raise ValueError(f"steps must be >= 1, got {steps}")
    sched = SharpnessSchedule() if schedule is None else schedule
    p = params.reshape(-1)
    losses: list[float] = [float(loss_fn(p))]
    ells: list[float] = []
    for _ in range(int(steps)):
        p, report = sharpness_scheduled_step(
            loss_fn, p, schedule=sched, krylov_dim=krylov_dim
        )
        ells.append(report.ell_k)
        losses.append(float(loss_fn(p)))
        if not math.isfinite(losses[-1]) or not bool(torch.isfinite(p).all()):
            break
    return p, losses, ells


__all__ = [
    "SharpnessReport",
    "SharpnessSchedule",
    "scheduled_value",
    "sharpness_lambda_max",
    "sharpness_scheduled_minimize",
    "sharpness_scheduled_step",
]
