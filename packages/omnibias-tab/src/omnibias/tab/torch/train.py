# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
r"""Training drivers for the soft-tree ensemble: exact second order + a first-order baseline.

The whole model -- oblique directions ``W``, thresholds ``t`` and leaves -- is trained
jointly by the exact-curvature optimizers of :mod:`omnibias.torch.optim`
(``TrustRegionNewtonCG`` / ``CubicNewton``, matrix-free exact Hessian; ``KFAC``, the
Kronecker-factored natural-gradient preconditioner on the additive Linear reparam). A
first-order Adam baseline (:func:`fit_first_order`) is provided so the "exact second order
beats first order on held-out" claim is measured, not asserted.

``beta`` (gate sharpness) is annealed ``beta_init -> beta_final`` across the run so the
soft splits harden toward hard ones -- **temperature collapse**, the feasibility sense,
distinct from the founding ``delta -> 0`` bias collapse.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
import torch.nn.functional as functional
from omnibias.tab._core.loss import metric as _metric
from omnibias.tab.torch.model import SoftTreeEnsemble
from torch import Tensor

_DTYPE = torch.float64
_SECOND_ORDER = ("trust_region", "cubic", "kfac")


@dataclass
class TrainResult:
    r"""Outcome of a training run (final losses + per-step history)."""

    optimizer: str
    steps: int
    train_loss: float
    val_metric: float | None = None
    history: list[float] = field(default_factory=list)
    betas: list[float] = field(default_factory=list)


def _as_xy(X: np.ndarray, y: np.ndarray, task: str) -> tuple[Tensor, Tensor]:
    Xt = torch.as_tensor(np.asarray(X, dtype=np.float64), dtype=_DTYPE)
    if task == "multiclass":
        yt = torch.as_tensor(np.asarray(y).reshape(-1), dtype=torch.long)
    else:
        yt = torch.as_tensor(np.asarray(y, dtype=np.float64), dtype=_DTYPE)
    return Xt, yt


def _task_loss(F: Tensor, y: Tensor, task: str) -> Tensor:
    if task == "binary":
        return functional.binary_cross_entropy_with_logits(F[:, 0], y.reshape(-1))
    if task == "multiclass":
        return functional.cross_entropy(F, y)
    return functional.mse_loss(F, y.reshape(F.shape))


def _val_metric(model: SoftTreeEnsemble, val: tuple[np.ndarray, np.ndarray] | None, beta: float) -> float | None:
    if val is None:
        return None
    Xv, yv = val
    F = model.score(Xv, beta=beta)
    return float(_metric(F, np.asarray(yv), model.config.task))


def fit_second_order(
    model: SoftTreeEnsemble,
    X: np.ndarray,
    y: np.ndarray,
    *,
    optimizer: str = "trust_region",
    steps: int = 60,
    anneal: bool = True,
    leaf_l2: float | None = None,
    weight_l2: float = 1e-4,
    val: tuple[np.ndarray, np.ndarray] | None = None,
    **opt_kwargs: Any,
) -> TrainResult:
    r"""Train the ensemble with an **exact second-order** optimizer.

    ``optimizer`` is one of ``"trust_region"`` (default; matrix-free exact-Hessian
    trust-region Newton-CG, all parameters), ``"cubic"`` (adaptive cubic-regularised
    Newton), or ``"kfac"`` (Kronecker-factored natural gradient on the additive Linear
    reparam -- requires ``depth == 1`` and trains at a fixed ``beta_final``). There is no
    learning rate for the exact-Hessian methods.
    """
    if optimizer not in _SECOND_ORDER:
        raise ValueError(f"optimizer must be one of {_SECOND_ORDER}, got {optimizer!r}")
    task = model.config.task
    l2 = model.config.leaf_l2 if leaf_l2 is None else float(leaf_l2)
    Xt, yt = _as_xy(X, y, task)

    if optimizer == "kfac":
        return _fit_kfac(
            model, Xt, yt, steps=steps, leaf_l2=l2, weight_l2=weight_l2, val=val, **opt_kwargs
        )

    from omnibias.torch.optim import CubicNewton, TrustRegionNewtonCG

    opt: torch.optim.Optimizer
    if optimizer == "trust_region":
        opt = TrustRegionNewtonCG(model.parameters(), **opt_kwargs)
    else:
        opt = CubicNewton(model.parameters(), **opt_kwargs)

    history: list[float] = []
    betas: list[float] = []
    for step in range(steps):
        beta = model.config.beta_at(step) if anneal else model.config.beta_final
        model.set_beta(beta)

        def closure() -> Tensor:
            F = model(Xt)
            loss = _task_loss(F, yt, task)
            loss = loss + l2 * (model.leaves**2).mean() + weight_l2 * (model.W**2).mean()
            return loss

        loss_t = opt.step(closure)  # type: ignore[arg-type]
        loss_val = float(loss_t) if loss_t is not None else float(closure().detach())
        history.append(loss_val)
        betas.append(beta)

    model.set_beta(model.config.beta_final)
    return TrainResult(
        optimizer=optimizer,
        steps=steps,
        train_loss=history[-1] if history else float("nan"),
        val_metric=_val_metric(model, val, model.config.beta_final),
        history=history,
        betas=betas,
    )


def _fit_kfac(
    model: SoftTreeEnsemble,
    Xt: Tensor,
    yt: Tensor,
    *,
    steps: int,
    leaf_l2: float,
    weight_l2: float,
    val: tuple[np.ndarray, np.ndarray] | None,
    lr: float = 0.1,
    damping: float = 1e-2,
    **opt_kwargs: object,
) -> TrainResult:
    if model.config.depth != 1:
        raise ValueError("optimizer='kfac' requires depth == 1 (the additive Linear reparam)")
    from omnibias.torch.optim import KFAC

    task = model.config.task
    beta = model.config.beta_final
    seq = model.to_additive_sequential(beta)
    opt = KFAC(seq, lr=lr, damping=damping, **opt_kwargs)

    history: list[float] = []
    for _ in range(steps):
        def closure() -> Tensor:
            F = seq(Xt)
            loss = _task_loss(F, yt, task)
            loss = loss + weight_l2 * (seq[0].weight**2).mean() + leaf_l2 * (seq[2].weight**2).mean()
            return loss

        loss_t = opt.step(closure)
        history.append(float(loss_t) if loss_t is not None else float(closure().detach()))
    opt.remove_hooks()
    model.load_from_additive_sequential(seq, beta)
    return TrainResult(
        optimizer="kfac",
        steps=steps,
        train_loss=history[-1] if history else float("nan"),
        val_metric=_val_metric(model, val, beta),
        history=history,
        betas=[beta] * steps,
    )


def fit_first_order(
    model: SoftTreeEnsemble,
    X: np.ndarray,
    y: np.ndarray,
    *,
    lr: float = 0.05,
    steps: int = 400,
    anneal: bool = True,
    leaf_l2: float | None = None,
    weight_l2: float = 0.0,
    val: tuple[np.ndarray, np.ndarray] | None = None,
) -> TrainResult:
    r"""First-order **Adam** baseline (the reference the second-order driver must beat)."""
    task = model.config.task
    l2 = model.config.leaf_l2 if leaf_l2 is None else float(leaf_l2)
    Xt, yt = _as_xy(X, y, task)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    history: list[float] = []
    betas: list[float] = []
    for step in range(steps):
        beta = model.config.beta_at(step) if anneal else model.config.beta_final
        model.set_beta(beta)
        opt.zero_grad(set_to_none=True)
        F = model(Xt)
        loss = _task_loss(F, yt, task) + l2 * (model.leaves**2).mean() + weight_l2 * (model.W**2).mean()
        loss.backward()
        opt.step()
        history.append(float(loss.detach()))
        betas.append(beta)

    model.set_beta(model.config.beta_final)
    return TrainResult(
        optimizer="adam",
        steps=steps,
        train_loss=history[-1] if history else float("nan"),
        val_metric=_val_metric(model, val, model.config.beta_final),
        history=history,
        betas=betas,
    )


__all__ = ["TrainResult", "fit_first_order", "fit_second_order"]
