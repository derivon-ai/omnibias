# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
r"""Stagewise Newton boosting of shallow soft trees -- the GBM-mirror.

This is the ``tab`` analogue of LightGBM / XGBoost, but with *soft* oblique trees and the
**closed-form function-space curvature** made explicit. At stage ``m`` the current
ensemble score ``F`` has a per-sample loss gradient ``g_i`` and Hessian ``h_i`` that are
closed form (the Riccati ``p (1 - p)`` tower; :func:`omnibias.tab._core.loss.score_grad_hess`).
A fresh weak learner is fit to the **Newton target** ``r_i = -g_i / h_i`` under the
Hessian weights ``h_i`` -- i.e. it minimises the second-order Taylor model of the loss,
exactly the Newton-boosting objective -- and is added with shrinkage ``learning_rate``.

The accumulated stages fold into a single :class:`~omnibias.tab.torch.model.SoftTreeEnsemble`
(leaves scaled by the shrinkage, the base score folded into ``b0``), so the boosted model
evaluates, exports and certifies through the same path as the jointly-trained one.

Terminology: each stage's gate ``sigmoid(beta (w.x - t))`` hardens as ``beta -> inf`` (the
feasibility / temperature sense of collapse), distinct from the founding ``delta -> 0``
bias collapse.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch
from omnibias.tab._core.config import SoftTreeConfig
from omnibias.tab._core.loss import loss_value, score_grad_hess
from omnibias.tab._core.loss import metric as _metric
from omnibias.tab.torch.model import SoftTreeEnsemble

_DTYPE = torch.float64
_EPS = 1e-12


@dataclass
class BoostResult:
    r"""Outcome of a Newton-boosting run."""

    n_stages: int
    learning_rate: float
    train_loss: float
    val_metric: float | None = None
    history: list[float] = field(default_factory=list)


def _base_score(y: np.ndarray, task: str, k: int) -> np.ndarray:
    if task == "binary":
        p = float(np.clip(np.mean(np.asarray(y, dtype=np.float64)), _EPS, 1.0 - _EPS))
        return np.array([np.log(p / (1.0 - p))], dtype=np.float64)
    if task == "multiclass":
        idx = np.asarray(y, dtype=np.int64).reshape(-1)
        freq = np.clip(np.bincount(idx, minlength=k).astype(np.float64) / idx.shape[0], _EPS, 1.0)
        return np.asarray(np.log(freq), dtype=np.float64)
    yv = np.asarray(y, dtype=np.float64).reshape(-1, k) if k > 1 else np.asarray(y, dtype=np.float64).reshape(-1, 1)
    base: np.ndarray = np.mean(yv, axis=0)
    return base


def _fit_weak_learner(
    cfg: SoftTreeConfig,
    X: np.ndarray,
    target: np.ndarray,
    weight: np.ndarray,
    *,
    steps: int,
    lr: float,
    seed: int,
) -> SoftTreeEnsemble:
    r"""Weighted least-squares fit of a fresh weak learner to the Newton target."""
    torch.manual_seed(seed)
    reg_cfg = SoftTreeConfig(
        n_features=cfg.n_features, n_trees=cfg.n_trees, depth=cfg.depth,
        task="regression", n_outputs=cfg.n_outputs, beta_final=cfg.beta_final, seed=seed,
    )
    model = SoftTreeEnsemble(reg_cfg)
    model.set_beta(cfg.beta_final)
    Xt = torch.as_tensor(np.asarray(X, dtype=np.float64), dtype=_DTYPE)
    rt = torch.as_tensor(np.asarray(target, dtype=np.float64), dtype=_DTYPE)
    wt = torch.as_tensor(np.asarray(weight, dtype=np.float64), dtype=_DTYPE)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for _ in range(steps):
        opt.zero_grad(set_to_none=True)
        F = model(Xt)
        loss = (wt * (F - rt) ** 2).mean()
        loss.backward()
        opt.step()
    return model


def fit_boosted(
    X: np.ndarray,
    y: np.ndarray,
    config: SoftTreeConfig,
    *,
    n_stages: int = 30,
    learning_rate: float = 0.3,
    inner_steps: int = 60,
    inner_lr: float = 0.05,
    val: tuple[np.ndarray, np.ndarray] | None = None,
) -> tuple[SoftTreeEnsemble, BoostResult]:
    r"""Fit a Newton-boosted soft-tree ensemble; returns ``(model, BoostResult)``.

    ``config`` describes **one stage** (its ``n_trees`` / ``depth`` are the weak-learner
    shape); the returned :class:`SoftTreeEnsemble` holds ``n_stages * config.n_trees`` trees.
    """
    task, k = config.task, config.n_outputs
    Xtr = np.asarray(X, dtype=np.float64)
    n = Xtr.shape[0]

    base = _base_score(y, task, k)  # (k,)
    F = np.tile(base[None, :], (n, 1))  # (n, k)
    Fval = None
    if val is not None:
        Fval = np.tile(base[None, :], (val[0].shape[0], 1))

    W_parts: list[np.ndarray] = []
    t_parts: list[np.ndarray] = []
    leaf_parts: list[np.ndarray] = []
    b0 = base.copy()
    history: list[float] = []

    for stage in range(n_stages):
        g, h = score_grad_hess(F, y, task)  # (n, k)
        target = -g / np.clip(h, _EPS, None)
        weak = _fit_weak_learner(
            config, Xtr, target, h, steps=inner_steps, lr=inner_lr, seed=config.seed + stage + 1
        )
        wp = weak.to_params()
        W_parts.append(wp.W)
        t_parts.append(wp.t)
        leaf_parts.append(learning_rate * wp.leaves)
        b0 = b0 + learning_rate * wp.b0

        F = F + learning_rate * weak.score(Xtr)
        if val is not None and Fval is not None:
            Fval = Fval + learning_rate * weak.score(val[0])
        history.append(loss_value(F, y, task))

    total_cfg = SoftTreeConfig(
        n_features=config.n_features,
        n_trees=n_stages * config.n_trees,
        depth=config.depth,
        task=task,
        n_outputs=k,
        beta_final=config.beta_final,
        seed=config.seed,
    )
    from omnibias.tab._core.params import TabParams

    params = TabParams(
        total_cfg,
        np.concatenate(W_parts, axis=0),
        np.concatenate(t_parts, axis=0),
        np.concatenate(leaf_parts, axis=0),
        b0,
    )
    model = SoftTreeEnsemble(total_cfg, params)
    model.set_beta(config.beta_final)

    val_metric = None
    if val is not None and Fval is not None:
        val_metric = _metric(Fval, np.asarray(val[1]), task)

    return model, BoostResult(
        n_stages=n_stages,
        learning_rate=learning_rate,
        train_loss=history[-1] if history else float("nan"),
        val_metric=val_metric,
        history=history,
    )


__all__ = ["BoostResult", "fit_boosted"]
