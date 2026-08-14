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
from torch import Tensor, nn

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


def _val_metric(
    model: SoftTreeEnsemble,
    val: tuple[np.ndarray, np.ndarray] | None,
    beta: float,
    *,
    graph: nn.Module | None = None,
) -> float | None:
    if val is None:
        return None
    Xv, yv = val
    if graph is None:
        F = model.score(Xv, beta=beta)
    else:
        Xt, _ = _as_xy(Xv, yv, model.config.task)
        p = next(graph.parameters())
        Xt = Xt.to(device=p.device, dtype=p.dtype)
        with torch.no_grad():
            F = graph(Xt).detach().cpu().numpy()
    return float(_metric(F, np.asarray(yv), model.config.task))


class _EncoderHead(nn.Module):
    """User graph ``head(encoder(X))`` for :func:`fit_joint` and optional ``encoder=``."""

    def __init__(self, encoder: nn.Module, head: nn.Module) -> None:
        super().__init__()
        self.encoder = encoder
        self.head = head

    def forward(self, x: Tensor) -> Tensor:
        return self.head(self.encoder(x))


def _compose_encoder(encoder: nn.Module | None, head: nn.Module) -> nn.Module:
    if encoder is None:
        return head
    p = next(head.parameters())
    encoder.to(device=p.device, dtype=p.dtype)
    return _EncoderHead(encoder, head)


def _maybe_move_xy(
    Xt: Tensor, yt: Tensor, graph: nn.Module, *, encoder: nn.Module | None
) -> tuple[Tensor, Tensor]:
    if encoder is None:
        return Xt, yt
    p = next(graph.parameters())
    Xt = Xt.to(device=p.device, dtype=p.dtype)
    if yt.is_floating_point():
        yt = yt.to(device=p.device, dtype=p.dtype)
    else:
        yt = yt.to(device=p.device)
    return Xt, yt


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
    patience: int | None = None,
    encoder: nn.Module | None = None,
    **opt_kwargs: Any,
) -> TrainResult:
    r"""Train the ensemble with an **exact second-order** optimizer.

    ``optimizer`` is one of ``"trust_region"`` (default; matrix-free exact-Hessian
    trust-region Newton-CG, all parameters), ``"cubic"`` (adaptive cubic-regularised
    Newton), or ``"kfac"`` (Kronecker-factored natural gradient on the additive Linear
    reparam -- requires ``depth == 1`` and trains at a fixed ``beta_final``). There is no
    learning rate for the exact-Hessian methods.

    Optional ``patience`` (with ``val``) restores the best validation checkpoint.
    Optional ``encoder`` jointly trains ``head(encoder(X))`` (``X`` is encoder input;
    head ``n_features`` is the encoder width). Default ``None`` is the tabular G3 path.
    """
    if optimizer not in _SECOND_ORDER:
        raise ValueError(f"optimizer must be one of {_SECOND_ORDER}, got {optimizer!r}")
    if encoder is not None and optimizer == "kfac":
        raise TypeError(
            "encoder= is incompatible with optimizer='kfac'; "
            "use fit_joint or optimizer='trust_region'"
        )
    task = model.config.task
    l2 = model.config.leaf_l2 if leaf_l2 is None else float(leaf_l2)
    graph = _compose_encoder(encoder, model)
    Xt, yt = _maybe_move_xy(*_as_xy(X, y, task), graph, encoder=encoder)

    if optimizer == "kfac":
        return _fit_kfac(
            model, Xt, yt, steps=steps, leaf_l2=l2, weight_l2=weight_l2, val=val, **opt_kwargs
        )

    from omnibias.torch.optim import CubicNewton, TrustRegionNewtonCG

    opt: torch.optim.Optimizer
    if optimizer == "trust_region":
        opt = TrustRegionNewtonCG(graph.parameters(), **opt_kwargs)
    else:
        opt = CubicNewton(graph.parameters(), **opt_kwargs)

    history: list[float] = []
    betas: list[float] = []
    use_es = val is not None and patience is not None
    best_state: dict[str, Tensor] | None = None
    best_val = _val_metric(model, val, model.config.beta_final, graph=graph if encoder is not None else None) if use_es else None
    stall = 0
    steps_run = 0
    for step in range(steps):
        beta = model.config.beta_at(step) if anneal else model.config.beta_final
        model.set_beta(beta)

        def closure() -> Tensor:
            F = graph(Xt)
            loss = _task_loss(F, yt, task)
            loss = loss + l2 * (model.leaves**2).mean() + weight_l2 * (model.W**2).mean()
            return loss

        loss_t = opt.step(closure)  # type: ignore[arg-type]
        loss_val = float(loss_t) if loss_t is not None else float(closure().detach())
        history.append(loss_val)
        betas.append(beta)
        steps_run = step + 1
        if use_es:
            cur = _val_metric(model, val, beta, graph=graph if encoder is not None else None)
            if cur is not None and (best_val is None or cur > best_val + 1e-12):
                best_val = cur
                best_state = {k: v.detach().clone() for k, v in graph.state_dict().items()}
                stall = 0
            else:
                stall += 1
                if stall >= max(1, int(patience or 1)):
                    break

    if use_es and best_state is not None:
        graph.load_state_dict(best_state)
    model.set_beta(model.config.beta_final)
    return TrainResult(
        optimizer=optimizer,
        steps=steps_run,
        train_loss=history[-1] if history else float("nan"),
        val_metric=_val_metric(
            model, val, model.config.beta_final, graph=graph if encoder is not None else None
        ),
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
    encoder: nn.Module | None = None,
) -> TrainResult:
    r"""First-order **Adam** baseline (the reference the second-order driver must beat).

    Optional ``encoder`` jointly trains ``head(encoder(X))``. Default ``None`` is
    the tabular G3 path.
    """
    task = model.config.task
    l2 = model.config.leaf_l2 if leaf_l2 is None else float(leaf_l2)
    graph = _compose_encoder(encoder, model)
    Xt, yt = _maybe_move_xy(*_as_xy(X, y, task), graph, encoder=encoder)
    opt = torch.optim.Adam(graph.parameters(), lr=lr)

    history: list[float] = []
    betas: list[float] = []
    for step in range(steps):
        beta = model.config.beta_at(step) if anneal else model.config.beta_final
        model.set_beta(beta)
        opt.zero_grad(set_to_none=True)
        F = graph(Xt)
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
        val_metric=_val_metric(
            model, val, model.config.beta_final, graph=graph if encoder is not None else None
        ),
        history=history,
        betas=betas,
    )


def _head_task(head: nn.Module) -> str:
    cfg = getattr(head, "config", None)
    if cfg is not None and hasattr(cfg, "task"):
        return str(cfg.task)
    task = getattr(head, "task", None)
    if task is not None:
        return str(task)
    members = getattr(head, "members", None)
    if members is not None and len(members) > 0:
        first = members[0]
        inner = getattr(first, "task", None)
        if inner is not None:
            return str(inner)
    return "binary"


def _joint_val_score(graph: nn.Module, Xv: Tensor, yv: Tensor, task: str) -> float:
    with torch.no_grad():
        F = graph(Xv)
        if task == "binary":
            pred = (F[..., 0] > 0).to(dtype=yv.dtype)
            return float((pred.reshape(-1) == yv.reshape(-1)).to(dtype=torch.float64).mean().item())
        if task == "multiclass":
            pred = F.argmax(dim=-1)
            return float((pred.reshape(-1) == yv.reshape(-1)).to(dtype=torch.float64).mean().item())
        return float(-functional.mse_loss(F, yv.reshape(F.shape)).item())


def fit_joint(
    encoder: nn.Module,
    head: nn.Module,
    X: np.ndarray,
    y: np.ndarray,
    *,
    optimizer: str = "adam",
    steps: int = 200,
    lr: float = 0.05,
    X_val: np.ndarray | None = None,
    y_val: np.ndarray | None = None,
    patience: int | None = 50,
    task: str | None = None,
    **opt_kwargs: Any,
) -> TrainResult:
    r"""Jointly train ``head(encoder(X))`` -- encoder and head parameters both move.

    Does **not** replace the tabular :func:`fit_arrangement` / :func:`fit_boosted`
    / :func:`fit_second_order` trainers (those stay numpy-in for LightGBM benches).
    ``as_head`` is the constructor for ``head``. ``optimizer`` is ``adam`` (default)
    or ``trust_region`` / ``cubic`` (exact Hessian over the composed graph).
    Early-stops on validation when ``X_val`` / ``y_val`` are given.
    """
    task_s = str(task) if task is not None else _head_task(head)
    graph = _EncoderHead(encoder, head)
    Xt, yt = _as_xy(X, y, task_s)
    p = next(graph.parameters())
    Xt = Xt.to(device=p.device, dtype=p.dtype)
    if yt.is_floating_point():
        yt = yt.to(device=p.device, dtype=p.dtype)
    else:
        yt = yt.to(device=p.device)

    val_xy: tuple[Tensor, Tensor] | None = None
    if X_val is not None or y_val is not None:
        if X_val is None or y_val is None:
            raise ValueError("X_val and y_val must be provided together")
        Xv, yv = _as_xy(X_val, y_val, task_s)
        Xv = Xv.to(device=p.device, dtype=p.dtype)
        yv = yv.to(device=p.device, dtype=yv.dtype if not yv.is_floating_point() else p.dtype)
        val_xy = (Xv, yv)

    name = str(optimizer)
    if name not in ("adam", "trust_region", "cubic"):
        raise ValueError(f"optimizer must be one of ('adam', 'trust_region', 'cubic'), got {name!r}")

    opt: torch.optim.Optimizer
    if name == "adam":
        opt = torch.optim.Adam(graph.parameters(), lr=float(lr), **opt_kwargs)
    elif name == "trust_region":
        from omnibias.torch.optim import TrustRegionNewtonCG

        opt = TrustRegionNewtonCG(graph.parameters(), **opt_kwargs)
    else:
        from omnibias.torch.optim import CubicNewton

        opt = CubicNewton(graph.parameters(), **opt_kwargs)

    history: list[float] = []
    use_es = val_xy is not None and patience is not None
    best_state: dict[str, Tensor] | None = None
    best_val = _joint_val_score(graph, val_xy[0], val_xy[1], task_s) if use_es and val_xy is not None else None
    stall = 0
    steps_run = 0
    for step in range(int(steps)):
        def closure() -> Tensor:
            return _task_loss(graph(Xt), yt, task_s)

        if name == "adam":
            opt.zero_grad(set_to_none=True)
            loss_t = closure()
            loss_t.backward()
            opt.step()
        else:
            loss_t = opt.step(closure)  # type: ignore[arg-type]
        if loss_t is None:
            loss_val = float(closure().detach())
        elif isinstance(loss_t, Tensor):
            loss_val = float(loss_t.detach())
        else:
            loss_val = float(loss_t)
        history.append(loss_val)
        steps_run = step + 1
        if use_es and val_xy is not None:
            cur = _joint_val_score(graph, val_xy[0], val_xy[1], task_s)
            if best_val is None or cur > best_val + 1e-12:
                best_val = cur
                best_state = {k: v.detach().clone() for k, v in graph.state_dict().items()}
                stall = 0
            else:
                stall += 1
                if stall >= max(1, int(patience or 1)):
                    break

    if use_es and best_state is not None:
        graph.load_state_dict(best_state)
    val_metric = (
        _joint_val_score(graph, val_xy[0], val_xy[1], task_s) if val_xy is not None else None
    )
    return TrainResult(
        optimizer=name,
        steps=steps_run,
        train_loss=history[-1] if history else float("nan"),
        val_metric=val_metric,
        history=history,
        betas=[],
    )


__all__ = ["TrainResult", "fit_first_order", "fit_joint", "fit_second_order"]
