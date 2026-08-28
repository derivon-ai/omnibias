# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
r"""Joint TabPOU trainer (theory 05-05): threshold polish, trained embed, POU residual.

Temperature collapse: ``sigmoid(beta (x[f] - t))`` hardens as ``beta -> inf``.
Founding ``delta -> 0`` bias collapse does not appear.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
import torch
from omnibias.tab._core.loss import loss_value, metric, score_grad_hess
from omnibias.tab.pou.config import TabPOUConfig
from omnibias.tab.pou.model import TabPOU, tokens_from_scaled
from omnibias.tab.pou.train import TabPOUResult, _fit_residual, fit_tabpou
from omnibias.tab.torch.model import SoftTreeEnsemble
from torch import Tensor, nn

_EPS = 1e-12
_DTYPE = torch.float64


@dataclass(frozen=True)
class TabPOUJointConfig:
    r"""05-05 knobs on top of a v0 :class:`TabPOUConfig`."""

    base: TabPOUConfig
    polish_thresholds: bool = False
    polish_sweeps: int = 4
    train_embed: bool = False
    grouped_splits: bool = False
    joint_residual: bool = False
    joint_steps: int = 40
    binarize_eval: bool = False
    pairwise_warmstart: bool = False

    def __post_init__(self) -> None:
        if self.polish_sweeps < 0:
            raise ValueError("polish_sweeps must be >= 0")
        if self.joint_steps < 0:
            raise ValueError("joint_steps must be >= 0")


@dataclass
class TabPOUJointResult(TabPOUResult):
    polished: bool = False
    joint_trained: bool = False
    binarize_eval: bool = False
    grouped_splits: bool = False
    train_embed: bool = False
    polish_loss_ratio: float = 1.0


def _forest_forward_t(
    *,
    W: Tensor,
    t: Tensor,
    leaves: Tensor,
    b0: Tensor,
    codes: Tensor,
    X: Tensor,
    beta: float,
) -> Tensor:
    z = torch.einsum("nd,mjd->nmj", X, W) - t.unsqueeze(0)
    g = torch.sigmoid(float(beta) * z)
    gexp = g.unsqueeze(2)
    bexp = codes.view(1, 1, codes.shape[0], codes.shape[1])
    factors = bexp * gexp + (1.0 - bexp) * (1.0 - gexp)
    memb = factors.prod(dim=-1)
    return torch.einsum("nml,mlk->nk", memb, leaves) + b0.unsqueeze(0)


def polish_thresholds(
    forest: SoftTreeEnsemble,
    X: np.ndarray,
    residual: np.ndarray,
    weight: np.ndarray,
    *,
    n_sweeps: int = 4,
) -> float:
    r"""08-07 block search on ``t`` only; ``W`` / leaves frozen. Never-worse.

    Returns ``loss_after / max(loss_before, eps)``.
    """
    from omnibias.core.line_search import JetLineSearchConfig
    from omnibias.torch.optim_block_search import block_exact_search

    Xt = torch.as_tensor(np.asarray(X, dtype=np.float64), dtype=_DTYPE)
    rt = torch.as_tensor(np.asarray(residual, dtype=np.float64), dtype=_DTYPE)
    wt = torch.as_tensor(np.asarray(weight, dtype=np.float64), dtype=_DTYPE)
    W = forest.W.detach()
    leaves = forest.leaves.detach()
    b0 = forest.b0.detach()
    codes = forest._codes.detach()
    beta = float(forest.beta)
    t_cur = forest.t.detach().reshape(-1).clone()

    def loss_fn(t_flat: Tensor) -> Tensor:
        t = t_flat.reshape(forest.t.shape)
        pred = _forest_forward_t(
            W=W, t=t, leaves=leaves, b0=b0, codes=codes, X=Xt, beta=beta
        )
        return (wt * (pred - rt) ** 2).mean()

    loss0 = float(loss_fn(t_cur).detach())
    best = loss0
    for _ in range(int(n_sweeps)):
        mask = torch.ones(t_cur.numel(), dtype=torch.bool, device=t_cur.device)
        t_new, _rep = block_exact_search(
            loss_fn, t_cur.detach(), mask=mask, config=JetLineSearchConfig(verify=True)
        )
        t_new = t_new.reshape(-1).to(dtype=_DTYPE).detach()
        loss_n = float(loss_fn(t_new).detach())
        if loss_n <= best + 1e-15:
            t_cur = t_new
            best = loss_n
        else:
            break
    with torch.no_grad():
        forest.t.copy_(t_cur.reshape(forest.t.shape))
    return float(best / max(loss0, 1e-30))


class _JointGraph(nn.Module):
    r"""``forest(tokens(embedder(Xs))) + residual``; ``W`` stays frozen one-hot."""

    def __init__(self, model: TabPOU, *, train_embed: bool, train_t: bool, train_residual: bool) -> None:
        super().__init__()
        self.tab = model
        model.tree.W.requires_grad_(False)
        model.tree.leaves.requires_grad_(False)
        model.tree.b0.requires_grad_(False)
        model.tree.t.requires_grad_(bool(train_t))
        if model.embedder is not None:
            for p in model.embedder.parameters():
                p.requires_grad_(bool(train_embed))
        if model.residual is not None:
            for p in model.residual.parameters():
                p.requires_grad_(bool(train_residual))

    def forward(self, xs: Tensor) -> Tensor:
        if self.tab.embedder is not None:
            emb = self.tab.embedder(xs)
            tok = torch.cat([emb, xs], dim=-1) if self.tab.config.concat_raw else emb
        else:
            tok = xs
        return self.tab.forward(tok)


def _task_loss(F: Tensor, y: Tensor, task: str) -> Tensor:
    import torch.nn.functional as functional

    if task == "binary":
        return functional.binary_cross_entropy_with_logits(F[:, 0], y.reshape(-1))
    if task == "multiclass":
        return functional.cross_entropy(F, y)
    return functional.mse_loss(F, y.reshape(F.shape))


def _joint_train(
    model: TabPOU,
    Xs: np.ndarray,
    y: np.ndarray,
    *,
    task: str,
    steps: int,
    train_embed: bool,
    train_t: bool,
    train_residual: bool,
    Xs_val: np.ndarray | None,
    y_val: np.ndarray | None,
) -> None:
    from omnibias.torch.optim import TrustRegionNewtonCG

    graph = _JointGraph(
        model, train_embed=train_embed, train_t=train_t, train_residual=train_residual
    )
    xt = torch.as_tensor(np.asarray(Xs, dtype=np.float64), dtype=_DTYPE)
    if task == "multiclass":
        yt = torch.as_tensor(np.asarray(y).reshape(-1), dtype=torch.long)
    else:
        yt = torch.as_tensor(np.asarray(y, dtype=np.float64), dtype=_DTYPE)
    trainable = [p for p in graph.parameters() if p.requires_grad]
    if not trainable or int(steps) < 1:
        return
    opt = TrustRegionNewtonCG(trainable, cg_max_iter=8, radius=1.0)
    best_state = {k: v.detach().clone() for k, v in graph.state_dict().items()}
    best = float("inf")
    stall = 0
    for _ in range(int(steps)):
        def closure() -> Tensor:
            return _task_loss(graph(xt), yt, task)

        opt.step(closure)
        if Xs_val is not None and y_val is not None:
            with torch.no_grad():
                xv = torch.as_tensor(np.asarray(Xs_val, dtype=np.float64), dtype=_DTYPE)
                if task == "multiclass":
                    yv = torch.as_tensor(np.asarray(y_val).reshape(-1), dtype=torch.long)
                    cur = float(_task_loss(graph(xv), yv, task).detach())
                else:
                    yv = torch.as_tensor(np.asarray(y_val, dtype=np.float64), dtype=_DTYPE)
                    cur = float(_task_loss(graph(xv), yv, task).detach())
            if cur < best - 1e-12:
                best = cur
                best_state = {k: v.detach().clone() for k, v in graph.state_dict().items()}
                stall = 0
            else:
                stall += 1
                if stall >= 8:
                    break
        else:
            with torch.no_grad():
                cur = float(closure().detach())
            if cur < best - 1e-12:
                best = cur
                best_state = {k: v.detach().clone() for k, v in graph.state_dict().items()}
    graph.load_state_dict(best_state)


def fit_tabpou_joint(
    X: np.ndarray,
    y: np.ndarray,
    config: TabPOUJointConfig,
    *,
    val: tuple[np.ndarray, np.ndarray] | None = None,
) -> tuple[TabPOU, TabPOUJointResult]:
    r"""Boost (v0 grower) then optional polish / joint residual / hard eval."""
    base = config.base
    seq_residual = bool(base.use_residual) and not bool(config.joint_residual)
    pou_cfg = replace(base, use_residual=seq_residual)
    model, res = fit_tabpou(
        X,
        y,
        pou_cfg,
        val=val,
        freeze_embed=not bool(config.train_embed),
        pairwise=bool(config.pairwise_warmstart),
        grouped_splits=bool(config.grouped_splits),
    )
    X_all = np.asarray(X, dtype=np.float64)
    y_all = np.asarray(y)
    Xs = model.preprocessor.transform(X_all) if model.preprocessor is not None else X_all
    Xtok = tokens_from_scaled(
        Xs, model.embedder, concat_raw=bool(model.config.concat_raw) and model.embedder is not None
    )
    ratio = 1.0
    if config.polish_thresholds and int(config.polish_sweeps) > 0:
        F = model.tree.score(Xtok)
        g, h = score_grad_hess(F, y_all, pou_cfg.task)
        target = -g / np.clip(h, _EPS, None)
        ratio = polish_thresholds(
            model.tree, Xtok, target, h, n_sweeps=int(config.polish_sweeps)
        )
    Xs_val = None
    y_val = None
    if val is not None:
        Xv_raw, y_val = val
        Xs_val = (
            model.preprocessor.transform(np.asarray(Xv_raw, dtype=np.float64))
            if model.preprocessor is not None
            else np.asarray(Xv_raw, dtype=np.float64)
        )
    if config.joint_residual and model.residual is None:
        n = int(Xtok.shape[0])
        k = int(pou_cfg.n_outputs)
        F = model.tree.score(Xtok)
        if pou_cfg.task == "regression":
            resid = np.asarray(y_all, dtype=np.float64).reshape(n, k) - F
        elif pou_cfg.task == "binary":
            resid = np.asarray(y_all, dtype=np.float64).reshape(n, 1) - (
                1.0 / (1.0 + np.exp(-F))
            )
        else:
            g_last, _h = score_grad_hess(F, y_all, pou_cfg.task)
            resid = -g_last
        from omnibias.tab._core.forward import leaf_memberships

        pou_w = leaf_memberships(model.tree.to_params(), Xtok, float(pou_cfg.beta_final)).mean(axis=1)
        model.residual = _fit_residual(
            Xtok,
            resid,
            ensemble_k=int(pou_cfg.residual_k),
            hidden=int(pou_cfg.residual_hidden),
            steps=max(4, int(pou_cfg.residual_steps) // 4),
            lr=float(pou_cfg.residual_lr),
            seed=int(pou_cfg.seed) + 10_000,
            Xval=None,
            resid_val=None,
            pou_weights=pou_w,
            pou_weights_val=None,
            n_regions=int(2 ** int(pou_cfg.depth)),
        )
    jointed = False
    if (config.train_embed or config.joint_residual) and int(config.joint_steps) > 0:
        _joint_train(
            model,
            Xs,
            y_all,
            task=str(pou_cfg.task),
            steps=int(config.joint_steps),
            train_embed=bool(config.train_embed) and model.embedder is not None,
            train_t=bool(config.polish_thresholds) or bool(config.joint_residual),
            train_residual=bool(config.joint_residual) and model.residual is not None,
            Xs_val=Xs_val,
            y_val=None if y_val is None else np.asarray(y_val),
        )
        jointed = True
    model.binarize_eval = bool(config.binarize_eval)
    val_metric = res.val_metric
    if val is not None:
        val_metric = metric(model.score(val[0]), np.asarray(val[1]), pou_cfg.task)
    return model, TabPOUJointResult(
        n_stages=res.n_stages,
        learning_rate=res.learning_rate,
        train_loss=float(loss_value(model.score(X_all), y_all, pou_cfg.task)),
        val_metric=val_metric,
        history=res.history,
        used_residual=model.residual is not None,
        n_features_tok=res.n_features_tok,
        polished=bool(config.polish_thresholds),
        joint_trained=jointed,
        binarize_eval=bool(config.binarize_eval),
        grouped_splits=bool(config.grouped_splits),
        train_embed=bool(config.train_embed),
        polish_loss_ratio=float(ratio),
    )


__all__ = [
    "TabPOUJointConfig",
    "TabPOUJointResult",
    "fit_tabpou_joint",
    "polish_thresholds",
]
