# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
r"""Stagewise Newton boosting for TabPOU (theory 05-04).

Each stage grows one (or ``trees_per_stage``) axis-aligned oblivious tree with
closed-form Newton leaves. Optional TabM residual is fit on ``y - F_boost``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from omnibias.tab._core.config import SoftTreeConfig
from omnibias.tab._core.forward import forward_np, leaf_memberships
from omnibias.tab._core.loss import loss_value, metric, score_grad_hess
from omnibias.tab._core.params import TabParams
from omnibias.tab.pou.config import TabPOUConfig
from omnibias.tab.pou.ensemble import TabMResidual
from omnibias.tab.pou.grow import fit_axis_tree, token_feature_groups
from omnibias.tab.pou.model import TabPOU, make_embedder, tokens_from_scaled
from omnibias.tab.pou.preprocess import TabPreprocessor
from omnibias.tab.torch.model import SoftTreeEnsemble

_EPS = 1e-12
_DTYPE = torch.float64


@dataclass
class TabPOUResult:
    n_stages: int
    learning_rate: float
    train_loss: float
    val_metric: float | None
    history: list[float]
    used_residual: bool
    n_features_tok: int


def _base_score(y: np.ndarray, task: str, k: int) -> np.ndarray:
    if task == "binary":
        p = float(np.clip(np.mean(np.asarray(y, dtype=np.float64)), _EPS, 1.0 - _EPS))
        return np.array([np.log(p / (1.0 - p))], dtype=np.float64)
    if task == "multiclass":
        idx = np.asarray(y, dtype=np.int64).reshape(-1)
        freq = np.clip(np.bincount(idx, minlength=k).astype(np.float64) / idx.shape[0], _EPS, 1.0)
        return np.asarray(np.log(freq), dtype=np.float64)
    yv = np.asarray(y, dtype=np.float64)
    yv = yv.reshape(-1, k) if k > 1 else yv.reshape(-1, 1)
    return np.mean(yv, axis=0)


def _concat_trees(parts: list[TabParams], base: np.ndarray, cfg: SoftTreeConfig) -> TabParams:
    W = np.concatenate([p.W for p in parts], axis=0)
    t = np.concatenate([p.t for p in parts], axis=0)
    leaves = np.concatenate([p.leaves for p in parts], axis=0)
    total = SoftTreeConfig(
        n_features=cfg.n_features,
        n_trees=sum(p.config.n_trees for p in parts),
        depth=cfg.depth,
        split_kind=cfg.split_kind,
        task=cfg.task,
        n_outputs=cfg.n_outputs,
        beta_final=cfg.beta_final,
        leaf_l2=cfg.leaf_l2,
        seed=cfg.seed,
    )
    return TabParams(total, W, t, leaves, base)


def _fit_residual(
    Xtok: np.ndarray,
    resid: np.ndarray,
    *,
    ensemble_k: int,
    hidden: int,
    steps: int,
    lr: float,
    seed: int,
    Xval: np.ndarray | None,
    resid_val: np.ndarray | None,
    pou_weights: np.ndarray | None = None,
    pou_weights_val: np.ndarray | None = None,
    n_regions: int | None = None,
) -> TabMResidual:
    torch.manual_seed(int(seed))
    n_out = int(resid.shape[1])
    model = TabMResidual(
        int(Xtok.shape[1]),
        n_out,
        ensemble_k=ensemble_k,
        hidden=hidden,
        n_regions=n_regions,
    )
    Xt = torch.as_tensor(Xtok, dtype=_DTYPE)
    yt = torch.as_tensor(resid, dtype=_DTYPE)
    Wt = None if pou_weights is None else torch.as_tensor(pou_weights, dtype=_DTYPE)
    from omnibias.torch.optim import TrustRegionNewtonCG

    opt = TrustRegionNewtonCG(model.parameters(), cg_max_iter=8, radius=max(float(lr), 1.0))
    best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    best = float("inf")
    stall = 0

    def _pred(x: torch.Tensor, w: torch.Tensor | None) -> torch.Tensor:
        if w is None:
            return model(x)
        return model(x, pou_weights=w)

    for _ in range(int(steps)):
        def closure() -> torch.Tensor:
            return ((_pred(Xt, Wt) - yt) ** 2).mean()

        opt.step(closure)
        if Xval is not None and resid_val is not None:
            with torch.no_grad():
                Wv = None if pou_weights_val is None else torch.as_tensor(pou_weights_val, dtype=_DTYPE)
                pv = _pred(torch.as_tensor(Xval, dtype=_DTYPE), Wv)
                cur = float(((pv - torch.as_tensor(resid_val, dtype=_DTYPE)) ** 2).mean())
            if cur < best - 1e-12:
                best = cur
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
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
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    return model


def fit_tabpou(
    X: np.ndarray,
    y: np.ndarray,
    config: TabPOUConfig,
    *,
    val: tuple[np.ndarray, np.ndarray] | None = None,
    feature_groups: list[np.ndarray] | None = None,
    freeze_embed: bool = True,
    pairwise: bool = False,
    grouped_splits: bool = False,
) -> tuple[TabPOU, TabPOUResult]:
    r"""Fit TabPOU; returns ``(model, TabPOUResult)``.

    Preprocess (robust scale + optional one-hot) is fit on ``X`` only. Band
    thresholds are frozen quantile edges of the scaled train matrix unless
    ``freeze_embed=False`` (theory 05-05; still quantile-init, then a later
    joint stage may train them).
    """
    task, k = config.task, config.n_outputs
    rng = np.random.default_rng(int(config.seed))
    X_all = np.asarray(X, dtype=np.float64)
    y_all = np.asarray(y)
    if val is None and config.patience is not None and X_all.shape[0] >= 10:
        n_all = X_all.shape[0]
        n_val = max(1, int(round(0.2 * n_all)))
        perm = rng.permutation(n_all)
        va_idx, tr_idx = perm[:n_val], perm[n_val:]
        Xtr, yv = X_all[tr_idx], y_all[tr_idx]
        val = (X_all[va_idx], y_all[va_idx])
    else:
        Xtr, yv = X_all, y_all
    prep: TabPreprocessor | None = None
    if config.robust_scale or config.onehot_max_card >= 2:
        prep = TabPreprocessor(
            clip=float(config.clip),
            onehot_max_card=int(config.onehot_max_card),
            robust_scale=bool(config.robust_scale),
        )
        Xs = prep.fit_transform(Xtr)
    else:
        Xs = Xtr
    embedder = make_embedder(Xs, config, freeze=freeze_embed) if config.use_embed else None
    Xtok = tokens_from_scaled(Xs, embedder, concat_raw=bool(config.concat_raw) and embedder is not None)
    if grouped_splits and config.use_embed and feature_groups is None:
        feature_groups = token_feature_groups(
            int(Xs.shape[1]),
            n_bins=int(config.n_bins),
            concat_raw=bool(config.concat_raw),
            use_embed=True,
        )
    n, d_tok = Xtok.shape
    base = _base_score(yv, task, k)
    F = np.tile(base[None, :], (n, 1))
    Fval = None
    Xtok_val = None
    if val is not None:
        Xv_raw, _y_val = val
        Xs_val = prep.transform(Xv_raw) if prep is not None else np.asarray(Xv_raw, dtype=np.float64)
        Xtok_val = tokens_from_scaled(
            Xs_val, embedder, concat_raw=bool(config.concat_raw) and embedder is not None
        )
        Fval = np.tile(base[None, :], (Xtok_val.shape[0], 1))

    tree_cfg = SoftTreeConfig(
        n_features=d_tok,
        n_trees=int(config.trees_per_stage),
        depth=int(config.depth),
        split_kind=str(config.split_kind),
        task=task,
        n_outputs=k,
        beta_final=float(config.beta_final),
        leaf_l2=float(config.leaf_l2),
        seed=int(config.seed),
    )
    nu = float(config.learning_rate)
    beta = float(config.beta_final)
    parts: list[TabParams] = []
    b0 = base.copy()
    history: list[float] = []
    best_val = None if Fval is None else loss_value(Fval, val[1], task)  # type: ignore[index]
    best_n = 0
    stall = 0
    n_keep_rows = max(1, int(round(float(config.subsample) * n)))

    for stage in range(int(config.n_stages)):
        g, h = score_grad_hess(F, yv, task)
        target = -g / np.clip(h, _EPS, None)
        if n_keep_rows < n:
            idx = rng.choice(n, size=n_keep_rows, replace=False)
            Xfit, rfit, hfit = Xtok[idx], target[idx], h[idx]
        else:
            Xfit, rfit, hfit = Xtok, target, h
        weak = fit_axis_tree(
            Xfit,
            rfit,
            hfit,
            depth=int(config.depth),
            beta=beta,
            leaf_l2=float(config.leaf_l2),
            n_quantiles=int(config.n_quantiles),
            colsample=float(config.colsample),
            rng=np.random.default_rng(int(config.seed) + stage + 1),
            n_trees=int(config.trees_per_stage),
            feature_groups=feature_groups,
            pairwise=bool(pairwise) and int(config.depth) == 2,
        )
        weak.leaves = nu * weak.leaves
        weak.b0 = nu * weak.b0
        parts.append(weak)
        b0 = b0 + weak.b0
        F = F + forward_np(weak, Xtok, beta)
        if Fval is not None and Xtok_val is not None:
            Fval = Fval + forward_np(weak, Xtok_val, beta)
        history.append(loss_value(F, yv, task))
        if Fval is not None and val is not None and config.patience is not None:
            cur = loss_value(Fval, val[1], task)
            if best_val is None or cur < best_val - 1e-12:
                best_val = cur
                best_n = stage + 1
                stall = 0
            else:
                stall += 1
                if stall >= max(1, int(config.patience)):
                    break

    keep = best_n if (val is not None and config.patience is not None and best_n > 0) else len(parts)
    keep = max(1, min(keep, len(parts)))
    parts = parts[:keep]
    b0 = base.copy()
    for p in parts:
        b0 = b0 + p.b0
    params = _concat_trees(parts, b0, tree_cfg)
    forest = SoftTreeEnsemble(params.config, params)
    forest.set_beta(beta)
    F = np.tile(base[None, :], (n, 1))
    for p in parts:
        F = F + forward_np(p, Xtok, beta)
    if Fval is not None and Xtok_val is not None:
        Fval = np.tile(base[None, :], (Xtok_val.shape[0], 1))
        for p in parts:
            Fval = Fval + forward_np(p, Xtok_val, beta)

    residual_mod = None
    if config.use_residual:
        if task == "regression":
            y_mat = np.asarray(yv, dtype=np.float64).reshape(n, k)
            resid = y_mat - F
        elif task == "binary":
            resid = np.asarray(yv, dtype=np.float64).reshape(n, 1) - (1.0 / (1.0 + np.exp(-F)))
        else:
            g_last, _h_last = score_grad_hess(F, yv, task)
            resid = -g_last
        resid_val = None
        if Xtok_val is not None and val is not None and task == "regression":
            yvm = np.asarray(val[1], dtype=np.float64).reshape(Xtok_val.shape[0], k)
            resid_val = yvm - Fval if Fval is not None else None
        pou = leaf_memberships(params, Xtok, beta).mean(axis=1)
        pou_val = (
            None
            if Xtok_val is None
            else leaf_memberships(params, Xtok_val, beta).mean(axis=1)
        )
        residual_mod = _fit_residual(
            Xtok,
            resid,
            ensemble_k=int(config.residual_k),
            hidden=int(config.residual_hidden),
            steps=int(config.residual_steps),
            lr=float(config.residual_lr),
            seed=int(config.seed) + 10_000,
            Xval=Xtok_val,
            resid_val=resid_val,
            pou_weights=pou,
            pou_weights_val=pou_val,
            n_regions=int(2 ** int(config.depth)),
        )

    model = TabPOU(
        config,
        forest,
        preprocessor=prep,
        embedder=embedder,
        residual=residual_mod,
    )
    val_metric = None
    if val is not None:
        val_metric = metric(model.score(val[0]), np.asarray(val[1]), task)
    return model, TabPOUResult(
        n_stages=keep,
        learning_rate=nu,
        train_loss=loss_value(F, yv, task) if parts else float("nan"),
        val_metric=val_metric,
        history=history[:keep] if history else history,
        used_residual=residual_mod is not None,
        n_features_tok=d_tok,
    )


__all__ = ["TabPOUResult", "fit_tabpou"]
