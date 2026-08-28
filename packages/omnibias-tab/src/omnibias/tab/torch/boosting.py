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


def _fit_weak_learner_closed_form(
    cfg: SoftTreeConfig,
    X: np.ndarray,
    target: np.ndarray,
    weight: np.ndarray,
    *,
    seed: int,
) -> SoftTreeEnsemble:
    r"""Frozen-gate (axis: greedy) closed-form Newton leaves; ``b0 = 0``."""
    from omnibias.tab._core.forward import gate_activations, memberships_from_gates
    from omnibias.tab._core.leaves import closed_form_leaves
    from omnibias.tab._core.params import init_params

    Xv = np.asarray(X, dtype=np.float64)
    rv = np.asarray(target, dtype=np.float64)
    hv = np.asarray(weight, dtype=np.float64)
    n = Xv.shape[0]
    k = int(cfg.n_outputs)
    rv = rv.reshape(n, k)
    hv = hv.reshape(n, k)
    if cfg.split_kind == "axis":
        from omnibias.tab.pou.grow import fit_axis_tree

        params = fit_axis_tree(
            Xv, rv, hv,
            depth=cfg.depth, beta=float(cfg.beta_final), leaf_l2=float(cfg.leaf_l2),
            n_quantiles=16, colsample=1.0,
            rng=np.random.default_rng(seed), n_trees=cfg.n_trees,
        )
        # Preserve the caller's task on the returned module (grow uses regression).
        params.config = SoftTreeConfig(
            n_features=cfg.n_features, n_trees=cfg.n_trees, depth=cfg.depth,
            split_kind="axis", task="regression", n_outputs=k,
            beta_final=cfg.beta_final, leaf_l2=cfg.leaf_l2, seed=seed,
        )
    else:
        params = init_params(cfg, seed)
        G = gate_activations(params, Xv, float(cfg.beta_final))
        P = memberships_from_gates(G, cfg.depth)
        params.leaves = closed_form_leaves(P, rv, hv, float(cfg.leaf_l2))
        params.b0[:] = 0.0
    model = SoftTreeEnsemble(
        SoftTreeConfig(
            n_features=cfg.n_features, n_trees=cfg.n_trees, depth=cfg.depth,
            split_kind=cfg.split_kind, task="regression", n_outputs=k,
            beta_final=cfg.beta_final, leaf_l2=cfg.leaf_l2, seed=seed,
        ),
        params,
    )
    model.set_beta(cfg.beta_final)
    return model


def _fit_weak_learner(
    cfg: SoftTreeConfig,
    X: np.ndarray,
    target: np.ndarray,
    weight: np.ndarray,
    *,
    steps: int,
    lr: float,
    seed: int,
    leaf_solver: str = "adam",
) -> SoftTreeEnsemble:
    r"""Weighted least-squares fit of a fresh weak learner to the Newton target."""
    if leaf_solver == "closed_form":
        return _fit_weak_learner_closed_form(cfg, X, target, weight, seed=seed)
    if leaf_solver != "adam":
        raise ValueError(f"leaf_solver must be 'adam' or 'closed_form', got {leaf_solver!r}")
    torch.manual_seed(seed)
    reg_cfg = SoftTreeConfig(
        n_features=cfg.n_features, n_trees=cfg.n_trees, depth=cfg.depth,
        split_kind=cfg.split_kind,
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


def _fit_boosted_impl(
    X: np.ndarray,
    y: np.ndarray,
    config: SoftTreeConfig,
    *,
    n_stages: int,
    learning_rate: float,
    inner_steps: int,
    inner_lr: float,
    val: tuple[np.ndarray, np.ndarray] | None,
    patience: int | None,
    sample_weight: np.ndarray | None,
    leaf_solver: str = "adam",
) -> tuple[SoftTreeEnsemble, BoostResult]:
    r"""Shared stagewise-Newton-boosting loop behind :func:`fit_boosted` and
    :func:`fit_boosted_heteroscedastic`.

    ``sample_weight`` is ``None`` for :func:`fit_boosted` (every stage's weak learner is
    fit under the Hessian weight ``h`` alone, unchanged) or a frozen per-row multiplier
    (``1 / s_hat**2``, theory 05-03 section 4(d)) that is folded in as ``h * sample_weight``
    for :func:`fit_boosted_heteroscedastic`'s ``weighting="gls"`` -- the Newton target
    ``-g/h`` itself is untouched either way, only the weak-learner's least-squares weight
    changes. ``sample_weight=None`` multiplies by exactly ``1.0`` nowhere (the branch is
    skipped entirely), so :func:`fit_boosted`'s bit-identical output is unaffected by this
    refactor.
    """
    task, k = config.task, config.n_outputs
    Xtr = np.asarray(X, dtype=np.float64)
    n = Xtr.shape[0]
    sw = None if sample_weight is None else np.asarray(sample_weight, dtype=np.float64).reshape(n, 1)

    base = _base_score(y, task, k)  # (k,)
    F = np.tile(base[None, :], (n, 1))  # (n, k)
    Fval = None
    if val is not None:
        Fval = np.tile(base[None, :], (val[0].shape[0], 1))

    W_parts: list[np.ndarray] = []
    t_parts: list[np.ndarray] = []
    leaf_parts: list[np.ndarray] = []
    b0 = base.copy()
    b0_parts: list[np.ndarray] = []
    history: list[float] = []
    best_val = None if val is None else loss_value(Fval, val[1], task)
    best_n = 0
    stall = 0

    for stage in range(n_stages):
        g, h = score_grad_hess(F, y, task)  # (n, k)
        target = -g / np.clip(h, _EPS, None)
        weight = h if sw is None else h * sw
        weak = _fit_weak_learner(
            config, Xtr, target, weight, steps=inner_steps, lr=inner_lr,
            seed=config.seed + stage + 1, leaf_solver=leaf_solver,
        )
        wp = weak.to_params()
        W_parts.append(wp.W)
        t_parts.append(wp.t)
        leaf_parts.append(learning_rate * wp.leaves)
        b0_delta = learning_rate * wp.b0
        b0_parts.append(b0_delta)
        b0 = b0 + b0_delta

        F = F + learning_rate * weak.score(Xtr)
        if val is not None and Fval is not None:
            Fval = Fval + learning_rate * weak.score(val[0])
        history.append(loss_value(F, y, task))
        if val is not None and Fval is not None and patience is not None:
            cur = loss_value(Fval, val[1], task)
            if best_val is None or cur < best_val - 1e-12:
                best_val = cur
                best_n = stage + 1
                stall = 0
            else:
                stall += 1
                if stall >= max(1, int(patience)):
                    break

    keep = best_n if (val is not None and patience is not None and best_n > 0) else len(W_parts)
    keep = max(1, min(keep, len(W_parts)))
    W_parts = W_parts[:keep]
    t_parts = t_parts[:keep]
    leaf_parts = leaf_parts[:keep]
    b0 = base.copy()
    for delta in b0_parts[:keep]:
        b0 = b0 + delta
    total_cfg = SoftTreeConfig(
        n_features=config.n_features,
        n_trees=keep * config.n_trees,
        depth=config.depth,
        split_kind=config.split_kind,
        task=task,
        n_outputs=k,
        beta_final=config.beta_final,
        leaf_l2=config.leaf_l2,
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
    if val is not None:
        val_metric = _metric(model.score(val[0]), np.asarray(val[1]), task)

    return model, BoostResult(
        n_stages=keep,
        learning_rate=learning_rate,
        train_loss=history[-1] if history else float("nan"),
        val_metric=val_metric,
        history=history,
    )


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
    patience: int | None = None,
    encoder: object | None = None,
    leaf_solver: str = "adam",
) -> tuple[SoftTreeEnsemble, BoostResult]:
    r"""Fit a Newton-boosted soft-tree ensemble; returns ``(model, BoostResult)``.

    ``config`` describes **one stage** (its ``n_trees`` / ``depth`` are the weak-learner
    shape); the returned :class:`SoftTreeEnsemble` holds ``n_stages * config.n_trees`` trees.
    Optional ``patience`` (with ``val``) keeps the prefix with best validation loss.

    ``encoder`` is rejected: stagewise numpy boosting does not jointly train an
    encoder. Use :func:`~omnibias.tab.torch.train.fit_joint` or
    :func:`~omnibias.tab.torch.train.fit_second_order`.
    """
    if encoder is not None:
        raise TypeError(
            "encoder= is not supported on the stagewise GBM-mirror trainers; "
            "use fit_joint or fit_second_order"
        )
    return _fit_boosted_impl(
        X, y, config,
        n_stages=n_stages, learning_rate=learning_rate,
        inner_steps=inner_steps, inner_lr=inner_lr,
        val=val, patience=patience, sample_weight=None, leaf_solver=leaf_solver,
    )


def fit_boosted_heteroscedastic(
    X: np.ndarray,
    y: np.ndarray,
    config: SoftTreeConfig,
    *,
    log_scale: np.ndarray,
    n_stages: int = 30,
    learning_rate: float = 0.3,
    inner_steps: int = 60,
    inner_lr: float = 0.05,
    weighting: str = "gls",
    val: tuple[np.ndarray, np.ndarray] | None = None,
    patience: int | None = None,
) -> tuple[SoftTreeEnsemble, BoostResult]:
    r"""``fit_boosted`` with the weak-learner weight ``h -> h / s_hat**2`` (theory 05-03
    section 4(d)) -- a generalized-least-squares reweighting of the boosting target, in the
    spirit of NGBoost (Duan et al. 2020): rows the frozen estimator ``s_hat`` calls
    low-noise get more say in each stage's weak-learner fit. This is a **different,
    older mechanism** than :func:`omnibias.tab.torch.heteroscedastic.fit_noise_aware`'s
    zero-gradient curvature penalty (section 4(b)) -- it changes the fitted function itself,
    not merely the optimizer's local curvature -- and is included because
    :func:`_fit_weak_learner`'s ``weight`` parameter already exists as the exact hook this
    needs (three lines, not new machinery), not presented as a novel idea.

    ``log_scale`` is a **frozen** per-row ``log(s)`` of length ``n`` (e.g.
    :func:`omnibias.tab.bench.fit_predict_catboost_uncertainty`'s output on train --
    section 4(c)'s anti-circularity requirement applies here exactly as it does to
    ``fit_noise_aware``). ``weighting="shrinkage"`` ignores ``log_scale`` and reproduces
    :func:`fit_boosted` bit-identically -- a second G0-style plumbing check, distinct from
    the ``weighting="gls"`` hypothesis gate (G3).
    """
    if weighting not in ("gls", "shrinkage"):
        raise ValueError(f"weighting must be 'gls' or 'shrinkage', got {weighting!r}")
    if config.task != "regression":
        raise ValueError(
            f"fit_boosted_heteroscedastic is regression-only (theory 05-03 section 10), "
            f"got task={config.task!r}"
        )
    n = int(np.asarray(X, dtype=np.float64).shape[0])
    if weighting == "shrinkage":
        sample_weight = None
    else:
        log_scale_arr = np.asarray(log_scale, dtype=np.float64).reshape(-1)
        if log_scale_arr.shape[0] != n:
            raise ValueError(f"log_scale must have length n={n}, got {log_scale_arr.shape[0]}")
        sample_weight = np.exp(-2.0 * log_scale_arr)  # 1 / s_hat**2
    return _fit_boosted_impl(
        X, y, config,
        n_stages=n_stages, learning_rate=learning_rate,
        inner_steps=inner_steps, inner_lr=inner_lr,
        val=val, patience=patience, sample_weight=sample_weight,
    )


__all__ = ["BoostResult", "fit_boosted", "fit_boosted_heteroscedastic"]
