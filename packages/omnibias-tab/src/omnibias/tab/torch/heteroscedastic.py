# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
r"""Noise-aware regression trainers for the soft-tree ensemble (theory 05-03).

Two independent mechanisms, both regression-only (section 10 of the spec):

1. :class:`HeteroscedasticHead` + :func:`fit_heteroscedastic` -- two
   :class:`~omnibias.tab.torch.model.SoftTreeEnsemble`\ s jointly fit by the
   closed-form Gaussian NLL of ``omnibias.tab._core.heteroscedastic``, predicting a
   mean **and** a per-row ``log_scale``.
2. :func:`fit_noise_aware` -- ordinary :func:`~omnibias.tab.torch.train.fit_second_order`
   plus a **zero-gradient Gauss-Newton penalty** added inside the closure. Let
   ``phi(x; theta)`` be the model score and ``s_i`` a *frozen*, externally supplied
   per-row noise estimate (e.g. :func:`omnibias.tab.bench.fit_predict_catboost_uncertainty`,
   never this model's own output -- anti-circularity is load-bearing, not decoration).
   Every ``closure()`` call adds

   .. code-block:: python

       resid = F - F.detach()  # an identity: exactly 0, at every theta, always
       penalty = (lam / n) * (s2 * resid * resid).sum()

   ``resid`` is *exactly* the zero tensor at whatever ``theta`` the closure is
   evaluated at (subtracting a detached copy of a tensor from itself), so ``penalty``
   contributes **exactly zero** to the loss value and to the gradient the optimizer
   follows -- but its second derivative, extracted by the trust-region / cubic-Newton
   optimizer's ``create_graph=True`` double-backward, is exactly
   ``lam * Sigma_noise`` with ``Sigma_noise = (2/n) sum_i s_i**2 g_i g_i^T`` and
   ``g_i = d phi_i/d theta``. This *shrinks* the Newton step specifically along
   directions high-noise rows are sensitive to. ``lam=0.0`` makes ``penalty``
   identically the Python float ``0.0`` (not merely small), so
   :func:`fit_noise_aware` reproduces :func:`~omnibias.tab.torch.train.fit_second_order`
   bit-for-bit -- gate G0, true by construction (see the module tests).

No collapse limit (founding or temperature) is invoked by either mechanism: ``lam``
is a fixed, chosen scalar, not a limit of anything, and the ``beta`` gate anneal
already present in :class:`~omnibias.tab._core.config.SoftTreeConfig` is the
(unrelated) temperature collapse, untouched here.
"""

from __future__ import annotations

import math
from dataclasses import replace
from typing import Any

import numpy as np
import torch
from omnibias.tab._core.config import SoftTreeConfig
from omnibias.tab.torch.model import SoftTreeEnsemble
from omnibias.tab.torch.train import TrainResult, _as_xy, _task_loss, _val_metric
from torch import Tensor, nn

_DTYPE = torch.float64
_HALF_LOG_2PI = 0.5 * math.log(2.0 * math.pi)
_NOISE_AWARE_OPTIMIZERS = ("trust_region", "cubic")


def gaussian_nll_elementwise(u: Tensor, v: Tensor, y: Tensor) -> Tensor:
    r"""Per-element heteroscedastic Gaussian NLL, the torch twin of
    :func:`omnibias.tab._core.heteroscedastic.gaussian_nll`, used inside a training
    closure so the exact-curvature optimizers can differentiate through it.
    """
    resid = y - u
    return _HALF_LOG_2PI + v + 0.5 * resid * resid * torch.exp(-2.0 * v)


class HeteroscedasticHead(nn.Module):
    r"""Two :class:`SoftTreeEnsemble`\ s sharing ``config``, predicting ``(f_hat, log_scale)``.

    Both heads use ``config.n_features`` / ``config.depth`` / the same ``beta`` anneal
    schedule; by default they also share ``config.n_trees`` (only their seeds differ,
    so the two heads do not start from identical parameters). ``scale_width``
    optionally narrows the ``log_scale`` head's ``n_trees`` -- the variance surface is
    often smoother than the mean, so a cheaper sub-network is a reasonable default
    knob, not a requirement.
    """

    def __init__(self, config: SoftTreeConfig, *, scale_width: int | None = None) -> None:
        super().__init__()
        if config.task != "regression":
            raise ValueError(f"HeteroscedasticHead is regression-only, got task={config.task!r}")
        self.config = config
        self.mean = SoftTreeEnsemble(config)
        scale_config = replace(
            config,
            n_trees=config.n_trees if scale_width is None else int(scale_width),
            seed=config.seed + 1,
        )
        self.log_scale_net = SoftTreeEnsemble(scale_config)

    def forward(self, X: Tensor) -> tuple[Tensor, Tensor]:
        r"""``(f_hat, log_scale)``, each of shape ``(..., n_outputs)``."""
        return self.mean(X), self.log_scale_net(X)

    def set_beta(self, beta: float) -> None:
        self.mean.set_beta(beta)
        self.log_scale_net.set_beta(beta)


def _nll_val_metric(
    model: HeteroscedasticHead, val: tuple[np.ndarray, np.ndarray] | None, beta: float
) -> float | None:
    if val is None:
        return None
    Xv, yv = val
    Xt, yt = _as_xy(Xv, yv, "regression")
    model.set_beta(beta)
    with torch.no_grad():
        u, v = model(Xt)
        nll = gaussian_nll_elementwise(u.reshape(-1), v.reshape(-1), yt.reshape(-1)).mean()
    return float(-nll.detach())


def fit_heteroscedastic(
    model: HeteroscedasticHead,
    X: np.ndarray,
    y: np.ndarray,
    *,
    steps: int = 60,
    optimizer: str = "trust_region",
    anneal: bool = True,
    leaf_l2: float | None = None,
    weight_l2: float = 1e-4,
    val: tuple[np.ndarray, np.ndarray] | None = None,
    patience: int | None = None,
    **opt_kwargs: Any,
) -> TrainResult:
    r"""Jointly train ``(f_hat, log_scale)`` by the closed-form Gaussian NLL (section 4(a)).

    ``val_metric`` in the returned :class:`TrainResult` is the **negative** mean NLL on
    ``val`` (higher is better, matching :func:`omnibias.tab._core.loss.metric`'s
    ``-rmse`` convention for plain regression).
    """
    if optimizer not in _NOISE_AWARE_OPTIMIZERS:
        raise ValueError(
            f"fit_heteroscedastic supports optimizer in {_NOISE_AWARE_OPTIMIZERS}, "
            f"got {optimizer!r} ('kfac' does not extend to a two-head model)"
        )
    config = model.config
    l2 = config.leaf_l2 if leaf_l2 is None else float(leaf_l2)
    Xt, yt = _as_xy(X, y, "regression")
    yt = yt.reshape(-1)

    from omnibias.torch.optim import CubicNewton, TrustRegionNewtonCG

    opt: torch.optim.Optimizer
    if optimizer == "trust_region":
        opt = TrustRegionNewtonCG(model.parameters(), **opt_kwargs)
    else:
        opt = CubicNewton(model.parameters(), **opt_kwargs)

    history: list[float] = []
    betas: list[float] = []
    use_es = val is not None and patience is not None
    best_state: dict[str, Tensor] | None = None
    best_val = _nll_val_metric(model, val, config.beta_final) if use_es else None
    stall = 0
    steps_run = 0
    for step in range(steps):
        beta = config.beta_at(step) if anneal else config.beta_final
        model.set_beta(beta)

        def closure() -> Tensor:
            u, v = model(Xt)
            nll = gaussian_nll_elementwise(u.reshape(-1), v.reshape(-1), yt).mean()
            reg = (
                l2 * (model.mean.leaves**2).mean()
                + weight_l2 * (model.mean.W**2).mean()
                + l2 * (model.log_scale_net.leaves**2).mean()
                + weight_l2 * (model.log_scale_net.W**2).mean()
            )
            return nll + reg

        loss_t = opt.step(closure)  # type: ignore[arg-type]
        loss_val = float(loss_t) if loss_t is not None else float(closure().detach())
        history.append(loss_val)
        betas.append(beta)
        steps_run = step + 1
        if use_es:
            cur = _nll_val_metric(model, val, beta)
            if cur is not None and (best_val is None or cur > best_val + 1e-12):
                best_val = cur
                best_state = {k: v2.detach().clone() for k, v2 in model.state_dict().items()}
                stall = 0
            else:
                stall += 1
                if stall >= max(1, int(patience or 1)):
                    break

    if use_es and best_state is not None:
        model.load_state_dict(best_state)
    model.set_beta(config.beta_final)
    return TrainResult(
        optimizer=optimizer,
        steps=steps_run,
        train_loss=history[-1] if history else float("nan"),
        val_metric=_nll_val_metric(model, val, config.beta_final),
        history=history,
        betas=betas,
    )


def fit_noise_aware(
    model: SoftTreeEnsemble,
    X: np.ndarray,
    y: np.ndarray,
    *,
    log_scale: np.ndarray,
    lam: float = 1.0,
    optimizer: str = "trust_region",
    steps: int = 60,
    anneal: bool = True,
    leaf_l2: float | None = None,
    weight_l2: float = 1e-4,
    val: tuple[np.ndarray, np.ndarray] | None = None,
    patience: int | None = None,
    **opt_kwargs: Any,
) -> TrainResult:
    r"""``fit_second_order`` plus the closure-local zero-gradient penalty (section 4(b)).

    ``log_scale`` is a **frozen** per-row ``log(s)`` array of length ``n`` (section
    4(c); never trained here -- pass an independent estimator's output, e.g.
    :func:`omnibias.tab.bench.fit_predict_catboost_uncertainty`). ``lam=0.0``
    reproduces :func:`~omnibias.tab.torch.train.fit_second_order` bit-identically
    (gate G0): the penalty term is then the exact Python float ``0.0`` added to the
    loss at every step, not merely a small number, so it changes neither the reported
    loss nor any parameter update. ``optimizer='kfac'`` is out of scope -- its
    additive-only reparam path does not carry the per-row score tensor this penalty
    needs.
    """
    if model.config.task != "regression":
        raise ValueError(
            f"fit_noise_aware is regression-only (theory 05-03 section 10), "
            f"got task={model.config.task!r}"
        )
    if optimizer not in _NOISE_AWARE_OPTIMIZERS:
        raise ValueError(
            f"fit_noise_aware supports optimizer in {_NOISE_AWARE_OPTIMIZERS}, got {optimizer!r} "
            "('kfac' is out of scope, see theory 05-03 section 6(a'))"
        )
    task = "regression"
    l2 = model.config.leaf_l2 if leaf_l2 is None else float(leaf_l2)
    Xt, yt = _as_xy(X, y, task)
    n = int(Xt.shape[0])
    s_raw = np.asarray(log_scale, dtype=np.float64)
    if s_raw.size != n:
        raise ValueError(f"log_scale must have length n={n}, got {s_raw.size}")
    s_arr = s_raw.reshape(n)
    s2 = torch.as_tensor(np.exp(2.0 * s_arr), dtype=_DTYPE).reshape(n, 1)
    lam_over_n = float(lam) / float(n)

    from omnibias.torch.optim import CubicNewton, TrustRegionNewtonCG

    opt: torch.optim.Optimizer
    if optimizer == "trust_region":
        opt = TrustRegionNewtonCG(model.parameters(), **opt_kwargs)
    else:
        opt = CubicNewton(model.parameters(), **opt_kwargs)

    history: list[float] = []
    betas: list[float] = []
    use_es = val is not None and patience is not None
    best_state: dict[str, Tensor] | None = None
    best_val = _val_metric(model, val, model.config.beta_final) if use_es else None
    stall = 0
    steps_run = 0
    for step in range(steps):
        beta = model.config.beta_at(step) if anneal else model.config.beta_final
        model.set_beta(beta)

        def closure() -> Tensor:
            F = model(Xt)
            loss = _task_loss(F, yt, task)
            resid = F - F.detach()  # exactly 0 at every theta -- see the module docstring
            penalty = lam_over_n * (s2 * resid * resid).sum()
            loss = loss + penalty + l2 * (model.leaves**2).mean() + weight_l2 * (model.W**2).mean()
            return loss

        loss_t = opt.step(closure)  # type: ignore[arg-type]
        loss_val = float(loss_t) if loss_t is not None else float(closure().detach())
        history.append(loss_val)
        betas.append(beta)
        steps_run = step + 1
        if use_es:
            cur = _val_metric(model, val, beta)
            if cur is not None and (best_val is None or cur > best_val + 1e-12):
                best_val = cur
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
                stall = 0
            else:
                stall += 1
                if stall >= max(1, int(patience or 1)):
                    break

    if use_es and best_state is not None:
        model.load_state_dict(best_state)
    model.set_beta(model.config.beta_final)
    return TrainResult(
        optimizer=optimizer,
        steps=steps_run,
        train_loss=history[-1] if history else float("nan"),
        val_metric=_val_metric(model, val, model.config.beta_final),
        history=history,
        betas=betas,
    )


__all__ = [
    "HeteroscedasticHead",
    "fit_heteroscedastic",
    "fit_noise_aware",
    "gaussian_nll_elementwise",
]
