# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Torch linearized multi-stage residual correction for CCF vorticity.

Torch twin of :func:`omnibias.pinn.jax.discovery.multistage.train_stage2_correction`:

* Fourier-feature stage-2 network with ``sigma = 2 π f_d`` from residual spectrum;
* linearized loss targeting ``R0 + eps * D[Phi0] Phi1 ≈ 0``;
* optional stage-2 normalization by ``|∂y Ω0|`` (DeepMind CCF follow-up).

Optimizer honesty
-----------------
Default stage-2 steps use Adam as a **labeled** ``stage2_heuristic`` (numpy FD
linearization does not expose a clean residual vector for Gauss–Newton).
Optional ``optimizer="gauss_newton"`` runs a **corr-matching quadratic proxy**
(``gauss_newton_corr_proxy``): fit ``Φ₁ ≈ -R₀/ε`` via
:class:`~omnibias.torch.optim.GaussNewton`. That is **not** linearized Wang
``R₀ + ε D[Φ₀]Φ₁`` Gauss–Newton — do not claim Martens–Grosse stage-2.
Neither mode forges Rung-1; reproduction gates on dense Wang residual only.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any, Literal

import numpy as np
import torch
import torch.nn as nn
from omnibias.pinn.jax.discovery.multistage import (
    MultiStageConfig,
    compose_profiles,
    dominant_residual_frequency,
)
from torch import Tensor

Stage2Optimizer = Literal["adam", "gauss_newton"]


class _FourierCorrector(nn.Module):
    def __init__(self, *, n_fourier: int, hidden: int, sigma: float, seed: int) -> None:
        super().__init__()
        g = torch.Generator().manual_seed(int(seed))
        b = torch.randn(int(n_fourier), generator=g, dtype=torch.float64) * float(sigma)
        self.register_buffer("B", b)
        self.W = nn.Linear(2 * int(n_fourier), int(hidden), bias=True, dtype=torch.float64)
        self.c = nn.Linear(int(hidden), 1, bias=True, dtype=torch.float64)
        nn.init.normal_(self.W.weight, std=0.1)
        nn.init.zeros_(self.W.bias)
        nn.init.normal_(self.c.weight, std=0.1)
        nn.init.zeros_(self.c.bias)

    def forward(self, y: Tensor) -> Tensor:
        proj = y.reshape(-1, 1) * self.B.reshape(1, -1)
        feat = torch.cat([torch.cos(proj), torch.sin(proj)], dim=-1)
        h = torch.tanh(self.W(feat))
        return self.c(h).reshape(-1)


def linearized_operator_action_numpy(
    residual_fn: Callable[[np.ndarray], np.ndarray],
    stage1: np.ndarray,
    direction: np.ndarray,
    *,
    fd_eps: float,
) -> np.ndarray:
    """Finite-difference action of ``D[Phi0]`` on ``direction``."""
    eps = float(fd_eps)
    rp = np.asarray(residual_fn(stage1 + eps * direction), dtype=float)
    rm = np.asarray(residual_fn(stage1 - eps * direction), dtype=float)
    return (rp - rm) / (2.0 * eps)


def _stage2_loss_tensors(
    model: _FourierCorrector,
    y_t: Tensor,
    *,
    stage1_np: np.ndarray,
    r0: np.ndarray,
    r0_t: Tensor,
    residual_fn: Callable[[np.ndarray], np.ndarray],
    cfg: MultiStageConfig,
    eps: float,
    fd: float,
    norm_np: np.ndarray | None,
) -> tuple[Tensor, float]:
    corr = model(y_t)
    corr_np = corr.detach().cpu().numpy()
    if cfg.linearized:
        d_action = linearized_operator_action_numpy(
            residual_fn, stage1_np, corr_np, fd_eps=fd
        )
        r_lin = r0 + eps * d_action
        if norm_np is not None:
            r_lin = r_lin / norm_np
        lin_loss = 0.5 * float(np.mean(r_lin * r_lin))
        target = -r0_t / max(eps, 1e-12)
        w = torch.as_tensor(
            np.abs(r_lin) / (float(np.mean(np.abs(r_lin))) + 1e-30),
            dtype=torch.float64,
        )
        loss = 0.5 * torch.mean(w.detach() * (corr - target.detach()) ** 2)
        return loss, lin_loss
    phi = stage1_np + eps * corr_np
    r_host = np.asarray(residual_fn(phi), dtype=float)
    if norm_np is not None:
        r_host = r_host / norm_np
    target = -torch.as_tensor(r_host.copy(), dtype=torch.float64) / max(eps, 1e-12)
    loss = 0.5 * torch.mean((corr - target.detach()) ** 2)
    return loss, float(loss.detach())


def correct_profile(
    y: np.ndarray,
    stage1: np.ndarray,
    residual_fn: Callable[[np.ndarray], np.ndarray],
    *,
    cfg: MultiStageConfig | None = None,
    omega_y0: np.ndarray | None = None,
    stage2_grad_norm_eps: float = 1e-6,
    optimizer: Stage2Optimizer = "adam",
) -> dict[str, Any]:
    """Run linearized Fourier stage-2 correction on a numpy profile."""
    cfg = cfg or MultiStageConfig(steps=80, hidden=24, n_fourier=12)
    y_np = np.asarray(y, dtype=float).reshape(-1)
    stage1_np = np.asarray(stage1, dtype=float).reshape(-1)
    r0 = np.asarray(residual_fn(stage1_np), dtype=float).reshape(-1)
    freq = float(dominant_residual_frequency(r0, y_np))
    sigma = max(2.0 * math.pi * max(freq, 1e-6), 1.0)
    eps = float(cfg.eps)
    fd = float(cfg.fd_eps)

    y_t = torch.as_tensor(y_np.copy(), dtype=torch.float64)
    r0_t = torch.as_tensor(r0.copy(), dtype=torch.float64)
    norm_np = None
    if omega_y0 is not None:
        omy = np.asarray(omega_y0, dtype=float).reshape(-1)
        norm_np = float(stage2_grad_norm_eps) + np.abs(omy)

    model = _FourierCorrector(
        n_fourier=cfg.n_fourier,
        hidden=cfg.hidden,
        sigma=sigma,
        seed=cfg.seed,
    )
    losses: list[float] = []
    if optimizer == "gauss_newton":
        from omnibias.torch.optim import GaussNewton, functional_residual_fn

        class _Proxy(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.model = model

            def forward(self) -> Tensor:
                # Corr-matching proxy only (Φ₁ ≈ -R₀/ε). Not Wang-linearized GN.
                # Do NOT call `_stage2_loss_tensors` here (it detaches to numpy).
                corr = self.model(y_t)
                target = -r0_t / max(eps, 1e-12)
                return (corr - target.detach()).reshape(-1)

        proxy = _Proxy()
        flat0, residual_vec_fn = functional_residual_fn(proxy)
        gn = GaussNewton(solver="qr", damping=1e-3, use_martens_grosse=False)
        params = flat0
        for _ in range(int(cfg.steps)):
            params, info = gn.step(residual_vec_fn, params)
            losses.append(float(info.loss))
        offset = 0
        with torch.no_grad():
            for p in proxy.parameters():
                n = p.numel()
                p.copy_(params[offset : offset + n].reshape(p.shape))
                offset += n
        optimizer_label = "gauss_newton_corr_proxy"
    else:
        opt = torch.optim.Adam(model.parameters(), lr=float(cfg.lr))
        for _ in range(int(cfg.steps)):
            opt.zero_grad(set_to_none=True)
            loss, lin_loss = _stage2_loss_tensors(
                model,
                y_t,
                stage1_np=stage1_np,
                r0=r0,
                r0_t=r0_t,
                residual_fn=residual_fn,
                cfg=cfg,
                eps=eps,
                fd=fd,
                norm_np=norm_np,
            )
            losses.append(lin_loss)
            loss.backward()
            opt.step()
        optimizer_label = "stage2_heuristic_adam"

    with torch.no_grad():
        corr_np = model(y_t).detach().cpu().numpy()
    composed = np.asarray(compose_profiles(stage1_np, corr_np, eps=eps), dtype=float)
    r1 = np.asarray(residual_fn(composed), dtype=float)
    return {
        "stage1": stage1_np,
        "stage2": np.asarray(corr_np, dtype=float),
        "composed": composed,
        "eps": eps,
        "sigma": sigma,
        "dominant_frequency": freq,
        "max_abs_residual_before": float(np.max(np.abs(r0))),
        "max_abs_residual_after": float(np.max(np.abs(r1))),
        "loss_history": losses,
        "linearized": bool(cfg.linearized),
        "optimizer": optimizer_label,
        "config": MultiStageConfig(
            hidden=cfg.hidden,
            n_fourier=cfg.n_fourier,
            sigma=sigma,
            eps=eps,
            steps=cfg.steps,
            lr=cfg.lr,
            seed=cfg.seed,
            fd_eps=cfg.fd_eps,
            linearized=cfg.linearized,
        ),
    }


def iterate_multistage(
    y: np.ndarray,
    stage1: np.ndarray,
    residual_fn: Callable[[np.ndarray], np.ndarray],
    *,
    rounds: int = 3,
    cfg: MultiStageConfig | None = None,
    omega_y0: np.ndarray | None = None,
    optimizer: Stage2Optimizer = "adam",
    improvement_tol: float = 0.98,
) -> dict[str, Any]:
    """Iterate stage-2 correction until residual plateaus."""
    profile = np.asarray(stage1, dtype=float).reshape(-1)
    omy = None if omega_y0 is None else np.asarray(omega_y0, dtype=float).reshape(-1)
    history: list[dict[str, float]] = []
    best = profile.copy()
    best_r = float(np.max(np.abs(residual_fn(best))))
    last: dict[str, Any] = {}
    for i in range(max(1, int(rounds))):
        last = correct_profile(
            y,
            profile,
            residual_fn,
            cfg=cfg,
            omega_y0=omy,
            optimizer=optimizer,
        )
        r_after = float(last["max_abs_residual_after"])
        history.append(
            {
                "round": float(i),
                "max_abs_before": float(last["max_abs_residual_before"]),
                "max_abs_after": r_after,
            }
        )
        if r_after < best_r * float(improvement_tol):
            best_r = r_after
            best = np.asarray(last["composed"], dtype=float)
            profile = best
            omy = np.gradient(profile, np.asarray(y, dtype=float))
        else:
            break
    return {
        "composed": best,
        "max_abs_residual_after": best_r,
        "rounds_run": len(history),
        "history": history,
        "last": last,
        "optimizer": last.get("optimizer", optimizer),
    }


__all__ = [
    "MultiStageConfig",
    "compose_profiles",
    "correct_profile",
    "dominant_residual_frequency",
    "iterate_multistage",
    "linearized_operator_action_numpy",
]
