# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Multi-stage residual correction (DeepMind linearized stage-2).

Stage-1 produces an approximate profile ``Phi0``. Stage-2 learns a correction
``eps * Phi1`` that solves the **linearized** residual equation

    -eps D[Phi0] Phi1 ≈ R^{stage-1}

(paper eq. 19), with a Fourier-feature network whose ``sigma`` is set from the
dominant residual frequency. The composed profile is ``Phi0 + eps * Phi1``.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

jax.config.update("jax_enable_x64", True)


@dataclass(frozen=True)
class MultiStageConfig:
    """Stage-2 Fourier-feature correction hyper-parameters."""

    hidden: int = 32
    n_fourier: int = 16
    sigma: float = 4.0
    eps: float = 1e-2
    steps: int = 200
    lr: float = 1e-2
    seed: int = 1
    fd_eps: float = 1e-5
    linearized: bool = True


def _fourier_features(x: Array, B: Array) -> Array:
    """``[cos(B x), sin(B x)]`` with ``B`` shape ``(n_fourier,)``."""
    proj = x[:, None] * B[None, :]
    return jnp.concatenate([jnp.cos(proj), jnp.sin(proj)], axis=-1)


def init_stage2_params(cfg: MultiStageConfig, *, n_in: int = 1) -> dict[str, Array]:
    del n_in
    key = jax.random.PRNGKey(cfg.seed)
    k_b, k_w, k_c = jax.random.split(key, 3)
    B = jax.random.normal(k_b, (cfg.n_fourier,), dtype=jnp.float64) * cfg.sigma
    W = jax.random.normal(k_w, (cfg.hidden, 2 * cfg.n_fourier), dtype=jnp.float64) * 0.1
    beta = jnp.zeros((cfg.hidden,), dtype=jnp.float64)
    c = jax.random.normal(k_c, (cfg.hidden,), dtype=jnp.float64) * 0.1
    b = jnp.zeros((), dtype=jnp.float64)
    return {"B": B, "W": W, "beta": beta, "c": c, "b": b}


def stage2_correction(params: dict[str, Array], y: Array) -> Array:
    """Scalar correction field on samples ``y``."""
    feat = _fourier_features(y, params["B"])
    h = jnp.tanh(feat @ params["W"].T + params["beta"])
    return params["b"] + h @ params["c"]


def compose_profiles(
    stage1: Array,
    stage2: Array,
    *,
    eps: float,
) -> Array:
    """``Phi = stage1 + eps * stage2``."""
    return stage1 + float(eps) * stage2


def dominant_residual_frequency(residual: Array, y: Array) -> float:
    """Heuristic dominant frequency (cycles per unit ``y``) from residual FFT."""
    r = jnp.asarray(residual)
    y = jnp.asarray(y)
    n = int(r.shape[0])
    if n < 4:
        return 1.0
    span = float(jnp.max(y) - jnp.min(y)) + 1e-12
    spec = jnp.abs(jnp.fft.rfft(r - jnp.mean(r)))
    k = int(jnp.argmax(spec[1:]) + 1) if spec.shape[0] > 1 else 1
    return float(k / span)


def linearized_operator_action(
    residual_fn: Callable[[Array], Array],
    stage1: Array,
    direction: Array,
    *,
    fd_eps: float,
) -> Array:
    """Finite-difference action of ``D[Phi0]`` on ``direction``."""
    eps = float(fd_eps)
    rp = residual_fn(stage1 + eps * direction)
    rm = residual_fn(stage1 - eps * direction)
    return (rp - rm) / (2.0 * eps)


def train_stage2_correction(
    *,
    y: Array,
    stage1_theta: Array,
    residual_fn: Callable[[Array], Array],
    cfg: MultiStageConfig | None = None,
) -> tuple[dict[str, Array], Array, dict[str, Any]]:
    """Train stage-2 against the linearized residual equation (or nonlinear).

    Linearized loss (default)::

        0.5 || R0 + eps * D[Phi0] Phi1 ||^2

    which is the discrete form of paper eq. 19.
    """
    cfg = MultiStageConfig() if cfg is None else cfg
    r0 = residual_fn(stage1_theta)
    freq = dominant_residual_frequency(r0, y)
    cfg = MultiStageConfig(
        hidden=cfg.hidden,
        n_fourier=cfg.n_fourier,
        sigma=max(cfg.sigma, 2.0 * math.pi * freq),
        eps=cfg.eps,
        steps=cfg.steps,
        lr=cfg.lr,
        seed=cfg.seed,
        fd_eps=cfg.fd_eps,
        linearized=cfg.linearized,
    )
    params = init_stage2_params(cfg)

    def loss_fn(p: dict[str, Array]) -> Array:
        corr = stage2_correction(p, y)
        if cfg.linearized:
            d_action = linearized_operator_action(
                residual_fn, stage1_theta, corr, fd_eps=cfg.fd_eps
            )
            # -eps D Phi1 ≈ R0  =>  R0 + eps D Phi1 ≈ 0
            r = r0 + float(cfg.eps) * d_action
        else:
            theta = compose_profiles(stage1_theta, corr, eps=cfg.eps)
            r = residual_fn(theta)
        return 0.5 * jnp.sum(r * r)

    loss_and_grad = jax.value_and_grad(loss_fn)
    b1, b2, eps_adam = 0.9, 0.999, 1e-8
    m = jax.tree_util.tree_map(jnp.zeros_like, params)
    v = jax.tree_util.tree_map(jnp.zeros_like, params)
    losses: list[float] = []

    for t in range(1, int(cfg.steps) + 1):
        loss, grad = loss_and_grad(params)
        m = jax.tree_util.tree_map(lambda mm, gg: b1 * mm + (1 - b1) * gg, m, grad)
        v = jax.tree_util.tree_map(lambda vv, gg: b2 * vv + (1 - b2) * gg * gg, v, grad)
        bc1 = 1.0 - b1**t
        bc2 = 1.0 - b2**t
        params = jax.tree_util.tree_map(
            lambda pp, mm, vv: pp - cfg.lr * (mm / bc1) / (jnp.sqrt(vv / bc2) + eps_adam),
            params, m, v,
        )
        losses.append(float(loss))

    info = {
        "eps": float(cfg.eps),
        "sigma": float(cfg.sigma),
        "dominant_frequency": float(freq),
        "final_loss": float(losses[-1]) if losses else float("nan"),
        "initial_loss": float(0.5 * jnp.sum(r0 * r0)),
        "linearized": bool(cfg.linearized),
    }
    return params, jnp.asarray(losses, dtype=jnp.float64), info


def refine_with_multistage(
    *,
    y: np.ndarray | Array,
    stage1_theta: np.ndarray | Array,
    residual_fn: Callable[[Array], Array],
    cfg: MultiStageConfig | None = None,
) -> dict[str, Any]:
    """Convenience wrapper returning numpy-friendly correction diagnostics."""
    y_j = jnp.asarray(y, dtype=jnp.float64)
    th0 = jnp.asarray(stage1_theta, dtype=jnp.float64)
    params, hist, info = train_stage2_correction(
        y=y_j, stage1_theta=th0, residual_fn=residual_fn, cfg=cfg
    )
    corr = stage2_correction(params, y_j)
    eps = float(info["eps"])
    theta = compose_profiles(th0, corr, eps=eps)
    return {
        "params": {k: np.asarray(v) for k, v in params.items()},
        "loss_history": np.asarray(hist),
        "correction": np.asarray(corr),
        "theta": np.asarray(theta),
        "info": info,
    }


__all__ = [
    "MultiStageConfig",
    "compose_profiles",
    "dominant_residual_frequency",
    "init_stage2_params",
    "linearized_operator_action",
    "refine_with_multistage",
    "stage2_correction",
    "train_stage2_correction",
]
