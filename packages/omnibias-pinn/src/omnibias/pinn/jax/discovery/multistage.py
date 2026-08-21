# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Multi-stage residual correction (DeepMind linearized stage-2 / MSNN).

Wang et al. (arXiv:2509.14185, 2511.22819) train a small tanh MLP on
compactified even coordinates, then a second Fourier-feature net that
corrects the leftover high-frequency residual. Stage-1 is ``Phi0``.
Stage-2 learns ``eps * Phi1`` for the linearized residual equation

    -eps D[Phi0] Phi1 ≈ R^{stage-1}

(paper eq. 19). The composed profile is ``Phi0 + eps * Phi1``.

For CCF the stage-2 *hat* must be even (same inductive bias as the paper:
the net sees only even coordinates). :func:`stage2_correction` on raw ``y``
is the generic 1-D toy path; :func:`stage2_even_hat` is the CCF-faithful
path. ``optimizer="martens_grosse"`` is the paper-faithful L2 / eq. 19
label (``wang_linearized_gn``). Default ``adam`` stays a labeled
heuristic. For the CCF 1601-pt L∞ stretch metric, measured winner
is the epigraph L∞ LP / ``linearized_linf_direction``, not L2 GN.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

jax.config.update("jax_enable_x64", True)

Stage2Optimizer = Literal["adam", "martens_grosse"]


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
    optimizer: Stage2Optimizer = "adam"
    even_q: bool = False
    lam: float = 0.6057
    identity_readout: bool = False


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
    if cfg.identity_readout:
        c = jnp.zeros((cfg.hidden,), dtype=jnp.float64)
        b = jnp.zeros((), dtype=jnp.float64)
    else:
        c = jax.random.normal(k_c, (cfg.hidden,), dtype=jnp.float64) * 0.1
        b = jnp.zeros((), dtype=jnp.float64)
    return {"B": B, "W": W, "beta": beta, "c": c, "b": b}


def stage2_correction(params: dict[str, Array], y: Array) -> Array:
    """Scalar correction field on samples ``y``.

    Raw ``y`` is *not* even. For CCF hats use :func:`stage2_even_hat`.
    """
    feat = _fourier_features(y, params["B"])
    h = jnp.tanh(feat @ params["W"].T + params["beta"])
    return params["b"] + h @ params["c"]


def even_compactified_coordinate(y: Array, *, lam: float) -> Array:
    """DeepMind even input ``q=(1+y^2)^{-α/2}``."""
    from omnibias.pinn.jax.equations.ccf_compactified import compactify_y_lambda

    return compactify_y_lambda(jnp.asarray(y, dtype=jnp.float64), lam)


def stage2_even_hat(params: dict[str, Array], y: Array, *, lam: float) -> Array:
    """Even hat correction: Fourier features of compactified ``q(y)``."""
    return stage2_correction(params, even_compactified_coordinate(y, lam=lam))


def stage2_field(params: dict[str, Array], y: Array, cfg: MultiStageConfig) -> Array:
    """Paper CCF path uses even ``q``; raw ``y`` is the generic 1-D toy path."""
    if cfg.even_q:
        return stage2_even_hat(params, y, lam=float(cfg.lam))
    return stage2_correction(params, y)


def gradient_normalize_residual(
    residual: Array,
    nn_core: Array,
    *,
    alpha: float = 2.0,
    eps: float = 1e-8,
) -> Array:
    """Wang et al. follow-up (arXiv:2511.22819): ``R / (eps + exp(α NN))``.

    Down-weights high-amplitude / high-gradient regions so the origin
    signal is not drowned. Score gates stay raw ``max|r|``.
    """
    return residual / (float(eps) + jnp.exp(float(alpha) * nn_core))


def deepmind_multistage_config(**overrides: Any) -> MultiStageConfig:
    """Paper-faithful stage-2: small Fourier net + exact-J Gauss–Newton.

    Wang et al. use a small second net so a (near) full Gauss–Newton matrix
    is feasible. This helper selects that size and
    ``optimizer="martens_grosse"`` (exact JVP). That is the paper
    L2 / eq. 19 trainer, not the measured CCF L∞ earn step
    (epigraph L∞). It does not forge stretch.
    """
    base = {
        "hidden": 16,
        "n_fourier": 12,
        "eps": 1.0,
        "steps": 40,
        "optimizer": "martens_grosse",
        "linearized": True,
        "seed": 1,
        "even_q": True,
        "identity_readout": False,
        "lam": 0.6057,
    }
    base.update(overrides)
    return MultiStageConfig(**base)


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


def wang_linearized_residual(
    residual_fn: Callable[[Array], Array],
    stage1: Array,
    direction: Array,
    *,
    r0: Array,
    eps: float,
    fd_eps: float,
    linearized: bool = True,
) -> Array:
    """Paper eq. 19 residual vector ``R0 + eps D[Phi0] Phi1`` (or nonlinear)."""
    if linearized:
        d_action = linearized_operator_action(
            residual_fn, stage1, direction, fd_eps=fd_eps
        )
        return r0 + float(eps) * d_action
    theta = compose_profiles(stage1, direction, eps=eps)
    return residual_fn(theta)


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

    ``optimizer="adam"`` (default) is a labeled ``stage2_heuristic``.
    ``optimizer="martens_grosse"`` runs residual-vector Gauss–Newton on
    that same eq. 19 vector and is labeled ``wang_linearized_gn``.
    Neither mode forges stretch / Rung-1.
    """
    cfg = MultiStageConfig() if cfg is None else cfg
    r0 = residual_fn(stage1_theta)
    freq = dominant_residual_frequency(r0, y)
    optimizer = str(cfg.optimizer)
    if optimizer not in ("adam", "martens_grosse"):
        raise ValueError(
            f"optimizer must be 'adam' or 'martens_grosse', got {optimizer!r}"
        )
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
        optimizer=optimizer,  # type: ignore[arg-type]
        even_q=bool(cfg.even_q),
        lam=float(cfg.lam),
        identity_readout=bool(cfg.identity_readout),
    )
    params = init_stage2_params(cfg)

    def residual_vec(p: dict[str, Array]) -> Array:
        corr = stage2_field(p, y, cfg)
        return wang_linearized_residual(
            residual_fn,
            stage1_theta,
            corr,
            r0=r0,
            eps=cfg.eps,
            fd_eps=cfg.fd_eps,
            linearized=cfg.linearized,
        )

    def loss_fn(p: dict[str, Array]) -> Array:
        r = residual_vec(p)
        return 0.5 * jnp.sum(r * r)

    losses: list[float]
    if optimizer == "martens_grosse":
        from omnibias.pinn.jax.discovery.train_gn import GNConfig, gauss_newton_minimize

        params, hist = gauss_newton_minimize(
            residual_vec,
            params,
            config=GNConfig(
                steps=int(cfg.steps),
                gamma=1e-3,
                use_martens_grosse=True,
                method="martens_grosse",
                solver="qr",
            ),
        )
        losses = [float(x) for x in hist]
        optimizer_label = "wang_linearized_gn"
    else:
        loss_and_grad = jax.value_and_grad(loss_fn)
        b1, b2, eps_adam = 0.9, 0.999, 1e-8
        m = jax.tree_util.tree_map(jnp.zeros_like, params)
        v = jax.tree_util.tree_map(jnp.zeros_like, params)
        losses = []
        for t in range(1, int(cfg.steps) + 1):
            loss, grad = loss_and_grad(params)
            m = jax.tree_util.tree_map(lambda mm, gg: b1 * mm + (1 - b1) * gg, m, grad)
            v = jax.tree_util.tree_map(
                lambda vv, gg: b2 * vv + (1 - b2) * gg * gg, v, grad
            )
            bc1 = 1.0 - b1**t
            bc2 = 1.0 - b2**t
            params = jax.tree_util.tree_map(
                lambda pp, mm, vv, c1=bc1, c2=bc2: pp
                - cfg.lr * (mm / c1) / (jnp.sqrt(vv / c2) + eps_adam),
                params,
                m,
                v,
            )
            losses.append(float(loss))
        optimizer_label = "stage2_heuristic_adam"

    info = {
        "eps": float(cfg.eps),
        "sigma": float(cfg.sigma),
        "dominant_frequency": float(freq),
        "final_loss": float(losses[-1]) if losses else float("nan"),
        "initial_loss": float(0.5 * jnp.sum(r0 * r0)),
        "linearized": bool(cfg.linearized),
        "optimizer": optimizer_label,
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
    corr = stage2_field(params, y_j, cfg if cfg is not None else MultiStageConfig())
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
    "Stage2Optimizer",
    "compose_profiles",
    "deepmind_multistage_config",
    "dominant_residual_frequency",
    "even_compactified_coordinate",
    "gradient_normalize_residual",
    "init_stage2_params",
    "linearized_operator_action",
    "refine_with_multistage",
    "stage2_correction",
    "stage2_even_hat",
    "stage2_field",
    "train_stage2_correction",
    "wang_linearized_residual",
]
