# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Line / compactified CCF discovery on the Cauchy-Hardy basis (jax).

Ansatz
------
``Theta(y) = sum_j c_j P_{a_j, alpha}(y)`` with ``alpha = 1/(1+lambda)``,
exact whole-line Hilbert, Gauss-Newton / Adam training, d0+d1+d2 residual
losses, adaptive collocation, and funnel ``lambda`` inference.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

# Enable float64 in-module (callers no longer need to remember).
jax.config.update("jax_enable_x64", True)

from omnibias.pinn.jax.discovery.funnel import (  # noqa: E402
    FunnelState,
    run_funnel_loop,
)
from omnibias.pinn.jax.discovery.train_gn import GNConfig, gauss_newton_minimize  # noqa: E402
from omnibias.pinn.jax.equations.ccf_compactified import (  # noqa: E402
    alpha_from_lambda,
    ccf_hardy_residual_samples,
    compactify_y_lambda,
    hardy_profile,
)


@dataclass(frozen=True)
class CCFLineDiscoveryConfig:
    """Configuration for Hardy-basis line-domain CCF discovery."""

    n_terms: int = 6
    n_grid: int = 128
    y_max: float = 20.0
    form: str = "transport"
    velocity_sign: float = 1.0
    lam_init: float = 0.6057
    train_lam: bool = False
    weight_kind: str = "one_plus_abs"
    norm_value: float = 1.0
    norm_weight: float = 1.0
    d1_weight: float = 0.1
    d2_weight: float = 0.01
    adaptive_power: float = 2.0
    adaptive_every: int = 25
    seed: int = 0
    optimizer: str = "gauss_newton"  # "adam" | "gauss_newton"
    gn_steps: int = 80
    gn_gamma: float = 1e-3


@dataclass(frozen=True)
class CCFLineDiscoveryResult:
    """Line-domain discovery result (numpy arrays for CAP export)."""

    lam: float
    y: np.ndarray
    q: np.ndarray
    theta: np.ndarray
    theta_y: np.ndarray
    residual: np.ndarray
    equation_residual: np.ndarray
    loss_history: np.ndarray
    diagnostics: dict[str, float]
    params: dict[str, np.ndarray]
    config: CCFLineDiscoveryConfig
    funnel: FunnelState | None = None
    extra: dict[str, Any] = field(default_factory=dict)


def init_params(cfg: CCFLineDiscoveryConfig) -> dict[str, Array]:
    key = jax.random.PRNGKey(cfg.seed)
    k_c, k_a = jax.random.split(key)
    coeffs = jax.random.normal(k_c, (cfg.n_terms,), dtype=jnp.float64) * 0.3
    coeffs = coeffs.at[0].set(cfg.norm_value)
    # positive scales via softplus of unconstrained vars
    log_scales = jnp.linspace(math.log(0.4), math.log(3.0), cfg.n_terms, dtype=jnp.float64)
    log_scales = log_scales + 0.05 * jax.random.normal(k_a, (cfg.n_terms,), dtype=jnp.float64)
    lam = jnp.asarray(cfg.lam_init, dtype=jnp.float64)
    return {"coeffs": coeffs, "log_scales": log_scales, "lam": lam}


def _scales(params: dict[str, Array]) -> Array:
    return jax.nn.softplus(params["log_scales"]) + 1e-3


def profile_from_params(
    params: dict[str, Array], y: Array
) -> tuple[Array, Array, Array, Array]:
    alpha = alpha_from_lambda(params["lam"])
    return hardy_profile(y, params["coeffs"], _scales(params), alpha)


def pde_residual_on_grid(
    params: dict[str, Array],
    y: Array,
    cfg: CCFLineDiscoveryConfig,
) -> tuple[Array, Array, dict[str, Array]]:
    eq, factored, _w, fields = ccf_hardy_residual_samples(
        y,
        params["coeffs"],
        _scales(params),
        params["lam"],
        form=cfg.form,
        velocity_sign=cfg.velocity_sign,
        weight_kind=cfg.weight_kind,
    )
    return eq, factored, fields


def _grid_derivatives(r: Array, y: Array) -> tuple[Array, Array]:
    """Central-difference d1/d2 of residual samples on a sorted grid."""
    y = jnp.asarray(y)
    r = jnp.asarray(r)
    # assume sorted y
    dy = jnp.diff(y)
    dy = jnp.concatenate([dy[:1], dy])
    d1 = jnp.gradient(r, y)
    d2 = jnp.gradient(d1, y)
    return d1, d2


def residual_vector(
    params: dict[str, Array],
    y: Array,
    cfg: CCFLineDiscoveryConfig,
) -> Array:
    """Stacked d0 + weighted d1/d2 + norm gauge residual."""
    _eq, factored, fields = pde_residual_on_grid(params, y, cfg)
    d1, d2 = _grid_derivatives(factored, y)
    theta0 = fields["theta"]
    # norm at the sample closest to origin
    i0 = int(jnp.argmin(jnp.abs(y)))
    norm = jnp.asarray([(theta0[i0] - cfg.norm_value) * math.sqrt(cfg.norm_weight)])
    parts = [factored, math.sqrt(cfg.d1_weight) * d1, math.sqrt(cfg.d2_weight) * d2, norm]
    return jnp.concatenate(parts)


def _adaptive_resample(
    y: Array,
    residual: Array,
    *,
    power: float,
    seed: int,
) -> Array:
    """Resample collocation points with probability ~ |R|^power (keep endpoints)."""
    w = jnp.abs(residual) ** float(power) + 1e-12
    w = w.at[0].set(0.0)
    w = w.at[-1].set(0.0)
    probs = w / jnp.sum(w)
    key = jax.random.PRNGKey(seed)
    n = int(y.shape[0])
    idx = jax.random.choice(key, n, shape=(n - 2,), p=probs, replace=True)
    y_new = jnp.sort(jnp.concatenate([y[:1], y[idx], y[-1:]]))
    # small jitter to avoid exact duplicates
    jitter = 1e-6 * (y[-1] - y[0]) * jax.random.normal(key, (n,))
    y_new = jnp.sort(y_new + jitter)
    return y_new


def dense_validation_residual(
    params: dict[str, Array],
    cfg: CCFLineDiscoveryConfig,
    *,
    n_val: int = 2001,
    y_max: float | None = None,
) -> dict[str, float]:
    """Unfactored residual on a fixed dense ``linspace`` (not the adaptive train grid)."""
    ymax = float(cfg.y_max if y_max is None else y_max)
    n = max(int(n_val), 51)
    if n % 2 == 0:
        n += 1  # include the origin
    y_val = jnp.linspace(-ymax, ymax, n, dtype=jnp.float64)
    equation, factored, _fields = pde_residual_on_grid(params, y_val, cfg)
    return {
        "dense_max_abs_equation": float(jnp.max(jnp.abs(equation))),
        "dense_rms_equation": float(jnp.sqrt(jnp.mean(equation * equation))),
        "dense_max_abs_factored": float(jnp.max(jnp.abs(factored))),
        "dense_n": float(n),
        "dense_y_max": ymax,
    }


def run_ccf_line_discovery(
    cfg: CCFLineDiscoveryConfig,
    *,
    steps: int = 400,
    lr: float = 3e-3,
    init: dict[str, Array] | None = None,
    funnel_updates: int = 0,
) -> CCFLineDiscoveryResult:
    """Run Hardy-basis line discovery (Adam or Gauss-Newton) with optional funnel."""
    y = jnp.linspace(-cfg.y_max, cfg.y_max, cfg.n_grid, dtype=jnp.float64)
    params = init_params(cfg) if init is None else dict(init)
    funnel_state: FunnelState | None = None
    loss_hist: list[float] = []

    def _train_fixed_lam(lam_fixed: float, n_steps: int, y_grid: Array) -> tuple[dict[str, Array], Array]:
        p = dict(params)
        p["lam"] = jnp.asarray(lam_fixed, dtype=jnp.float64)
        y_cur = y_grid

        if cfg.optimizer == "gauss_newton":
            chunk = max(int(cfg.adaptive_every), 1)
            remaining = int(n_steps)
            t = 0
            # Freeze lambda in the GN pytree when not training it (Adam already
            # zeros the grad; GN would otherwise free lam and collapse).
            train_lam = bool(cfg.train_lam) and funnel_updates <= 0
            while remaining > 0:
                n_chunk = min(chunk, remaining)
                lam_hold = float(p["lam"])

                def r_fn(pp: object, y_ref: Array = y_cur) -> Array:
                    assert isinstance(pp, dict)
                    if not train_lam:
                        pp = {**pp, "lam": jnp.asarray(lam_hold, dtype=jnp.float64)}
                    return residual_vector(pp, y_ref, cfg)

                gn_params = p if train_lam else {k: v for k, v in p.items() if k != "lam"}
                trained, hist = gauss_newton_minimize(
                    r_fn,
                    gn_params,
                    config=GNConfig(
                        steps=min(n_chunk, cfg.gn_steps),
                        gamma=cfg.gn_gamma,
                        use_martens_grosse=True,
                    ),
                )
                loss_hist.extend([float(x) for x in np.asarray(hist)])
                assert isinstance(trained, dict)
                p = dict(trained)
                p["lam"] = jnp.asarray(lam_hold if not train_lam else float(p.get("lam", lam_hold)), dtype=jnp.float64)
                _, factored, _ = pde_residual_on_grid(p, y_cur, cfg)
                y_cur = _adaptive_resample(
                    y_cur, factored, power=cfg.adaptive_power, seed=cfg.seed + t
                )
                remaining -= n_chunk
                t += 1
            return p, y_cur

        # Adam
        def loss_fn(pp: dict[str, Array], y_ref: Array) -> Array:
            r = residual_vector(pp, y_ref, cfg)
            return 0.5 * jnp.sum(r * r)

        b1, b2, eps = 0.9, 0.999, 1e-8
        m = jax.tree_util.tree_map(jnp.zeros_like, p)
        v = jax.tree_util.tree_map(jnp.zeros_like, p)
        train_lam = cfg.train_lam and funnel_updates <= 0
        y_cur = y_grid
        for t in range(1, int(n_steps) + 1):
            loss, grad = jax.value_and_grad(lambda pp: loss_fn(pp, y_cur))(p)
            if not train_lam:
                grad = {**grad, "lam": jnp.zeros_like(grad["lam"])}
            m = jax.tree_util.tree_map(lambda mm, gg: b1 * mm + (1 - b1) * gg, m, grad)
            v = jax.tree_util.tree_map(lambda vv, gg: b2 * vv + (1 - b2) * gg * gg, v, grad)
            bc1 = 1.0 - b1**t
            bc2 = 1.0 - b2**t
            p = jax.tree_util.tree_map(
                lambda pp, mm, vv: pp - lr * (mm / bc1) / (jnp.sqrt(vv / bc2) + eps),
                p, m, v,
            )
            loss_hist.append(float(loss))
            if t % max(cfg.adaptive_every, 1) == 0:
                _, factored, _ = pde_residual_on_grid(p, y_cur, cfg)
                y_cur = _adaptive_resample(
                    y_cur, factored, power=cfg.adaptive_power, seed=cfg.seed + t
                )
        return p, y_cur

    if funnel_updates > 0:
        def train_and_residual(lam: float) -> tuple[float, object, object]:
            nonlocal params, y
            params, y = _train_fixed_lam(lam, max(steps // max(funnel_updates, 1), 20), y)
            _eq, factored, _ = pde_residual_on_grid(params, y, cfg)
            return float(params["lam"]), y, factored

        funnel_state = run_funnel_loop(
            lam0=float(cfg.lam_init),
            train_and_residual=train_and_residual,
            n_updates=funnel_updates,
        )
        lam_final = float(funnel_state.lambdas[-1])
        params, y = _train_fixed_lam(lam_final, max(steps // max(funnel_updates, 1), 20), y)
    else:
        params, y = _train_fixed_lam(float(params["lam"]), steps, y)

    equation, factored, fields = pde_residual_on_grid(params, y, cfg)
    dense = dense_validation_residual(params, cfg)
    q = compactify_y_lambda(y, params["lam"])
    diagnostics = {
        "max_abs_residual": float(jnp.max(jnp.abs(factored))),
        "rms_residual": float(jnp.sqrt(jnp.mean(factored * factored))),
        "max_abs_equation_residual": float(jnp.max(jnp.abs(equation))),
        "dense_max_abs_equation": dense["dense_max_abs_equation"],
        "dense_rms_equation": dense["dense_rms_equation"],
        "dense_max_abs_factored": dense["dense_max_abs_factored"],
        "lam": float(params["lam"]),
        "alpha": float(alpha_from_lambda(params["lam"])),
        "theta_peak": float(jnp.max(jnp.abs(fields["theta"]))),
        "domain": 1.0,
    }
    scales = np.asarray(_scales(params))
    return CCFLineDiscoveryResult(
        lam=float(params["lam"]),
        y=np.asarray(y),
        q=np.asarray(q),
        theta=np.asarray(fields["theta"]),
        theta_y=np.asarray(fields["theta_y"]),
        residual=np.asarray(factored),
        equation_residual=np.asarray(equation),
        loss_history=np.asarray(loss_hist, dtype=float),
        diagnostics=diagnostics,
        params={
            "coeffs": np.asarray(params["coeffs"]),
            "log_scales": np.asarray(params["log_scales"]),
            "scales": scales,
            "lam": np.asarray(params["lam"]),
        },
        config=cfg,
        funnel=funnel_state,
        extra={
            "domain": "line_compactified",
            "optimizer": cfg.optimizer,
            "hilbert_convention": "hardy_exact",
            "ansatz": "cauchy_hardy",
            "compactification": "lambda_tied_eq5",
        },
    )


__all__ = [
    "CCFLineDiscoveryConfig",
    "CCFLineDiscoveryResult",
    "dense_validation_residual",
    "init_params",
    "pde_residual_on_grid",
    "profile_from_params",
    "residual_vector",
    "run_ccf_line_discovery",
]
