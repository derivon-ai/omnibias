# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Wang / DeepMind vorticity-form CCF discovery on an Omega-primary Hardy basis.

Published unstable profiles are discovered in **vorticity form**
(Wang et al., arXiv:2509.14185 / 2511.22819):

    Ω + ((1+λ) y - U) Ω_y - Ω U_y = 0,
    U'(y) = (H Ω)(y),   U(0) = 0,

with odd ``Ω`` and gauge ``Ω(y_g) = g_val`` (default ``Ω(0.5)=0.05``).

Ansatz (far-field corrected)
----------------------------
``Ω(y) = sum_j c_j Q_{a_j, γ_j}(y)`` with ``γ ∈ {k α}`` (optionally offset for
``k≥2``), ``α = 1/(1+λ)``. Exact Hilbert: ``HΩ = -P_{a,γ}``. Exact velocity:
``U = -Q_{a, γ-1}/(γ-1)`` (``γ ≠ 1``). Leading decay is ``|y|^{-α}``, which
cancels the linear far-field operator (``1 - (1+λ)α = 0``).

The older Θ-even sum ``Θ = Σ c P`` produced ``Ω = Θ' ∼ |y|^{-(α+1)}`` and is
retained only as :func:`hardy_theta_profile` for diagnostics / back-compat.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

jax.config.update("jax_enable_x64", True)

from omnibias.pinn.jax.discovery.train_gn import GNConfig, gauss_newton_minimize  # noqa: E402
from omnibias.pinn.jax.equations.ccf_compactified import (  # noqa: E402
    alpha_from_lambda,
    hardy_even,
    hardy_even_deriv,
    hardy_odd,
    hardy_odd_deriv,
)


@dataclass(frozen=True)
class CCFVorticityDiscoveryConfig:
    """Omega-primary Hardy / vorticity discovery configuration."""

    n_scales: int = 8
    n_gamma_multiples: int = 4  # γ ≈ k*α for k=1..K
    n_grid: int = 401
    y_max: float = 40.0
    lam: float = 0.6057
    gauge_point: float = 0.5
    gauge_value: float = 0.05
    gauge_weight: float = 50.0
    seed: int = 0
    gn_steps: int = 80
    gn_gamma: float = 1e-3
    coeff_l2: float = 1e-6
    scale_lo: float = 0.25
    scale_hi: float = 8.0
    free_gamma_offsets: bool = False
    hard_gauge_rescale: bool = True
    near_field_power: float = 0.0
    # Independent Hardy atoms: n_terms free (a_j, γ_j) instead of scales×gamma grid.
    independent_terms: bool = False
    n_terms: int = 16
    n_alpha_offsets: int = 1
    theta0_weight: float = 0.0
    enforce_theta0: bool = False
    # Peak-focused / adaptive collocation (DeepMind-style residual^p sampling).
    adaptive_rounds: int = 0
    adaptive_power: float = 4.0
    peak_weight_power: float = 0.0
    origin_fraction: float = 0.25
    # Soft floor on max|Ω| (after expand). With hard_gauge_rescale, shapes that
    # peak only at the gauge point get omega_max≈gauge_value and stall; a floor
    # pushes the optimizer toward profiles with interior peaks.
    omega_max_floor: float = 0.0
    omega_max_weight: float = 0.0


@dataclass(frozen=True)
class CCFVorticityDiscoveryResult:
    """Vorticity-form discovery result."""

    lam: float
    coeffs: np.ndarray
    scales: np.ndarray
    alphas: np.ndarray
    y: np.ndarray
    omega: np.ndarray
    residual: np.ndarray
    diagnostics: dict[str, float]
    config: CCFVorticityDiscoveryConfig
    extra: dict[str, Any] = field(default_factory=dict)
    params: dict[str, Any] | None = None


def _dictionary(
    cfg: CCFVorticityDiscoveryConfig,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Sparse scales × γ = k·α ladder (well-conditioned vs dense geomspace)."""
    alpha0 = float(alpha_from_lambda(cfg.lam))
    if cfg.n_scales == 1:
        scales_base = np.array([1.3], dtype=float)
    else:
        scales_base = np.linspace(float(cfg.scale_lo), float(cfg.scale_hi), cfg.n_scales)
    gammas = alpha0 * np.arange(1, cfg.n_gamma_multiples + 1, dtype=float)
    scales = np.repeat(scales_base, cfg.n_gamma_multiples)
    alphas = np.tile(gammas, cfg.n_scales)
    return scales, alphas, alpha0


def hardy_omega_profile(
    y: Array,
    coeffs: Array,
    scales: Array,
    gammas: Array,
) -> tuple[Array, Array, Array, Array]:
    """Return ``(Omega, Omega_y, U, U_y=HOmega)`` for an odd Hardy-Ω sum.

    Vectorized over atoms (broadcast ``y[:,None]`` against atom axes).
    """
    y = jnp.asarray(y, dtype=jnp.float64).reshape(-1)
    coeffs = jnp.asarray(coeffs, dtype=jnp.float64).reshape(-1)
    scales = jnp.asarray(scales, dtype=jnp.float64).reshape(-1)
    gammas = jnp.asarray(gammas, dtype=jnp.float64).reshape(-1)
    # Shapes: y (N,), atoms (M,) → fields (N, M)
    yy = y[:, None]
    aa = scales[None, :]
    gg = gammas[None, :]
    cc = coeffs[None, :]
    r = jnp.hypot(aa, yy)
    phi = jnp.arctan2(yy, aa)
    # Ω = Σ c Q_{a,γ}; Q = r^{-γ} sin(γ φ)
    om_atm = (r ** (-gg)) * jnp.sin(gg * phi)
    # Ω_y = Σ c γ P_{a,γ+1}; P = r^{-(γ+1)} cos((γ+1) φ)
    omy_atm = gg * (r ** (-(gg + 1.0))) * jnp.cos((gg + 1.0) * phi)
    # HΩ = -P_{a,γ}
    uy_atm = -(r ** (-gg)) * jnp.cos(gg * phi)
    # U = -Q_{a,γ-1}/(γ-1)  (γ≠1) or -arctan(y/a) (γ=1)
    near1 = jnp.abs(gg - 1.0) < 1e-12
    u_gen = -(r ** (-(gg - 1.0))) * jnp.sin((gg - 1.0) * phi) / (gg - 1.0)
    u_g1 = -jnp.arctan(yy / aa)
    u_atm = jnp.where(near1, u_g1, u_gen)
    om = jnp.sum(cc * om_atm, axis=1)
    omy = jnp.sum(cc * omy_atm, axis=1)
    u = jnp.sum(cc * u_atm, axis=1)
    uy = jnp.sum(cc * uy_atm, axis=1)
    return om, omy, u, uy


def hardy_theta_profile(
    y: Array,
    coeffs: Array,
    scales: Array,
    alphas: Array,
) -> tuple[Array, Array, Array, Array, Array]:
    """Legacy even-Θ sum (diagnostic). Prefer :func:`hardy_omega_profile`."""
    y = jnp.asarray(y).reshape(-1)
    coeffs = jnp.asarray(coeffs, dtype=jnp.float64).reshape(-1)
    scales = jnp.asarray(scales, dtype=jnp.float64).reshape(-1)
    alphas = jnp.asarray(alphas, dtype=jnp.float64).reshape(-1)
    th = jnp.zeros_like(y, dtype=jnp.float64)
    om = jnp.zeros_like(y, dtype=jnp.float64)
    omy = jnp.zeros_like(y, dtype=jnp.float64)
    u = jnp.zeros_like(y, dtype=jnp.float64)
    uy = jnp.zeros_like(y, dtype=jnp.float64)
    for c, a, al in zip(coeffs, scales, alphas, strict=True):
        th = th + c * hardy_even(y, a, al)
        om = om + c * hardy_even_deriv(y, a, al)
        omy = omy + c * (-al * (al + 1.0) * hardy_even(y, a, al + 2.0))
        u = u + c * hardy_odd(y, a, al)
        uy = uy + c * hardy_odd_deriv(y, a, al)
    return th, om, omy, u, uy


def hardy_odd_profile(
    y: Array,
    coeffs: Array,
    scales: Array,
    alphas: Array,
) -> tuple[Array, Array, Array, Array]:
    """Q-basis helper: ``(Omega, Omega_y, HOmega, zeros)``."""
    om, omy, _u, uy = hardy_omega_profile(y, coeffs, scales, alphas)
    return om, omy, uy, jnp.zeros_like(om)


def leading_mode_far_field_cancel(
    *,
    a: float = 1.3,
    lam: float = 0.6057,
    y_lo: float = 40.0,
    y_hi: float = 160.0,
    n: int = 400,
) -> dict[str, float]:
    """Regression helper: leading γ=α mode cancels linear far-field operator."""
    alpha = float(alpha_from_lambda(lam))
    yd = jnp.linspace(y_lo, y_hi, n, dtype=jnp.float64)
    om, omy, _, _ = hardy_omega_profile(
        yd,
        jnp.asarray([1.0]),
        jnp.asarray([a]),
        jnp.asarray([alpha]),
    )
    lin = om + (1.0 + lam) * yd * omy
    tail = lin[-n // 5 :]
    yt = yd[-n // 5 :]
    return {
        "far_lin_max": float(jnp.max(jnp.abs(tail))),
        "far_lin_times_y_2alpha": float(
            jnp.max(jnp.abs(tail * jnp.power(yt, 2.0 * alpha)))
        ),
        "alpha": alpha,
        "expected_cancel_factor": float(1.0 - (1.0 + lam) * alpha),
    }


def vorticity_residual_samples(
    y: Array,
    coeffs: Array,
    scales: Array,
    alphas: Array,
    lam: float,
) -> tuple[Array, dict[str, Array]]:
    """Wang vorticity residual on samples (exact Hardy-Ω Hilbert)."""
    om, omy, u, uy = hardy_omega_profile(y, coeffs, scales, alphas)
    r = om + ((1.0 + lam) * y - u) * omy - om * uy
    th = jnp.zeros_like(om)
    return r, {
        "theta": th,
        "omega": om,
        "omega_y": omy,
        "U": u,
        "U_y": uy,
    }


def project_omega_hardy_numpy(
    y: np.ndarray,
    omega: np.ndarray,
    *,
    scales: np.ndarray,
    gammas: np.ndarray,
) -> tuple[np.ndarray, float, dict[str, np.ndarray]]:
    """Least-squares project ``Ω`` onto the Ω-primary Hardy span (numpy twin)."""
    y = np.asarray(y, dtype=float).reshape(-1)
    omega = np.asarray(omega, dtype=float).reshape(-1)
    scales = np.asarray(scales, dtype=float).reshape(-1)
    gammas = np.asarray(gammas, dtype=float).reshape(-1)
    phi = np.column_stack(
        [
            np.asarray(hardy_odd(jnp.asarray(y), float(a), float(g)))
            for a, g in zip(scales, gammas, strict=True)
        ]
    )
    coeffs, *_ = np.linalg.lstsq(phi, omega, rcond=None)
    om_hat = phi @ coeffs
    defect = float(np.max(np.abs(om_hat - omega)))
    om, omy, u, uy = hardy_omega_profile(
        jnp.asarray(y),
        jnp.asarray(coeffs),
        jnp.asarray(scales),
        jnp.asarray(gammas),
    )
    return (
        np.asarray(coeffs, dtype=float),
        defect,
        {
            "omega_proj": om_hat,
            "H": np.asarray(uy),
            "U": np.asarray(u),
            "omega_y_proj": np.asarray(omy),
        },
    )


def dense_vorticity_residual(
    coeffs: Array | np.ndarray,
    scales: Array | np.ndarray,
    alphas: Array | np.ndarray,
    lam: float,
    *,
    n_val: int = 4001,
    y_max: float = 40.0,
) -> dict[str, float]:
    """Dense fixed-grid vorticity residual (Rung-1 metric)."""
    n = max(int(n_val), 51)
    if n % 2 == 0:
        n += 1
    y = jnp.linspace(-float(y_max), float(y_max), n, dtype=jnp.float64)
    r, fields = vorticity_residual_samples(
        y,
        jnp.asarray(coeffs, dtype=jnp.float64),
        jnp.asarray(scales, dtype=jnp.float64),
        jnp.asarray(alphas, dtype=jnp.float64),
        float(lam),
    )
    return {
        "dense_max_abs_vorticity": float(jnp.max(jnp.abs(r))),
        "dense_rms_vorticity": float(jnp.sqrt(jnp.mean(r * r))),
        "dense_n": float(n),
        "dense_y_max": float(y_max),
        "omega_gauge_sample": float(jnp.interp(0.5, y, fields["omega"])),
        "omega_max_abs": float(jnp.max(jnp.abs(fields["omega"]))),
        "theta0": 0.0,
    }


def init_params(cfg: CCFVorticityDiscoveryConfig) -> dict[str, Array]:
    key = jax.random.PRNGKey(cfg.seed)
    if cfg.independent_terms:
        n = int(cfg.n_terms)
        k1, k2 = jax.random.split(key)
        coeffs = jax.random.normal(k1, (n,), dtype=jnp.float64) * 0.05
        coeffs = coeffs.at[0].add(0.35)
        # Scales spread across [scale_lo, scale_hi]
        t = jnp.linspace(0.0, 1.0, n, dtype=jnp.float64)
        scales0 = cfg.scale_lo * (cfg.scale_hi / cfg.scale_lo) ** t
        log_scales = jnp.log(scales0)
        # γ_0 = α; others start near 2α, 3α, ...
        alpha0 = alpha_from_lambda(cfg.lam)
        gamma0 = alpha0 * jnp.arange(1, n + 1, dtype=jnp.float64)
        # Store log(γ/α) with first frozen via expand
        log_gamma_over_alpha = jnp.log(jnp.maximum(gamma0 / alpha0, 1e-8))
        return {
            "coeffs": coeffs,
            "log_scales": log_scales,
            "log_gamma_over_alpha": log_gamma_over_alpha,
        }
    scales, _alphas, _ = _dictionary(cfg)
    coeffs = jax.random.normal(key, (scales.shape[0],), dtype=jnp.float64) * 0.05
    coeffs = coeffs.at[:: cfg.n_gamma_multiples].add(0.3)
    log_scales = jnp.log(jnp.asarray(scales[:: cfg.n_gamma_multiples], dtype=jnp.float64))
    params: dict[str, Array] = {"coeffs": coeffs, "log_scales_base": log_scales}
    if cfg.free_gamma_offsets and cfg.n_gamma_multiples > 1:
        params["gamma_offset_raw"] = jnp.zeros(
            (cfg.n_gamma_multiples - 1,), dtype=jnp.float64
        )
    return params


def _expand(
    params: dict[str, Array], cfg: CCFVorticityDiscoveryConfig
) -> tuple[Array, Array, Array]:
    alpha0 = alpha_from_lambda(cfg.lam)
    if cfg.independent_terms:
        log_lo = math.log(max(float(cfg.scale_lo), 1e-8))
        log_hi = math.log(max(float(cfg.scale_hi), float(cfg.scale_lo) * 1.01))
        scales = jnp.exp(jnp.clip(params["log_scales"], log_lo, log_hi))
        # Pin leading γ to α for far-field cancel; free the rest via log(γ/α).
        lga = params["log_gamma_over_alpha"]
        lga = lga.at[0].set(0.0)
        # Keep γ/α ≥ 0.5 to avoid singular U formulas near γ=0
        gammas = alpha0 * jnp.exp(jnp.clip(lga, math.log(0.5), math.log(20.0)))
        coeffs = params["coeffs"]
    else:
        k = jnp.arange(1, cfg.n_gamma_multiples + 1, dtype=jnp.float64)
        if (
            cfg.free_gamma_offsets
            and "gamma_offset_raw" in params
            and cfg.n_gamma_multiples > 1
        ):
            raw = params["gamma_offset_raw"]
            delta = jax.nn.softplus(raw) - jax.nn.softplus(jnp.zeros_like(raw))
            delta = jnp.concatenate([jnp.zeros((1,), dtype=jnp.float64), delta])
            gammas = alpha0 * (k + delta)
        else:
            gammas = alpha0 * k
        log_lo = math.log(max(float(cfg.scale_lo), 1e-8))
        log_hi = math.log(max(float(cfg.scale_hi), float(cfg.scale_lo) * 1.01))
        scales_base = jnp.exp(jnp.clip(params["log_scales_base"], log_lo, log_hi))
        scales = jnp.repeat(scales_base, cfg.n_gamma_multiples)
        gammas = jnp.tile(gammas, cfg.n_scales)
        coeffs = params["coeffs"]
    alphas = gammas
    if cfg.hard_gauge_rescale:
        y_g = jnp.asarray([cfg.gauge_point], dtype=jnp.float64)
        om_g, _, _, _ = hardy_omega_profile(y_g, coeffs, scales, alphas)
        scale = cfg.gauge_value / (om_g[0] + 1e-30)
        coeffs = coeffs * scale
    return coeffs, scales, alphas


def residual_vector(
    params: dict[str, Array],
    y: Array,
    cfg: CCFVorticityDiscoveryConfig,
) -> Array:
    coeffs, scales, alphas = _expand(params, cfg)
    r, fields = vorticity_residual_samples(y, coeffs, scales, alphas, cfg.lam)
    if cfg.near_field_power > 0.0:
        r = r * jnp.power(1.0 + jnp.abs(y), -float(cfg.near_field_power))
    if cfg.peak_weight_power > 0.0:
        # Emphasize residual peaks without changing the zero set.
        r = r * jnp.power(1.0 + jnp.abs(r), float(cfg.peak_weight_power))
    om_g = jnp.interp(cfg.gauge_point, y, fields["omega"])
    gauge = jnp.asarray([(om_g - cfg.gauge_value) * math.sqrt(cfg.gauge_weight)])
    parts = [r, gauge]
    if cfg.omega_max_floor > 0.0 and cfg.omega_max_weight > 0.0:
        omax = jnp.max(jnp.abs(fields["omega"]))
        # Soft hinge: penalize only when max|Ω| is below the floor.
        parts.append(
            jnp.asarray(
                [
                    math.sqrt(cfg.omega_max_weight)
                    * jnp.minimum(omax - float(cfg.omega_max_floor), 0.0)
                ]
            )
        )
    l2 = math.sqrt(cfg.coeff_l2) * coeffs
    parts.append(l2)
    return jnp.concatenate(parts)


def _collocation_grid(
    cfg: CCFVorticityDiscoveryConfig,
    *,
    residual_weights: np.ndarray | None = None,
    y_pool: np.ndarray | None = None,
    seed: int = 0,
) -> Array:
    """Build a mixed origin + residual-adaptive collocation grid on ``[-y_max,y_max]``."""
    n = int(cfg.n_grid)
    if n % 2 == 0:
        n += 1
    rng = np.random.default_rng(int(seed))
    n_origin = max(3, int(round(float(cfg.origin_fraction) * n)))
    if n_origin % 2 == 0:
        n_origin += 1
    n_adapt = max(1, n - n_origin)
    y_ori = np.concatenate(
        [
            rng.uniform(-1.0, 1.0, size=n_origin // 2),
            rng.uniform(0.15, 1.25, size=n_origin - n_origin // 2),
        ]
    )
    if residual_weights is None or y_pool is None or y_pool.size < 2:
        y_ad = rng.uniform(-cfg.y_max, cfg.y_max, size=n_adapt)
    else:
        w = np.asarray(residual_weights, dtype=float).reshape(-1)
        w = np.maximum(w, 0.0)
        if float(w.sum()) <= 0.0:
            w = np.ones_like(w)
        w = w / w.sum()
        idx = rng.choice(y_pool.size, size=n_adapt, replace=True, p=w)
        y_ad = y_pool[idx] + rng.normal(0.0, 0.02 * cfg.y_max / max(n, 1), size=n_adapt)
        y_ad = np.clip(y_ad, -cfg.y_max, cfg.y_max)
    y = np.sort(np.concatenate([y_ori, y_ad, [0.0, cfg.gauge_point, -cfg.gauge_point]]))
    # Keep odd length uniqueness for FD-friendly grids.
    y = np.unique(np.round(y, decimals=10))
    if y.size < n:
        extra = rng.uniform(-cfg.y_max, cfg.y_max, size=n - y.size)
        y = np.unique(np.concatenate([y, extra]))
    if y.size > n:
        # Prefer keeping origin neighborhood.
        keep = np.argsort(np.abs(y))[:n]
        y = np.sort(y[keep])
    return jnp.asarray(y, dtype=jnp.float64)


def run_ccf_vorticity_discovery(
    cfg: CCFVorticityDiscoveryConfig | None = None,
    *,
    steps: int | None = None,
    warm_params: dict[str, Array] | None = None,
) -> CCFVorticityDiscoveryResult:
    """Gauss-Newton discovery of a Hardy-Ω vorticity profile at fixed ``lambda``."""
    cfg = cfg or CCFVorticityDiscoveryConfig()
    n_steps = int(cfg.gn_steps if steps is None else steps)
    rounds = max(1, int(cfg.adaptive_rounds) + 1)
    steps_per = max(8, n_steps // rounds)
    # Initial grid: clustered near origin via t**1.5 on the half-line.
    t = jnp.linspace(0.0, 1.0, cfg.n_grid, dtype=jnp.float64)
    y_pos = cfg.y_max * (t**1.5)
    y = jnp.concatenate([-y_pos[:0:-1], y_pos])
    params = init_params(cfg) if warm_params is None else warm_params
    hist_all: list[float] = []
    trained: dict[str, Array] = params

    for rnd in range(rounds):
        y_round = y

        def r_fn(pp: object, y_grid: Array = y_round) -> Array:
            assert isinstance(pp, dict)
            return residual_vector(pp, y_grid, cfg)

        trained, hist = gauss_newton_minimize(
            r_fn,
            trained,
            config=GNConfig(
                steps=steps_per,
                gamma=cfg.gn_gamma,
                use_martens_grosse=True,
                solver="qr",
                seed=int(cfg.seed) + rnd,
            ),
        )
        assert isinstance(trained, dict)
        hist_all.extend([float(h) for h in hist])
        if rnd + 1 >= rounds or int(cfg.adaptive_rounds) <= 0:
            break
        # Resample toward residual peaks on a dense probe pool.
        export_cfg = cfg if cfg.hard_gauge_rescale else replace(cfg, hard_gauge_rescale=True)
        c_tmp, s_tmp, g_tmp = _expand(trained, export_cfg)
        y_pool = np.linspace(-float(cfg.y_max), float(cfg.y_max), max(int(cfg.n_grid) * 4, 401))
        r_pool, _ = vorticity_residual_samples(
            jnp.asarray(y_pool), c_tmp, s_tmp, g_tmp, cfg.lam
        )
        weights = np.abs(np.asarray(r_pool, dtype=float)) ** float(cfg.adaptive_power)
        y = _collocation_grid(
            cfg,
            residual_weights=weights,
            y_pool=y_pool,
            seed=int(cfg.seed) + 17 * (rnd + 1),
        )

    export_cfg = cfg if cfg.hard_gauge_rescale else replace(cfg, hard_gauge_rescale=True)
    coeffs, scales, alphas = _expand(trained, export_cfg)
    r, fields = vorticity_residual_samples(y, coeffs, scales, alphas, cfg.lam)
    dense = dense_vorticity_residual(coeffs, scales, alphas, cfg.lam, y_max=cfg.y_max)
    ff = leading_mode_far_field_cancel(lam=cfg.lam)
    diagnostics = {
        "max_abs_vorticity_residual": float(jnp.max(jnp.abs(r))),
        "rms_vorticity_residual": float(jnp.sqrt(jnp.mean(r * r))),
        "lam": float(cfg.lam),
        "alpha0": float(alpha_from_lambda(cfg.lam)),
        "loss_final": float(hist_all[-1]) if hist_all else float("nan"),
        "loss_initial": float(hist_all[0]) if hist_all else float("nan"),
        "far_field_cancel_factor": ff["expected_cancel_factor"],
        "adaptive_rounds": float(rounds - 1),
        **dense,
    }
    return CCFVorticityDiscoveryResult(
        lam=float(cfg.lam),
        coeffs=np.asarray(coeffs),
        scales=np.asarray(scales),
        alphas=np.asarray(alphas),
        y=np.asarray(y),
        omega=np.asarray(fields["omega"]),
        residual=np.asarray(r),
        diagnostics=diagnostics,
        config=cfg,
        extra={
            "hilbert_convention": "hardy_exact_omega",
            "train_hilbert": "hardy_exact_omega",
            "residual_form": "wang_vorticity",
            "ansatz": "omega_primary_hardy_odd_gamma_k_alpha",
            "gauge": f"Omega({cfg.gauge_point})={cfg.gauge_value}",
            "optimizer": "martens_grosse_gn",
            "gn_solver": "qr",
            "adaptive_collocation": bool(cfg.adaptive_rounds > 0),
        },
        params={k: np.asarray(v) for k, v in trained.items()},
    )


__all__ = [
    "CCFVorticityDiscoveryConfig",
    "CCFVorticityDiscoveryResult",
    "dense_vorticity_residual",
    "hardy_odd_profile",
    "hardy_omega_profile",
    "hardy_theta_profile",
    "init_params",
    "leading_mode_far_field_cancel",
    "project_omega_hardy_numpy",
    "residual_vector",
    "run_ccf_vorticity_discovery",
    "vorticity_residual_samples",
]
