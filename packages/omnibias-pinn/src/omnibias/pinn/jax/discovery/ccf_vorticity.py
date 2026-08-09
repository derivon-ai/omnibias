# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Wang / DeepMind vorticity-form CCF discovery on a Hardy-even Theta basis.

The active-scalar transport residual
``(1+λ) y Θ' - λ Θ + s (HΘ) Θ'`` is structurally nonzero at the origin for any
even profile with ``Θ(0) ≠ 0`` (it equals ``-λ Θ(0)``). Published unstable
profiles are discovered in **vorticity form** (Wang et al., arXiv:2509.14185 /
2511.22819):

    Ω + ((1+λ) y - U) Ω_y - Ω U_y = 0,
    U = HΘ,   Ω = Θ',   U_y = (HΘ)',

with odd ``Ω`` and gauge ``Ω(y_g) = g_val`` (default ``Ω(0.5)=0.05``).

Ansatz
------
``Θ(y) = sum_j c_j P_{a_j, α_j}(y)`` (even Hardy sum) with exact
``HΘ = sum c_j Q`` and ``Ω = Θ'``.  Optional ``Theta(0)=0`` nullspace
constraint (required for a true transport zero; compatible with vorticity).

Rung-1 absolute gates are evaluated on a dense fixed ``linspace`` of this
vorticity residual — never on the adaptive train grid alone.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
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
    """Hardy-even Theta / vorticity discovery configuration."""

    n_scales: int = 16
    n_alpha_offsets: int = 3  # alpha0 + 2m
    n_grid: int = 401
    y_max: float = 40.0
    lam: float = 0.6057
    gauge_point: float = 0.5
    gauge_value: float = 0.05
    gauge_weight: float = 50.0
    theta0_weight: float = 20.0
    enforce_theta0: bool = True
    seed: int = 0
    gn_steps: int = 80
    gn_gamma: float = 1e-3
    coeff_l2: float = 1e-6


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


def _dictionary(
    cfg: CCFVorticityDiscoveryConfig,
) -> tuple[np.ndarray, np.ndarray, float]:
    alpha0 = float(alpha_from_lambda(cfg.lam))
    offsets = np.arange(cfg.n_alpha_offsets, dtype=float) * 2.0
    scales_base = np.geomspace(0.35, 8.0, cfg.n_scales)
    scales = np.repeat(scales_base, cfg.n_alpha_offsets)
    alphas = alpha0 + np.tile(offsets, cfg.n_scales)
    return scales, alphas, alpha0


def hardy_theta_profile(
    y: Array,
    coeffs: Array,
    scales: Array,
    alphas: Array,
) -> tuple[Array, Array, Array, Array, Array]:
    """Return ``(Theta, Omega, Omega_y, U, U_y)`` for an even Hardy sum.

    Exact Hilbert: ``U = HTheta``, ``U_y = (HTheta)'``; ``Omega = Theta'``.
    """
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
        # Theta'' = -al*(al+1) P_{a, al+2}
        omy = omy + c * (-al * (al + 1.0) * hardy_even(y, a, al + 2.0))
        u = u + c * hardy_odd(y, a, al)
        uy = uy + c * hardy_odd_deriv(y, a, al)
    return th, om, omy, u, uy


# Back-compat alias used by tests / callers expecting Q-basis naming.
def hardy_odd_profile(
    y: Array,
    coeffs: Array,
    scales: Array,
    alphas: Array,
) -> tuple[Array, Array, Array, Array]:
    """Legacy Q-basis helper: ``(Omega, Omega_y, HOmega, Theta_placeholder)``."""
    om = jnp.zeros_like(jnp.asarray(y).reshape(-1), dtype=jnp.float64)
    omy = jnp.zeros_like(om)
    hom = jnp.zeros_like(om)
    for c, a, al in zip(
        jnp.asarray(coeffs).reshape(-1),
        jnp.asarray(scales).reshape(-1),
        jnp.asarray(alphas).reshape(-1),
        strict=True,
    ):
        from omnibias.pinn.jax.equations.ccf_compactified import hardy_odd as _Q

        om = om + c * _Q(y, a, al)
        omy = omy + c * al * hardy_even(y, a, al + 1.0)
        hom = hom + c * (-hardy_even(y, a, al))
    return om, omy, hom, jnp.zeros_like(om)


def vorticity_residual_samples(
    y: Array,
    coeffs: Array,
    scales: Array,
    alphas: Array,
    lam: float,
) -> tuple[Array, dict[str, Array]]:
    """Wang vorticity residual on samples (exact ``U = HTheta``)."""
    th, om, omy, u, uy = hardy_theta_profile(y, coeffs, scales, alphas)
    r = om + ((1.0 + lam) * y - u) * omy - om * uy
    return r, {
        "theta": th,
        "omega": om,
        "omega_y": omy,
        "U": u,
        "U_y": uy,
    }


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
        "theta0": float(fields["theta"][n // 2]),
    }


def init_params(cfg: CCFVorticityDiscoveryConfig) -> dict[str, Array]:
    scales, alphas, _ = _dictionary(cfg)
    key = jax.random.PRNGKey(cfg.seed)
    coeffs = jax.random.normal(key, (scales.shape[0],), dtype=jnp.float64) * 0.05
    coeffs = coeffs.at[:: cfg.n_alpha_offsets].add(0.1)
    log_scales = jnp.log(
        jnp.asarray(scales[:: cfg.n_alpha_offsets], dtype=jnp.float64)
    )
    return {"coeffs": coeffs, "log_scales_base": log_scales}


def _expand(
    params: dict[str, Array], cfg: CCFVorticityDiscoveryConfig
) -> tuple[Array, Array, Array]:
    alpha0 = alpha_from_lambda(cfg.lam)
    offsets = jnp.arange(cfg.n_alpha_offsets, dtype=jnp.float64) * 2.0
    scales_base = jnp.exp(jnp.clip(params["log_scales_base"], -4.0, 4.0))
    scales = jnp.repeat(scales_base, cfg.n_alpha_offsets)
    alphas = alpha0 + jnp.tile(offsets, cfg.n_scales)
    coeffs = params["coeffs"]
    if cfg.enforce_theta0:
        # Null Theta(0): P(0)=a^{-alpha}; solve for first coeff.
        th0 = jnp.power(scales, -alphas)
        c0 = -jnp.dot(th0[1:], coeffs[1:]) / (th0[0] + 1e-30)
        coeffs = coeffs.at[0].set(c0)
    # soft gauge rescale toward Omega(y_g)=g_val
    y_g = jnp.asarray([cfg.gauge_point], dtype=jnp.float64)
    _, om_g, _, _, _ = hardy_theta_profile(y_g, coeffs, scales, alphas)
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
    om_g = jnp.interp(cfg.gauge_point, y, fields["omega"])
    gauge = jnp.asarray(
        [(om_g - cfg.gauge_value) * math.sqrt(cfg.gauge_weight)]
    )
    extras = [r, gauge]
    if cfg.enforce_theta0:
        extras.append(
            jnp.asarray(
                [fields["theta"][y.shape[0] // 2] * math.sqrt(cfg.theta0_weight)]
            )
        )
    l2 = math.sqrt(cfg.coeff_l2) * coeffs
    extras.append(l2)
    return jnp.concatenate(extras)


def run_ccf_vorticity_discovery(
    cfg: CCFVorticityDiscoveryConfig | None = None,
    *,
    steps: int | None = None,
) -> CCFVorticityDiscoveryResult:
    """Gauss-Newton discovery of a Hardy vorticity profile at fixed ``lambda``."""
    cfg = cfg or CCFVorticityDiscoveryConfig()
    n_steps = int(cfg.gn_steps if steps is None else steps)
    t = jnp.linspace(0.0, 1.0, cfg.n_grid, dtype=jnp.float64)
    y_pos = cfg.y_max * (t**1.5)
    y = jnp.concatenate([-y_pos[:0:-1], y_pos])
    params = init_params(cfg)

    def r_fn(pp: object) -> Array:
        assert isinstance(pp, dict)
        return residual_vector(pp, y, cfg)

    trained, hist = gauss_newton_minimize(
        r_fn,
        params,
        config=GNConfig(
            steps=n_steps,
            gamma=cfg.gn_gamma,
            use_martens_grosse=True,
        ),
    )
    assert isinstance(trained, dict)
    coeffs, scales, alphas = _expand(trained, cfg)
    r, fields = vorticity_residual_samples(y, coeffs, scales, alphas, cfg.lam)
    dense = dense_vorticity_residual(coeffs, scales, alphas, cfg.lam, y_max=cfg.y_max)
    diagnostics = {
        "max_abs_vorticity_residual": float(jnp.max(jnp.abs(r))),
        "rms_vorticity_residual": float(jnp.sqrt(jnp.mean(r * r))),
        "lam": float(cfg.lam),
        "alpha0": float(alpha_from_lambda(cfg.lam)),
        "loss_final": float(hist[-1]) if len(hist) else float("nan"),
        "loss_initial": float(hist[0]) if len(hist) else float("nan"),
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
            "hilbert_convention": "hardy_exact_theta",
            "residual_form": "wang_vorticity",
            "ansatz": "multi_alpha_hardy_even_theta",
            "gauge": f"Omega({cfg.gauge_point})={cfg.gauge_value}",
            "enforce_theta0": bool(cfg.enforce_theta0),
        },
    )


__all__ = [
    "CCFVorticityDiscoveryConfig",
    "CCFVorticityDiscoveryResult",
    "dense_vorticity_residual",
    "hardy_odd_profile",
    "hardy_theta_profile",
    "init_params",
    "residual_vector",
    "run_ccf_vorticity_discovery",
    "vorticity_residual_samples",
]
