# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Signed PirateNet hat on the official CCF envelope (free Ω).

jaxpi PirateNets (Wang–Li–Chen–Perdikaris, arXiv:2402.00326) are gated
PI-ResNet blocks with identity-init skips. This module ports **only** that
backbone onto the official Wang lift ``Ω = y · E · hat`` with a *signed*
hat (no softplus). Official ``CompactifiedOmegaOMBU`` defaults
``exp_core=True`` (softplus), which cannot represent a signed champ hat.

PI-init least-squares the linear readout to a target even hat (DeepMind /
jaxpi physics-informed init). Gauss–Newton then uses
:func:`free_omega_vorticity_residual`. This is not a stretch or Rung-1
claim until the measured dense residual clears the named gates.
``navier_stokes_proof_claim`` stays false.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
from jax import Array
from omnibias.jax.architectures.piratenet import (
    PirateNetConfig,
    pirate_apply,
)
from omnibias.jax.architectures.piratenet import (
    init_pirate_params as init_shared_pirate_params,
)
from omnibias.jax.architectures.piratenet import (
    pirate_features as pirate_features_shared,
)
from omnibias.pinn.jax.discovery.ccf_hat_homotopy import (
    GAUGE,
    GAUGE_PT,
    hard_gauge,
    honesty_flags,
    train_nodes,
)
from omnibias.pinn.jax.discovery.ccf_vorticity import (
    free_omega_vorticity_residual,
)
from omnibias.pinn.jax.discovery.train_gn import (
    GNConfig,
    gauss_newton_minimize,
)
from omnibias.pinn.jax.equations.ccf_compactified import (
    alpha_from_lambda,
    apply_envelope,
    compactify_y_lambda,
)

jax.config.update("jax_enable_x64", True)

LAM_DEFAULT = 0.6057


@dataclass(frozen=True)
class PirateHatConfig:
    """Signed PirateNet hat on compactified ``q``."""

    hidden: int = 24
    n_layers: int = 2
    lam: float = LAM_DEFAULT
    gauge_point: float = GAUGE_PT
    gauge_value: float = GAUGE
    n_grid: int = 121
    y_max: float = 40.0
    seed: int = 0
    gn_steps: int = 12
    gn_gamma: float = 1e-3


N_FEATURES = 3  # (q, y^2/(1+y^2), even bump); q alone crushes the near field


def even_features(
    y: Array,
    *,
    lam: float,
    bump_c: float = 1.35,
    bump_w: float = 0.40,
) -> Array:
    """Even-in-``y`` coordinates. ``q`` is DeepMind's compactification."""
    y = jnp.asarray(y, dtype=jnp.float64)
    q = compactify_y_lambda(y, lam)
    xi = (y * y) / (1.0 + y * y)
    # Even C^∞ bump in y^2 (do not use |y|: that kink pollutes Ω_y).
    bump = jnp.exp(-((y * y - float(bump_c) ** 2) / float(bump_w)) ** 2)
    return jnp.stack([q, xi, bump], axis=-1)


def init_pirate_params(
    cfg: PirateHatConfig,
    *,
    key: Array | None = None,
) -> dict[str, Any]:
    """Identity-skip PirateNet (``alpha=0``) plus a zero readout."""
    return init_shared_pirate_params(
        PirateNetConfig(
            in_dim=N_FEATURES,
            hidden=int(cfg.hidden),
            n_layers=int(cfg.n_layers),
            out_dim=1,
            seed=int(cfg.seed),
        ),
        key=key,
    )


def pirate_features(params: dict[str, Any], coords: Array) -> Array:
    """Penultimate features, shape ``(..., hidden)``. ``alpha=0`` is identity."""
    return pirate_features_shared(params, coords)


def pirate_hat_coords(params: dict[str, Any], coords: Array) -> Array:
    """Signed hat from even coordinates (scalar-batch or ``(n, 2)``)."""
    return pirate_apply(params, coords)


def pirate_hat_y(params: dict[str, Any], y: Array, *, lam: float) -> Array:
    """Signed even hat as a function of physical ``y``."""
    return pirate_hat_coords(params, even_features(y, lam=lam))


def hat_from_pirate(
    params: dict[str, Any],
    y: Array,
    *,
    lam: float,
) -> tuple[Array, Array]:
    """Even signed hat and analytic ``hat_y`` via JVP on ``y``."""
    y = jnp.asarray(y, dtype=jnp.float64)

    def _hat_one(yi: Array) -> Array:
        return pirate_hat_y(params, yi, lam=lam)

    return jax.vmap(
        lambda yi: jax.jvp(_hat_one, (yi,), (jnp.ones((), dtype=jnp.float64),))
    )(y)


def omega_from_pirate(
    params: dict[str, Any],
    y: Array,
    *,
    lam: float,
) -> tuple[Array, Array]:
    """Official envelope lift of a signed PirateNet hat."""
    hat, hat_y = hat_from_pirate(params, y, lam=lam)
    alpha = float(alpha_from_lambda(lam))
    psi, psi_y = apply_envelope(y, hat, hat_y, power=alpha + 1.0)
    return y * psi, psi + y * psi_y


def fit_pi_init(
    params: dict[str, Any],
    y: Array,
    target_hat: Array,
    *,
    lam: float,
) -> dict[str, Any]:
    """Least-squares linear readout to a target even hat (jaxpi PI-init)."""
    feat = pirate_features(params, even_features(y, lam=lam))
    ones = jnp.ones((feat.shape[0], 1), dtype=jnp.float64)
    a = jnp.concatenate([feat, ones], axis=1)
    coef, *_ = jnp.linalg.lstsq(a, jnp.asarray(target_hat, dtype=jnp.float64), rcond=None)
    out = dict(params)
    out["Wout"] = coef[:-1]
    out["bout"] = coef[-1]
    return out


def fit_hat_l2(
    params: dict[str, Any],
    y: Array,
    target_hat: Array,
    *,
    lam: float,
    steps: int = 8,
    gamma: float = 1e-4,
) -> tuple[dict[str, Any], Array]:
    """Gauss–Newton match of the signed hat (no Hilbert). Stronger than readout-only PI-init."""
    target = jnp.asarray(target_hat, dtype=jnp.float64)
    yy = jnp.asarray(y, dtype=jnp.float64)

    def r_fn(th: dict[str, Any]) -> Array:
        return pirate_hat_y(th, yy, lam=lam) - target

    return gauss_newton_minimize(  # type: ignore[return-value]
        r_fn,
        params,
        config=GNConfig(
            steps=int(steps),
            gamma=float(gamma),
            use_martens_grosse=True,
            method="martens_grosse",
            solver="qr",
        ),
    )


def score_pirate(
    params: dict[str, Any],
    *,
    cfg: PirateHatConfig | None = None,
    n_score: int = 1601,
) -> dict[str, Any]:
    """Official-path dense Wang residual on ``|y|<38``."""
    cfg = PirateHatConfig() if cfg is None else cfg
    y_train = train_nodes(int(cfg.n_grid), cfg.y_max, core=2.0, n_core=81)
    y_score = jnp.linspace(-40.0, 40.0, int(n_score), dtype=jnp.float64)

    def omega_fn(tt: Array, th: dict[str, Any] = params) -> Array:
        om, _ = omega_from_pirate(th, tt, lam=cfg.lam)
        g = jnp.interp(cfg.gauge_point, y_train, omega_from_pirate(th, y_train, lam=cfg.lam)[0])
        return om * (cfg.gauge_value / (g + 1e-30))

    om, omy = omega_from_pirate(params, y_score, lam=cfg.lam)
    om, omy = hard_gauge(y_score, om, omy, point=cfg.gauge_point, value=cfg.gauge_value)
    r, fields = free_omega_vorticity_residual(
        y_score, om, omy, omega_fn, lam=cfg.lam, y_trunc=40.0
    )
    mask = jnp.abs(y_score) < 38.0
    dense = float(jnp.max(jnp.abs(r[mask])))
    omax = float(jnp.max(jnp.abs(fields["omega"])))
    og = float(jnp.interp(cfg.gauge_point, y_score, fields["omega"]))
    anti = abs(og - cfg.gauge_value) <= 0.01 and omax >= 0.02
    out: dict[str, Any] = {
        "dense_max_abs": dense,
        "h0": float(jnp.interp(0.0, y_score, fields["U_y"])),
        "h0_target": float(0.5 * (2.0 + cfg.lam)),
        "omega_max": omax,
        "omega_gauge": og,
        "anti_ghost": bool(anti),
        "peak_y": float(y_score[jnp.argmax(jnp.abs(jnp.where(mask, r, 0.0)))]),
    }
    out.update(honesty_flags(dense_max_abs=dense, anti_ghost=anti))
    return out


def compose_hats(
    params: dict[str, Any],
    y: Array,
    *,
    lam: float,
    base_hat: Array,
    base_hat_y: Array,
) -> tuple[Array, Array]:
    """``base + pirate``. Zero readout is an exact identity correction."""
    hat, hat_y = hat_from_pirate(params, y, lam=lam)
    return base_hat + hat, base_hat_y + hat_y


def residual_vector(params: dict[str, Any], y: Array, *, cfg: PirateHatConfig) -> Array:
    """Hard-gauged official residual on the train grid (GN objective)."""

    def omega_fn(tt: Array, th: dict[str, Any] = params) -> Array:
        om, _ = omega_from_pirate(th, tt, lam=cfg.lam)
        g = jnp.interp(cfg.gauge_point, y, omega_from_pirate(th, y, lam=cfg.lam)[0])
        return om * (cfg.gauge_value / (g + 1e-30))

    om, omy = omega_from_pirate(params, y, lam=cfg.lam)
    om, omy = hard_gauge(y, om, omy, point=cfg.gauge_point, value=cfg.gauge_value)
    r, _ = free_omega_vorticity_residual(y, om, omy, omega_fn, lam=cfg.lam, y_trunc=40.0)
    return r


def run_pirate_gn(
    params0: dict[str, Any],
    *,
    cfg: PirateHatConfig | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Martens–Grosse on the signed PirateNet hat. Prototype; no stretch forge."""
    cfg = PirateHatConfig() if cfg is None else cfg
    y = train_nodes(int(cfg.n_grid), cfg.y_max, core=2.0, n_core=81)

    def r_fn(th: dict[str, Any]) -> Array:
        return residual_vector(th, y, cfg=cfg)

    trained, hist = gauss_newton_minimize(
        r_fn,
        params0,
        config=GNConfig(
            steps=int(cfg.gn_steps),
            gamma=float(cfg.gn_gamma),
            use_martens_grosse=True,
            method="martens_grosse",
            solver="qr",
        ),
    )
    scored = score_pirate(trained, cfg=cfg)
    scored["gn_loss_hist"] = [float(x) for x in hist]
    return trained, scored


__all__ = [
    "LAM_DEFAULT",
    "N_FEATURES",
    "PirateHatConfig",
    "compose_hats",
    "even_features",
    "fit_hat_l2",
    "fit_pi_init",
    "hat_from_pirate",
    "init_pirate_params",
    "omega_from_pirate",
    "pirate_features",
    "pirate_hat_coords",
    "pirate_hat_y",
    "residual_vector",
    "run_pirate_gn",
    "score_pirate",
]
