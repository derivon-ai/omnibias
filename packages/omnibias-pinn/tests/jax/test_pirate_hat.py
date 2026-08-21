# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Signed PirateNet hat: identity skip, PI-init, honesty locks."""

from __future__ import annotations

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from omnibias.pinn.jax.discovery.pirate_hat import (  # noqa: E402
    PirateHatConfig,
    compose_hats,
    even_features,
    fit_hat_l2,
    fit_pi_init,
    hat_from_pirate,
    init_pirate_params,
    omega_from_pirate,
    pirate_features,
    pirate_hat_y,
    score_pirate,
)

LAM = 0.6057


def test_alpha_zero_features_are_embedding() -> None:
    cfg = PirateHatConfig(hidden=8, n_layers=2, seed=1)
    params = init_pirate_params(cfg)
    assert float(jnp.max(jnp.abs(params["alpha"]))) == 0.0
    y = jnp.linspace(-4.0, 4.0, 17, dtype=jnp.float64)
    coords = even_features(y, lam=LAM)
    feat = pirate_features(params, coords)
    emb = jnp.tanh(coords @ params["We"].T + params["be"])
    assert float(jnp.max(jnp.abs(feat - emb))) < 1e-14


def test_pi_init_matches_even_target_hat() -> None:
    cfg = PirateHatConfig(hidden=12, n_layers=1, seed=2)
    params = init_pirate_params(cfg)
    y = jnp.linspace(-6.0, 6.0, 81, dtype=jnp.float64)
    coords = even_features(y, lam=LAM)
    target = 1.0 - 0.35 * coords[:, 0] * coords[:, 0]
    fitted = fit_pi_init(params, y, target, lam=LAM)
    pred = pirate_hat_y(fitted, y, lam=LAM)
    rel = float(jnp.max(jnp.abs(pred - target))) / (float(jnp.max(jnp.abs(target))) + 1e-12)
    assert rel < 0.05


def test_signed_hat_is_even_and_omega_odd() -> None:
    cfg = PirateHatConfig(hidden=8, n_layers=1, seed=3)
    params = init_pirate_params(cfg)
    y = jnp.linspace(-5.0, 5.0, 51, dtype=jnp.float64)
    coords = even_features(y, lam=LAM)
    target = 0.4 - coords[:, 0]
    params = fit_pi_init(params, y, target, lam=LAM)
    hat, hat_y = hat_from_pirate(params, y, lam=LAM)
    om, omy = omega_from_pirate(params, y, lam=LAM)
    assert float(jnp.max(jnp.abs(hat - hat[::-1]))) < 1e-12
    assert float(jnp.max(jnp.abs(hat_y + hat_y[::-1]))) < 1e-12
    assert float(jnp.max(jnp.abs(om + om[::-1]))) < 1e-12
    assert float(jnp.max(jnp.abs(omy - omy[::-1]))) < 1e-10


def test_hat_l2_tightens_readout_pi_init() -> None:
    cfg = PirateHatConfig(hidden=8, n_layers=1, seed=5)
    params = init_pirate_params(cfg)
    y = jnp.linspace(-6.0, 6.0, 65, dtype=jnp.float64)
    coords = even_features(y, lam=LAM)
    target = 0.7 - coords[:, 0] + 0.2 * coords[:, 1]
    readout = fit_pi_init(params, y, target, lam=LAM)
    err0 = float(jnp.max(jnp.abs(pirate_hat_y(readout, y, lam=LAM) - target)))
    refined, _hist = fit_hat_l2(readout, y, target, lam=LAM, steps=4)
    err1 = float(jnp.max(jnp.abs(pirate_hat_y(refined, y, lam=LAM) - target)))
    assert err1 <= err0 + 1e-12
    assert err1 < 1.0


def test_zero_readout_compose_is_identity() -> None:
    cfg = PirateHatConfig(hidden=6, n_layers=1, seed=6)
    params = init_pirate_params(cfg)
    y = jnp.linspace(-4.0, 4.0, 33, dtype=jnp.float64)
    base = 1.0 + 0.2 * y * y
    base_y = 0.4 * y
    hat, hat_y = compose_hats(params, y, lam=LAM, base_hat=base, base_hat_y=base_y)
    assert float(jnp.max(jnp.abs(hat - base))) < 1e-14
    assert float(jnp.max(jnp.abs(hat_y - base_y))) < 1e-14


def test_score_never_forges_rung1_or_ns() -> None:
    cfg = PirateHatConfig(hidden=6, n_layers=1, n_grid=41, seed=4)
    params = init_pirate_params(cfg)
    y = jnp.linspace(-8.0, 8.0, 41, dtype=jnp.float64)
    params = fit_pi_init(params, y, jnp.ones_like(y), lam=LAM)
    scored = score_pirate(params, cfg=cfg, n_score=201)
    assert scored["navier_stokes_proof_claim"] is False
    assert scored["rung1_1e-11_report"] is False
    assert scored["stretch_1e-13_cleared"] is False
    assert scored["anti_ghost"] is True
    assert scored["dense_max_abs"] > 1e-6
