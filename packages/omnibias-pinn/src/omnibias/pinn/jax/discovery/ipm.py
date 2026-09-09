# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""IPM self-similar discovery smoke harness (jax, streamfunction form)."""

from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np
from jax import Array
from omnibias.pinn.jax.equations.ipm_selfsimilar import ipm_selfsimilar_residual_samples


@dataclass(frozen=True)
class IPMDiscoveryConfig:
    n: int = 16
    lam_init: float = 0.5
    seed: int = 0
    steps: int = 50
    lr: float = 1e-2  # retained for API compatibility; unused on the CubicGN path
    method: str = "cubic"


def run_ipm_discovery(cfg: IPMDiscoveryConfig) -> dict[str, object]:
    """Fit Gaussian amplitudes for (Theta, Psi) by CubicGN (Adam forbidden)."""
    from omnibias.pinn.jax.discovery.train_gn import GNConfig, gauss_newton_minimize

    xs = jnp.linspace(-1.0, 1.0, cfg.n)
    y1, y2 = jnp.meshgrid(xs, xs, indexing="ij")
    y1 = y1.reshape(-1)
    y2 = y2.reshape(-1)
    a_th = jnp.asarray(1.0, dtype=jnp.float64)
    a_psi = jnp.asarray(0.5, dtype=jnp.float64)
    lam = jnp.asarray(cfg.lam_init, dtype=jnp.float64)

    def fields(ath: Array, apsi: Array) -> tuple[Array, ...]:
        g = jnp.exp(-(y1 * y1 + y2 * y2))
        theta = ath * g
        ty1 = -2.0 * y1 * theta
        ty2 = -2.0 * y2 * theta
        # Psi ~ y2 * bump so U is nontrivial; lap of Gaussian*poly is analytic
        psi = apsi * y2 * g
        psi_y1 = -2.0 * y1 * psi
        psi_y2 = apsi * g * (1.0 - 2.0 * y2 * y2)
        psi_lap = apsi * g * (-6.0 * y2 + 4.0 * y2 * (y1 * y1 + y2 * y2))
        return theta, ty1, ty2, psi, psi_y1, psi_y2, psi_lap

    def residual_fn(params: tuple[Array, Array]) -> Array:
        ath, apsi = params
        th, ty1, ty2, psi, py1, py2, plap = fields(ath, apsi)
        rt, rp = ipm_selfsimilar_residual_samples(
            y1, y2, th, ty1, ty2, psi, py1, py2, plap, lam
        )
        return jnp.concatenate([rt, rp])

    method = str(cfg.method)
    if method.lower() in {"adam", "sgd"}:
        raise ValueError(
            "run_ipm_discovery earn path forbids Adam/SGD; use CubicGN "
            f"(got method={method!r})"
        )
    params0: tuple[Array, Array] = (a_th, a_psi)
    params, _losses = gauss_newton_minimize(
        residual_fn,
        params0,
        config=GNConfig(steps=int(cfg.steps), method="cubic", seed=int(cfg.seed)),
    )
    th, ty1, ty2, psi, py1, py2, plap = fields(*params)
    rt, rp = ipm_selfsimilar_residual_samples(
        y1, y2, th, ty1, ty2, psi, py1, py2, plap, lam
    )
    return {
        "lam": float(lam),
        "amp_theta": float(params[0]),
        "amp_psi": float(params[1]),
        "max_abs_residual_theta": float(jnp.max(jnp.abs(rt))),
        "max_abs_residual_psi": float(jnp.max(jnp.abs(rp))),
        "honesty": {"unproven_claim": False, "navier_stokes_proof_claim": False},
        "optimizer": "cubic_gn",
        "y1": np.asarray(y1),
        "y2": np.asarray(y2),
        "theta": np.asarray(th),
        "psi": np.asarray(psi),
        "residual_theta": np.asarray(rt),
        "residual_psi": np.asarray(rp),
        "validation_inputs": {
            "y1": np.asarray(y1).tolist(),
            "y2": np.asarray(y2).tolist(),
            "theta": np.asarray(th).tolist(),
            "theta_y1": np.asarray(ty1).tolist(),
            "theta_y2": np.asarray(ty2).tolist(),
            "psi": np.asarray(psi).tolist(),
            "psi_y1": np.asarray(py1).tolist(),
            "psi_y2": np.asarray(py2).tolist(),
            "psi_lap": np.asarray(plap).tolist(),
            "lambda": float(lam),
        },
    }


__all__ = ["IPMDiscoveryConfig", "run_ipm_discovery"]
