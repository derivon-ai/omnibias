# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Boussinesq self-similar discovery smoke harness (jax, streamfunction form)."""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from omnibias.pinn.jax.equations.boussinesq_selfsimilar import (
    boussinesq_selfsimilar_residual_samples,
    infer_lambda_from_streamfunction_u1_y1,
)


@dataclass(frozen=True)
class BoussinesqDiscoveryConfig:
    n: int = 16
    lam_init: float = 1.5
    seed: int = 0
    steps: int = 50
    lr: float = 1e-2


def run_boussinesq_discovery(cfg: BoussinesqDiscoveryConfig) -> dict[str, object]:
    xs = jnp.linspace(0.0, 1.0, cfg.n)
    y1, y2 = jnp.meshgrid(xs, xs, indexing="ij")
    y1 = y1.reshape(-1)
    y2 = y2.reshape(-1)
    a_om = jnp.asarray(1.0, dtype=jnp.float64)
    a_th = jnp.asarray(0.5, dtype=jnp.float64)
    a_psi = jnp.asarray(0.3, dtype=jnp.float64)
    lam = jnp.asarray(cfg.lam_init, dtype=jnp.float64)

    def fields(ao: Array, at: Array, ap: Array) -> tuple[Array, ...]:
        g = jnp.exp(-(y1 * y1 + y2 * y2))
        omega = ao * y1 * g
        ow1 = ao * g * (1.0 - 2.0 * y1 * y1)
        ow2 = -2.0 * y2 * omega
        theta = at * g
        tw1 = -2.0 * y1 * theta
        tw2 = -2.0 * y2 * theta
        psi = ap * y1 * y2 * g
        psi_y1 = ap * y2 * g * (1.0 - 2.0 * y1 * y1)
        psi_y2 = ap * y1 * g * (1.0 - 2.0 * y2 * y2)
        # analytic laplacian of y1 y2 e^{-r^2}
        psi_lap = ap * g * (
            -8.0 * y1 * y2 + 4.0 * y1 * y2 * (y1 * y1 + y2 * y2)
        )
        return omega, ow1, ow2, theta, tw1, tw2, psi, psi_y1, psi_y2, psi_lap

    def loss(params: tuple[Array, Array, Array]) -> Array:
        ao, at, ap = params
        om, ow1, ow2, th, tw1, tw2, psi, py1, py2, plap = fields(ao, at, ap)
        ro, rt, rp = boussinesq_selfsimilar_residual_samples(
            y1, y2, om, ow1, ow2, th, tw1, tw2, psi, py1, py2, plap, lam
        )
        return jnp.mean(ro * ro + rt * rt + rp * rp)

    params: tuple[Array, Array, Array] = (a_om, a_th, a_psi)
    for _ in range(cfg.steps):
        g = jax.grad(loss)(params)
        params = (
            params[0] - cfg.lr * g[0],
            params[1] - cfg.lr * g[1],
            params[2] - cfg.lr * g[2],
        )
    om, ow1, ow2, th, tw1, tw2, psi, py1, py2, plap = fields(*params)
    ro, rt, rp = boussinesq_selfsimilar_residual_samples(
        y1, y2, om, ow1, ow2, th, tw1, tw2, psi, py1, py2, plap, lam
    )
    # U1 = psi_y2; d_y1 U1 at origin ~ evaluate mixed derivative of ansatz at 0
    u1_y1_0 = float(params[2])  # ap * d/dy1 (y1 g (1-2 y2^2)) at 0 = ap
    lam_inferred = float(infer_lambda_from_streamfunction_u1_y1(u1_y1_0))
    return {
        "lam": float(lam),
        "lam_inferred": lam_inferred,
        "max_abs_residual_omega": float(jnp.max(jnp.abs(ro))),
        "max_abs_residual_theta": float(jnp.max(jnp.abs(rt))),
        "max_abs_residual_psi": float(jnp.max(jnp.abs(rp))),
        "honesty": {"unproven_claim": False, "navier_stokes_proof_claim": False},
        "lambda_n_hypothesis": {
            "formula": "1/(1.4187*n + 1.0863) + 1",
            "status": "empirical_hypothesis_not_theorem",
            "init_only": True,
        },
        "residual_omega": np.asarray(ro),
        "residual_theta": np.asarray(rt),
        "residual_psi": np.asarray(rp),
        "validation_inputs": {
            "y1": np.asarray(y1).tolist(),
            "y2": np.asarray(y2).tolist(),
            "omega": np.asarray(om).tolist(),
            "omega_y1": np.asarray(ow1).tolist(),
            "omega_y2": np.asarray(ow2).tolist(),
            "theta": np.asarray(th).tolist(),
            "theta_y1": np.asarray(tw1).tolist(),
            "theta_y2": np.asarray(tw2).tolist(),
            "psi": np.asarray(psi).tolist(),
            "psi_y1": np.asarray(py1).tolist(),
            "psi_y2": np.asarray(py2).tolist(),
            "psi_lap": np.asarray(plap).tolist(),
            "lambda": float(lam),
        },
    }


__all__ = ["BoussinesqDiscoveryConfig", "run_boussinesq_discovery"]
