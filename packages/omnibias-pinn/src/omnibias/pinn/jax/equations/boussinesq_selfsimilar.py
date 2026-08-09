# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""2-D Boussinesq (with boundary) self-similar residual (jax).

Streamfunction-as-unknown system matching the paper / LucasAschenbach form:

* ``R_Omega = U·∇Ω - (1+λ) y·∇Ω + Ω - ∂_{y1} Θ``
* ``R_Theta = U·∇Θ - (1+λ) y·∇Θ + (1-λ) Θ``
* ``R_Psi   = ΔΨ - Ω``, with ``U = (∂_{y2} Ψ, -∂_{y1} Ψ)``

Lambda-inference relation (paper): ``λ = -3 - 2 ∂_{y1} U_1(0,0)`` with
``U_1 = ∂_{y2} Ψ``.

Uses an explicit Laplacian field (closed-form tower / analytic), not stacked AD.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp
from jax import Array


def infer_lambda_from_streamfunction_u1_y1(u1_y1_at_origin: Array | float) -> Array:
    """Paper relation ``lam = -3 - 2 * d_y1 U_1(0,0)``."""
    return -3.0 - 2.0 * jnp.asarray(u1_y1_at_origin)


def boussinesq_selfsimilar_residual_samples(
    y1: Array,
    y2: Array,
    omega: Array,
    omega_y1: Array,
    omega_y2: Array,
    theta: Array,
    theta_y1: Array,
    theta_y2: Array,
    psi: Array,
    psi_y1: Array,
    psi_y2: Array,
    psi_lap: Array,
    lam: Array | float,
) -> tuple[Array, Array, Array]:
    """Return ``(R_omega, R_theta, R_psi)``."""
    u1 = psi_y2
    u2 = -psi_y1
    r_omega = (
        u1 * omega_y1
        + u2 * omega_y2
        - (1.0 + lam) * (y1 * omega_y1 + y2 * omega_y2)
        + omega
        - theta_y1
    )
    r_theta = (
        u1 * theta_y1
        + u2 * theta_y2
        - (1.0 + lam) * (y1 * theta_y1 + y2 * theta_y2)
        + (1.0 - lam) * theta
    )
    r_psi = psi_lap - omega
    return r_omega, r_theta, r_psi


@dataclass
class BoussinesqSelfSimilar:
    lam: float = 1.5

    def residual(
        self,
        y1: Array,
        y2: Array,
        omega: Array,
        omega_y1: Array,
        omega_y2: Array,
        theta: Array,
        theta_y1: Array,
        theta_y2: Array,
        psi: Array,
        psi_y1: Array,
        psi_y2: Array,
        psi_lap: Array,
    ) -> tuple[Array, Array, Array]:
        return boussinesq_selfsimilar_residual_samples(
            y1,
            y2,
            omega,
            omega_y1,
            omega_y2,
            theta,
            theta_y1,
            theta_y2,
            psi,
            psi_y1,
            psi_y2,
            psi_lap,
            self.lam,
        )


__all__ = [
    "BoussinesqSelfSimilar",
    "boussinesq_selfsimilar_residual_samples",
    "infer_lambda_from_streamfunction_u1_y1",
]
