# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Incompressible porous media (IPM) self-similar residual (jax).

Streamfunction-as-unknown formulation (paper-faithful nonlocality):

* vorticity from density: ``Omega = -d_{y1} Theta``
* Poisson: ``Delta Psi = Omega``
* velocity: ``U = (d_{y2} Psi, -d_{y1} Psi)``
* density transport residual with self-similar linear terms

This replaces the incorrect local curl proxy that treated ``Theta`` as its own
streamfunction. Discovery-stage only; not a Navier-Stokes claim.
"""

from __future__ import annotations

from dataclasses import dataclass

from jax import Array


def ipm_selfsimilar_residual_samples(
    y1: Array,
    y2: Array,
    theta: Array,
    theta_y1: Array,
    theta_y2: Array,
    psi: Array,
    psi_y1: Array,
    psi_y2: Array,
    psi_lap: Array,
    lam: Array | float,
) -> tuple[Array, Array]:
    """Return ``(R_theta, R_psi)`` with ``R_psi = Delta Psi + d_y1 Theta``."""
    omega = -theta_y1
    u1 = psi_y2
    u2 = -psi_y1
    adv = u1 * theta_y1 + u2 * theta_y2
    # IPM self-similar density residual (powers of (1-t) cancelled)
    r_theta = (1.0 + lam) * (y1 * theta_y1 + y2 * theta_y2) - lam * theta + adv
    r_psi = psi_lap - omega
    return r_theta, r_psi


@dataclass
class IPMSelfSimilar:
    lam: float = 0.5

    def residual(
        self,
        y1: Array,
        y2: Array,
        theta: Array,
        theta_y1: Array,
        theta_y2: Array,
        psi: Array,
        psi_y1: Array,
        psi_y2: Array,
        psi_lap: Array,
    ) -> tuple[Array, Array]:
        return ipm_selfsimilar_residual_samples(
            y1, y2, theta, theta_y1, theta_y2, psi, psi_y1, psi_y2, psi_lap, self.lam
        )


__all__ = ["IPMSelfSimilar", "ipm_selfsimilar_residual_samples"]
