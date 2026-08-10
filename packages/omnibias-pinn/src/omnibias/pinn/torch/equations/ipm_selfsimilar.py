# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Incompressible porous media (IPM) self-similar residual (torch twin)."""

from __future__ import annotations

from dataclasses import dataclass

from torch import Tensor


def ipm_selfsimilar_residual_samples(
    y1: Tensor,
    y2: Tensor,
    theta: Tensor,
    theta_y1: Tensor,
    theta_y2: Tensor,
    psi: Tensor,
    psi_y1: Tensor,
    psi_y2: Tensor,
    psi_lap: Tensor,
    lam: Tensor | float,
) -> tuple[Tensor, Tensor]:
    omega = -theta_y1
    u1 = psi_y2
    u2 = -psi_y1
    adv = u1 * theta_y1 + u2 * theta_y2
    r_theta = (1.0 + lam) * (y1 * theta_y1 + y2 * theta_y2) - lam * theta + adv
    r_psi = psi_lap - omega
    return r_theta, r_psi


@dataclass
class IPMSelfSimilar:
    lam: float = 0.5

    def residual(
        self,
        y1: Tensor,
        y2: Tensor,
        theta: Tensor,
        theta_y1: Tensor,
        theta_y2: Tensor,
        psi: Tensor,
        psi_y1: Tensor,
        psi_y2: Tensor,
        psi_lap: Tensor,
    ) -> tuple[Tensor, Tensor]:
        return ipm_selfsimilar_residual_samples(
            y1, y2, theta, theta_y1, theta_y2, psi, psi_y1, psi_y2, psi_lap, self.lam
        )


__all__ = ["IPMSelfSimilar", "ipm_selfsimilar_residual_samples"]
