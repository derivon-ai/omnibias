# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""2-D Boussinesq self-similar residual (torch twin)."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


def infer_lambda_from_streamfunction_u1_y1(u1_y1_at_origin: Tensor | float) -> Tensor:
    return -3.0 - 2.0 * torch.as_tensor(u1_y1_at_origin)


def boussinesq_selfsimilar_residual_samples(
    y1: Tensor,
    y2: Tensor,
    omega: Tensor,
    omega_y1: Tensor,
    omega_y2: Tensor,
    theta: Tensor,
    theta_y1: Tensor,
    theta_y2: Tensor,
    psi: Tensor,
    psi_y1: Tensor,
    psi_y2: Tensor,
    psi_lap: Tensor,
    lam: Tensor | float,
) -> tuple[Tensor, Tensor, Tensor]:
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
        y1: Tensor,
        y2: Tensor,
        omega: Tensor,
        omega_y1: Tensor,
        omega_y2: Tensor,
        theta: Tensor,
        theta_y1: Tensor,
        theta_y2: Tensor,
        psi: Tensor,
        psi_y1: Tensor,
        psi_y2: Tensor,
        psi_lap: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
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
