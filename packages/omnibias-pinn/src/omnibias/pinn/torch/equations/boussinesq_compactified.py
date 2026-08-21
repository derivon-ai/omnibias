# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""2-D Boussinesq compactification and envelopes (torch twin)."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True)
class HatJet:
    """Value and ``(q, β)`` jet of a hat (up to second order)."""

    value: Tensor
    d_q: Tensor
    d_beta: Tensor
    d_qq: Tensor
    d_q_beta: Tensor
    d_beta_beta: Tensor


def compactify_yb_lambda(
    y1: Tensor,
    y2: Tensor,
    lam: Tensor | float,
) -> tuple[Tensor, Tensor, Tensor]:
    """Return ``(q, β, r2)`` for paper eq. 5."""
    y1 = torch.as_tensor(y1, dtype=torch.float64)
    y2 = torch.as_tensor(y2, dtype=torch.float64)
    alpha = 1.0 / (1.0 + torch.as_tensor(lam, dtype=torch.float64))
    r2 = 1.0 + y1 * y1 + y2 * y2
    q = torch.pow(r2, -0.5 * alpha)
    beta = y2 / torch.sqrt(r2)
    return q, beta, r2


def odd_y1_prefactor(y1: Tensor, r2: Tensor) -> Tensor:
    """Odd-in-``y1`` symmetry factor ``y1 / sqrt(1+|y|^2)``."""
    return torch.as_tensor(y1, dtype=torch.float64) / torch.sqrt(
        torch.as_tensor(r2, dtype=torch.float64)
    )


def affine_hat_jet(
    q: Tensor,
    beta: Tensor,
    c0: Tensor | float,
    cq: Tensor | float = 0.0,
    cb: Tensor | float = 0.0,
) -> HatJet:
    """Linear hat ``c0 + cq q + cb β`` (second derivatives vanish)."""
    q = torch.as_tensor(q, dtype=torch.float64)
    beta = torch.as_tensor(beta, dtype=torch.float64)
    ones = torch.ones_like(q)
    zeros = torch.zeros_like(q)
    c0_a = torch.as_tensor(c0, dtype=torch.float64)
    cq_a = torch.as_tensor(cq, dtype=torch.float64)
    cb_a = torch.as_tensor(cb, dtype=torch.float64)
    return HatJet(
        value=c0_a * ones + cq_a * q + cb_a * beta,
        d_q=cq_a * ones,
        d_beta=cb_a * ones,
        d_qq=zeros,
        d_q_beta=zeros,
        d_beta_beta=zeros,
    )


def chart_jets(y1: Tensor, y2: Tensor, lam: Tensor | float) -> dict[str, Tensor]:
    """Closed-form first and second chart derivatives."""
    y1 = torch.as_tensor(y1, dtype=torch.float64)
    y2 = torch.as_tensor(y2, dtype=torch.float64)
    alpha = 1.0 / (1.0 + torch.as_tensor(lam, dtype=torch.float64))
    r2 = 1.0 + y1 * y1 + y2 * y2
    r = torch.sqrt(r2)
    q = torch.pow(r2, -0.5 * alpha)
    beta = y2 / r
    odd = y1 / r
    r3 = r * r2
    r5 = r3 * r2
    r2sq = r2 * r2
    q_y1 = -alpha * y1 * q / r2
    q_y2 = -alpha * y2 * q / r2
    q_y1y1 = -alpha * q / r2 + alpha * (alpha + 2.0) * y1 * y1 * q / r2sq
    q_y2y2 = -alpha * q / r2 + alpha * (alpha + 2.0) * y2 * y2 * q / r2sq
    q_y1y2 = alpha * (alpha + 2.0) * y1 * y2 * q / r2sq
    return {
        "q": q,
        "beta": beta,
        "r2": r2,
        "odd": odd,
        "q_y1": q_y1,
        "q_y2": q_y2,
        "q_y1y1": q_y1y1,
        "q_y2y2": q_y2y2,
        "q_y1y2": q_y1y2,
        "beta_y1": -y1 * y2 / r3,
        "beta_y2": (1.0 + y1 * y1) / r3,
        "beta_y1y1": -y2 / r3 + 3.0 * y1 * y1 * y2 / r5,
        "beta_y1y2": -y1 / r3 + 3.0 * y1 * y2 * y2 / r5,
        "beta_y2y2": -3.0 * y2 * (1.0 + y1 * y1) / r5,
        "odd_y1": (1.0 + y2 * y2) / r3,
        "odd_y2": -y1 * y2 / r3,
    }


def _hat_physical(
    hat: HatJet, jets: dict[str, Tensor]
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    h = hat.value
    h_y1 = hat.d_q * jets["q_y1"] + hat.d_beta * jets["beta_y1"]
    h_y2 = hat.d_q * jets["q_y2"] + hat.d_beta * jets["beta_y2"]
    h_y1y1 = (
        hat.d_qq * jets["q_y1"] * jets["q_y1"]
        + 2.0 * hat.d_q_beta * jets["q_y1"] * jets["beta_y1"]
        + hat.d_beta_beta * jets["beta_y1"] * jets["beta_y1"]
        + hat.d_q * jets["q_y1y1"]
        + hat.d_beta * jets["beta_y1y1"]
    )
    h_y2y2 = (
        hat.d_qq * jets["q_y2"] * jets["q_y2"]
        + 2.0 * hat.d_q_beta * jets["q_y2"] * jets["beta_y2"]
        + hat.d_beta_beta * jets["beta_y2"] * jets["beta_y2"]
        + hat.d_q * jets["q_y2y2"]
        + hat.d_beta * jets["beta_y2y2"]
    )
    return h, h_y1, h_y2, h_y1y1, h_y2y2


def _power_envelope(
    q: Tensor,
    jets: dict[str, Tensor],
    power: float,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    p = float(power)
    ones = torch.ones_like(q)
    zeros = torch.zeros_like(q)
    if abs(p) < 1e-15:
        return ones, zeros, zeros, zeros, zeros
    e = torch.pow(q, p)
    e_y1 = p * torch.pow(q, p - 1.0) * jets["q_y1"]
    e_y2 = p * torch.pow(q, p - 1.0) * jets["q_y2"]
    e_y1y1 = (
        p * (p - 1.0) * torch.pow(q, p - 2.0) * jets["q_y1"] * jets["q_y1"]
        + p * torch.pow(q, p - 1.0) * jets["q_y1y1"]
    )
    e_y2y2 = (
        p * (p - 1.0) * torch.pow(q, p - 2.0) * jets["q_y2"] * jets["q_y2"]
        + p * torch.pow(q, p - 1.0) * jets["q_y2y2"]
    )
    return e, e_y1, e_y2, e_y1y1, e_y2y2


def lift_even_envelope(
    hat: HatJet,
    jets: dict[str, Tensor],
    *,
    power: float = 0.0,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """``field = h · q^p`` with first derivatives and Laplacian."""
    h, h_y1, h_y2, h_y1y1, h_y2y2 = _hat_physical(hat, jets)
    e, e_y1, e_y2, e_y1y1, e_y2y2 = _power_envelope(jets["q"], jets, power)
    field = e * h
    field_y1 = e_y1 * h + e * h_y1
    field_y2 = e_y2 * h + e * h_y2
    field_y1y1 = e_y1y1 * h + 2.0 * e_y1 * h_y1 + e * h_y1y1
    field_y2y2 = e_y2y2 * h + 2.0 * e_y2 * h_y2 + e * h_y2y2
    return field, field_y1, field_y2, field_y1y1 + field_y2y2


def lift_omega_envelope(hat: HatJet, jets: dict[str, Tensor]) -> tuple[Tensor, Tensor, Tensor]:
    """``Ω = odd · q · hΩ`` with first physical derivatives."""
    h, h_y1, h_y2, _h11, _h22 = _hat_physical(hat, jets)
    env = jets["odd"] * jets["q"]
    env_y1 = jets["odd_y1"] * jets["q"] + jets["odd"] * jets["q_y1"]
    env_y2 = jets["odd_y2"] * jets["q"] + jets["odd"] * jets["q_y2"]
    omega = env * h
    return omega, env_y1 * h + env * h_y1, env_y2 * h + env * h_y2


def compose_boussinesq_envelope_fields(
    y1: Tensor,
    y2: Tensor,
    lam: Tensor | float,
    *,
    hat_omega: HatJet,
    hat_theta: HatJet,
    hat_psi: HatJet,
    theta_decay_power: float = 0.0,
    psi_decay_power: float = 0.0,
) -> dict[str, Tensor]:
    """Lift hats through the paper envelopes; residual-ready fields."""
    jets = chart_jets(y1, y2, lam)
    omega, omega_y1, omega_y2 = lift_omega_envelope(hat_omega, jets)
    theta, theta_y1, theta_y2, _ = lift_even_envelope(
        hat_theta, jets, power=theta_decay_power
    )
    psi, psi_y1, psi_y2, psi_lap = lift_even_envelope(
        hat_psi, jets, power=psi_decay_power
    )
    return {
        "q": jets["q"],
        "beta": jets["beta"],
        "omega": omega,
        "omega_y1": omega_y1,
        "omega_y2": omega_y2,
        "theta": theta,
        "theta_y1": theta_y1,
        "theta_y2": theta_y2,
        "psi": psi,
        "psi_y1": psi_y1,
        "psi_y2": psi_y2,
        "psi_lap": psi_lap,
    }


__all__ = [
    "HatJet",
    "affine_hat_jet",
    "chart_jets",
    "compactify_yb_lambda",
    "compose_boussinesq_envelope_fields",
    "lift_even_envelope",
    "lift_omega_envelope",
    "odd_y1_prefactor",
]
