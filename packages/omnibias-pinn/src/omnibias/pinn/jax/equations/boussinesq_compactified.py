# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""2-D Boussinesq compactification and envelopes (jax).

Wang et al. (arXiv:2509.14185) eqs. 5–6 / LucasAschenbach chart:

* ``q = (1+|y|^2)^{-1/(2(1+λ))}``
* ``β = y2 / sqrt(1+|y|^2)``
* ``Ω = odd_y1 · hΩ(q,β) · q``
* ``Θ = hΘ(q,β) · q^{θ_pow}``
* ``Ψ = hΨ(q,β) · q^{ψ_pow}``

Envelope and chart derivatives are closed-form product / chain rule.
Hats enter as a :class:`HatJet` on ``(q, β)`` (affine, JetMLP, or OMBU).
This is not a DeepMind residual claim and not Navier–Stokes.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp
from jax import Array


@dataclass(frozen=True)
class HatJet:
    """Value and ``(q, β)`` jet of a hat (up to second order)."""

    value: Array
    d_q: Array
    d_beta: Array
    d_qq: Array
    d_q_beta: Array
    d_beta_beta: Array


def compactify_yb_lambda(
    y1: Array,
    y2: Array,
    lam: Array | float,
) -> tuple[Array, Array, Array]:
    """Return ``(q, β, r2)`` for paper eq. 5."""
    y1 = jnp.asarray(y1, dtype=jnp.float64)
    y2 = jnp.asarray(y2, dtype=jnp.float64)
    alpha = 1.0 / (1.0 + jnp.asarray(lam, dtype=jnp.float64))
    r2 = 1.0 + y1 * y1 + y2 * y2
    q = jnp.power(r2, -0.5 * alpha)
    beta = y2 / jnp.sqrt(r2)
    return q, beta, r2


def odd_y1_prefactor(y1: Array, r2: Array) -> Array:
    """Odd-in-``y1`` symmetry factor ``y1 / sqrt(1+|y|^2)``."""
    return jnp.asarray(y1, dtype=jnp.float64) / jnp.sqrt(jnp.asarray(r2, dtype=jnp.float64))


def affine_hat_jet(
    q: Array,
    beta: Array,
    c0: Array | float,
    cq: Array | float = 0.0,
    cb: Array | float = 0.0,
) -> HatJet:
    """Linear hat ``c0 + cq q + cb β`` (second derivatives vanish)."""
    q = jnp.asarray(q, dtype=jnp.float64)
    beta = jnp.asarray(beta, dtype=jnp.float64)
    ones = jnp.ones_like(q)
    zeros = jnp.zeros_like(q)
    c0_a = jnp.asarray(c0, dtype=jnp.float64)
    cq_a = jnp.asarray(cq, dtype=jnp.float64)
    cb_a = jnp.asarray(cb, dtype=jnp.float64)
    return HatJet(
        value=c0_a * ones + cq_a * q + cb_a * beta,
        d_q=cq_a * ones,
        d_beta=cb_a * ones,
        d_qq=zeros,
        d_q_beta=zeros,
        d_beta_beta=zeros,
    )


def chart_jets(y1: Array, y2: Array, lam: Array | float) -> dict[str, Array]:
    """Closed-form first and second chart derivatives."""
    y1 = jnp.asarray(y1, dtype=jnp.float64)
    y2 = jnp.asarray(y2, dtype=jnp.float64)
    alpha = 1.0 / (1.0 + jnp.asarray(lam, dtype=jnp.float64))
    r2 = 1.0 + y1 * y1 + y2 * y2
    r = jnp.sqrt(r2)
    q = jnp.power(r2, -0.5 * alpha)
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


def _hat_physical(hat: HatJet, jets: dict[str, Array]) -> tuple[Array, Array, Array, Array, Array]:
    """Chain-rule ``h`` and first / unmixed second physical derivatives."""
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
    q: Array,
    jets: dict[str, Array],
    power: float,
) -> tuple[Array, Array, Array, Array, Array]:
    """``E = q^p`` and physical derivatives (``p=0`` is identically 1)."""
    p = float(power)
    ones = jnp.ones_like(q)
    zeros = jnp.zeros_like(q)
    if abs(p) < 1e-15:
        return ones, zeros, zeros, zeros, zeros
    e = jnp.power(q, p)
    e_y1 = p * jnp.power(q, p - 1.0) * jets["q_y1"]
    e_y2 = p * jnp.power(q, p - 1.0) * jets["q_y2"]
    e_y1y1 = (
        p * (p - 1.0) * jnp.power(q, p - 2.0) * jets["q_y1"] * jets["q_y1"]
        + p * jnp.power(q, p - 1.0) * jets["q_y1y1"]
    )
    e_y2y2 = (
        p * (p - 1.0) * jnp.power(q, p - 2.0) * jets["q_y2"] * jets["q_y2"]
        + p * jnp.power(q, p - 1.0) * jets["q_y2y2"]
    )
    return e, e_y1, e_y2, e_y1y1, e_y2y2


def lift_even_envelope(
    hat: HatJet,
    jets: dict[str, Array],
    *,
    power: float = 0.0,
) -> tuple[Array, Array, Array, Array]:
    """``field = h · q^p`` with first derivatives and Laplacian."""
    h, h_y1, h_y2, h_y1y1, h_y2y2 = _hat_physical(hat, jets)
    e, e_y1, e_y2, e_y1y1, e_y2y2 = _power_envelope(jets["q"], jets, power)
    field = e * h
    field_y1 = e_y1 * h + e * h_y1
    field_y2 = e_y2 * h + e * h_y2
    field_y1y1 = e_y1y1 * h + 2.0 * e_y1 * h_y1 + e * h_y1y1
    field_y2y2 = e_y2y2 * h + 2.0 * e_y2 * h_y2 + e * h_y2y2
    return field, field_y1, field_y2, field_y1y1 + field_y2y2


def lift_omega_envelope(hat: HatJet, jets: dict[str, Array]) -> tuple[Array, Array, Array]:
    """``Ω = odd · q · hΩ`` with first physical derivatives."""
    h, h_y1, h_y2, _h11, _h22 = _hat_physical(hat, jets)
    env = jets["odd"] * jets["q"]
    env_y1 = jets["odd_y1"] * jets["q"] + jets["odd"] * jets["q_y1"]
    env_y2 = jets["odd_y2"] * jets["q"] + jets["odd"] * jets["q_y2"]
    omega = env * h
    return omega, env_y1 * h + env * h_y1, env_y2 * h + env * h_y2


def compose_boussinesq_envelope_fields(
    y1: Array,
    y2: Array,
    lam: Array | float,
    *,
    hat_omega: HatJet,
    hat_theta: HatJet,
    hat_psi: HatJet,
    theta_decay_power: float = 0.0,
    psi_decay_power: float = 0.0,
) -> dict[str, Array]:
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
