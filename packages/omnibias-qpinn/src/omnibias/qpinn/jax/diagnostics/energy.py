# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Energy diagnostics for the jax backend."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

import jax.numpy as jnp
from jax import Array
from omnibias.qpinn._core.complex import apply_hamiltonian, psi_value

if TYPE_CHECKING:  # pragma: no cover
    from omnibias.pinn._core.state import FieldState


def expectation_value(
    state: FieldState,
    *,
    operator_action: Callable[[FieldState], tuple[Array, Array]],
    group: str = "psi",
    quadrature_weights: Array | None = None,
) -> Array:
    r"""JAX twin of :func:`omnibias.qpinn.torch.diagnostics.expectation_value`."""
    psi_re, psi_im = psi_value(state, group)
    O_re, O_im = operator_action(state)
    integrand = psi_re * O_re + psi_im * O_im
    density = psi_re * psi_re + psi_im * psi_im
    if quadrature_weights is None:
        num = jnp.mean(integrand)
        den = jnp.mean(density)
    else:
        if quadrature_weights.shape != integrand.shape:
            raise ValueError(
                f"quadrature_weights shape {tuple(quadrature_weights.shape)} "
                f"!= integrand shape {tuple(integrand.shape)}"
            )
        num = jnp.sum(quadrature_weights * integrand)
        den = jnp.sum(quadrature_weights * density)
    return num / (den + jnp.finfo(num.dtype).tiny)


def expected_energy(
    state: FieldState,
    *,
    hbar: float = 1.0,
    mass: float = 1.0,
    potential: Callable[[FieldState], Array] | None = None,
    group: str = "psi",
    quadrature_weights: Array | None = None,
) -> Array:
    def _Hpsi(s: FieldState) -> tuple[Array, Array]:
        return apply_hamiltonian(
            s, group=group, hbar=hbar, mass=mass, potential=potential,
        )
    return expectation_value(
        state, operator_action=_Hpsi, group=group,
        quadrature_weights=quadrature_weights,
    )


def energy_variance(
    state: FieldState,
    *,
    hbar: float = 1.0,
    mass: float = 1.0,
    potential: Callable[[FieldState], Array] | None = None,
    group: str = "psi",
    quadrature_weights: Array | None = None,
) -> Array:
    psi_re, psi_im = psi_value(state, group)
    H_re, H_im = apply_hamiltonian(
        state, group=group, hbar=hbar, mass=mass, potential=potential,
    )
    density = psi_re * psi_re + psi_im * psi_im
    H_dot_psi = psi_re * H_re + psi_im * H_im
    if quadrature_weights is None:
        num = jnp.mean(H_dot_psi)
        den = jnp.mean(density)
        h2_num = jnp.mean(H_re * H_re + H_im * H_im)
    else:
        num = jnp.sum(quadrature_weights * H_dot_psi)
        den = jnp.sum(quadrature_weights * density)
        h2_num = jnp.sum(quadrature_weights * (H_re * H_re + H_im * H_im))
    eps = jnp.finfo(num.dtype).tiny
    E = num / (den + eps)
    Hsq = h2_num / (den + eps)
    return Hsq - E * E


__all__ = [
    "energy_variance",
    "expectation_value",
    "expected_energy",
]
