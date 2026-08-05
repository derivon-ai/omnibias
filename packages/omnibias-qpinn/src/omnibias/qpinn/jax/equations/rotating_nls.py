# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Stationary 2D rotating-frame Gross-Pitaevskii residual (jax twin).

Bit-parity twin of :class:`omnibias.qpinn.torch.equations.RotatingNLS`.
See that module for the equation derivation; this file only swaps
``torch`` for ``jax.numpy``.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import jax.numpy as jnp
from jax import Array
from omnibias.pinn._core.state import FieldState
from omnibias.qpinn._core.complex import (
    apply_angular_momentum_z,
    apply_hamiltonian,
    psi_density,
    psi_value,
)
from omnibias.qpinn.jax.equations._types import RotatingNLSOutput


@dataclass
class RotatingNLS:
    """JAX twin of :class:`omnibias.qpinn.torch.equations.RotatingNLS`."""

    g: float = 1.0
    omega_rot: float = 0.0
    mu: float | Array = 0.0
    hbar: float = 1.0
    mass: float = 1.0
    psi: str = "psi"
    x_axis: int = 0
    y_axis: int = 1
    potential: Callable[[FieldState], Array] | None = None
    source: Callable[[FieldState], Array] | None = None

    def __call__(self, state: FieldState) -> RotatingNLSOutput:
        if state.coordinate_spec.ndim < 2:
            raise ValueError(
                "RotatingNLS requires at least 2 spatial axes; "
                f"got coordinate_spec.ndim = {state.coordinate_spec.ndim}"
            )
        if self.x_axis == self.y_axis:
            raise ValueError(
                f"x_axis ({self.x_axis}) and y_axis ({self.y_axis}) must differ"
            )
        re_name = f"{self.psi}_re"
        if not state.components.is_component(re_name):
            raise KeyError(
                f"component {re_name!r} not found; build the field with "
                "omnibias.qpinn.make_psi_components"
            )

        psi_re, psi_im = psi_value(state, self.psi)
        H_re, H_im = apply_hamiltonian(
            state, group=self.psi,
            hbar=self.hbar, mass=self.mass, potential=self.potential,
        )
        density = psi_density(state, self.psi)
        mu = self.mu
        if not isinstance(mu, Array):
            mu = jnp.asarray(mu, dtype=psi_re.dtype)
        nl_re = self.g * density * psi_re
        nl_im = self.g * density * psi_im
        Lz_psi_re, Lz_psi_im = apply_angular_momentum_z(
            state, group=self.psi, hbar=1.0,
            x_axis=self.x_axis, y_axis=self.y_axis,
        )

        res_re = H_re + nl_re - self.omega_rot * Lz_psi_re - mu * psi_re
        res_im = H_im + nl_im - self.omega_rot * Lz_psi_im - mu * psi_im
        residual = jnp.stack([res_re, res_im], axis=-1)
        if self.source is not None:
            residual = residual - self.source(state)

        rotation_energy_density = self.omega_rot * (
            psi_re * Lz_psi_im - psi_im * Lz_psi_re
        )
        diag = {
            "mean_sq_residual": jnp.mean(residual * residual),
            "mean_density": jnp.mean(density),
            "mean_rotation_energy_density": jnp.mean(rotation_energy_density),
            "nonlinear_energy": jnp.mean(self.g * density * density / 2),
        }
        return RotatingNLSOutput(residual=residual, density=density, diag=diag)


def rotating_nls(
    state: FieldState,
    *,
    g: float = 1.0,
    omega_rot: float = 0.0,
    mu: float | Array = 0.0,
    hbar: float = 1.0,
    mass: float = 1.0,
    psi: str = "psi",
    x_axis: int = 0,
    y_axis: int = 1,
    potential: Callable[[FieldState], Array] | None = None,
    source: Callable[[FieldState], Array] | None = None,
) -> RotatingNLSOutput:
    """Stateless one-shot wrapper around :class:`RotatingNLS`."""
    return RotatingNLS(
        g=g, omega_rot=omega_rot, mu=mu, hbar=hbar, mass=mass, psi=psi,
        x_axis=x_axis, y_axis=y_axis, potential=potential, source=source,
    )(state)


__all__ = ["RotatingNLS", "rotating_nls"]
