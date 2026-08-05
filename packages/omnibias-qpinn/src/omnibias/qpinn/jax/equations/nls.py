# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Nonlinear Schrodinger / Gross-Pitaevskii residual (jax twin)."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import jax.numpy as jnp
from jax import Array
from omnibias.pinn._core.state import FieldState
from omnibias.qpinn._core.complex import apply_hamiltonian, psi_density, psi_value
from omnibias.qpinn.jax.equations._types import NLSOutput


@dataclass
class NLS:
    """JAX twin of :class:`omnibias.qpinn.torch.equations.NLS`."""

    g: float = 1.0
    hbar: float = 1.0
    mass: float = 1.0
    psi: str = "psi"
    potential: Callable[[FieldState], Array] | None = None
    source: Callable[[FieldState], Array] | None = None

    def __call__(self, state: FieldState) -> NLSOutput:
        time = state.coordinate_spec.time_axis
        if time is None:
            raise ValueError(
                "NLS residual requires a time axis in the coordinate spec"
            )
        re_name = f"{self.psi}_re"
        im_name = f"{self.psi}_im"
        if not state.components.is_component(re_name):
            raise KeyError(
                f"component {re_name!r} not found; build the field with "
                "omnibias.qpinn.make_psi_components"
            )

        psi_re, psi_im = psi_value(state, self.psi)
        psi_re_t = state.ops.derivative(state, re_name, axis=time, order=1)
        psi_im_t = state.ops.derivative(state, im_name, axis=time, order=1)

        H_re, H_im = apply_hamiltonian(
            state, group=self.psi,
            hbar=self.hbar, mass=self.mass, potential=self.potential,
        )
        density = psi_density(state, self.psi)
        nl_re = self.g * density * psi_re
        nl_im = self.g * density * psi_im

        res_re = -self.hbar * psi_im_t - (H_re + nl_re)
        res_im = self.hbar * psi_re_t - (H_im + nl_im)
        residual = jnp.stack([res_re, res_im], axis=-1)
        if self.source is not None:
            residual = residual - self.source(state)

        diag = {
            "mean_sq_residual": jnp.mean(residual * residual),
            "mean_density": jnp.mean(density),
            "nonlinear_energy": jnp.mean(self.g * density * density / 2),
        }
        return NLSOutput(residual=residual, diag=diag)


def nls(
    state: FieldState,
    *,
    g: float = 1.0,
    hbar: float = 1.0,
    mass: float = 1.0,
    psi: str = "psi",
    potential: Callable[[FieldState], Array] | None = None,
    source: Callable[[FieldState], Array] | None = None,
) -> NLSOutput:
    """Stateless one-shot wrapper around :class:`NLS`."""
    return NLS(
        g=g, hbar=hbar, mass=mass, psi=psi,
        potential=potential, source=source,
    )(state)


__all__ = ["NLS", "nls"]
