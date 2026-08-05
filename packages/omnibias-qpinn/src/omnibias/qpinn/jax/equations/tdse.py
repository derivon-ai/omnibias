# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Time-dependent Schrodinger equation residual (jax twin)."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import jax.numpy as jnp
from jax import Array
from omnibias.pinn._core.state import FieldState
from omnibias.qpinn._core.complex import apply_hamiltonian
from omnibias.qpinn.jax.equations._types import TDSEOutput


@dataclass
class TDSE:
    """JAX twin of :class:`omnibias.qpinn.torch.equations.TDSE`."""

    hbar: float = 1.0
    mass: float = 1.0
    psi: str = "psi"
    potential: Callable[[FieldState], Array] | None = None
    source: Callable[[FieldState], Array] | None = None

    def __call__(self, state: FieldState) -> TDSEOutput:
        time = state.coordinate_spec.time_axis
        if time is None:
            raise ValueError(
                "TDSE residual requires a time axis in the coordinate spec"
            )
        re_name = f"{self.psi}_re"
        im_name = f"{self.psi}_im"
        if not state.components.is_component(re_name):
            raise KeyError(
                f"component {re_name!r} not found; build the field with "
                "omnibias.qpinn.make_psi_components"
            )
        psi_re_t = state.ops.derivative(state, re_name, axis=time, order=1)
        psi_im_t = state.ops.derivative(state, im_name, axis=time, order=1)
        H_re, H_im = apply_hamiltonian(
            state, group=self.psi,
            hbar=self.hbar, mass=self.mass, potential=self.potential,
        )
        res_re = -self.hbar * psi_im_t - H_re
        res_im = self.hbar * psi_re_t - H_im
        residual = jnp.stack([res_re, res_im], axis=-1)
        if self.source is not None:
            residual = residual - self.source(state)
        return TDSEOutput(
            residual=residual,
            diag={"mean_sq_residual": jnp.mean(residual * residual)},
        )


def tdse(
    state: FieldState,
    *,
    hbar: float = 1.0,
    mass: float = 1.0,
    psi: str = "psi",
    potential: Callable[[FieldState], Array] | None = None,
    source: Callable[[FieldState], Array] | None = None,
) -> TDSEOutput:
    """Stateless one-shot wrapper around :class:`TDSE`."""
    return TDSE(
        hbar=hbar, mass=mass, psi=psi,
        potential=potential, source=source,
    )(state)


__all__ = ["TDSE", "tdse"]
