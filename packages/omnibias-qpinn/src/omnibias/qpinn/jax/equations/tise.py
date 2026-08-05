# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Time-independent Schrodinger equation residual (jax twin)."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import jax.numpy as jnp
from jax import Array
from omnibias.pinn._core.state import FieldState
from omnibias.qpinn._core.complex import apply_hamiltonian, psi_value
from omnibias.qpinn.jax.equations._types import TISEOutput


@dataclass
class TISE:
    """JAX twin of :class:`omnibias.qpinn.torch.equations.TISE`."""

    energy: float | Array = 0.0
    hbar: float = 1.0
    mass: float = 1.0
    psi: str = "psi"
    potential: Callable[[FieldState], Array] | None = None
    quadrature_weights: Array | None = None

    def __call__(self, state: FieldState) -> TISEOutput:
        psi_re, psi_im = psi_value(state, self.psi)
        H_re, H_im = apply_hamiltonian(
            state, group=self.psi,
            hbar=self.hbar, mass=self.mass, potential=self.potential,
        )
        E = jnp.asarray(self.energy, dtype=psi_re.dtype)
        res_re = H_re - E * psi_re
        res_im = H_im - E * psi_im
        residual = jnp.stack([res_re, res_im], axis=-1)

        energy_estimate: Array | None = None
        diag: dict[str, float] = {
            "mean_sq_residual": jnp.mean(residual * residual),
        }
        if self.quadrature_weights is not None:
            w = self.quadrature_weights
            if w.shape != psi_re.shape:
                raise ValueError(
                    f"quadrature_weights shape {tuple(w.shape)} != "
                    f"psi shape {tuple(psi_re.shape)}"
                )
            density = psi_re * psi_re + psi_im * psi_im
            num = jnp.sum(w * (psi_re * H_re + psi_im * H_im))
            den = jnp.sum(w * density)
            energy_estimate = num / (den + jnp.finfo(psi_re.dtype).tiny)
            diag["energy_estimate"] = float(energy_estimate)
            diag["norm_squared"] = float(den)
        return TISEOutput(
            residual=residual,
            energy_estimate=energy_estimate,
            diag=diag,
        )


def tise(
    state: FieldState,
    *,
    energy: float | Array = 0.0,
    hbar: float = 1.0,
    mass: float = 1.0,
    psi: str = "psi",
    potential: Callable[[FieldState], Array] | None = None,
    quadrature_weights: Array | None = None,
) -> TISEOutput:
    """Stateless one-shot wrapper around :class:`TISE`."""
    return TISE(
        energy=energy, hbar=hbar, mass=mass, psi=psi,
        potential=potential, quadrature_weights=quadrature_weights,
    )(state)


__all__ = ["TISE", "tise"]
