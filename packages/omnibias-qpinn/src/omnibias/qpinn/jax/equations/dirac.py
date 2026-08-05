# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Dirac equation residual (jax twin)."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

import jax.numpy as jnp
from jax import Array
from omnibias.pinn._core.state import FieldState
from omnibias.qpinn._core.spinor import gamma_partial_psi, spinor_value


@dataclass
class DiracOutput:
    residual: Array
    diag: dict[str, float]


@dataclass
class Dirac:
    """JAX twin of :class:`omnibias.qpinn.torch.equations.Dirac`."""

    mass: float = 1.0
    representation: Literal["dirac", "weyl"] = "dirac"
    spinor: str = "spinor"
    source: Callable[[FieldState], Array] | None = None

    def __call__(self, state: FieldState) -> DiracOutput:
        gp = gamma_partial_psi(
            state, spinor_group=self.spinor, representation=self.representation,
        )
        cols: list[Array] = []
        for a in range(4):
            psi_re_a, psi_im_a = spinor_value(state, self.spinor, a)
            gp_re_a, gp_im_a = gp[a]
            r_re = -gp_im_a - self.mass * psi_re_a
            r_im = gp_re_a - self.mass * psi_im_a
            cols.append(r_re)
            cols.append(r_im)
        residual = jnp.stack(cols, axis=-1)
        if self.source is not None:
            residual = residual - self.source(state)
        return DiracOutput(
            residual=residual,
            diag={"mean_sq_residual": jnp.mean(residual * residual)},
        )


def dirac(
    state: FieldState,
    *,
    mass: float = 1.0,
    representation: Literal["dirac", "weyl"] = "dirac",
    spinor: str = "spinor",
    source: Callable[[FieldState], Array] | None = None,
) -> DiracOutput:
    """Stateless one-shot wrapper around :class:`Dirac`."""
    return Dirac(
        mass=mass, representation=representation, spinor=spinor, source=source,
    )(state)


__all__ = ["Dirac", "DiracOutput", "dirac"]
