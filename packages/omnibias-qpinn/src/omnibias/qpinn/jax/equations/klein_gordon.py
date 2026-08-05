# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Klein-Gordon equation residual (jax twin)."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import jax.numpy as jnp
from jax import Array
from omnibias.pinn._core.state import FieldState


@dataclass
class KleinGordonOutput:
    residual: Array
    diag: dict[str, float]


@dataclass
class KleinGordon:
    """JAX twin of :class:`omnibias.qpinn.torch.equations.KleinGordon`."""

    mass: float = 1.0
    lambda_phi4: float = 0.0
    component: str = "phi"
    source: Callable[[FieldState], Array] | None = None

    def __call__(self, state: FieldState) -> KleinGordonOutput:
        time = state.coordinate_spec.time_axis
        if time is None:
            raise ValueError(
                "Klein-Gordon equation requires a time axis (Box = -d_tt + lap)"
            )
        phi = state.ops.value(state, self.component)
        # Box = -d_tt + lap = d'Alembertian with c=1 in the mostly-plus signature.
        box = state.ops.dalembertian(state, self.component, c=1.0)
        m_sq = self.mass * self.mass
        residual = box - m_sq * phi - self.lambda_phi4 * phi * phi * phi
        if self.source is not None:
            residual = residual - self.source(state)
        return KleinGordonOutput(
            residual=residual,
            diag={"mean_sq_residual": jnp.mean(residual * residual)},
        )


def klein_gordon(
    state: FieldState,
    *,
    mass: float = 1.0,
    lambda_phi4: float = 0.0,
    component: str = "phi",
    source: Callable[[FieldState], Array] | None = None,
) -> KleinGordonOutput:
    """Stateless one-shot wrapper around :class:`KleinGordon`."""
    return KleinGordon(
        mass=mass, lambda_phi4=lambda_phi4, component=component, source=source,
    )(state)


__all__ = ["KleinGordon", "KleinGordonOutput", "klein_gordon"]
