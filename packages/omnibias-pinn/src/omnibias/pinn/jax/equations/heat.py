# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Heat equation residual (jax twin)."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import jax.numpy as jnp
from jax import Array
from omnibias.pinn._core.state import FieldState
from omnibias.pinn.jax.equations._types import HeatOutput


@dataclass
class Heat:
    alpha: float = 1.0
    component: str = "u"
    source: Callable[[FieldState], Array] | None = None

    def __call__(self, state: FieldState) -> HeatOutput:
        time = state.coordinate_spec.time_axis
        if time is None:
            raise ValueError("Heat equation requires a time axis")
        u_t = state.ops.derivative(state, self.component, axis=time, order=1)
        lap_u = state.ops.laplacian(state, self.component)
        residual = u_t - self.alpha * lap_u
        if self.source is not None:
            residual = residual - self.source(state)
        diag = {"mean_sq_residual": jnp.mean(residual * residual)}
        return HeatOutput(residual=residual, diag=diag)


def heat(
    state: FieldState,
    *,
    alpha: float = 1.0,
    component: str = "u",
    source: Callable[[FieldState], Array] | None = None,
) -> HeatOutput:
    """Stateless one-shot wrapper around :class:`Heat`."""
    return Heat(alpha=alpha, component=component, source=source)(state)


__all__ = ["Heat", "heat"]
