# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Biharmonic equation residual (jax twin)."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import jax.numpy as jnp
from jax import Array
from omnibias.pinn._core.state import FieldState
from omnibias.pinn.jax.equations._types import BiharmonicOutput


@dataclass
class Biharmonic:
    component: str = "u"
    include_time: bool = False
    source: Callable[[FieldState], Array] | None = None

    def __call__(self, state: FieldState) -> BiharmonicOutput:
        bih = state.ops.biharmonic(state, self.component)
        if self.include_time:
            time = state.coordinate_spec.time_axis
            if time is None:
                raise ValueError(
                    "Biharmonic(include_time=True) requires a time axis"
                )
            u_t = state.ops.derivative(state, self.component, axis=time, order=1)
            residual = u_t + bih
        else:
            residual = bih
        if self.source is not None:
            residual = residual - self.source(state)
        return BiharmonicOutput(
            residual=residual,
            diag={"mean_sq_residual": jnp.mean(residual * residual)},
        )


def biharmonic(
    state: FieldState,
    *,
    component: str = "u",
    include_time: bool = False,
    source: Callable[[FieldState], Array] | None = None,
) -> BiharmonicOutput:
    """Stateless one-shot wrapper around :class:`Biharmonic`."""
    return Biharmonic(
        component=component, include_time=include_time, source=source,
    )(state)


__all__ = ["Biharmonic", "biharmonic"]
