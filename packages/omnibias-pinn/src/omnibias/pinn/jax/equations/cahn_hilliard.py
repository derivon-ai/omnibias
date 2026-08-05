# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Cahn-Hilliard equation residual (jax twin)."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

import jax.numpy as jnp
from jax import Array
from omnibias.pinn._core.state import FieldState
from omnibias.pinn.jax.equations._types import CHOutput


class Potential(Protocol):
    def f(self, c: Array) -> Array: ...

    def df(self, c: Array) -> Array: ...

    def d2f(self, c: Array) -> Array: ...

    def d3f(self, c: Array) -> Array: ...


@dataclass(frozen=True)
class GinzburgLandauPotential:
    """``f(c) = W * (c^2 - 1)^2 / 4`` -- canonical double-well potential."""

    W: float = 1.0

    def f(self, c: Array) -> Array:
        x2 = c * c - 1.0
        return self.W * 0.25 * x2 * x2

    def df(self, c: Array) -> Array:
        return self.W * (c * c * c - c)

    def d2f(self, c: Array) -> Array:
        return self.W * (3.0 * c * c - 1.0)

    def d3f(self, c: Array) -> Array:
        return self.W * 6.0 * c


@dataclass
class CahnHilliard:
    M: float = 1.0
    kappa: float = 1e-3
    component: str = "c"
    potential: Potential = None  # type: ignore[assignment]
    forcing: Callable[[FieldState], Array] | None = None

    def __post_init__(self) -> None:
        if self.potential is None:
            object.__setattr__(self, "potential", GinzburgLandauPotential())

    def __call__(self, state: FieldState) -> CHOutput:
        time = state.coordinate_spec.time_axis
        if time is None:
            raise ValueError("Cahn-Hilliard equation requires a time axis")
        c = state.ops.value(state, self.component)
        grad_c = state.ops.gradient(state, self.component)
        lap_c = state.ops.laplacian(state, self.component)
        bih_c = state.ops.biharmonic(state, self.component)
        c_t = state.ops.derivative(state, self.component, axis=time, order=1)

        f2 = self.potential.d2f(c)
        f3 = self.potential.d3f(c)
        grad_sq = jnp.sum(grad_c * grad_c, axis=-1)

        Delta_fprime = f2 * lap_c + f3 * grad_sq
        residual = c_t - self.M * Delta_fprime + self.M * self.kappa * bih_c
        if self.forcing is not None:
            residual = residual - self.forcing(state)
        return CHOutput(
            residual=residual,
            diag={"mean_sq_residual": jnp.mean(residual * residual)},
        )


def cahn_hilliard(
    state: FieldState,
    *,
    M: float = 1.0,
    kappa: float = 1e-3,
    component: str = "c",
    potential: Potential | None = None,
    forcing: Callable[[FieldState], Array] | None = None,
) -> CHOutput:
    """Stateless one-shot wrapper around :class:`CahnHilliard`."""
    return CahnHilliard(
        M=M, kappa=kappa, component=component,
        potential=potential if potential is not None else GinzburgLandauPotential(),
        forcing=forcing,
    )(state)


__all__ = [
    "CahnHilliard",
    "GinzburgLandauPotential",
    "Potential",
    "cahn_hilliard",
]
