# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Navier-Stokes equation residual (jax twin)."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import jax.numpy as jnp
from jax import Array
from omnibias.pinn._core.state import FieldState
from omnibias.pinn.jax.equations._types import NavierStokesOutput


@dataclass
class NavierStokes:
    """Configurable Navier-Stokes residual (jax)."""

    viscosity: float = 1e-3
    density: float = 1.0
    form: str = "primitive_3d"
    velocity: tuple[str, ...] = field(default=("u", "v", "w"))
    pressure: str = "p"
    streamfunction: str = "psi"
    forcing: Callable[[FieldState], Array] | None = None
    incompressibility: str = "soft"

    def __call__(self, state: FieldState) -> NavierStokesOutput:
        if self.form == "primitive_3d":
            return self._primitive(state, dim=3)
        if self.form == "primitive_2d":
            return self._primitive(state, dim=2)
        if self.form == "vorticity_stream_2d":
            return self._vorticity_stream_2d(state)
        raise ValueError(
            f"NavierStokes form must be 'primitive_3d' | 'primitive_2d' | "
            f"'vorticity_stream_2d'; got {self.form!r}"
        )

    def _primitive(self, state: FieldState, *, dim: int) -> NavierStokesOutput:
        time = state.coordinate_spec.time_axis
        if time is None:
            raise ValueError("Navier-Stokes (primitive) requires a time axis")
        spatial = state.coordinate_spec.spatial_axes
        if len(spatial) != dim:
            raise ValueError(
                f"primitive_{dim}d form requires {dim} spatial axes, got "
                f"{len(spatial)} ({spatial!r})"
            )
        comps = self.velocity[:dim]
        if len(comps) != dim:
            raise ValueError(
                f"velocity must have at least {dim} components for "
                f"primitive_{dim}d, got {self.velocity!r}"
            )

        u_t = state.ops.vector_derivative(state, comps, axis=time, order=1)
        adv = state.ops.advection(state, velocity=comps)
        grad_p = state.ops.gradient(state, self.pressure)
        lap_vec = state.ops.vector_laplacian(state, comps)

        residual = (
            self.density * (u_t + adv)
            + grad_p
            - self.viscosity * lap_vec
        )
        if self.forcing is not None:
            residual = residual - self.forcing(state)

        if self.incompressibility == "soft":
            continuity = state.ops.divergence(state, comps)
        elif self.incompressibility == "hard":
            continuity = jnp.zeros_like(residual[..., 0])
        else:
            raise ValueError(
                f"incompressibility must be 'soft' or 'hard', got "
                f"{self.incompressibility!r}"
            )

        diag = {
            "mean_sq_residual": jnp.mean(residual * residual),
            "mean_sq_continuity": jnp.mean(continuity * continuity),
        }
        return NavierStokesOutput(
            residual=residual, continuity=continuity, diag=diag,
        )

    def _vorticity_stream_2d(self, state: FieldState) -> NavierStokesOutput:
        time = state.coordinate_spec.time_axis
        if time is None:
            raise ValueError("vorticity_stream_2d form requires a time axis")
        spatial = state.coordinate_spec.spatial_axes
        if len(spatial) != 2:
            raise ValueError(
                f"vorticity_stream_2d form requires 2 spatial axes, got "
                f"{len(spatial)} ({spatial!r})"
            )
        ax_x, ax_y = spatial
        psi = self.streamfunction

        psi_x = state.ops.derivative(state, psi, axis=ax_x, order=1)
        psi_y = state.ops.derivative(state, psi, axis=ax_y, order=1)

        mp = state.ops.mixed_partial
        psi_xxx = mp(state, psi, (ax_x,), (3,))
        psi_yyx = mp(state, psi, (ax_x, ax_y), (1, 2))
        psi_xxy = mp(state, psi, (ax_x, ax_y), (2, 1))
        psi_yyy = mp(state, psi, (ax_y,), (3,))
        omega_x = -(psi_xxx + psi_yyx)
        omega_y = -(psi_xxy + psi_yyy)

        bih_psi = state.ops.biharmonic(state, psi)
        lap_omega = -bih_psi

        psi_xxt = mp(state, psi, (ax_x, time), (2, 1))
        psi_yyt = mp(state, psi, (ax_y, time), (2, 1))
        omega_t = -(psi_xxt + psi_yyt)

        residual = (
            omega_t
            + psi_y * omega_x
            - psi_x * omega_y
            - self.viscosity * lap_omega
        )
        if self.forcing is not None:
            residual = residual - self.forcing(state)

        continuity = jnp.zeros_like(residual)
        diag = {
            "mean_sq_residual": jnp.mean(residual * residual),
            "mean_sq_continuity": 0.0,
        }
        return NavierStokesOutput(
            residual=residual, continuity=continuity, diag=diag,
        )

    def momentum_residual(self, state: FieldState) -> Array:
        return self(state).residual

    def continuity_residual(self, state: FieldState) -> Array:
        return self(state).continuity


def navier_stokes(
    state: FieldState,
    *,
    viscosity: float = 1e-3,
    density: float = 1.0,
    form: str = "primitive_3d",
    velocity: tuple[str, ...] = ("u", "v", "w"),
    pressure: str = "p",
    streamfunction: str = "psi",
    forcing: Callable[[FieldState], Array] | None = None,
    incompressibility: str = "soft",
) -> NavierStokesOutput:
    """Stateless one-shot wrapper around :class:`NavierStokes`."""
    return NavierStokes(
        viscosity=viscosity, density=density, form=form,
        velocity=velocity, pressure=pressure, streamfunction=streamfunction,
        forcing=forcing, incompressibility=incompressibility,
    )(state)


__all__ = ["NavierStokes", "navier_stokes"]
