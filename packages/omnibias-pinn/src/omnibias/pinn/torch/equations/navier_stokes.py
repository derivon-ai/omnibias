# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Navier-Stokes equation residual.

Two forms supported:

* **Primitive variables** (``form="primitive_3d"`` / ``"primitive_2d"``):
  field carries velocity components ``u, v, [w]`` plus pressure ``p``.
  The momentum residual is

  .. math::

     R_i \\;=\\; \\rho\\, (D u_i / D t)  + \\partial_i p
                \\;-\\; \\mu\\, \\Delta u_i \\;-\\; f_i,

  where :math:`Du_i/Dt = \\partial_t u_i + (u \\cdot \\nabla) u_i`.
  Continuity :math:`\\nabla \\cdot u` is also exposed for soft enforcement
  (or for verification when the field is wrapped in a divergence-free
  cage).

* **Vorticity-streamfunction 2D** (``form="vorticity_stream_2d"``):
  field carries a single component ``psi`` (the stream function); the
  vorticity is :math:`\\omega = -\\Delta \\psi` and the velocity is
  :math:`u = \\partial_y \\psi,\\, v = -\\partial_x \\psi`. The residual is

  .. math::

     R \\;=\\; \\partial_t \\omega
              \\;+\\; \\psi_y\\, \\omega_x \\;-\\; \\psi_x\\, \\omega_y
              \\;-\\; \\nu\\, \\Delta \\omega \\;-\\; f_{\\omega}.

  Substituting :math:`\\omega = -\\Delta \\psi` reduces the viscous
  term to the *biharmonic* of :math:`\\psi`. This is the canonical
  4th-order omnibias-friendly form used by
  :mod:`research.experiments.navier_stokes_2d`.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import torch
from omnibias.pinn._core.state import FieldState
from omnibias.pinn.torch.equations._types import NavierStokesOutput
from torch import Tensor


@dataclass
class NavierStokes:
    """Configurable Navier-Stokes residual.

    Parameters
    ----------
    viscosity
        Kinematic viscosity :math:`\\nu = \\mu / \\rho` (or absolute :math:`\\mu`,
        depending on density). Default 1e-3.
    density
        Density :math:`\\rho`. Default 1.0 (incompressible scaling).
    form
        ``"primitive_3d"`` (default), ``"primitive_2d"``, or
        ``"vorticity_stream_2d"``.
    velocity
        Tuple of velocity component names. For 3D primitive defaults to
        ``("u", "v", "w")``; for 2D primitive ``("u", "v")``. Ignored for
        vorticity-stream form.
    pressure
        Pressure component name (primitive forms only). Default ``"p"``.
    streamfunction
        Stream-function component name (vorticity-stream form). Default ``"psi"``.
    forcing
        Optional callable :math:`f(state) \\to (B, D)` external body force
        for primitive form, or :math:`f_\\omega(state) \\to (B,)` for
        vorticity-stream form (the curl of the body force).
    incompressibility
        ``"soft"`` (default) returns a non-zero ``continuity`` to be
        added as a penalty; ``"hard"`` skips computing continuity (the
        field is assumed wrapped in :class:`StreamfunctionField` or
        :class:`VectorPotentialField`).
    """

    viscosity: float = 1e-3
    density: float = 1.0
    form: str = "primitive_3d"
    velocity: tuple[str, ...] = field(default=("u", "v", "w"))
    pressure: str = "p"
    streamfunction: str = "psi"
    forcing: Callable[[FieldState], Tensor] | None = None
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

    # ------- primitive form (any D) ----------------------------------

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

        # D u / D t = u_t + (u . grad) u  (shape (B, D))
        u_t = state.ops.vector_derivative(state, comps, axis=time, order=1)
        adv = state.ops.advection(state, velocity=comps)
        # nabla p (shape (B, D))
        grad_p = state.ops.gradient(state, self.pressure)
        # vector Laplacian of u (shape (B, D))
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
            continuity = torch.zeros_like(residual[..., 0])
        else:
            raise ValueError(
                f"incompressibility must be 'soft' or 'hard', got "
                f"{self.incompressibility!r}"
            )

        diag = {
            "mean_sq_residual": float((residual.detach() ** 2).mean()),
            "mean_sq_continuity": float((continuity.detach() ** 2).mean()),
        }
        return NavierStokesOutput(
            residual=residual, continuity=continuity, diag=diag,
        )

    # ------- vorticity-stream 2D --------------------------------------

    def _vorticity_stream_2d(self, state: FieldState) -> NavierStokesOutput:
        time = state.coordinate_spec.time_axis
        if time is None:
            raise ValueError(
                "vorticity_stream_2d form requires a time axis"
            )
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

        # omega = -Laplace(psi); omega_a = -d_a Laplace(psi).
        # Use mixed partials for second-derivative wrt one axis * one of
        # (xx, yy). This stays closed-form for spectral / chebyshev
        # fields (avoids re-forming the gradient of the Laplacian field).
        mp = state.ops.mixed_partial
        psi_xxx = mp(state, psi, (ax_x,), (3,))
        psi_yyx = mp(state, psi, (ax_x, ax_y), (1, 2))
        psi_xxy = mp(state, psi, (ax_x, ax_y), (2, 1))
        psi_yyy = mp(state, psi, (ax_y,), (3,))
        omega_x = -(psi_xxx + psi_yyx)
        omega_y = -(psi_xxy + psi_yyy)

        # Laplace(omega) = -bih(psi).
        bih_psi = state.ops.biharmonic(state, psi)
        lap_omega = -bih_psi

        # omega_t = -Laplace(psi_t) = -(psi_xxt + psi_yyt)
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

        # Continuity is identically zero by construction (psi is a stream
        # function): u = psi_y, v = -psi_x ⇒ ∂_x u + ∂_y v = psi_yx - psi_xy = 0.
        continuity = torch.zeros_like(residual)
        diag = {
            "mean_sq_residual": float((residual.detach() ** 2).mean()),
            "mean_sq_continuity": 0.0,
        }
        return NavierStokesOutput(
            residual=residual, continuity=continuity, diag=diag,
        )

    # ------- explicit accessors --------------------------------------

    def momentum_residual(self, state: FieldState) -> Tensor:
        """Compute and return only ``out.residual``."""
        return self(state).residual

    def continuity_residual(self, state: FieldState) -> Tensor:
        """Compute and return only ``out.continuity``."""
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
    forcing: Callable[[FieldState], Tensor] | None = None,
    incompressibility: str = "soft",
) -> NavierStokesOutput:
    """Stateless one-shot wrapper around :class:`NavierStokes`."""
    return NavierStokes(
        viscosity=viscosity, density=density, form=form,
        velocity=velocity, pressure=pressure, streamfunction=streamfunction,
        forcing=forcing, incompressibility=incompressibility,
    )(state)


__all__ = ["NavierStokes", "navier_stokes"]
