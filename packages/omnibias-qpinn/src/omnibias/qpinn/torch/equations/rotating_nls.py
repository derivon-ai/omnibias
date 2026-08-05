# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Stationary 2D rotating-frame Gross-Pitaevskii residual (torch backend).

The 2D rotating-frame Gross-Pitaevskii equation, evaluated at a
stationary state with chemical potential :math:`\mu`, is

.. math::

    \big[-\tfrac{\hbar^2}{2m}\,\nabla^2 + V(x, y) - \Omega\,L_z
         + g\,|\psi|^2\big]\,\psi \;=\; \mu\,\psi,

with the planar angular-momentum operator
:math:`L_z = -i\,(x\,\partial_y - y\,\partial_x)`. Splitting
:math:`\psi = \psi_R + i\,\psi_I` and using the elementary identities

.. math::

    \text{Re}(L_z\psi) &= x\,\partial_y\psi_I - y\,\partial_x\psi_I,\\
    \text{Im}(L_z\psi) &= -\,(x\,\partial_y\psi_R - y\,\partial_x\psi_R),

the residual decouples into two real equations that consume only
*standard* omnibias-pinn derivatives (Laplacian + first partial)
of each channel. No autograd-Hessian round-trip is needed and the
omnibias closed-form derivative tower is preserved across both terms.

This is the canonical *headline* residual of the qpinn vortex-lattice
demo; for the
time-dependent rotating frame use :class:`NLS` with a custom
:math:`-\Omega L_z\psi` source term.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import torch
from omnibias.pinn._core.state import FieldState
from omnibias.qpinn._core.complex import (
    apply_angular_momentum_z,
    apply_hamiltonian,
    psi_density,
    psi_value,
)
from omnibias.qpinn.torch.equations._types import RotatingNLSOutput
from torch import Tensor


@dataclass
class RotatingNLS:
    r"""Configurable 2D rotating-frame stationary Gross-Pitaevskii residual.

    Parameters
    ----------
    g
        Mean-field interaction strength. Positive = repulsive
        (BEC of bosons with positive scattering length).
    omega_rot
        Rotation frequency :math:`\Omega` of the trap.
    mu
        Chemical potential. Accepts either a Python ``float`` (fixed)
        or a 0-d ``Tensor`` (trainable Lagrange parameter).
    hbar
        Planck constant. Default 1.0 (oscillator units).
    mass
        Particle mass. Default 1.0 (oscillator units).
    psi
        Wavefunction group name on the :class:`FieldState`. Default
        ``"psi"``.
    x_axis, y_axis
        Coordinate axes that define the plane of rotation. Default
        ``(0, 1)``, i.e. axes ``"x"`` and ``"y"`` in a
        :class:`CoordinateSpec` built as ``("x", "y", ...)``.
    potential
        Callable ``V(state) -> Tensor`` of shape ``(B,)``. The
        rotating-frame *centrifugal* term :math:`-\tfrac{1}{2}\Omega^2 r^2`
        is **not** absorbed; pass it yourself if you want the
        non-inertial pseudo-potential.
    source
        Optional callable ``s(state) -> Tensor`` of shape ``(B, 2)``
        subtracted from the residual.
    """

    g: float = 1.0
    omega_rot: float = 0.0
    mu: float | Tensor = 0.0
    hbar: float = 1.0
    mass: float = 1.0
    psi: str = "psi"
    x_axis: int = 0
    y_axis: int = 1
    potential: Callable[[FieldState], Tensor] | None = None
    source: Callable[[FieldState], Tensor] | None = None

    def __call__(self, state: FieldState) -> RotatingNLSOutput:
        if state.coordinate_spec.ndim < 2:
            raise ValueError(
                "RotatingNLS requires at least 2 spatial axes; "
                f"got coordinate_spec.ndim = {state.coordinate_spec.ndim}"
            )
        if self.x_axis == self.y_axis:
            raise ValueError(
                f"x_axis ({self.x_axis}) and y_axis ({self.y_axis}) must differ"
            )
        re_name = f"{self.psi}_re"
        if not state.components.is_component(re_name):
            raise KeyError(
                f"component {re_name!r} not found; build the field with "
                "omnibias.qpinn.make_psi_components"
            )

        psi_re, psi_im = psi_value(state, self.psi)
        H_re, H_im = apply_hamiltonian(
            state, group=self.psi,
            hbar=self.hbar, mass=self.mass, potential=self.potential,
        )
        density = psi_density(state, self.psi)
        # Linear contributions H psi + g |psi|^2 psi - mu psi.
        mu = self.mu
        if not isinstance(mu, Tensor):
            mu = torch.as_tensor(mu, dtype=psi_re.dtype, device=psi_re.device)
        nl_re = self.g * density * psi_re
        nl_im = self.g * density * psi_im
        # Angular-momentum contributions L_z = -i (x d_y - y d_x).
        Lz_psi_re, Lz_psi_im = apply_angular_momentum_z(
            state, group=self.psi, hbar=1.0,
            x_axis=self.x_axis, y_axis=self.y_axis,
        )

        res_re = H_re + nl_re - self.omega_rot * Lz_psi_re - mu * psi_re
        res_im = H_im + nl_im - self.omega_rot * Lz_psi_im - mu * psi_im
        residual = torch.stack([res_re, res_im], dim=-1)
        if self.source is not None:
            residual = residual - self.source(state)

        # <psi | L_z | psi> = Im(psi* L_z psi).
        # For our split-real psi, psi* L_z psi has imaginary part
        # = psi_re * Im(L_z psi) - psi_im * Re(L_z psi).
        rotation_energy_density = self.omega_rot * (
            psi_re * Lz_psi_im - psi_im * Lz_psi_re
        )
        diag = {
            "mean_sq_residual": float((residual.detach() ** 2).mean()),
            "mean_density": float(density.detach().mean()),
            "mean_rotation_energy_density": float(rotation_energy_density.detach().mean()),
            "nonlinear_energy": float((self.g * density * density / 2).detach().mean()),
        }
        return RotatingNLSOutput(residual=residual, density=density, diag=diag)


def rotating_nls(
    state: FieldState,
    *,
    g: float = 1.0,
    omega_rot: float = 0.0,
    mu: float | Tensor = 0.0,
    hbar: float = 1.0,
    mass: float = 1.0,
    psi: str = "psi",
    x_axis: int = 0,
    y_axis: int = 1,
    potential: Callable[[FieldState], Tensor] | None = None,
    source: Callable[[FieldState], Tensor] | None = None,
) -> RotatingNLSOutput:
    """Stateless one-shot wrapper around :class:`RotatingNLS`."""
    return RotatingNLS(
        g=g, omega_rot=omega_rot, mu=mu, hbar=hbar, mass=mass, psi=psi,
        x_axis=x_axis, y_axis=y_axis, potential=potential, source=source,
    )(state)


__all__ = ["RotatingNLS", "rotating_nls"]
