# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Time-dependent Schrodinger equation residual (torch backend).

The TDSE on :math:`\mathbb{R}^D \times \mathbb{R}_+` reads

.. math::

    i\hbar\,\partial_t \psi(x, t) \;=\; \hat H \psi(x, t),
    \qquad
    \hat H = -\frac{\hbar^2}{2m}\,\nabla^2 + V(x, t).

With :math:`\psi = \psi_R + i\,\psi_I` and grouping by real / imaginary
parts:

.. math::

    -\hbar\,\partial_t \psi_I &= (\hat H \psi)_R, \\
    +\hbar\,\partial_t \psi_R &= (\hat H \psi)_I.

The residual is the LHS - RHS for each channel, stacked into ``(B, 2)``.
For a real Hermitian potential :math:`V(x, t)` the Hamiltonian acts
independently on the two channels (no ``i`` factor), so
:math:`(H\psi)_R = -\frac{\hbar^2}{2m}\Delta\psi_R + V\psi_R` and
similarly for :math:`(H\psi)_I`.

The companion :class:`NLS` residual extends this with a nonlinear
``g |psi|^2 psi`` term.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import torch
from omnibias.pinn._core.state import FieldState
from omnibias.qpinn._core.complex import apply_hamiltonian
from omnibias.qpinn.torch.equations._types import TDSEOutput
from torch import Tensor


@dataclass
class TDSE:
    r"""Configurable time-dependent Schrodinger residual.

    Parameters
    ----------
    hbar
        Planck constant in the chosen unit system. Default 1.0.
    mass
        Particle mass. Default 1.0.
    psi
        Wavefunction group name on the :class:`FieldState`. Default
        ``"psi"``.
    potential
        Callable ``V(state) -> Tensor of shape (B,)``. ``None``
        corresponds to the free-particle TDSE.
    source
        Optional callable ``s(state) -> Tensor of shape (B, 2)`` added
        to the residual (used for manufactured-solution tests).
    """

    hbar: float = 1.0
    mass: float = 1.0
    psi: str = "psi"
    potential: Callable[[FieldState], Tensor] | None = None
    source: Callable[[FieldState], Tensor] | None = None

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
        residual = torch.stack([res_re, res_im], dim=-1)
        if self.source is not None:
            residual = residual - self.source(state)
        return TDSEOutput(
            residual=residual,
            diag={"mean_sq_residual": float((residual.detach() ** 2).mean())},
        )


def tdse(
    state: FieldState,
    *,
    hbar: float = 1.0,
    mass: float = 1.0,
    psi: str = "psi",
    potential: Callable[[FieldState], Tensor] | None = None,
    source: Callable[[FieldState], Tensor] | None = None,
) -> TDSEOutput:
    """Stateless one-shot wrapper around :class:`TDSE`."""
    return TDSE(
        hbar=hbar, mass=mass, psi=psi,
        potential=potential, source=source,
    )(state)


__all__ = ["TDSE", "tdse"]
