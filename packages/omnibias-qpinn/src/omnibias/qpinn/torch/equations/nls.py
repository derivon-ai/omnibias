# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Nonlinear Schrodinger / Gross-Pitaevskii residual (torch backend).

The nonlinear Schrodinger equation (NLSE), also known as the
Gross-Pitaevskii equation when modelling a Bose-Einstein condensate, is

.. math::

    i\hbar\,\partial_t \psi
      \;=\;
      -\frac{\hbar^2}{2m}\,\nabla^2 \psi
      \;+\; V(x, t)\,\psi
      \;+\; g\,|\psi|^2\,\psi,

where :math:`g > 0` gives a repulsive (defocusing) nonlinearity and
:math:`g < 0` an attractive (focusing) one. With
:math:`\psi = \psi_R + i\,\psi_I` and using :math:`|\psi|^2 = \psi_R^2
+ \psi_I^2` the nonlinear term acts as a real, density-dependent
potential :math:`V_{NL}(x, t) = g\,|\psi(x, t)|^2`, so the residual
shape matches :class:`TDSE` with the added :math:`V_{NL}` contribution.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import torch
from omnibias.pinn._core.state import FieldState
from omnibias.qpinn._core.complex import apply_hamiltonian, psi_density, psi_value
from omnibias.qpinn.torch.equations._types import NLSOutput
from torch import Tensor


@dataclass
class NLS:
    r"""Configurable nonlinear-Schrodinger / Gross-Pitaevskii residual.

    Parameters
    ----------
    g
        Nonlinearity coupling. Positive for defocusing (repulsive
        bosonic interaction), negative for focusing.
    hbar
        Planck constant. Default 1.0.
    mass
        Particle mass. Default 1.0.
    psi
        Wavefunction group name. Default ``"psi"``.
    potential
        Optional callable ``V(state) -> Tensor of shape (B,)`` for the
        linear (external) part of the potential. The nonlinear
        :math:`g\,|\psi|^2` part is added automatically.
    source
        Optional callable ``s(state) -> Tensor of shape (B, 2)`` added
        to the residual.
    """

    g: float = 1.0
    hbar: float = 1.0
    mass: float = 1.0
    psi: str = "psi"
    potential: Callable[[FieldState], Tensor] | None = None
    source: Callable[[FieldState], Tensor] | None = None

    def __call__(self, state: FieldState) -> NLSOutput:
        time = state.coordinate_spec.time_axis
        if time is None:
            raise ValueError(
                "NLS residual requires a time axis in the coordinate spec"
            )
        re_name = f"{self.psi}_re"
        im_name = f"{self.psi}_im"
        if not state.components.is_component(re_name):
            raise KeyError(
                f"component {re_name!r} not found; build the field with "
                "omnibias.qpinn.make_psi_components"
            )

        psi_re, psi_im = psi_value(state, self.psi)
        psi_re_t = state.ops.derivative(state, re_name, axis=time, order=1)
        psi_im_t = state.ops.derivative(state, im_name, axis=time, order=1)

        # Linear Hamiltonian H = T + V.
        H_re, H_im = apply_hamiltonian(
            state, group=self.psi,
            hbar=self.hbar, mass=self.mass, potential=self.potential,
        )
        # Nonlinear contribution: V_NL = g |psi|^2 (real, density-dependent).
        density = psi_density(state, self.psi)
        nl_re = self.g * density * psi_re
        nl_im = self.g * density * psi_im

        res_re = -self.hbar * psi_im_t - (H_re + nl_re)
        res_im = self.hbar * psi_re_t - (H_im + nl_im)
        residual = torch.stack([res_re, res_im], dim=-1)
        if self.source is not None:
            residual = residual - self.source(state)

        diag = {
            "mean_sq_residual": float((residual.detach() ** 2).mean()),
            "mean_density": float(density.detach().mean()),
            "nonlinear_energy": float((self.g * density * density / 2).detach().mean()),
        }
        return NLSOutput(residual=residual, diag=diag)


def nls(
    state: FieldState,
    *,
    g: float = 1.0,
    hbar: float = 1.0,
    mass: float = 1.0,
    psi: str = "psi",
    potential: Callable[[FieldState], Tensor] | None = None,
    source: Callable[[FieldState], Tensor] | None = None,
) -> NLSOutput:
    """Stateless one-shot wrapper around :class:`NLS`."""
    return NLS(
        g=g, hbar=hbar, mass=mass, psi=psi,
        potential=potential, source=source,
    )(state)


__all__ = ["NLS", "nls"]
