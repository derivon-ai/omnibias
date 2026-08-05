# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Time-independent Schrodinger equation residual (torch backend).

The TISE on :math:`\mathbb{R}^D` is the eigenvalue problem

.. math::

    \hat H \psi(x) = E\,\psi(x),

with the standard one-body Hamiltonian

.. math::

    \hat H = -\frac{\hbar^2}{2m}\,\nabla^2 \;+\; V(x).

We write the wavefunction as :math:`\psi = \psi_R + i\,\psi_I` and
stack the real/imaginary residuals into a ``(B, 2)`` tensor for the
caller's loss reduction. For purely real eigenstates (e.g. ground
states of one-body Hermitian Hamiltonians) the imaginary residual
trivially carries the same information as the real one; for general
complex eigenstates both channels are independent.

The eigenvalue ``E`` can be a fixed scalar (when the user wants the
residual at a known eigenvalue) or a trainable scalar tensor (when
the user wants to learn the ground-state energy variationally).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import torch
from omnibias.pinn._core.state import FieldState
from omnibias.qpinn._core.complex import apply_hamiltonian, psi_value
from omnibias.qpinn.torch.equations._types import TISEOutput
from torch import Tensor


@dataclass
class TISE:
    r"""Configurable time-independent Schrodinger residual.

    Parameters
    ----------
    energy
        Eigenvalue :math:`E`. Accepts either a Python ``float`` (fixed
        eigenvalue) or a 0-d ``Tensor`` (trainable). Default 0.0.
    hbar
        Planck constant in the chosen unit system. Default 1.0.
    mass
        Particle mass. Default 1.0 (electron mass in atomic units).
    psi
        Wavefunction group name on the :class:`FieldState`. Default
        ``"psi"``; the group must carry exactly two real components in
        ``(re, im)`` order as built by
        :func:`omnibias.qpinn.make_psi_components`.
    potential
        Callable ``V(state) -> Tensor of shape (B,)``. ``None``
        corresponds to the free-particle case. The potential is taken
        to be real (Hermitian).
    quadrature_weights
        Optional ``(B,)`` quadrature weights. If provided, the residual
        output additionally exposes ``energy_estimate``, the variational
        Rayleigh quotient :math:`\langle\psi|\hat H|\psi\rangle /
        \langle\psi|\psi\rangle` averaged over the collocation points.
        ``None`` (default) skips the estimate.
    """

    energy: float | Tensor = 0.0
    hbar: float = 1.0
    mass: float = 1.0
    psi: str = "psi"
    potential: Callable[[FieldState], Tensor] | None = None
    quadrature_weights: Tensor | None = None

    def __call__(self, state: FieldState) -> TISEOutput:
        psi_re, psi_im = psi_value(state, self.psi)
        H_re, H_im = apply_hamiltonian(
            state, group=self.psi,
            hbar=self.hbar, mass=self.mass, potential=self.potential,
        )
        E = self.energy
        if not isinstance(E, Tensor):
            E = torch.as_tensor(E, dtype=psi_re.dtype, device=psi_re.device)
        res_re = H_re - E * psi_re
        res_im = H_im - E * psi_im
        residual = torch.stack([res_re, res_im], dim=-1)

        energy_estimate: Tensor | None = None
        diag: dict[str, float] = {
            "mean_sq_residual": float((residual.detach() ** 2).mean()),
        }
        if self.quadrature_weights is not None:
            w = self.quadrature_weights
            if w.shape != psi_re.shape:
                raise ValueError(
                    f"quadrature_weights shape {tuple(w.shape)} != "
                    f"psi shape {tuple(psi_re.shape)}"
                )
            density = psi_re * psi_re + psi_im * psi_im
            num = (w * (psi_re * H_re + psi_im * H_im)).sum()
            den = (w * density).sum()
            energy_estimate = num / (den + torch.finfo(psi_re.dtype).tiny)
            diag["energy_estimate"] = float(energy_estimate.detach())
            diag["norm_squared"] = float(den.detach())
        return TISEOutput(
            residual=residual,
            energy_estimate=energy_estimate,
            diag=diag,
        )


def tise(
    state: FieldState,
    *,
    energy: float | Tensor = 0.0,
    hbar: float = 1.0,
    mass: float = 1.0,
    psi: str = "psi",
    potential: Callable[[FieldState], Tensor] | None = None,
    quadrature_weights: Tensor | None = None,
) -> TISEOutput:
    """Stateless one-shot wrapper around :class:`TISE`."""
    return TISE(
        energy=energy, hbar=hbar, mass=mass, psi=psi,
        potential=potential, quadrature_weights=quadrature_weights,
    )(state)


__all__ = ["TISE", "tise"]
