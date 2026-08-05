# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Energy diagnostics for the torch backend.

We expose three standard variational quantities:

- :func:`expectation_value`: :math:`\langle\psi|\hat O|\psi\rangle /
  \langle\psi|\psi\rangle` for a user-supplied operator action
  :math:`\hat O\psi`.
- :func:`expected_energy`: shortcut that evaluates the Schrodinger
  Hamiltonian via :func:`omnibias.qpinn._core.complex.apply_hamiltonian`.
- :func:`energy_variance`: :math:`\langle (H - E)^2\rangle`, useful as
  an excited-state objective.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

import torch
from omnibias.qpinn._core.complex import apply_hamiltonian, psi_value
from torch import Tensor

if TYPE_CHECKING:  # pragma: no cover -- typing-only import
    from omnibias.pinn._core.state import FieldState


def expectation_value(
    state: FieldState,
    *,
    operator_action: Callable[[FieldState], tuple[Tensor, Tensor]],
    group: str = "psi",
    quadrature_weights: Tensor | None = None,
) -> Tensor:
    r"""Compute :math:`\langle\psi|\hat O|\psi\rangle / \langle\psi|\psi\rangle`.

    The operator is supplied as a callable returning the real and
    imaginary parts of :math:`\hat O\psi` at every collocation point;
    the integrand is then
    :math:`\psi_R\,(O\psi)_R + \psi_I\,(O\psi)_I` (Hermitian operators
    have a real expectation value, so we use the symmetrised form).

    Parameters
    ----------
    state
        :class:`FieldState` carrying the wavefunction group.
    operator_action
        Callable ``O(state) -> (re, im)`` returning the action of the
        operator on the wavefunction as a pair of ``(B,)`` tensors.
    group
        Wavefunction group name.
    quadrature_weights
        Per-collocation weights ``(B,)``. If ``None``, uniform.

    Returns
    -------
    Tensor
        Zero-dimensional ``Tensor`` with the expectation value.
    """
    psi_re, psi_im = psi_value(state, group)
    O_re, O_im = operator_action(state)
    integrand = psi_re * O_re + psi_im * O_im
    density = psi_re * psi_re + psi_im * psi_im
    if quadrature_weights is None:
        num = integrand.mean()
        den = density.mean()
    else:
        if quadrature_weights.shape != integrand.shape:
            raise ValueError(
                f"quadrature_weights shape {tuple(quadrature_weights.shape)} "
                f"!= integrand shape {tuple(integrand.shape)}"
            )
        num = (quadrature_weights * integrand).sum()
        den = (quadrature_weights * density).sum()
    return num / (den + torch.finfo(num.dtype).tiny)


def expected_energy(
    state: FieldState,
    *,
    hbar: float = 1.0,
    mass: float = 1.0,
    potential: Callable[[FieldState], Tensor] | None = None,
    group: str = "psi",
    quadrature_weights: Tensor | None = None,
) -> Tensor:
    r"""Variational energy :math:`\langle\psi|\hat H|\psi\rangle` for the
    one-body Schrodinger Hamiltonian.

    Parameters
    ----------
    state
        :class:`FieldState`.
    hbar, mass
        Atomic-units defaults. Set ``hbar = 1, mass = 1`` for the
        electron-mass scale.
    potential
        Callable ``V(state) -> Tensor(B,)``. ``None`` for free particle.
    group
        Wavefunction group name.
    quadrature_weights
        Per-collocation weights.
    """
    def _Hpsi(s: FieldState) -> tuple[Tensor, Tensor]:
        return apply_hamiltonian(
            s, group=group, hbar=hbar, mass=mass, potential=potential,
        )
    return expectation_value(
        state, operator_action=_Hpsi, group=group,
        quadrature_weights=quadrature_weights,
    )


def energy_variance(
    state: FieldState,
    *,
    hbar: float = 1.0,
    mass: float = 1.0,
    potential: Callable[[FieldState], Tensor] | None = None,
    group: str = "psi",
    quadrature_weights: Tensor | None = None,
) -> Tensor:
    r"""Variance :math:`\langle (\hat H - \langle\hat H\rangle)^2\rangle`.

    A standard variational excited-state objective: when the ansatz is
    an exact eigenstate, the variance is zero. The function returns the
    *non-negative* variance, useful as a direct loss term.

    Parameters
    ----------
    state
        :class:`FieldState`.
    hbar, mass, potential, group, quadrature_weights
        Same as :func:`expected_energy`.
    """
    psi_re, psi_im = psi_value(state, group)
    H_re, H_im = apply_hamiltonian(
        state, group=group, hbar=hbar, mass=mass, potential=potential,
    )
    density = psi_re * psi_re + psi_im * psi_im
    H_dot_psi = psi_re * H_re + psi_im * H_im
    if quadrature_weights is None:
        num = H_dot_psi.mean()
        den = density.mean()
        h2_num = (H_re * H_re + H_im * H_im).mean()
    else:
        num = (quadrature_weights * H_dot_psi).sum()
        den = (quadrature_weights * density).sum()
        h2_num = (quadrature_weights * (H_re * H_re + H_im * H_im)).sum()
    eps = torch.finfo(num.dtype).tiny
    E = num / (den + eps)
    Hsq = h2_num / (den + eps)
    return Hsq - E * E


__all__ = [
    "energy_variance",
    "expectation_value",
    "expected_energy",
]
