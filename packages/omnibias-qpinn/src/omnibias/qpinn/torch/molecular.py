# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Molecular (Born-Oppenheimer) electronic-structure local energy (torch).

Bit-identical twin of :mod:`omnibias.qpinn.jax.molecular`. See that module for
the physics and the closed-form-scope honesty note. The electronic Hamiltonian
in atomic units is ``H = -1/2 sum_j nabla_j^2 + V_Coulomb`` and the local energy
is ``E_L = T_L + V`` with the drift-form kinetic term

.. math::

    T_L = -\tfrac12\bigl(\nabla^2 \log|\psi| + \lVert\nabla\log|\psi|\rVert^2\bigr).

The Laplacian of ``log|psi|`` comes from the closed-form multivariate jet tower
:func:`omnibias.torch.mlp_jet_mv` (no autodiff, no finite differences).

Closed-form scope
-----------------
The kinetic term and the Coulomb potential are closed form / exact. The
*variational* solution of the Schrodinger equation is **not** in scope here:
VMC Monte-Carlo sampling, SCF/HF/CI/CC self-consistency, and Gaussian-basis
electron-repulsion integrals are iterative / stochastic numerics and are
labelled accordingly (see ``docs/scope-and-guarantees.md``).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import torch
from omnibias.torch.jet_mv import jet_gradient, jet_hessian, mlp_jet_mv
from torch import Tensor

LayerSpec = tuple[Any, Any, Any]

_COULOMB_EPS2 = 1e-30


def coulomb_potential(R: Tensor, r_flat: Tensor, charges: Tensor, n_e: int) -> Tensor:
    r"""Bare Coulomb potential energy ``V(R, r)`` in atomic units.

    ``R`` are nuclear positions ``(n_atoms*3,)``, ``r_flat`` electron positions
    ``(n_e*3,)``, ``charges`` the nuclear charges ``Z_a`` ``(n_atoms,)``. Includes
    the electron-nucleus, electron-electron, and nucleus-nucleus terms.
    """
    charges = torch.as_tensor(charges)
    n_atoms = charges.shape[0]
    r_atoms = torch.as_tensor(R).reshape((n_atoms, 3))
    r = torch.as_tensor(r_flat).reshape((n_e, 3))

    d_en = torch.sqrt(
        torch.sum((r[:, None, :] - r_atoms[None, :, :]) ** 2, dim=-1) + _COULOMB_EPS2
    )
    v_en = -torch.sum(charges[None, :] / d_en)

    if n_e > 1:
        d_ee = torch.sqrt(
            torch.sum((r[:, None, :] - r[None, :, :]) ** 2, dim=-1) + _COULOMB_EPS2
        )
        v_ee = torch.sum(torch.triu(1.0 / d_ee, diagonal=1))
    else:
        v_ee = torch.zeros((), dtype=r.dtype)

    if n_atoms > 1:
        d_nn = torch.sqrt(
            torch.sum((r_atoms[:, None, :] - r_atoms[None, :, :]) ** 2, dim=-1)
            + _COULOMB_EPS2
        )
        zz = charges[:, None] * charges[None, :]
        v_nn = torch.sum(torch.triu(zz / d_nn, diagonal=1))
    else:
        v_nn = torch.zeros((), dtype=r_atoms.dtype)

    return v_en + v_ee + v_nn


def local_kinetic_energy(grad_log_psi: Tensor, lap_log_psi: Tensor) -> Tensor:
    r"""``T_L = -1/2 (nabla^2 log|psi| + ||nabla log|psi|||^2)`` (drift form)."""
    grad = torch.as_tensor(grad_log_psi)
    return -0.5 * (torch.as_tensor(lap_log_psi) + torch.sum(grad * grad))


def local_energy(
    grad_log_psi: Tensor, lap_log_psi: Tensor, potential: Tensor
) -> Tensor:
    r"""Local energy ``E_L = T_L + V`` for a supplied scalar potential value."""
    return local_kinetic_energy(grad_log_psi, lap_log_psi) + torch.as_tensor(potential)


def log_psi_derivatives(
    x0: Tensor, layers: Sequence[LayerSpec], *, order: int = 2
) -> tuple[Tensor, Tensor]:
    r"""Closed-form ``(nabla log|psi|, nabla^2 log|psi|)`` for an MLP ``log|psi|``.

    ``layers`` is the :func:`omnibias.torch.mlp_jet_mv` layer stack whose scalar
    output is ``log|psi|(x)``; the gradient and Laplacian come from the exact
    multivariate jet tower.
    """
    x0 = torch.as_tensor(x0)
    dim = int(x0.shape[-1])
    jet = mlp_jet_mv(x0, layers, order)
    grad = jet_gradient(jet, dim, order)  # (D, C)
    hess = jet_hessian(jet, dim, order)  # (D, D, C)
    lap = torch.diagonal(hess, dim1=0, dim2=1).sum(dim=-1)  # (C,)
    if grad.ndim == 2 and grad.shape[-1] == 1:
        grad = grad[:, 0]
        lap = lap[0]
    return grad, lap


def molecular_local_energy(
    grad_log_psi: Tensor,
    lap_log_psi: Tensor,
    R: Tensor,
    r_flat: Tensor,
    charges: Tensor,
    n_e: int,
) -> Tensor:
    r"""``E_L = T_L + V_Coulomb`` from the derivatives of ``log|psi|``."""
    return local_energy(
        grad_log_psi, lap_log_psi, coulomb_potential(R, r_flat, charges, n_e)
    )


@dataclass(frozen=True)
class MolecularHamiltonian:
    """A fixed-nuclei electronic Hamiltonian ``H = T + V_Coulomb``.

    Parameters
    ----------
    charges
        Nuclear charges ``Z_a`` ``(n_atoms,)``.
    n_e
        Number of electrons.
    """

    charges: Tensor
    n_e: int

    def potential(self, R: Tensor, r_flat: Tensor) -> Tensor:
        """Bare Coulomb potential ``V(R, r)``."""
        return coulomb_potential(R, r_flat, self.charges, self.n_e)

    def local_energy(
        self, grad_log_psi: Tensor, lap_log_psi: Tensor, R: Tensor, r_flat: Tensor
    ) -> Tensor:
        """Local energy from the derivatives of ``log|psi|``."""
        return molecular_local_energy(
            grad_log_psi, lap_log_psi, R, r_flat, self.charges, self.n_e
        )


__all__ = [
    "MolecularHamiltonian",
    "coulomb_potential",
    "local_energy",
    "local_kinetic_energy",
    "log_psi_derivatives",
    "molecular_local_energy",
]
