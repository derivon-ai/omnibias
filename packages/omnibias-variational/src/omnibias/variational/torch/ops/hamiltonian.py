# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Hamiltonian / energy quantities of a Lagrangian trajectory (torch).

- the conjugate momentum ``p_i = dL/dqdot_i`` (Legendre variable),
- the energy function / Hamiltonian ``H = sum_i p_i qdot_i - L`` (the Jacobi
  integral; the conserved energy for an autonomous ``L``),
- the energy / Hamilton residual ``sum_i qdot_i EL_i = dH/dt + dL/dt``, which is
  zero exactly on a solution (energy conservation for autonomous systems).

The momentum and Hamiltonian use ``torch.func`` autodiff of the user Lagrangian
for its ``qdot`` gradient; the trajectory ``q``, ``qdot`` remain closed-form.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from omnibias.variational.torch.ops.euler_lagrange import (
    euler_lagrange_residual,
    trajectory,
)
from torch import Tensor
from torch.func import jacrev, vmap

if TYPE_CHECKING:  # pragma: no cover
    from omnibias.fields._core.state import FieldState
    from omnibias.variational._core.lagrangian import Lagrangian


def conjugate_momentum(state: FieldState, lagrangian: Lagrangian) -> Tensor:
    r"""Conjugate momentum ``p_i = dL/dqdot_i``, shape ``(B, n_dof)``."""
    q, qdot, _qddot, t = trajectory(state, lagrangian)
    return vmap(jacrev(lagrangian.fn, argnums=1))(q, qdot, t)


def hamiltonian(state: FieldState, lagrangian: Lagrangian) -> Tensor:
    r"""Energy function ``H = sum_i p_i qdot_i - L``, shape ``(B,)``.

    The Legendre transform evaluated along the trajectory (no inversion of the
    momentum map needed). For an autonomous ``L`` this is the conserved energy.
    """
    q, qdot, _qddot, t = trajectory(state, lagrangian)
    p = vmap(jacrev(lagrangian.fn, argnums=1))(q, qdot, t)
    lval = lagrangian.fn(q, qdot, t)
    return torch.einsum("bi,bi->b", p, qdot) - lval


def energy(state: FieldState, lagrangian: Lagrangian) -> Tensor:
    r"""Total mechanical energy along the trajectory (alias of :func:`hamiltonian`)."""
    return hamiltonian(state, lagrangian)


def hamiltons_equations_residual(state: FieldState, lagrangian: Lagrangian) -> Tensor:
    r"""Energy / Hamilton residual ``sum_i qdot_i EL_i = dH/dt + dL/dt``, ``(B,)``.

    A scalar consequence of Hamilton's equations: it vanishes exactly when the
    equations of motion hold, so on an autonomous solution it certifies
    ``dH/dt = 0`` (energy conservation). Off a solution it equals the
    ``qdot``-projection of the Euler-Lagrange residual.
    """
    _q, qdot, _qddot, _t = trajectory(state, lagrangian)
    el = euler_lagrange_residual(state, lagrangian)
    return torch.einsum("bi,bi->b", qdot, el)


__all__ = [
    "conjugate_momentum",
    "energy",
    "hamiltonian",
    "hamiltons_equations_residual",
]
