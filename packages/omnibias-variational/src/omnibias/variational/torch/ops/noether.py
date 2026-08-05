# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Noether charges of a continuous symmetry (torch).

For an infinitesimal symmetry ``q_i -> q_i + eps * X_i(q, qdot, t)`` that leaves
the action invariant (no boundary term), Noether's theorem gives the conserved
charge

.. math::

    Q = \sum_i \frac{\partial L}{\partial\dot q_i}\,X_i = p\cdot X.

Time-translation invariance instead conserves the energy (see
:func:`omnibias.variational.torch.ops.hamiltonian.hamiltonian`).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from omnibias.variational.torch.ops.euler_lagrange import trajectory
from omnibias.variational.torch.ops.hamiltonian import conjugate_momentum
from torch import Tensor

if TYPE_CHECKING:  # pragma: no cover
    from collections.abc import Callable

    from omnibias.fields._core.state import FieldState
    from omnibias.variational._core.lagrangian import Lagrangian


def noether_charge(
    state: FieldState,
    lagrangian: Lagrangian,
    generator: Tensor | Callable[[Tensor, Tensor, Tensor], Tensor],
) -> Tensor:
    r"""Conserved charge ``Q = p . X`` of the symmetry ``generator`` ``X``, ``(B,)``.

    Parameters
    ----------
    state, lagrangian
        The trajectory and its Lagrangian.
    generator
        The symmetry direction ``X_i`` -- either a ``(B, n_dof)`` tensor or a
        callable ``X(q, qdot, t) -> (B, n_dof)`` (e.g. translations ``X = 1`` or
        a rotation ``X = (-q_y, q_x)``).
    """
    q, qdot, _qddot, t = trajectory(state, lagrangian)
    p = conjugate_momentum(state, lagrangian)
    x = generator(q, qdot, t) if callable(generator) else generator
    return torch.einsum("bi,bi->b", p, x)


__all__ = ["noether_charge"]
