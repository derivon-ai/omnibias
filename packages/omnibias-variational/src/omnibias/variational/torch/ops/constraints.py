# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Constrained variational calculus (torch).

Two flavours of constrained stationarity:

* **Holonomic** ``g_a(q, t) = 0``. Constrained Hamilton's principle adds a
  Lagrange-multiplier force to the Euler-Lagrange equation:

  .. math::

      \frac{d}{dt}\frac{\partial L}{\partial\dot q_i}-\frac{\partial L}{\partial q_i}
        = \sum_a \lambda_a\,\frac{\partial g_a}{\partial q_i},

  so ``constrained_euler_lagrange_residual`` returns
  ``euler_lagrange_residual - sum_a lambda_a dg_a/dq`` (the constraint gradient
  is autodiff of the pure-Python :class:`~omnibias.variational.Constraint`) along
  with the constraint values ``g(q, t)``.

* **Isoperimetric** ``int g\,dt = C``. Stationarity of the constrained action is
  stationarity of the *augmented* action ``int (L - lambda g)\,dt`` for a constant
  multiplier ``lambda``, so ``augmented_lagrangian`` returns the combined
  :class:`~omnibias.variational.Lagrangian` ``L' = L - lambda g`` (here ``g`` is
  itself a Lagrangian-shaped integrand, and may depend on ``qdot`` -- e.g. arc
  length); feed it back to ``euler_lagrange_residual``.

Honesty: the trajectory derivatives ``q, qdot, qddot`` are closed form; the
Lagrangian / constraint partials are ``torch.func`` autodiff.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from omnibias.fields.torch.ops.basic import stack_components
from omnibias.variational._core.lagrangian import Lagrangian
from omnibias.variational.torch.ops.euler_lagrange import euler_lagrange_residual
from torch import Tensor
from torch.func import jacrev, vmap

if TYPE_CHECKING:  # pragma: no cover
    from omnibias.fields._core.state import FieldState
    from omnibias.variational._core.constraint import Constraint


def constrained_euler_lagrange_residual(
    state: FieldState,
    lagrangian: Lagrangian,
    constraint: Constraint,
    multipliers: Tensor,
) -> tuple[Tensor, Tensor]:
    r"""Holonomic constrained residual, ``(eom_residual, constraint_values)``.

    ``eom_residual`` has shape ``(B, n_dof)`` and equals
    ``euler_lagrange_residual - sum_a lambda_a dg_a/dq``; ``constraint_values``
    has shape ``(B, n_c)`` and equals ``g(q, t)``. Both are zero (to machine
    precision) exactly on a constrained solution with the given
    ``multipliers`` (shape ``(B, n_c)``, the Lagrange multipliers ``lambda_a(t)``).
    """
    el = euler_lagrange_residual(state, lagrangian)  # (B, n_dof)
    q = stack_components(state, lagrangian.dof)  # (B, n_dof)
    idx = state.coordinate_spec.axis_index(lagrangian.time_axis)
    t = state.coords[:, idx].unsqueeze(-1)  # (B, 1)
    g = constraint.fn(q, t)  # (B, n_c)
    dg = vmap(jacrev(constraint.fn, argnums=0))(q, t)  # (B, n_c, n_dof)
    force = torch.einsum("bc,bci->bi", multipliers, dg)
    return el - force, g


def augmented_lagrangian(
    lagrangian: Lagrangian, constraint: Lagrangian, multiplier: float
) -> Lagrangian:
    r"""Augmented Lagrangian ``L' = L - multiplier * g`` for ``int g dt = C``.

    ``constraint`` is the isoperimetric *integrand* as a
    :class:`~omnibias.variational.Lagrangian` (so it may depend on ``qdot``, e.g.
    arc length ``sqrt(1 + qdot^2)``); ``multiplier`` is the constant Lagrange
    multiplier ``lambda``. The stationary trajectories of the returned Lagrangian
    are the isoperimetric extremals -- feed it to ``euler_lagrange_residual``.

    ``lagrangian`` and ``constraint`` must share ``dof``, ``time_axis``, and
    ``order``.
    """
    if lagrangian.dof != constraint.dof:
        raise ValueError("lagrangian and constraint must share dof")
    if lagrangian.time_axis != constraint.time_axis:
        raise ValueError("lagrangian and constraint must share time_axis")
    if lagrangian.order != constraint.order:
        raise ValueError("lagrangian and constraint must share order")
    lfn, gfn = lagrangian.fn, constraint.fn

    def fn(*args: Tensor) -> Tensor:
        return lfn(*args) - multiplier * gfn(*args)

    return Lagrangian(
        fn, dof=lagrangian.dof, time_axis=lagrangian.time_axis, order=lagrangian.order
    )


__all__ = ["augmented_lagrangian", "constrained_euler_lagrange_residual"]
