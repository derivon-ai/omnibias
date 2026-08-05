# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Training losses for the two least-action methods (torch).

- **Direct (Ritz) method**: minimise the action itself. Parameterise the
  trajectory / field by a network, evaluate :func:`action_minimization_loss`,
  and call ``.backward()``.
- **Indirect method**: drive the Euler-Lagrange residual to zero as a PINN loss
  (:func:`euler_lagrange_loss`, :func:`field_euler_lagrange_loss`).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from omnibias.variational.torch.ops.action import action
from omnibias.variational.torch.ops.dynamics import acceleration
from omnibias.variational.torch.ops.euler_lagrange import euler_lagrange_residual
from omnibias.variational.torch.ops.field_theory import field_euler_lagrange_residual
from torch import Tensor

if TYPE_CHECKING:  # pragma: no cover
    from omnibias.fields._core.quadrature import QuadratureSpec
    from omnibias.fields._core.state import FieldState
    from omnibias.variational._core.lagrangian import Lagrangian, LagrangianDensity


def _reduce(sq: Tensor, reduction: str) -> Tensor:
    if reduction == "mean":
        return sq.mean()
    if reduction == "sum":
        return sq.sum()
    raise ValueError(f"reduction must be 'mean' or 'sum', got {reduction!r}")


def action_minimization_loss(
    state: FieldState, lagrangian: Lagrangian, *, rule: QuadratureSpec,
) -> Tensor:
    r"""The action ``S`` as a loss for the direct (Ritz) method (a scalar)."""
    return action(state, lagrangian, rule=rule)


def euler_lagrange_loss(
    state: FieldState, lagrangian: Lagrangian, *, reduction: str = "mean",
) -> Tensor:
    r"""Mean/sum squared Euler-Lagrange residual (indirect PINN loss)."""
    res = euler_lagrange_residual(state, lagrangian)
    return _reduce(res**2, reduction)


def field_euler_lagrange_loss(
    state: FieldState, density: LagrangianDensity, *, reduction: str = "mean",
) -> Tensor:
    r"""Mean/sum squared field Euler-Lagrange residual (indirect PINN loss)."""
    res = field_euler_lagrange_residual(state, density)
    return _reduce(res**2, reduction)


def lagrangian_dynamics_loss(
    lagrangian: Lagrangian,
    q: Tensor,
    qdot: Tensor,
    qddot_target: Tensor,
    t: Tensor,
    *,
    reduction: str = "mean",
) -> Tensor:
    r"""Mean/sum squared acceleration error -- the Lagrangian Neural Network loss.

    ``|| acceleration(L; q, qdot, t) - qddot_target ||^2`` on ``(q, qdot, t)``
    state samples with observed accelerations ``qddot_target`` (each ``(B, n)``,
    ``t`` is ``(B, 1)``): fit a Lagrangian ``L`` whose forward dynamics reproduce
    the data. Differentiable w.r.t. the parameters of ``L`` (an omnibias field /
    a network closed over by ``lagrangian.fn``).
    """
    pred = acceleration(lagrangian, q, qdot, t)
    return _reduce((pred - qddot_target) ** 2, reduction)


__all__ = [
    "action_minimization_loss",
    "euler_lagrange_loss",
    "field_euler_lagrange_loss",
    "lagrangian_dynamics_loss",
]
