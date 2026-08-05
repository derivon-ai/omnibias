# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Training losses for the two least-action methods (jax). Bit-identical twin."""

from __future__ import annotations

from typing import TYPE_CHECKING

from jax import Array
from omnibias.variational.jax.ops.action import action
from omnibias.variational.jax.ops.dynamics import acceleration
from omnibias.variational.jax.ops.euler_lagrange import euler_lagrange_residual
from omnibias.variational.jax.ops.field_theory import field_euler_lagrange_residual

if TYPE_CHECKING:  # pragma: no cover
    from omnibias.fields._core.quadrature import QuadratureSpec
    from omnibias.fields._core.state import FieldState
    from omnibias.variational._core.lagrangian import Lagrangian, LagrangianDensity


def _reduce(sq: Array, reduction: str) -> Array:
    if reduction == "mean":
        return sq.mean()
    if reduction == "sum":
        return sq.sum()
    raise ValueError(f"reduction must be 'mean' or 'sum', got {reduction!r}")


def action_minimization_loss(
    state: FieldState, lagrangian: Lagrangian, *, rule: QuadratureSpec,
) -> Array:
    r"""The action ``S`` as a loss for the direct (Ritz) method (a scalar)."""
    return action(state, lagrangian, rule=rule)


def euler_lagrange_loss(
    state: FieldState, lagrangian: Lagrangian, *, reduction: str = "mean",
) -> Array:
    r"""Mean/sum squared Euler-Lagrange residual (indirect PINN loss)."""
    res = euler_lagrange_residual(state, lagrangian)
    return _reduce(res**2, reduction)


def field_euler_lagrange_loss(
    state: FieldState, density: LagrangianDensity, *, reduction: str = "mean",
) -> Array:
    r"""Mean/sum squared field Euler-Lagrange residual (indirect PINN loss)."""
    res = field_euler_lagrange_residual(state, density)
    return _reduce(res**2, reduction)


def lagrangian_dynamics_loss(
    lagrangian: Lagrangian,
    q: Array,
    qdot: Array,
    qddot_target: Array,
    t: Array,
    *,
    reduction: str = "mean",
) -> Array:
    r"""Mean/sum squared acceleration error -- the Lagrangian Neural Network loss.

    ``|| acceleration(L; q, qdot, t) - qddot_target ||^2`` on ``(q, qdot, t)``
    state samples with observed accelerations ``qddot_target``.
    """
    pred = acceleration(lagrangian, q, qdot, t)
    return _reduce((pred - qddot_target) ** 2, reduction)


__all__ = [
    "action_minimization_loss",
    "euler_lagrange_loss",
    "field_euler_lagrange_loss",
    "lagrangian_dynamics_loss",
]
