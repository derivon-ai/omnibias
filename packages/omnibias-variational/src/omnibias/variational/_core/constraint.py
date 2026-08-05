# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Backend-agnostic holonomic-constraint schema.

A :class:`Constraint` describes one or more *holonomic* constraints
``g_a(q, t) = 0`` on a trajectory by a callable ``fn(q, t) -> (..., n_c)`` plus
its count. Constrained stationarity of the action then reads

.. math::

    \\frac{d}{dt}\\frac{\\partial L}{\\partial\\dot q_i}-\\frac{\\partial L}{\\partial q_i}
    = \\sum_a \\lambda_a \\frac{\\partial g_a}{\\partial q_i},

with Lagrange multipliers ``lambda_a(t)`` -- see
``omnibias.variational.<backend>.ops.constrained_euler_lagrange_residual``.

Pure Python (no torch / jax), exactly like :class:`Lagrangian`. Velocity-
dependent (non-holonomic) and isoperimetric-integrand constraints are out of
scope here; the isoperimetric case is handled by ``augmented_lagrangian``.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

#: ``g(q, t)``: ``q`` of shape ``(..., n_dof)``, ``t`` of shape ``(..., 1)`` ->
#: constraint values of shape ``(..., n_constraints)``.
ConstraintFn = Callable[[Any, Any], Any]


@dataclass(frozen=True)
class Constraint:
    """One or more holonomic constraints ``g_a(q, t) = 0``.

    Parameters
    ----------
    fn
        Callable ``fn(q, t) -> g``. ``q`` has shape ``(..., n_dof)`` and ``t``
        shape ``(..., 1)``; the output holds the ``count`` constraint values with
        shape ``(..., count)``. Write it with a trailing constraint axis (e.g.
        ``stack`` the scalar constraints on ``-1``) so it is ``jacrev``-friendly.
    count
        Number of scalar constraints ``n_c`` (the size of ``fn``'s last axis).
    """

    fn: ConstraintFn
    count: int = 1

    def __post_init__(self) -> None:
        if not callable(self.fn):
            raise TypeError("Constraint.fn must be callable")
        if not isinstance(self.count, int) or isinstance(self.count, bool) or self.count < 1:
            raise ValueError(f"count must be an integer >= 1, got {self.count!r}")


__all__ = ["Constraint", "ConstraintFn"]
