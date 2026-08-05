# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Constrained variational calculus (jax).

Bit-identical twin of the torch module: holonomic constrained residual with
Lagrange-multiplier forces, and the augmented Lagrangian for isoperimetric
``int g dt = C`` constraints. The trajectory derivatives are closed form; the
Lagrangian / constraint partials are ``jax`` autodiff.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax.numpy as jnp
from jax import Array, jacrev, vmap
from omnibias.fields.jax.ops.basic import stack_components
from omnibias.variational._core.lagrangian import Lagrangian
from omnibias.variational.jax.ops.euler_lagrange import euler_lagrange_residual

if TYPE_CHECKING:  # pragma: no cover
    from omnibias.fields._core.state import FieldState
    from omnibias.variational._core.constraint import Constraint


def constrained_euler_lagrange_residual(
    state: FieldState,
    lagrangian: Lagrangian,
    constraint: Constraint,
    multipliers: Array,
) -> tuple[Array, Array]:
    r"""Holonomic constrained residual, ``(eom_residual, constraint_values)``.

    ``eom_residual`` (shape ``(B, n_dof)``) equals
    ``euler_lagrange_residual - sum_a lambda_a dg_a/dq``; ``constraint_values``
    (shape ``(B, n_c)``) equals ``g(q, t)``. Zero on a constrained solution.
    """
    el = euler_lagrange_residual(state, lagrangian)
    q = stack_components(state, lagrangian.dof)
    idx = state.coordinate_spec.axis_index(lagrangian.time_axis)
    t = state.coords[:, idx][:, None]
    g = constraint.fn(q, t)
    dg = vmap(jacrev(constraint.fn, argnums=0))(q, t)
    force = jnp.einsum("bc,bci->bi", multipliers, dg)
    return el - force, g


def augmented_lagrangian(
    lagrangian: Lagrangian, constraint: Lagrangian, multiplier: float
) -> Lagrangian:
    r"""Augmented Lagrangian ``L' = L - multiplier * g`` for ``int g dt = C``.

    ``constraint`` is the isoperimetric integrand as a
    :class:`~omnibias.variational.Lagrangian` (may depend on ``qdot``);
    ``multiplier`` is the constant Lagrange multiplier. Feed the result to
    ``euler_lagrange_residual``. ``lagrangian`` and ``constraint`` must share
    ``dof``, ``time_axis``, and ``order``.
    """
    if lagrangian.dof != constraint.dof:
        raise ValueError("lagrangian and constraint must share dof")
    if lagrangian.time_axis != constraint.time_axis:
        raise ValueError("lagrangian and constraint must share time_axis")
    if lagrangian.order != constraint.order:
        raise ValueError("lagrangian and constraint must share order")
    lfn, gfn = lagrangian.fn, constraint.fn

    def fn(*args: Array) -> Array:
        return lfn(*args) - multiplier * gfn(*args)

    return Lagrangian(
        fn, dof=lagrangian.dof, time_axis=lagrangian.time_axis, order=lagrangian.order
    )


__all__ = ["augmented_lagrangian", "constrained_euler_lagrange_residual"]
