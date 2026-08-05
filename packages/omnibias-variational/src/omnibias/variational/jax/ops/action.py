# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""The action integral ``S = integral L(q, qdot, t) dt`` (jax).

Bit-identical twin of the torch module.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax.numpy as jnp
from jax import Array
from omnibias.fields._core.quadrature import QuadratureSpec
from omnibias.fields.jax.ops.basic import stack_components, vector_derivative

if TYPE_CHECKING:  # pragma: no cover
    from omnibias.fields._core.state import FieldState
    from omnibias.variational._core.lagrangian import Lagrangian


def integrate_values(values: Array, *, rule: QuadratureSpec) -> Array:
    r"""Quadrature integral of a raw ``(n_nodes, ...)`` integrand tensor."""
    if values.shape[0] != rule.n_nodes:
        raise ValueError(
            f"integrate_values: integrand has {values.shape[0]} points but rule "
            f"has {rule.n_nodes} nodes; evaluate the field at quadrature_nodes(rule)"
        )
    w = jnp.asarray(rule.weights, dtype=values.dtype)
    return jnp.tensordot(w, values, axes=([0], [0]))


def _time_column(state: FieldState, time_axis: str) -> Array:
    idx = state.coordinate_spec.axis_index(time_axis)
    return state.coords[:, idx][:, None]


def lagrangian_values(state: FieldState, lagrangian: Lagrangian) -> Array:
    r"""Evaluate ``L(q(t), qdot(t), t)`` along the trajectory, shape ``(B,)``."""
    q = stack_components(state, lagrangian.dof)
    qdot = vector_derivative(state, lagrangian.dof, axis=lagrangian.time_axis, order=1)
    t = _time_column(state, lagrangian.time_axis)
    return lagrangian.fn(q, qdot, t)


def action(state: FieldState, lagrangian: Lagrangian, *, rule: QuadratureSpec) -> Array:
    r"""The action :math:`S = \int L(q,\dot q,t)\,dt`, a scalar array."""
    return integrate_values(lagrangian_values(state, lagrangian), rule=rule)


__all__ = ["action", "integrate_values", "lagrangian_values"]
