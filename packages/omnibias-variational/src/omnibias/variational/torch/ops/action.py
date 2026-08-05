# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""The action integral ``S = integral L(q, qdot, t) dt`` (torch).

The trajectory ``q(t)`` is an omnibias field, so ``q`` and ``qdot`` are the
*closed-form* value / derivative of the field at the quadrature nodes; the
Lagrangian ``L`` is the user callable; the integral is a *quadrature* of ``L``
over the rule's nodes. Minimising this scalar w.r.t. the field parameters is the
direct (Ritz) form of the least-action principle.

Evaluate the field at the rule's nodes first::

    rule = gauss_legendre([(0.0, T)], 64)
    state = field(quadrature_nodes(rule, like=coords))
    S = action(state, lagrangian, rule=rule)      # scalar tensor
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from omnibias.fields._core.quadrature import QuadratureSpec
from omnibias.fields.torch.ops.basic import stack_components, vector_derivative
from torch import Tensor

if TYPE_CHECKING:  # pragma: no cover
    from omnibias.fields._core.state import FieldState
    from omnibias.variational._core.lagrangian import Lagrangian


def integrate_values(values: Tensor, *, rule: QuadratureSpec) -> Tensor:
    r"""Quadrature integral of a raw ``(n_nodes, ...)`` integrand tensor.

    Contracts the batch (node) axis with the rule's weights,
    :math:`\int_\Omega f\,dx \approx \sum_q w_q\,f(x_q)`. ``values`` must have
    been produced at ``rule``'s nodes (``values.shape[0] == rule.n_nodes``).
    This is the field-agnostic companion of
    :func:`omnibias.fields.torch.ops.integral.integrate`, which only integrates a
    *named* field component; a Lagrangian is an on-the-fly tensor.
    """
    if values.shape[0] != rule.n_nodes:
        raise ValueError(
            f"integrate_values: integrand has {values.shape[0]} points but rule "
            f"has {rule.n_nodes} nodes; evaluate the field at quadrature_nodes(rule)"
        )
    w = torch.as_tensor(rule.weights, dtype=values.dtype, device=values.device)
    return torch.tensordot(w, values, dims=([0], [0]))


def _time_column(state: FieldState, time_axis: str) -> Tensor:
    idx = state.coordinate_spec.axis_index(time_axis)
    return state.coords[:, idx].unsqueeze(-1)


def lagrangian_values(state: FieldState, lagrangian: Lagrangian) -> Tensor:
    r"""Evaluate ``L(q(t), qdot(t), t)`` along the trajectory, shape ``(B,)``.

    ``q`` is the closed-form value of the ``dof`` components and ``qdot`` their
    closed-form first time-derivative.
    """
    q = stack_components(state, lagrangian.dof)
    qdot = vector_derivative(state, lagrangian.dof, axis=lagrangian.time_axis, order=1)
    t = _time_column(state, lagrangian.time_axis)
    return lagrangian.fn(q, qdot, t)


def action(state: FieldState, lagrangian: Lagrangian, *, rule: QuadratureSpec) -> Tensor:
    r"""The action :math:`S = \int L(q,\dot q,t)\,dt`, a scalar tensor.

    Parameters
    ----------
    state
        Trajectory field state evaluated at the quadrature ``rule``'s nodes.
    lagrangian
        The :class:`~omnibias.variational.Lagrangian`.
    rule
        Quadrature rule over the time interval (typically 1-D Gauss-Legendre).
    """
    return integrate_values(lagrangian_values(state, lagrangian), rule=rule)


__all__ = ["action", "integrate_values", "lagrangian_values"]
