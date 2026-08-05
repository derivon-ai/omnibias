# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""The Euler-Lagrange operator (torch).

For a trajectory ``q(t)`` the stationary-action condition ``delta S = 0`` gives

.. math::

    \mathrm{EL}_i = \frac{d}{dt}\frac{\partial L}{\partial\dot q_i}
                    - \frac{\partial L}{\partial q_i}.

We expand the total time-derivative by the chain rule so the *outer* ``d/dt``
rides on the **closed-form** trajectory derivatives ``qdot``, ``qddot``:

.. math::

    \mathrm{EL}_i = \sum_j \frac{\partial^2 L}{\partial\dot q_i\partial q_j}\dot q_j
       + \sum_j \frac{\partial^2 L}{\partial\dot q_i\partial\dot q_j}\ddot q_j
       + \frac{\partial^2 L}{\partial\dot q_i\partial t}
       - \frac{\partial L}{\partial q_i}.

``qdot`` (order 1) and ``qddot`` (order 2) are the sigma-tower closed-form time
derivatives of the field; the Lagrangian's own gradient / Hessian in the
``(q, qdot, t)`` arguments are ``torch.func`` autodiff of the user callable.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from omnibias.fields.torch.ops.basic import stack_components, vector_derivative
from torch import Tensor
from torch.func import jacrev, vmap

if TYPE_CHECKING:  # pragma: no cover
    from omnibias.fields._core.state import FieldState
    from omnibias.variational._core.lagrangian import Lagrangian


def trajectory(state: FieldState, lagrangian: Lagrangian) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    r"""Closed-form ``(q, qdot, qddot, t)`` along the trajectory, each ``(B, n_dof)``.

    ``t`` has shape ``(B, 1)``. ``q`` is the field value; ``qdot`` / ``qddot`` are
    the closed-form first / second time-derivatives.
    """
    dof = lagrangian.dof
    ax = lagrangian.time_axis
    q = stack_components(state, dof)
    qdot = vector_derivative(state, dof, axis=ax, order=1)
    qddot = vector_derivative(state, dof, axis=ax, order=2)
    idx = state.coordinate_spec.axis_index(ax)
    t = state.coords[:, idx].unsqueeze(-1)
    return q, qdot, qddot, t


def lagrangian_partials(
    lagrangian: Lagrangian, q: Tensor, qdot: Tensor, t: Tensor,
) -> tuple[Tensor, Tensor]:
    r"""Autodiff partials ``(dL/dq, dL/dqdot)``, each shape ``(B, n_dof)``."""
    fn = lagrangian.fn
    g_q = vmap(jacrev(fn, argnums=0))(q, qdot, t)
    g_v = vmap(jacrev(fn, argnums=1))(q, qdot, t)
    return g_q, g_v


def euler_lagrange_residual(state: FieldState, lagrangian: Lagrangian) -> Tensor:
    r"""Euler-Lagrange residual ``d/dt(dL/dqdot) - dL/dq``, shape ``(B, n_dof)``.

    Zero (to machine precision) exactly on a solution of the equations of motion.
    For a higher-order Lagrangian (``order >= 2``) this is the Euler-Poisson
    residual ``-functional_derivative`` (see
    :func:`omnibias.variational.torch.ops.functional.functional_derivative`).
    """
    if lagrangian.order != 1:
        from omnibias.variational.torch.ops.functional import functional_derivative

        return -functional_derivative(state, lagrangian)
    q, qdot, qddot, t = trajectory(state, lagrangian)
    fn = lagrangian.fn
    g_q = vmap(jacrev(fn, argnums=0))(q, qdot, t)          # (B, n)
    dl_dv = jacrev(fn, argnums=1)
    h_vq = vmap(jacrev(dl_dv, argnums=0))(q, qdot, t)      # (B, n, n)
    h_vv = vmap(jacrev(dl_dv, argnums=1))(q, qdot, t)      # (B, n, n)
    h_vt = vmap(jacrev(dl_dv, argnums=2))(q, qdot, t)      # (B, n, 1)
    dp_dt = (
        torch.einsum("bij,bj->bi", h_vq, qdot)
        + torch.einsum("bij,bj->bi", h_vv, qddot)
        + h_vt[..., 0]
    )
    return dp_dt - g_q


__all__ = ["euler_lagrange_residual", "lagrangian_partials", "trajectory"]
