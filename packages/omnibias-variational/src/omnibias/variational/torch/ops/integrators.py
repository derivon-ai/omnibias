# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Variational / symplectic integrators (torch).

Discretising the *action* rather than the equations of motion yields
structure-preserving integrators. With the midpoint discrete Lagrangian

.. math::

    L_d(q_k, q_{k+1}) = \Delta t\;
        L\!\Big(\tfrac{q_k+q_{k+1}}2, \tfrac{q_{k+1}-q_k}{\Delta t}\Big),

the discrete Euler-Lagrange (DEL) equations

.. math::

    D_2 L_d(q_{k-1}, q_k) + D_1 L_d(q_k, q_{k+1}) = 0

define the trajectory; :func:`discrete_euler_lagrange_residual` returns their
left-hand side. For a separable ``L = 1/2 |qdot|^2 - V(q)`` the DEL map is the
explicit :func:`stormer_verlet_step` (leapfrog), whose energy error stays
bounded over exponentially long times -- unlike a non-symplectic method.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch import Tensor
from torch.func import jacrev, vmap

if TYPE_CHECKING:  # pragma: no cover
    from collections.abc import Callable

    from omnibias.variational._core.lagrangian import Lagrangian


def _discrete_lagrangian(fn, dt: float):  # type: ignore[no-untyped-def]
    def ld(a: Tensor, b: Tensor) -> Tensor:
        qmid = 0.5 * (a + b)
        v = (b - a) / dt
        tzero = torch.zeros(1, dtype=a.dtype, device=a.device)
        return dt * fn(qmid, v, tzero)

    return ld


def discrete_euler_lagrange_residual(
    q_prev: Tensor,
    q: Tensor,
    q_next: Tensor,
    *,
    lagrangian: Lagrangian,
    dt: float,
) -> Tensor:
    r"""Discrete Euler-Lagrange residual for a trajectory triple, shape ``(B, n_dof)``.

    ``D_2 L_d(q_prev, q) + D_1 L_d(q, q_next)``; zero on a discrete solution. The
    three positions are ``(B, n_dof)`` tensors at times ``t - dt``, ``t``,
    ``t + dt``.
    """
    ld = _discrete_lagrangian(lagrangian.fn, dt)
    d1_next = vmap(jacrev(ld, argnums=0))(q, q_next)
    d2_prev = vmap(jacrev(ld, argnums=1))(q_prev, q)
    return d2_prev + d1_next


def stormer_verlet_step(
    q: Tensor,
    v: Tensor,
    *,
    grad_potential: Callable[[Tensor], Tensor],
    dt: float,
) -> tuple[Tensor, Tensor]:
    r"""One Stormer-Verlet (leapfrog) step for ``L = 1/2 |qdot|^2 - V(q)``.

    Parameters
    ----------
    q, v
        Position and velocity, shape ``(B, n_dof)`` (unit mass, so ``v`` is the
        momentum).
    grad_potential
        Callable ``grad V(q) -> (B, n_dof)``.
    dt
        Time step.

    Returns
    -------
    tuple[Tensor, Tensor]
        The updated ``(q_next, v_next)``. The map is symplectic.
    """
    v_half = v - 0.5 * dt * grad_potential(q)
    q_next = q + dt * v_half
    v_next = v_half - 0.5 * dt * grad_potential(q_next)
    return q_next, v_next


__all__ = ["discrete_euler_lagrange_residual", "stormer_verlet_step"]
