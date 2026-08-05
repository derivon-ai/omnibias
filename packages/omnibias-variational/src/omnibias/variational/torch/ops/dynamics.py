# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Forward Lagrangian dynamics -- the equations of motion (torch).

For a Lagrangian ``L(q, qdot, t)`` the Euler-Lagrange equations
``d/dt(dL/dqdot) - dL/dq = 0`` expand (chain rule on the outer ``d/dt``) into

.. math::

    M(q,\dot q,t)\,\ddot q = F(q,\dot q,t),\qquad
    M = \frac{\partial^2 L}{\partial\dot q\,\partial\dot q},\quad
    F = \frac{\partial L}{\partial q}
        - \frac{\partial^2 L}{\partial\dot q\,\partial q}\,\dot q
        - \frac{\partial^2 L}{\partial\dot q\,\partial t}.

Solving for the acceleration ``qddot = M^{-1} F`` turns a Lagrangian into its
equations of motion -- the forward map behind **Lagrangian Neural Networks**
(learn ``L`` so its predicted ``qddot`` matches data, energy-conserving by
construction). It is the dual of
:func:`omnibias.variational.torch.ops.euler_lagrange.euler_lagrange_residual`,
which instead *substitutes* a trajectory's closed-form ``qddot`` to test whether
the equations hold.

These ops are **array-level**: they take ``(q, qdot, t)`` state samples (``q`` /
``qdot`` of shape ``(B, n_dof)``, ``t`` of shape ``(B, 1)``) -- the natural LNN
interface (state points, not necessarily a ``q(t)`` field). The Lagrangian's
partials are ``torch.func`` autodiff of the user callable; the ``M^{-1}F`` solve
is an exact linear solve (``M`` must be invertible -- positive definite for a
physical Lagrangian). Only ``order == 1`` Lagrangians are supported.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import torch
from torch import Tensor
from torch.func import jacrev, vmap

if TYPE_CHECKING:  # pragma: no cover
    from omnibias.fields._core.state import FieldState
    from omnibias.variational._core.lagrangian import Lagrangian


def _require_first_order(lagrangian: Lagrangian) -> None:
    if lagrangian.order != 1:
        raise NotImplementedError(
            "forward Lagrangian dynamics is implemented for order == 1 Lagrangians "
            f"only; got order = {lagrangian.order}. Use functional_derivative / "
            "euler_lagrange_residual for the higher-order (Euler-Poisson) residual."
        )


def _dynamics_terms(
    lagrangian: Lagrangian, q: Tensor, qdot: Tensor, t: Tensor,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    r"""Autodiff terms ``(g_q, h_vq, h_vv, h_vt)`` of ``L`` at ``(q, qdot, t)``.

    ``g_q = dL/dq`` ``(B, n)``; ``h_vq = d2L/dqdot dq`` ``(B, n, n)``;
    ``h_vv = d2L/dqdot dqdot`` ``(B, n, n)``; ``h_vt = d2L/dqdot dt`` ``(B, n, 1)``.
    """
    fn = lagrangian.fn
    g_q = vmap(jacrev(fn, argnums=0))(q, qdot, t)
    dl_dv = jacrev(fn, argnums=1)
    h_vq = vmap(jacrev(dl_dv, argnums=0))(q, qdot, t)
    h_vv = vmap(jacrev(dl_dv, argnums=1))(q, qdot, t)
    h_vt = vmap(jacrev(dl_dv, argnums=2))(q, qdot, t)
    return g_q, h_vq, h_vv, h_vt


def _force(g_q: Tensor, h_vq: Tensor, h_vt: Tensor, qdot: Tensor) -> Tensor:
    return g_q - torch.einsum("bij,bj->bi", h_vq, qdot) - h_vt[..., 0]


def mass_matrix(lagrangian: Lagrangian, q: Tensor, qdot: Tensor, t: Tensor) -> Tensor:
    r"""Generalized mass (velocity Hessian) ``M = d2L/dqdot^2``, shape ``(B, n, n)``."""
    _require_first_order(lagrangian)
    dl_dv = jacrev(lagrangian.fn, argnums=1)
    return cast(Tensor, vmap(jacrev(dl_dv, argnums=1))(q, qdot, t))


def generalized_force(lagrangian: Lagrangian, q: Tensor, qdot: Tensor, t: Tensor) -> Tensor:
    r"""Generalized force ``F = dL/dq - (d2L/dqdot dq) qdot - d2L/dqdot dt``, ``(B, n)``.

    The right-hand side of ``M qddot = F``.
    """
    _require_first_order(lagrangian)
    g_q, h_vq, _h_vv, h_vt = _dynamics_terms(lagrangian, q, qdot, t)
    return _force(g_q, h_vq, h_vt, qdot)


def acceleration(lagrangian: Lagrangian, q: Tensor, qdot: Tensor, t: Tensor) -> Tensor:
    r"""Acceleration ``qddot = M^{-1} F`` implied by the Lagrangian, shape ``(B, n)``.

    The forward equations of motion (the Lagrangian Neural Network map). ``M``
    must be invertible (positive definite for a physical Lagrangian).
    """
    _require_first_order(lagrangian)
    g_q, h_vq, h_vv, h_vt = _dynamics_terms(lagrangian, q, qdot, t)
    force = _force(g_q, h_vq, h_vt, qdot)
    return cast(Tensor, torch.linalg.solve(h_vv, force.unsqueeze(-1)).squeeze(-1))


def dynamics_rhs(
    lagrangian: Lagrangian, q: Tensor, qdot: Tensor, t: Tensor,
) -> tuple[Tensor, Tensor]:
    r"""Second-order ODE right-hand side ``(qdot, qddot)`` for rollouts, each ``(B, n)``."""
    return qdot, acceleration(lagrangian, q, qdot, t)


def inverse_dynamics(
    lagrangian: Lagrangian, q: Tensor, qdot: Tensor, qddot: Tensor, t: Tensor,
) -> Tensor:
    r"""Applied generalized force ``tau = M qddot - F`` realising ``qddot``, ``(B, n)``.

    Robotics inverse dynamics; identically the Euler-Lagrange residual evaluated
    with the supplied ``qddot`` (zero exactly when ``qddot`` solves the equations
    of motion, so ``inverse_dynamics(L, q, qdot, acceleration(L, q, qdot, t), t)``
    is zero to machine precision).
    """
    _require_first_order(lagrangian)
    g_q, h_vq, h_vv, h_vt = _dynamics_terms(lagrangian, q, qdot, t)
    return (
        torch.einsum("bij,bj->bi", h_vv, qddot)
        + torch.einsum("bij,bj->bi", h_vq, qdot)
        + h_vt[..., 0]
        - g_q
    )


def predicted_acceleration(state: FieldState, lagrangian: Lagrangian) -> Tensor:
    r"""The Lagrangian's acceleration at a trajectory's ``(q, qdot, t)``, ``(B, n)``.

    A ``FieldState`` convenience wrapper: pulls the closed-form ``(q, qdot, t)``
    off the trajectory and returns :func:`acceleration`. On a true solution it
    equals the field's own closed-form ``qddot``, so
    ``predicted_acceleration(state, L) - trajectory(state, L)[2]`` is the
    equations-of-motion error.
    """
    from omnibias.variational.torch.ops.euler_lagrange import trajectory

    q, qdot, _qddot, t = trajectory(state, lagrangian)
    return acceleration(lagrangian, q, qdot, t)


__all__ = [
    "acceleration",
    "dynamics_rhs",
    "generalized_force",
    "inverse_dynamics",
    "mass_matrix",
    "predicted_acceleration",
]
