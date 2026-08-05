# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Forward Lagrangian dynamics -- the equations of motion (jax).

Bit-identical twin of the torch module (the ``M^{-1}F`` solve is each backend's
native primitive, so the two agree to ``rtol=1e-12`` in float64). See
:mod:`omnibias.variational.torch.ops.dynamics` for the full derivation.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import jax.numpy as jnp
from jax import Array, jacrev, vmap

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
    lagrangian: Lagrangian, q: Array, qdot: Array, t: Array,
) -> tuple[Array, Array, Array, Array]:
    r"""Autodiff terms ``(g_q, h_vq, h_vv, h_vt)`` of ``L`` at ``(q, qdot, t)``."""
    fn = lagrangian.fn
    g_q = vmap(jacrev(fn, argnums=0))(q, qdot, t)
    dl_dv = jacrev(fn, argnums=1)
    h_vq = vmap(jacrev(dl_dv, argnums=0))(q, qdot, t)
    h_vv = vmap(jacrev(dl_dv, argnums=1))(q, qdot, t)
    h_vt = vmap(jacrev(dl_dv, argnums=2))(q, qdot, t)
    return g_q, h_vq, h_vv, h_vt


def _force(g_q: Array, h_vq: Array, h_vt: Array, qdot: Array) -> Array:
    return g_q - jnp.einsum("bij,bj->bi", h_vq, qdot) - h_vt[..., 0]


def mass_matrix(lagrangian: Lagrangian, q: Array, qdot: Array, t: Array) -> Array:
    r"""Generalized mass (velocity Hessian) ``M = d2L/dqdot^2``, shape ``(B, n, n)``."""
    _require_first_order(lagrangian)
    dl_dv = jacrev(lagrangian.fn, argnums=1)
    return cast(Array, vmap(jacrev(dl_dv, argnums=1))(q, qdot, t))


def generalized_force(lagrangian: Lagrangian, q: Array, qdot: Array, t: Array) -> Array:
    r"""Generalized force ``F = dL/dq - (d2L/dqdot dq) qdot - d2L/dqdot dt``, ``(B, n)``."""
    _require_first_order(lagrangian)
    g_q, h_vq, _h_vv, h_vt = _dynamics_terms(lagrangian, q, qdot, t)
    return _force(g_q, h_vq, h_vt, qdot)


def acceleration(lagrangian: Lagrangian, q: Array, qdot: Array, t: Array) -> Array:
    r"""Acceleration ``qddot = M^{-1} F`` implied by the Lagrangian, shape ``(B, n)``."""
    _require_first_order(lagrangian)
    g_q, h_vq, h_vv, h_vt = _dynamics_terms(lagrangian, q, qdot, t)
    force = _force(g_q, h_vq, h_vt, qdot)
    return cast(Array, jnp.linalg.solve(h_vv, force[..., None])[..., 0])


def dynamics_rhs(
    lagrangian: Lagrangian, q: Array, qdot: Array, t: Array,
) -> tuple[Array, Array]:
    r"""Second-order ODE right-hand side ``(qdot, qddot)`` for rollouts, each ``(B, n)``."""
    return qdot, acceleration(lagrangian, q, qdot, t)


def inverse_dynamics(
    lagrangian: Lagrangian, q: Array, qdot: Array, qddot: Array, t: Array,
) -> Array:
    r"""Applied generalized force ``tau = M qddot - F`` realising ``qddot``, ``(B, n)``."""
    _require_first_order(lagrangian)
    g_q, h_vq, h_vv, h_vt = _dynamics_terms(lagrangian, q, qdot, t)
    return (
        jnp.einsum("bij,bj->bi", h_vv, qddot)
        + jnp.einsum("bij,bj->bi", h_vq, qdot)
        + h_vt[..., 0]
        - g_q
    )


def predicted_acceleration(state: FieldState, lagrangian: Lagrangian) -> Array:
    r"""The Lagrangian's acceleration at a trajectory's ``(q, qdot, t)``, ``(B, n)``."""
    from omnibias.variational.jax.ops.euler_lagrange import trajectory

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
