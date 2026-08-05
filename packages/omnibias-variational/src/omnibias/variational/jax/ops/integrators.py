# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Variational / symplectic integrators (jax). Bit-identical twin."""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax.numpy as jnp
from jax import Array, jacrev, vmap

if TYPE_CHECKING:  # pragma: no cover
    from collections.abc import Callable

    from omnibias.variational._core.lagrangian import Lagrangian


def _discrete_lagrangian(fn, dt: float):  # type: ignore[no-untyped-def]
    def ld(a: Array, b: Array) -> Array:
        qmid = 0.5 * (a + b)
        v = (b - a) / dt
        tzero = jnp.zeros(1, dtype=a.dtype)
        return dt * fn(qmid, v, tzero)

    return ld


def discrete_euler_lagrange_residual(
    q_prev: Array,
    q: Array,
    q_next: Array,
    *,
    lagrangian: Lagrangian,
    dt: float,
) -> Array:
    r"""Discrete Euler-Lagrange residual for a trajectory triple, shape ``(B, n_dof)``."""
    ld = _discrete_lagrangian(lagrangian.fn, dt)
    d1_next = vmap(jacrev(ld, argnums=0))(q, q_next)
    d2_prev = vmap(jacrev(ld, argnums=1))(q_prev, q)
    return d2_prev + d1_next


def stormer_verlet_step(
    q: Array,
    v: Array,
    *,
    grad_potential: Callable[[Array], Array],
    dt: float,
) -> tuple[Array, Array]:
    r"""One Stormer-Verlet (leapfrog) step for ``L = 1/2 |qdot|^2 - V(q)``."""
    v_half = v - 0.5 * dt * grad_potential(q)
    q_next = q + dt * v_half
    v_next = v_half - 0.5 * dt * grad_potential(q_next)
    return q_next, v_next


__all__ = ["discrete_euler_lagrange_residual", "stormer_verlet_step"]
