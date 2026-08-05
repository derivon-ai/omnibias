# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Legendre transform: the Hamiltonian bridge (jax).

Bit-identical twin of the torch module (the Newton ``M^{-1}`` solve is each
backend's native primitive, so the two agree to ``rtol=1e-12`` in float64). See
:mod:`omnibias.variational.torch.ops.legendre` for the full derivation.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import jax.numpy as jnp
from jax import Array, jacrev, vmap
from omnibias.variational._core.hamiltonian import Hamiltonian
from omnibias.variational.jax.ops.dynamics import _require_first_order

if TYPE_CHECKING:  # pragma: no cover
    from omnibias.variational._core.lagrangian import Lagrangian, LagrangianFn


def _velocity_single(fn: LagrangianFn, q: Array, p: Array, t: Array, iters: int) -> Array:
    r"""Newton-invert ``p = dL/dqdot`` for a single sample ``(n,)`` -> ``qdot (n,)``."""
    dl_dv = jacrev(fn, argnums=1)
    qdot = jnp.zeros_like(p)
    for _ in range(iters):
        resid = p - dl_dv(q, qdot, t)
        m = jacrev(dl_dv, argnums=1)(q, qdot, t)
        qdot = qdot + jnp.linalg.solve(m, resid)
    return qdot


def _legendre_single(fn: LagrangianFn, q: Array, p: Array, t: Array, iters: int) -> Array:
    r"""Single-sample Legendre transform ``H = p . qdot(p) - L`` -> scalar."""
    qdot = _velocity_single(fn, q, p, t, iters)
    return cast(Array, (p * qdot).sum(-1) - fn(q, qdot, t))


def momentum(lagrangian: Lagrangian, q: Array, qdot: Array, t: Array) -> Array:
    r"""Conjugate momentum ``p = dL/dqdot`` at ``(q, qdot, t)``, shape ``(B, n)``."""
    _require_first_order(lagrangian)
    return cast(Array, vmap(jacrev(lagrangian.fn, argnums=1))(q, qdot, t))


def velocity_from_momentum(
    lagrangian: Lagrangian, q: Array, p: Array, t: Array, *, iters: int = 8,
) -> Array:
    r"""Invert ``p = dL/dqdot`` for ``qdot(q, p, t)`` by Newton iteration, ``(B, n)``."""
    _require_first_order(lagrangian)
    fn = lagrangian.fn
    return vmap(lambda qi, pi, ti: _velocity_single(fn, qi, pi, ti, iters))(q, p, t)


def legendre_transform(
    lagrangian: Lagrangian, q: Array, p: Array, t: Array, *, iters: int = 8,
) -> Array:
    r"""The Hamiltonian ``H(q, p, t) = p . qdot(p) - L(q, qdot(p), t)``, shape ``(B,)``."""
    _require_first_order(lagrangian)
    fn = lagrangian.fn
    return vmap(lambda qi, pi, ti: _legendre_single(fn, qi, pi, ti, iters))(q, p, t)


def hamiltonian_from_lagrangian(lagrangian: Lagrangian, *, iters: int = 8) -> Hamiltonian:
    r"""Build the :class:`Hamiltonian` Legendre-dual to ``lagrangian``."""
    _require_first_order(lagrangian)
    fn = lagrangian.fn

    def h_fn(q: Array, p: Array, t: Array) -> Array:
        return _legendre_single(fn, q, p, t, iters)

    return Hamiltonian(fn=h_fn, dof=lagrangian.dof, time_axis=lagrangian.time_axis)


def canonical_equations(
    hamiltonian: Hamiltonian, q: Array, p: Array, t: Array,
) -> tuple[Array, Array]:
    r"""Hamilton's equations ``(qdot, pdot) = (dH/dp, -dH/dq)``, each shape ``(B, n)``."""
    fn = hamiltonian.fn
    qdot = vmap(jacrev(fn, argnums=1))(q, p, t)
    pdot = -vmap(jacrev(fn, argnums=0))(q, p, t)
    return qdot, pdot


__all__ = [
    "canonical_equations",
    "hamiltonian_from_lagrangian",
    "legendre_transform",
    "momentum",
    "velocity_from_momentum",
]
