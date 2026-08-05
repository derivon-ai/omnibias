# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Legendre transform: the Hamiltonian bridge (torch).

The Legendre transform sends a Lagrangian ``L(q, qdot, t)`` to the phase-space
Hamiltonian ``H(q, p, t)``. The conjugate momentum is ``p = dL/dqdot``; inverting
this relation for ``qdot(q, p, t)`` (a Newton solve whose Jacobian is the mass
matrix ``M = d2L/dqdot^2``) gives

.. math::

    H(q, p, t) = p \cdot \dot q(p) - L\big(q, \dot q(p), t\big),

and Hamilton's canonical equations ``qdot = dH/dp``, ``pdot = -dH/dq`` generate
the same dynamics as the Lagrangian forward map, in phase-space form. Unlike
:func:`omnibias.variational.torch.ops.hamiltonian.hamiltonian` (which evaluates
the energy *along a supplied trajectory* using its ``qdot``), this is the genuine
``H(q, p, t)`` on phase space.

Array-level ops (``q``, ``p``, ``qdot`` of shape ``(B, n_dof)``, ``t`` of shape
``(B, 1)``). The partials are ``torch.func`` autodiff of the user callable;
``velocity_from_momentum`` is a fixed-``iters`` Newton solve -- exact in one step
when ``L`` is quadratic in ``qdot``, convergent for a convex-in-velocity ``L``.
Only ``order == 1`` Lagrangians are supported.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import torch
from omnibias.variational._core.hamiltonian import Hamiltonian
from omnibias.variational.torch.ops.dynamics import _require_first_order
from torch import Tensor
from torch.func import jacrev, vmap

if TYPE_CHECKING:  # pragma: no cover
    from omnibias.variational._core.lagrangian import Lagrangian, LagrangianFn


def _velocity_single(fn: LagrangianFn, q: Tensor, p: Tensor, t: Tensor, iters: int) -> Tensor:
    r"""Newton-invert ``p = dL/dqdot`` for a single sample ``(n,)`` -> ``qdot (n,)``."""
    dl_dv = jacrev(fn, argnums=1)
    qdot = torch.zeros_like(p)
    for _ in range(iters):
        resid = p - dl_dv(q, qdot, t)
        m = jacrev(dl_dv, argnums=1)(q, qdot, t)
        qdot = qdot + torch.linalg.solve(m, resid)
    return qdot


def _legendre_single(fn: LagrangianFn, q: Tensor, p: Tensor, t: Tensor, iters: int) -> Tensor:
    r"""Single-sample Legendre transform ``H = p . qdot(p) - L`` -> scalar."""
    qdot = _velocity_single(fn, q, p, t, iters)
    return cast(Tensor, (p * qdot).sum(-1) - fn(q, qdot, t))


def momentum(lagrangian: Lagrangian, q: Tensor, qdot: Tensor, t: Tensor) -> Tensor:
    r"""Conjugate momentum ``p = dL/dqdot`` at ``(q, qdot, t)``, shape ``(B, n)``."""
    _require_first_order(lagrangian)
    return cast(Tensor, vmap(jacrev(lagrangian.fn, argnums=1))(q, qdot, t))


def velocity_from_momentum(
    lagrangian: Lagrangian, q: Tensor, p: Tensor, t: Tensor, *, iters: int = 8,
) -> Tensor:
    r"""Invert ``p = dL/dqdot`` for ``qdot(q, p, t)`` by Newton iteration, ``(B, n)``.

    The Jacobian ``dp/dqdot = M`` (the mass matrix); the update is
    ``qdot <- qdot + M^{-1}(p - dL/dqdot)`` from ``qdot = 0``. Exact in one step
    for a Lagrangian quadratic in ``qdot``; convergent for a convex-in-velocity
    ``L`` (``iters`` fixed Newton steps -- no data-dependent stopping, so the op
    is ``vmap``/``jit``-safe and torch/jax agree).
    """
    _require_first_order(lagrangian)
    fn = lagrangian.fn
    return cast(Tensor, vmap(lambda qi, pi, ti: _velocity_single(fn, qi, pi, ti, iters))(q, p, t))


def legendre_transform(
    lagrangian: Lagrangian, q: Tensor, p: Tensor, t: Tensor, *, iters: int = 8,
) -> Tensor:
    r"""The Hamiltonian ``H(q, p, t) = p . qdot(p) - L(q, qdot(p), t)``, shape ``(B,)``."""
    _require_first_order(lagrangian)
    fn = lagrangian.fn
    return cast(Tensor, vmap(lambda qi, pi, ti: _legendre_single(fn, qi, pi, ti, iters))(q, p, t))


def hamiltonian_from_lagrangian(lagrangian: Lagrangian, *, iters: int = 8) -> Hamiltonian:
    r"""Build the :class:`Hamiltonian` Legendre-dual to ``lagrangian``.

    The returned Hamiltonian's callable is the single-sample Legendre transform
    (``vmap``/``jacrev``-friendly like any :class:`Lagrangian` ``fn``), so its
    :func:`canonical_equations` reproduce the Lagrangian's forward dynamics.
    """
    _require_first_order(lagrangian)
    fn = lagrangian.fn

    def h_fn(q: Tensor, p: Tensor, t: Tensor) -> Tensor:
        return _legendre_single(fn, q, p, t, iters)

    return Hamiltonian(fn=h_fn, dof=lagrangian.dof, time_axis=lagrangian.time_axis)


def canonical_equations(
    hamiltonian: Hamiltonian, q: Tensor, p: Tensor, t: Tensor,
) -> tuple[Tensor, Tensor]:
    r"""Hamilton's equations ``(qdot, pdot) = (dH/dp, -dH/dq)``, each shape ``(B, n)``.

    The phase-space flow of a :class:`Hamiltonian` (autodiff of its callable).
    With ``p = momentum(L, q, qdot, t)`` and ``H = hamiltonian_from_lagrangian(L)``,
    ``qdot`` reproduces the Lagrangian velocity and ``pdot`` its ``dL/dq``.
    """
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
