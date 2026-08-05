# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Functional (variational) derivative and first variation (jax).

Bit-identical twin of the torch module: the Euler-Poisson operator

.. math::

    \frac{\delta S}{\delta q_i}
      = \sum_{k=0}^{n} (-1)^k \frac{d^k}{dt^k}\frac{\partial L}{\partial q_i^{(k)}},

with the outer total time derivatives riding on the closed-form ``q^(k)`` (up to
order ``2n``) and the Lagrangian partials by autodiff. ``first_variation`` is the
exact weak Gateaux pairing against a perturbation field.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

from jax import Array, jacrev, vmap
from omnibias.fields.jax.ops.basic import stack_components, vector_derivative
from omnibias.variational.jax.ops.action import integrate_values
from omnibias.variational.jax.ops.euler_lagrange import euler_lagrange_residual

if TYPE_CHECKING:  # pragma: no cover
    from omnibias.fields._core.quadrature import QuadratureSpec
    from omnibias.fields._core.state import FieldState
    from omnibias.variational._core.lagrangian import Lagrangian, LagrangianFn


def _total_derivative(h: Callable[..., Array], m: int) -> Callable[..., Array]:
    r"""Total time-derivative operator ``D`` on a single-sample function."""

    def dh(*args: Array) -> Array:
        a = args[:-1]
        t = args[-1]
        inner = (*a[: m + 1], t)
        out = jacrev(h, argnums=m + 1)(*inner)[..., 0]
        for j in range(m + 1):
            out = out + jacrev(h, argnums=j)(*inner) @ a[j + 1]
        return out

    return dh


def _euler_poisson_single(fn: LagrangianFn, order: int) -> Callable[..., Array]:
    r"""Single-sample Euler-Poisson operator for a Lagrangian of ``order``."""

    def euler_poisson(*args: Array) -> Array:
        a = args[:-1]
        t = args[-1]
        total: Array | None = None
        for k in range(order + 1):
            h: Callable[..., Array] = jacrev(fn, argnums=k)
            m = order
            for _ in range(k):
                h = _total_derivative(h, m)
                m += 1
            term = h(*a[: order + k + 1], t)
            if k % 2 == 1:
                term = -term
            total = term if total is None else total + term
        assert total is not None
        return total

    return euler_poisson


def _closed_form_derivatives(state: FieldState, lagrangian: Lagrangian, up_to: int) -> list[Array]:
    """``[q, q^(1), ..., q^(up_to)]`` closed form, each ``(B, n_dof)``."""
    dof = lagrangian.dof
    ax = lagrangian.time_axis
    out = [stack_components(state, dof)]
    out += [vector_derivative(state, dof, axis=ax, order=j) for j in range(1, up_to + 1)]
    return out


def _time_column(state: FieldState, time_axis: str) -> Array:
    idx = state.coordinate_spec.axis_index(time_axis)
    return state.coords[:, idx][:, None]


def functional_derivative(state: FieldState, lagrangian: Lagrangian) -> Array:
    r"""Variational derivative ``delta S / delta q``, shape ``(B, n_dof)``.

    Euler-Poisson operator for any ``order``; equals ``-euler_lagrange_residual``.
    """
    n = lagrangian.order
    if n == 1:
        return -euler_lagrange_residual(state, lagrangian)
    derivs = _closed_form_derivatives(state, lagrangian, 2 * n)
    t = _time_column(state, lagrangian.time_axis)
    return vmap(_euler_poisson_single(lagrangian.fn, n))(*derivs, t)


def first_variation(
    state: FieldState,
    lagrangian: Lagrangian,
    perturbation: FieldState,
    *,
    rule: QuadratureSpec,
) -> Array:
    r"""First (Gateaux) variation ``delta S[q; eta]``, a scalar array.

    Exact weak pairing ``int sum_k (dL/dq^(k)) . eta^(k) dt`` (boundary terms
    retained), with ``eta`` the ``perturbation`` field on the same nodes.
    """
    n = lagrangian.order
    q_derivs = _closed_form_derivatives(state, lagrangian, n)
    eta_derivs = _closed_form_derivatives(perturbation, lagrangian, n)
    t = _time_column(state, lagrangian.time_axis)
    fn = lagrangian.fn
    integrand: Array | None = None
    for k in range(n + 1):
        g_k = vmap(jacrev(fn, argnums=k))(*q_derivs, t)
        pair = (g_k * eta_derivs[k]).sum(-1)
        integrand = pair if integrand is None else integrand + pair
    assert integrand is not None
    return integrate_values(integrand, rule=rule)


__all__ = ["first_variation", "functional_derivative"]
