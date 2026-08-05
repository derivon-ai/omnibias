# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Functional (variational) derivative and first variation (torch).

The variational derivative of the action ``S[q] = int L dt`` is the
Euler-Poisson operator -- the arbitrary-order generalisation of Euler-Lagrange:

.. math::

    \frac{\delta S}{\delta q_i}
      = \sum_{k=0}^{n} (-1)^k \frac{d^k}{dt^k}\frac{\partial L}{\partial q_i^{(k)}},

for a Lagrangian ``L(q, q^(1), ..., q^(n), t)`` of order ``n``. The outer total
time derivatives ``d^k/dt^k`` ride on the **closed-form** trajectory
derivatives ``q^(j)`` (up to order ``2n``, all from the sigma-tower); the
Lagrangian's own partials are ``torch.func`` autodiff of the callable.

Sign convention: ``functional_derivative == -euler_lagrange_residual`` (the
residual is the equation-of-motion form ``d/dt(dL/dqdot) - dL/dq``, the negative
of ``delta S / delta q``).

``first_variation`` is the Gateaux derivative paired against a perturbation
``eta`` (itself an omnibias field), computed as the exact weak form
``delta S[q; eta] = int sum_k (dL/dq^(k)) . eta^(k) dt`` -- no boundary terms
dropped.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

from omnibias.fields.torch.ops.basic import stack_components, vector_derivative
from omnibias.variational.torch.ops.action import integrate_values
from omnibias.variational.torch.ops.euler_lagrange import euler_lagrange_residual
from torch import Tensor
from torch.func import jacrev, vmap

if TYPE_CHECKING:  # pragma: no cover
    from omnibias.fields._core.quadrature import QuadratureSpec
    from omnibias.fields._core.state import FieldState
    from omnibias.variational._core.lagrangian import Lagrangian, LagrangianFn


def _total_derivative(h: Callable[..., Tensor], m: int) -> Callable[..., Tensor]:
    r"""Total time-derivative operator ``D`` on a single-sample function.

    ``h`` maps ``(a_0, ..., a_m, t) -> (n_dof,)``; the returned ``Dh`` maps
    ``(a_0, ..., a_{m+1}, t) -> (n_dof,)`` via the chain rule
    ``D h = sum_j (dh/da_j) a_{j+1} + dh/dt`` (so ``a_{j+1}`` are the next-order
    closed-form derivatives). Applying ``D`` k times consumes ``a`` up to order
    ``m + k``.
    """

    def dh(*args: Tensor) -> Tensor:
        a = args[:-1]
        t = args[-1]
        inner = (*a[: m + 1], t)
        out = jacrev(h, argnums=m + 1)(*inner)[..., 0]
        for j in range(m + 1):
            out = out + jacrev(h, argnums=j)(*inner) @ a[j + 1]
        return out

    return dh


def _euler_poisson_single(fn: LagrangianFn, order: int) -> Callable[..., Tensor]:
    r"""Single-sample Euler-Poisson operator for a Lagrangian of ``order``.

    Returns a function of ``(a_0, ..., a_{2*order}, t)`` (each ``a_k`` shape
    ``(n_dof,)``, ``t`` shape ``(1,)``) giving ``delta S / delta q`` of shape
    ``(n_dof,)``.
    """

    def euler_poisson(*args: Tensor) -> Tensor:
        a = args[:-1]
        t = args[-1]
        total: Tensor | None = None
        for k in range(order + 1):
            h: Callable[..., Tensor] = jacrev(fn, argnums=k)  # dL/da_k
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


def _closed_form_derivatives(state: FieldState, lagrangian: Lagrangian, up_to: int) -> list[Tensor]:
    """``[q, q^(1), ..., q^(up_to)]`` closed form, each ``(B, n_dof)``."""
    dof = lagrangian.dof
    ax = lagrangian.time_axis
    out = [stack_components(state, dof)]
    out += [vector_derivative(state, dof, axis=ax, order=j) for j in range(1, up_to + 1)]
    return out


def _time_column(state: FieldState, time_axis: str) -> Tensor:
    idx = state.coordinate_spec.axis_index(time_axis)
    return state.coords[:, idx].unsqueeze(-1)


def functional_derivative(state: FieldState, lagrangian: Lagrangian) -> Tensor:
    r"""Variational derivative ``delta S / delta q``, shape ``(B, n_dof)``.

    The Euler-Poisson operator of :class:`~omnibias.variational.Lagrangian` of
    any ``order``. Equals ``-euler_lagrange_residual`` (and is exactly that fast
    path for the first-order default); zero on a stationary trajectory.
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
) -> Tensor:
    r"""First (Gateaux) variation ``delta S[q; eta]``, a scalar tensor.

    The exact weak pairing ``int sum_{k=0}^{n} (dL/dq^(k)) . eta^(k) dt``, where
    ``eta`` is the ``perturbation`` field (same ``dof``) evaluated at the same
    quadrature nodes as ``state``; its derivatives ``eta^(k)`` are closed form.
    No integration by parts, so boundary terms are retained -- this equals
    ``d/deps S[q + eps*eta]`` at ``eps = 0``.
    """
    n = lagrangian.order
    q_derivs = _closed_form_derivatives(state, lagrangian, n)
    eta_derivs = _closed_form_derivatives(perturbation, lagrangian, n)
    t = _time_column(state, lagrangian.time_axis)
    fn = lagrangian.fn
    integrand: Tensor | None = None
    for k in range(n + 1):
        g_k = vmap(jacrev(fn, argnums=k))(*q_derivs, t)  # (B, n_dof)
        pair = (g_k * eta_derivs[k]).sum(-1)  # (B,)
        integrand = pair if integrand is None else integrand + pair
    assert integrand is not None
    return integrate_values(integrand, rule=rule)


__all__ = ["first_variation", "functional_derivative"]
