# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Classical field theory from a Lagrangian density (torch).

For a density ``L(phi, d phi, x)`` the field action is ``S = integral L d^n x`` and
the Euler-Lagrange field equations are

.. math::

    \sum_\mu \partial_\mu\frac{\partial L}{\partial(\partial_\mu\phi_a)}
        - \frac{\partial L}{\partial\phi_a} = 0 .

The outer space-time divergence ``d_mu`` rides on the **closed-form** field
gradient ``d_mu phi`` and Hessian ``d_mu d_nu phi`` (via the omnibias field ops);
the density's own partials are ``torch.func`` autodiff. The canonical
stress-energy tensor ``T^mu_nu = sum_a (dL/d(d_mu phi_a)) d_nu phi_a
- delta^mu_nu L`` is the Noether current of space-time translations.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from omnibias.fields.torch.ops.basic import stack_components
from omnibias.fields.torch.ops.high_order import jacobian, vector_hessian
from omnibias.variational.torch.ops.action import integrate_values
from torch import Tensor
from torch.func import jacrev, vmap

if TYPE_CHECKING:  # pragma: no cover
    from omnibias.fields._core.quadrature import QuadratureSpec
    from omnibias.fields._core.state import FieldState
    from omnibias.variational._core.lagrangian import LagrangianDensity


def _field_tensors(state: FieldState, fields: tuple[str, ...]) -> tuple[Tensor, Tensor, Tensor]:
    """``(phi, dphi, x)`` -- closed-form value / all-axis gradient / coords."""
    phi = stack_components(state, fields)          # (B, nf)
    dphi = jacobian(state, fields)                 # (B, nf, D) over all axes
    x = state.coords                               # (B, D)
    return phi, dphi, x


def density_values(state: FieldState, density: LagrangianDensity) -> Tensor:
    r"""Evaluate the density ``L(phi, d phi, x)`` at the nodes, shape ``(B,)``."""
    phi, dphi, x = _field_tensors(state, density.fields)
    return density.fn(phi, dphi, x)


def action_density(
    state: FieldState, density: LagrangianDensity, *, rule: QuadratureSpec,
) -> Tensor:
    r"""Field action :math:`S = \int L\,d^n x`, a scalar tensor."""
    return integrate_values(density_values(state, density), rule=rule)


def field_euler_lagrange_residual(state: FieldState, density: LagrangianDensity) -> Tensor:
    r"""Field Euler-Lagrange residual, shape ``(B, n_fields)``.

    Zero (to machine precision) exactly on a solution of the field equations.
    """
    phi, dphi, x = _field_tensors(state, density.fields)
    hess = vector_hessian(state, density.fields)   # (B, nf, D, D) = d_mu d_nu phi
    fn = density.fn

    g_phi = vmap(jacrev(fn, argnums=0))(phi, dphi, x)          # (B, nf)
    dl_dgrad = jacrev(fn, argnums=1)                            # -> (nf, D)
    h_pi_phi = vmap(jacrev(dl_dgrad, argnums=0))(phi, dphi, x)  # (B, nf, D, nf)
    h_pi_grad = vmap(jacrev(dl_dgrad, argnums=1))(phi, dphi, x)  # (B, nf, D, nf, D)
    h_pi_x = vmap(jacrev(dl_dgrad, argnums=2))(phi, dphi, x)    # (B, nf, D, D)

    div_pi = (
        torch.einsum("bamc,bcm->ba", h_pi_phi, dphi)
        + torch.einsum("bamcn,bcmn->ba", h_pi_grad, hess)
        + torch.einsum("bamm->ba", h_pi_x)
    )
    return div_pi - g_phi


def field_functional_derivative(state: FieldState, density: LagrangianDensity) -> Tensor:
    r"""Field variational derivative ``delta S / delta phi``, shape ``(B, n_fields)``.

    Equals ``-field_euler_lagrange_residual``; zero on a solution of the field
    equations. The particle analogue is
    :func:`omnibias.variational.torch.ops.functional.functional_derivative`.
    """
    return -field_euler_lagrange_residual(state, density)


def first_variation_density(
    state: FieldState,
    density: LagrangianDensity,
    perturbation: FieldState,
    *,
    rule: QuadratureSpec,
) -> Tensor:
    r"""First variation ``delta S[phi; eta]`` of the field action, a scalar tensor.

    The exact weak pairing
    ``int [ dL/dphi . eta + sum_mu dL/d(d_mu phi) . d_mu eta ] d^n x`` with ``eta``
    the ``perturbation`` field (same components) on the same nodes; its gradient
    ``d_mu eta`` is closed form. No integration by parts -- boundary terms are
    retained -- so this equals ``d/deps S[phi + eps*eta]`` at ``eps = 0``.
    """
    phi, dphi, x = _field_tensors(state, density.fields)
    fn = density.fn
    g_phi = vmap(jacrev(fn, argnums=0))(phi, dphi, x)   # (B, nf)
    pi = vmap(jacrev(fn, argnums=1))(phi, dphi, x)      # (B, nf, D)
    eta = stack_components(perturbation, density.fields)  # (B, nf)
    deta = jacobian(perturbation, density.fields)        # (B, nf, D)
    integrand = torch.einsum("ba,ba->b", g_phi, eta) + torch.einsum("bam,bam->b", pi, deta)
    return integrate_values(integrand, rule=rule)


def stress_energy_tensor(state: FieldState, density: LagrangianDensity) -> Tensor:
    r"""Canonical stress-energy tensor ``T^mu_nu``, shape ``(B, D, D)``.

    ``T[b, mu, nu] = sum_a (dL/d(d_mu phi_a)) d_nu phi_a - delta_{mu nu} L``.
    """
    phi, dphi, x = _field_tensors(state, density.fields)
    fn = density.fn
    pi = vmap(jacrev(fn, argnums=1))(phi, dphi, x)   # (B, nf, D)
    lval = fn(phi, dphi, x)                           # (B,)
    d = dphi.shape[-1]
    eye = torch.eye(d, dtype=lval.dtype, device=lval.device)
    return torch.einsum("bam,ban->bmn", pi, dphi) - eye * lval[:, None, None]


__all__ = [
    "action_density",
    "density_values",
    "field_euler_lagrange_residual",
    "field_functional_derivative",
    "first_variation_density",
    "stress_energy_tensor",
]
