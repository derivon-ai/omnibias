# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Classical field theory from a Lagrangian density (jax).

Bit-identical twin of the torch module.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax.numpy as jnp
from jax import Array, jacrev, vmap
from omnibias.fields.jax.ops.basic import stack_components
from omnibias.fields.jax.ops.high_order import jacobian, vector_hessian
from omnibias.variational.jax.ops.action import integrate_values

if TYPE_CHECKING:  # pragma: no cover
    from omnibias.fields._core.quadrature import QuadratureSpec
    from omnibias.fields._core.state import FieldState
    from omnibias.variational._core.lagrangian import LagrangianDensity


def _field_tensors(state: FieldState, fields: tuple[str, ...]) -> tuple[Array, Array, Array]:
    """``(phi, dphi, x)`` -- closed-form value / all-axis gradient / coords."""
    phi = stack_components(state, fields)
    dphi = jacobian(state, fields)
    x = state.coords
    return phi, dphi, x


def density_values(state: FieldState, density: LagrangianDensity) -> Array:
    r"""Evaluate the density ``L(phi, d phi, x)`` at the nodes, shape ``(B,)``."""
    phi, dphi, x = _field_tensors(state, density.fields)
    return density.fn(phi, dphi, x)


def action_density(
    state: FieldState, density: LagrangianDensity, *, rule: QuadratureSpec,
) -> Array:
    r"""Field action :math:`S = \int L\,d^n x`, a scalar array."""
    return integrate_values(density_values(state, density), rule=rule)


def field_euler_lagrange_residual(state: FieldState, density: LagrangianDensity) -> Array:
    r"""Field Euler-Lagrange residual, shape ``(B, n_fields)``."""
    phi, dphi, x = _field_tensors(state, density.fields)
    hess = vector_hessian(state, density.fields)
    fn = density.fn

    g_phi = vmap(jacrev(fn, argnums=0))(phi, dphi, x)
    dl_dgrad = jacrev(fn, argnums=1)
    h_pi_phi = vmap(jacrev(dl_dgrad, argnums=0))(phi, dphi, x)
    h_pi_grad = vmap(jacrev(dl_dgrad, argnums=1))(phi, dphi, x)
    h_pi_x = vmap(jacrev(dl_dgrad, argnums=2))(phi, dphi, x)

    div_pi = (
        jnp.einsum("bamc,bcm->ba", h_pi_phi, dphi)
        + jnp.einsum("bamcn,bcmn->ba", h_pi_grad, hess)
        + jnp.einsum("bamm->ba", h_pi_x)
    )
    return div_pi - g_phi


def field_functional_derivative(state: FieldState, density: LagrangianDensity) -> Array:
    r"""Field variational derivative ``delta S / delta phi``, shape ``(B, n_fields)``.

    Equals ``-field_euler_lagrange_residual``; zero on a solution.
    """
    return -field_euler_lagrange_residual(state, density)


def first_variation_density(
    state: FieldState,
    density: LagrangianDensity,
    perturbation: FieldState,
    *,
    rule: QuadratureSpec,
) -> Array:
    r"""First variation ``delta S[phi; eta]`` of the field action, a scalar array.

    Exact weak pairing ``int [ dL/dphi . eta + sum_mu dL/d(d_mu phi) . d_mu eta ] d^n x``
    with ``eta`` the ``perturbation`` field on the same nodes (boundary terms
    retained).
    """
    phi, dphi, x = _field_tensors(state, density.fields)
    fn = density.fn
    g_phi = vmap(jacrev(fn, argnums=0))(phi, dphi, x)
    pi = vmap(jacrev(fn, argnums=1))(phi, dphi, x)
    eta = stack_components(perturbation, density.fields)
    deta = jacobian(perturbation, density.fields)
    integrand = jnp.einsum("ba,ba->b", g_phi, eta) + jnp.einsum("bam,bam->b", pi, deta)
    return integrate_values(integrand, rule=rule)


def stress_energy_tensor(state: FieldState, density: LagrangianDensity) -> Array:
    r"""Canonical stress-energy tensor ``T^mu_nu``, shape ``(B, D, D)``."""
    phi, dphi, x = _field_tensors(state, density.fields)
    fn = density.fn
    pi = vmap(jacrev(fn, argnums=1))(phi, dphi, x)
    lval = fn(phi, dphi, x)
    d = dphi.shape[-1]
    eye = jnp.eye(d, dtype=lval.dtype)
    return jnp.einsum("bam,ban->bmn", pi, dphi) - eye * lval[:, None, None]


__all__ = [
    "action_density",
    "density_values",
    "field_euler_lagrange_residual",
    "field_functional_derivative",
    "first_variation_density",
    "stress_energy_tensor",
]
