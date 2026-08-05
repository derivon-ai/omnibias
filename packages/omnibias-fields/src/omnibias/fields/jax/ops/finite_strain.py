# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Finite-strain (nonlinear) solid mechanics (jax).

Bit-identical twin of :mod:`omnibias.fields.torch.ops.finite_strain`; see that
module for the full docstring, index conventions and honesty note.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import jax
import jax.numpy as jnp
from jax import Array
from omnibias.fields.jax.ops.basic import stack_components
from omnibias.fields.jax.ops.high_order import vector_hessian
from omnibias.fields.jax.ops.tensor import (
    tensor_determinant,
    tensor_inverse,
    tensor_matmul,
    tensor_trace,
    tensor_transpose,
)
from omnibias.fields.jax.ops.vector import deformation_gradient

if TYPE_CHECKING:  # pragma: no cover

    from omnibias.fields._core.state import FieldState

EnergyFn = "Callable[[Array], Array]"


def _eye_like(f: Array) -> Array:
    d = f.shape[-1]
    return jnp.eye(d, dtype=f.dtype)


# ----------------------------------------------------------------------
# Kinematics
# ----------------------------------------------------------------------
def right_cauchy_green(f: Array) -> Array:
    r"""Right Cauchy-Green deformation tensor :math:`C = F^\top F`, ``(..., d, d)``."""
    return tensor_matmul(tensor_transpose(f), f)


def left_cauchy_green(f: Array) -> Array:
    r"""Left Cauchy-Green (Finger) tensor :math:`b = F F^\top`, ``(..., d, d)``."""
    return tensor_matmul(f, tensor_transpose(f))


def green_lagrange_strain(f: Array) -> Array:
    r"""Green-Lagrange strain :math:`E = \tfrac12 (F^\top F - I)`, ``(..., d, d)``."""
    return 0.5 * (right_cauchy_green(f) - _eye_like(f))


def jacobian_det(f: Array) -> Array:
    r"""Volume ratio :math:`J = \det F`, shape ``(...,)``."""
    return tensor_determinant(f)


# ----------------------------------------------------------------------
# Hyperelastic stored-energy densities
# ----------------------------------------------------------------------
def st_venant_kirchhoff_energy(f: Array, *, lam: float = 1.0, mu: float = 1.0) -> Array:
    r"""St Venant-Kirchhoff energy :math:`W = \tfrac{\lambda}{2}(\operatorname{tr}E)^2 + \mu\,\operatorname{tr}(E^2)`."""
    e = green_lagrange_strain(f)
    tr_e = tensor_trace(e)
    return 0.5 * lam * tr_e**2 + mu * tensor_trace(tensor_matmul(e, e))


def neo_hookean_energy(f: Array, *, lam: float = 1.0, mu: float = 1.0) -> Array:
    r"""Compressible neo-Hookean energy :math:`W = \tfrac{\mu}{2}(I_1-d) - \mu\ln J + \tfrac{\lambda}{2}(\ln J)^2`."""
    c = right_cauchy_green(f)
    d = f.shape[-1]
    i1 = tensor_trace(c)
    ln_j = jnp.log(tensor_determinant(f))
    return 0.5 * mu * (i1 - d) - mu * ln_j + 0.5 * lam * ln_j**2


def mooney_rivlin_energy(
    f: Array, *, c1: float = 1.0, c2: float = 1.0, kappa: float = 1.0,
) -> Array:
    r"""Compressible (3-D) Mooney-Rivlin energy with an isochoric-volumetric split."""
    if f.shape[-1] != 3:
        raise ValueError("mooney_rivlin_energy is defined for 3-D deformations only")
    c = right_cauchy_green(f)
    j = tensor_determinant(f)
    i1 = tensor_trace(c)
    i2 = 0.5 * (i1**2 - tensor_trace(tensor_matmul(c, c)))
    j_23 = j ** (-2.0 / 3.0)
    i1_bar = j_23 * i1
    i2_bar = j_23**2 * i2
    return c1 * (i1_bar - 3.0) + c2 * (i2_bar - 3.0) + 0.5 * kappa * (j - 1.0) ** 2


# ----------------------------------------------------------------------
# Stress from the stored energy
# ----------------------------------------------------------------------
def pk1_stress(f: Array, energy_fn: EnergyFn) -> Array:  # type: ignore[valid-type]
    r"""First Piola-Kirchhoff stress :math:`P = \partial W/\partial F`, ``(B, d, d)``."""
    if f.ndim != 3:
        raise ValueError(f"pk1_stress expects F of shape (B, d, d); got {tuple(f.shape)}")
    return jax.vmap(jax.jacrev(energy_fn))(f)


def pk2_stress(f: Array, energy_fn: EnergyFn) -> Array:  # type: ignore[valid-type]
    r"""Second Piola-Kirchhoff stress :math:`S = F^{-1}P`, ``(B, d, d)``."""
    p = pk1_stress(f, energy_fn)
    return tensor_matmul(tensor_inverse(f), p)


def cauchy_stress(f: Array, energy_fn: EnergyFn) -> Array:  # type: ignore[valid-type]
    r"""Cauchy (true) stress :math:`\sigma = J^{-1} P F^\top`, ``(B, d, d)``."""
    p = pk1_stress(f, energy_fn)
    j = tensor_determinant(f)
    return tensor_matmul(p, tensor_transpose(f)) / j[..., None, None]


# ----------------------------------------------------------------------
# Closed-form stress references
# ----------------------------------------------------------------------
def st_venant_kirchhoff_pk2(f: Array, *, lam: float = 1.0, mu: float = 1.0) -> Array:
    r"""Closed-form StVK second Piola-Kirchhoff stress :math:`S = \lambda(\operatorname{tr}E)I + 2\mu E`."""
    e = green_lagrange_strain(f)
    tr_e = tensor_trace(e)
    return lam * tr_e[..., None, None] * _eye_like(f) + 2.0 * mu * e


def neo_hookean_pk2(f: Array, *, lam: float = 1.0, mu: float = 1.0) -> Array:
    r"""Closed-form neo-Hookean second Piola-Kirchhoff stress :math:`S = \mu(I - C^{-1}) + \lambda\ln J\,C^{-1}`."""
    c_inv = tensor_inverse(right_cauchy_green(f))
    ln_j = jnp.log(tensor_determinant(f))
    eye = _eye_like(f)
    return mu * (eye - c_inv) + lam * ln_j[..., None, None] * c_inv


def isotropic_stiffness(d: int, *, lam: float = 1.0, mu: float = 1.0,
                        dtype: Any = None) -> Array:
    r"""Isotropic 4th-order elasticity tensor :math:`C_{ijkl}=\lambda\delta_{ij}\delta_{kl}+\mu(\delta_{ik}\delta_{jl}+\delta_{il}\delta_{jk})`."""
    dt = dtype if dtype is not None else jnp.zeros(()).dtype
    eye = jnp.eye(d, dtype=dt)
    return (
        lam * jnp.einsum("ij,kl->ijkl", eye, eye)
        + mu * jnp.einsum("ik,jl->ijkl", eye, eye)
        + mu * jnp.einsum("il,jk->ijkl", eye, eye)
    )


def hooke_stress_general(strain: Array, stiffness: Array) -> Array:
    r"""General anisotropic Hooke law :math:`\sigma_{ij}=C_{ijkl}\varepsilon_{kl}`, ``(B, d, d)``."""
    return jnp.einsum("...ijkl,...kl->...ij", stiffness, strain)


# ----------------------------------------------------------------------
# Field kinematics + balance laws
# ----------------------------------------------------------------------
def deformation_gradient_finite(state: FieldState, names: tuple[str, ...]) -> Array:
    r"""Finite deformation gradient :math:`F = I + \nabla u`, shape ``(B, d, d)``."""
    h = deformation_gradient(state, names)
    if h.shape[-1] != h.shape[-2]:
        raise ValueError(
            f"deformation_gradient_finite requires a square gradient; got {tuple(h.shape)}"
        )
    return h + _eye_like(h)


def _body_force(
    state: FieldState, body_force: tuple[str, ...] | Array | None, d: int,
) -> Array | float:
    if body_force is None:
        return 0.0
    if isinstance(body_force, tuple):
        return stack_components(state, body_force)
    return jnp.asarray(body_force)


def _pk1_divergence(state: FieldState, names: tuple[str, ...], energy_fn: EnergyFn) -> Array:  # type: ignore[valid-type]
    r"""``Div(P)_i = A_{iJkL}\, d^2 u_k/dx_J dx_L`` (autodiff tangent x closed-form Hessian)."""
    sa = tuple(state.coordinate_spec.spatial_axes)
    f = deformation_gradient_finite(state, names)
    tangent = jax.vmap(jax.jacrev(jax.jacrev(energy_fn)))(f)   # (B, i, J, k, L)
    hess = vector_hessian(state, names, axes=sa)               # (B, k, J, L)
    return jnp.einsum("bijkl,bkjl->bi", tangent, hess)


def finite_strain_residual(
    state: FieldState,
    names: tuple[str, ...],
    energy_fn: EnergyFn,  # type: ignore[valid-type]
    *,
    body_force: tuple[str, ...] | Array | None = None,
) -> Array:
    r"""Finite-strain equilibrium residual :math:`\operatorname{Div}P + f`, shape ``(B, d)``."""
    div_p = _pk1_divergence(state, names, energy_fn)
    f = _body_force(state, body_force, len(names))
    return div_p + f


def elastodynamic_residual(
    state: FieldState,
    names: tuple[str, ...],
    energy_fn: EnergyFn,  # type: ignore[valid-type]
    *,
    density: float = 1.0,
    body_force: tuple[str, ...] | Array | None = None,
) -> Array:
    r"""Elastodynamic residual :math:`\rho\,u_{tt} - \operatorname{Div}P - f`, shape ``(B, d)``."""
    from omnibias.fields.jax.ops.basic import derivative

    time = state.coordinate_spec.time_axis
    if time is None:
        raise ValueError("elastodynamic_residual requires a time axis in the coordinate spec")
    u_tt = jnp.stack([derivative(state, n, axis=time, order=2) for n in names], axis=-1)
    div_p = _pk1_divergence(state, names, energy_fn)
    f = _body_force(state, body_force, len(names))
    return density * u_tt - div_p - f


__all__ = [
    "cauchy_stress",
    "deformation_gradient_finite",
    "elastodynamic_residual",
    "finite_strain_residual",
    "green_lagrange_strain",
    "hooke_stress_general",
    "isotropic_stiffness",
    "jacobian_det",
    "left_cauchy_green",
    "mooney_rivlin_energy",
    "neo_hookean_energy",
    "neo_hookean_pk2",
    "pk1_stress",
    "pk2_stress",
    "right_cauchy_green",
    "st_venant_kirchhoff_energy",
    "st_venant_kirchhoff_pk2",
]
