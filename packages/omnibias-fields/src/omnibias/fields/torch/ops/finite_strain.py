# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Finite-strain (nonlinear) solid mechanics (torch).

Twin of :mod:`omnibias.fields.jax.ops.finite_strain`. Extends the small-strain
:mod:`omnibias.fields.torch.ops.mechanics` surface to *finite* deformations:

- **kinematics** -- the finite deformation gradient ``F = I + grad u``, the right
  Cauchy-Green tensor ``C = F^T F``, the Green-Lagrange strain ``E = 1/2 (C - I)``
  and the Jacobian ``J = det F``;
- **hyperelastic energies** -- St Venant-Kirchhoff, compressible neo-Hookean and
  (3-D) Mooney-Rivlin stored-energy densities ``W(F)``;
- **stress** -- the first / second Piola-Kirchhoff and Cauchy stresses, obtained
  as the *exact* derivative of the algebraic stored energy ``P = dW/dF`` (a
  machine-precision autodiff of the closed-form energy, **not** a finite
  difference), plus hand-derived closed-form references for StVK / neo-Hookean
  and the general anisotropic Hooke law;
- **balance laws** -- the finite-strain equilibrium residual ``Div(P) + f`` and
  the elastodynamic residual ``rho u_tt - Div(P) - f``. The stress divergence is
  assembled from the *tangent modulus* ``A = dP/dF`` (autodiff-exact) contracted
  with the **closed-form** second spatial derivatives of the displacement field
  ``d^2 u_k / dx_J dx_L`` (the omnibias sigma-tower).

Honesty
-------
Elasticity / hyperelasticity / elastodynamics here are exact: the constitutive
tangent is autodiff of an algebraic energy (exact to machine precision) and the
kinematic derivatives are the closed-form activation tower. History-dependent
inelasticity (plasticity return maps, viscoelastic internal-variable evolution)
is **not** closed-form; those solvers are iterative/`numerical` and are out of
scope for this module (only the differentiable yield/flow *residuals* would be).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from omnibias.fields.torch.ops.basic import stack_components
from omnibias.fields.torch.ops.high_order import vector_hessian
from omnibias.fields.torch.ops.tensor import (
    tensor_determinant,
    tensor_inverse,
    tensor_matmul,
    tensor_trace,
    tensor_transpose,
)
from omnibias.fields.torch.ops.vector import deformation_gradient
from torch import Tensor
from torch.func import jacrev, vmap

if TYPE_CHECKING:  # pragma: no cover

    from omnibias.fields._core.state import FieldState

EnergyFn = "Callable[[Tensor], Tensor]"


def _eye_like(f: Tensor) -> Tensor:
    d = f.shape[-1]
    return torch.eye(d, dtype=f.dtype, device=f.device)


# ----------------------------------------------------------------------
# Kinematics (pure tensor ops on F of shape (..., d, d))
# ----------------------------------------------------------------------
def right_cauchy_green(f: Tensor) -> Tensor:
    r"""Right Cauchy-Green deformation tensor :math:`C = F^\top F`, ``(..., d, d)``."""
    return tensor_matmul(tensor_transpose(f), f)


def left_cauchy_green(f: Tensor) -> Tensor:
    r"""Left Cauchy-Green (Finger) tensor :math:`b = F F^\top`, ``(..., d, d)``."""
    return tensor_matmul(f, tensor_transpose(f))


def green_lagrange_strain(f: Tensor) -> Tensor:
    r"""Green-Lagrange strain :math:`E = \tfrac12 (F^\top F - I)`, ``(..., d, d)``.

    Vanishes for a rigid-body motion (``F`` orthogonal), unlike the engineering
    strain, so it is the correct finite-deformation measure.
    """
    return 0.5 * (right_cauchy_green(f) - _eye_like(f))


def jacobian_det(f: Tensor) -> Tensor:
    r"""Volume ratio :math:`J = \det F`, shape ``(...,)`` (``J = 1`` incompressible)."""
    return tensor_determinant(f)


# ----------------------------------------------------------------------
# Hyperelastic stored-energy densities  W(F) -> (...,)
# ----------------------------------------------------------------------
def st_venant_kirchhoff_energy(f: Tensor, *, lam: float = 1.0, mu: float = 1.0) -> Tensor:
    r"""St Venant-Kirchhoff energy :math:`W = \tfrac{\lambda}{2}(\operatorname{tr}E)^2 + \mu\,\operatorname{tr}(E^2)`."""
    e = green_lagrange_strain(f)
    tr_e = tensor_trace(e)
    return 0.5 * lam * tr_e**2 + mu * tensor_trace(tensor_matmul(e, e))


def neo_hookean_energy(f: Tensor, *, lam: float = 1.0, mu: float = 1.0) -> Tensor:
    r"""Compressible neo-Hookean energy.

    :math:`W = \tfrac{\mu}{2}(I_1 - d) - \mu\ln J + \tfrac{\lambda}{2}(\ln J)^2`,
    with :math:`I_1 = \operatorname{tr} C` and :math:`J = \det F`.
    """
    c = right_cauchy_green(f)
    d = f.shape[-1]
    i1 = tensor_trace(c)
    ln_j = torch.log(tensor_determinant(f))
    return 0.5 * mu * (i1 - d) - mu * ln_j + 0.5 * lam * ln_j**2


def mooney_rivlin_energy(
    f: Tensor, *, c1: float = 1.0, c2: float = 1.0, kappa: float = 1.0,
) -> Tensor:
    r"""Compressible (3-D) Mooney-Rivlin energy with an isochoric-volumetric split.

    :math:`W = c_1(\bar I_1 - 3) + c_2(\bar I_2 - 3) + \tfrac{\kappa}{2}(J-1)^2`,
    with the isochoric invariants :math:`\bar I_1 = J^{-2/3} I_1`,
    :math:`\bar I_2 = J^{-4/3} I_2`. Defined for ``d = 3`` only.
    """
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
# Stress from the stored energy (P = dW/dF, exact autodiff of the algebra)
# ----------------------------------------------------------------------
def pk1_stress(f: Tensor, energy_fn: EnergyFn) -> Tensor:  # type: ignore[valid-type]
    r"""First Piola-Kirchhoff stress :math:`P = \partial W/\partial F`, ``(B, d, d)``.

    ``energy_fn`` maps a single ``(d, d)`` deformation gradient to the scalar
    stored energy; ``P`` is its exact gradient (reverse-mode autodiff of the
    algebraic energy -- reverse mode is both natural for a scalar energy and
    avoids a ``vmap``-of-forward-mode ``linalg.det`` batching defect).
    """
    if f.ndim != 3:
        raise ValueError(f"pk1_stress expects F of shape (B, d, d); got {tuple(f.shape)}")
    return vmap(jacrev(energy_fn))(f)  # type: ignore[no-any-return]


def pk2_stress(f: Tensor, energy_fn: EnergyFn) -> Tensor:  # type: ignore[valid-type]
    r"""Second Piola-Kirchhoff stress :math:`S = F^{-1}P`, ``(B, d, d)`` (symmetric)."""
    p = pk1_stress(f, energy_fn)
    return tensor_matmul(tensor_inverse(f), p)


def cauchy_stress(f: Tensor, energy_fn: EnergyFn) -> Tensor:  # type: ignore[valid-type]
    r"""Cauchy (true) stress :math:`\sigma = J^{-1} P F^\top`, ``(B, d, d)`` (symmetric)."""
    p = pk1_stress(f, energy_fn)
    j = tensor_determinant(f)
    return tensor_matmul(p, tensor_transpose(f)) / j[..., None, None]


# ----------------------------------------------------------------------
# Closed-form stress references (validated against pk*_stress in tests)
# ----------------------------------------------------------------------
def st_venant_kirchhoff_pk2(f: Tensor, *, lam: float = 1.0, mu: float = 1.0) -> Tensor:
    r"""Closed-form StVK second Piola-Kirchhoff stress :math:`S = \lambda(\operatorname{tr}E)I + 2\mu E`."""
    e = green_lagrange_strain(f)
    tr_e = tensor_trace(e)
    return lam * tr_e[..., None, None] * _eye_like(f) + 2.0 * mu * e


def neo_hookean_pk2(f: Tensor, *, lam: float = 1.0, mu: float = 1.0) -> Tensor:
    r"""Closed-form neo-Hookean second Piola-Kirchhoff stress :math:`S = \mu(I - C^{-1}) + \lambda\ln J\,C^{-1}`."""
    c_inv = tensor_inverse(right_cauchy_green(f))
    ln_j = torch.log(tensor_determinant(f))
    eye = _eye_like(f)
    return mu * (eye - c_inv) + lam * ln_j[..., None, None] * c_inv


def isotropic_stiffness(d: int, *, lam: float = 1.0, mu: float = 1.0,
                        dtype: torch.dtype | None = None) -> Tensor:
    r"""Isotropic 4th-order elasticity tensor :math:`C_{ijkl}=\lambda\delta_{ij}\delta_{kl}+\mu(\delta_{ik}\delta_{jl}+\delta_{il}\delta_{jk})`."""
    dt = dtype or torch.get_default_dtype()
    eye = torch.eye(d, dtype=dt)
    return (
        lam * torch.einsum("ij,kl->ijkl", eye, eye)
        + mu * torch.einsum("ik,jl->ijkl", eye, eye)
        + mu * torch.einsum("il,jk->ijkl", eye, eye)
    )


def hooke_stress_general(strain: Tensor, stiffness: Tensor) -> Tensor:
    r"""General anisotropic Hooke law :math:`\sigma_{ij}=C_{ijkl}\varepsilon_{kl}`, ``(B, d, d)``."""
    return torch.einsum("...ijkl,...kl->...ij", stiffness, strain)


# ----------------------------------------------------------------------
# Field kinematics + balance laws (consume a FieldState)
# ----------------------------------------------------------------------
def deformation_gradient_finite(state: FieldState, names: tuple[str, ...]) -> Tensor:
    r"""Finite deformation gradient :math:`F = I + \nabla u`, shape ``(B, d, d)``.

    Unlike :func:`omnibias.fields.torch.ops.deformation_gradient` (which returns
    the displacement gradient :math:`\nabla u`), this adds the identity.
    """
    h = deformation_gradient(state, names)
    if h.shape[-1] != h.shape[-2]:
        raise ValueError(
            f"deformation_gradient_finite requires a square gradient; got {tuple(h.shape)}"
        )
    return h + _eye_like(h)


def _body_force(
    state: FieldState, body_force: tuple[str, ...] | Tensor | None, d: int,
) -> Tensor | float:
    if body_force is None:
        return 0.0
    if isinstance(body_force, tuple):
        return stack_components(state, body_force)
    return torch.as_tensor(body_force)


def _pk1_divergence(state: FieldState, names: tuple[str, ...], energy_fn: EnergyFn) -> Tensor:  # type: ignore[valid-type]
    r"""``Div(P)_i = d_J P_{iJ} = A_{iJkL}\, d^2 u_k/dx_J dx_L``.

    The tangent modulus ``A = dP/dF`` is autodiff-exact; the second spatial
    derivatives of the displacement are closed-form (sigma-tower).
    """
    sa = tuple(state.coordinate_spec.spatial_axes)
    f = deformation_gradient_finite(state, names)
    tangent = vmap(jacrev(jacrev(energy_fn)))(f)          # (B, i, J, k, L)
    hess = vector_hessian(state, names, axes=sa)          # (B, k, J, L)
    return torch.einsum("bijkl,bkjl->bi", tangent, hess)


def finite_strain_residual(
    state: FieldState,
    names: tuple[str, ...],
    energy_fn: EnergyFn,  # type: ignore[valid-type]
    *,
    body_force: tuple[str, ...] | Tensor | None = None,
) -> Tensor:
    r"""Finite-strain equilibrium residual :math:`\operatorname{Div}P + f`, shape ``(B, d)``.

    In the small-strain limit with a St Venant-Kirchhoff energy this reduces to
    :func:`omnibias.fields.torch.ops.navier_cauchy_residual`.
    """
    div_p = _pk1_divergence(state, names, energy_fn)
    f = _body_force(state, body_force, len(names))
    return div_p + f


def elastodynamic_residual(
    state: FieldState,
    names: tuple[str, ...],
    energy_fn: EnergyFn,  # type: ignore[valid-type]
    *,
    density: float = 1.0,
    body_force: tuple[str, ...] | Tensor | None = None,
) -> Tensor:
    r"""Elastodynamic residual :math:`\rho\,u_{tt} - \operatorname{Div}P - f`, shape ``(B, d)``.

    ``u_tt`` is the closed-form second time derivative of the displacement field;
    requires the coordinate spec to carry a time axis.
    """
    from omnibias.fields.torch.ops.basic import derivative

    time = state.coordinate_spec.time_axis
    if time is None:
        raise ValueError("elastodynamic_residual requires a time axis in the coordinate spec")
    u_tt = torch.stack([derivative(state, n, axis=time, order=2) for n in names], dim=-1)
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
