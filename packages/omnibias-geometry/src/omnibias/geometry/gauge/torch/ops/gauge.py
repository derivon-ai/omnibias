# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Gauge transformations and gauge-invariance checks (torch).

Two physically essential, exactly testable transformations:

* the **infinitesimal** local transformation ``delta A_mu = D_mu omega`` (the
  covariant derivative of the gauge parameter) under which ``F`` transforms
  *homogeneously*, ``delta F_{mu nu}^a = g f^{abc} omega^b F_{mu nu}^c``, leaving
  the action invariant to first order;
* the **finite global** transformation ``A_mu -> U A_mu U^{-1}`` for a constant
  group element ``U`` (fundamental representation), which leaves the action
  exactly invariant.
"""

from __future__ import annotations

import torch
from omnibias.geometry.gauge._core import kernels
from omnibias.geometry.gauge._core.lie_algebra import LieAlgebra
from omnibias.geometry.gauge.torch.ops.algebra import from_matrix, structure_constants, to_matrix
from omnibias.geometry.gauge.torch.ops.hodge import signature_diagonal
from torch import Tensor


def infinitesimal_gauge_variation(
    A: Tensor,
    omega: Tensor,
    domega: Tensor,
    *,
    algebra: LieAlgebra,
    coupling: float,
) -> Tensor:
    r"""``delta A_mu^a = (D_mu omega)^a = d_mu omega^a + g f^{abc} A_mu^b omega^c``."""
    f = structure_constants(algebra, dtype=A.dtype, device=A.device)
    return kernels.covariant_derivative_adjoint(torch, omega, domega, A, f, coupling)


def gauge_variation_field_strength(
    F: Tensor, omega: Tensor, *, algebra: LieAlgebra, coupling: float
) -> Tensor:
    r"""Homogeneous variation ``delta F_{mu nu}^a = g f^{abc} omega^b F_{mu nu}^c``."""
    f = structure_constants(algebra, dtype=F.dtype, device=F.device)
    return kernels.gauge_variation_field_strength(torch, F, omega, f, coupling)


def global_gauge_transform(A: Tensor, U: Tensor, *, algebra: LieAlgebra) -> Tensor:
    r"""Finite global transform ``A_mu -> U A_mu U^{-1}`` for constant ``U`` -> ``(B, d, n)``."""
    a_mat = to_matrix(A, algebra)
    u = U.to(a_mat.dtype)
    u_dag = u.conj().transpose(-1, -2)
    transformed = torch.matmul(torch.matmul(u, a_mat), u_dag)
    return from_matrix(transformed, algebra).to(A.dtype)


def gauge_invariance_defect(
    F: Tensor,
    omega: Tensor,
    *,
    algebra: LieAlgebra,
    coupling: float,
    signature: tuple[int, ...],
) -> Tensor:
    r"""First-order change of the action density under ``delta F`` -> ``(B,)`` (~ 0).

    ``delta[(1/4) F^2] = (1/2) eta_mu eta_nu F^a_{mu nu} delta F^a_{mu nu}``, which
    vanishes identically by antisymmetry of ``f^{abc}``.
    """
    dF = gauge_variation_field_strength(F, omega, algebra=algebra, coupling=coupling)
    eta = signature_diagonal(signature, dtype=F.dtype, device=F.device)
    return 0.5 * torch.einsum("m,n,Bmna,Bmna->B", eta, eta, F, dF)


__all__ = [
    "gauge_invariance_defect",
    "gauge_variation_field_strength",
    "global_gauge_transform",
    "infinitesimal_gauge_variation",
]
