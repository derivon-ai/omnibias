# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Gauge transformations and gauge-invariance checks (jax).

The jax twin of :mod:`omnibias.geometry.gauge.torch.ops.gauge`.
"""

from __future__ import annotations

from typing import Any

import jax.numpy as jnp
from omnibias.geometry.gauge._core import kernels
from omnibias.geometry.gauge._core.lie_algebra import LieAlgebra
from omnibias.geometry.gauge.jax.ops.algebra import from_matrix, structure_constants, to_matrix
from omnibias.geometry.gauge.jax.ops.hodge import signature_diagonal

Array = Any


def infinitesimal_gauge_variation(
    A: Array,
    omega: Array,
    domega: Array,
    *,
    algebra: LieAlgebra,
    coupling: float,
) -> Array:
    r"""``delta A_mu^a = (D_mu omega)^a = d_mu omega^a + g f^{abc} A_mu^b omega^c``."""
    f = structure_constants(algebra, dtype=A.dtype)
    return kernels.covariant_derivative_adjoint(jnp, omega, domega, A, f, coupling)


def gauge_variation_field_strength(
    F: Array, omega: Array, *, algebra: LieAlgebra, coupling: float
) -> Array:
    r"""Homogeneous variation ``delta F_{mu nu}^a = g f^{abc} omega^b F_{mu nu}^c``."""
    f = structure_constants(algebra, dtype=F.dtype)
    return kernels.gauge_variation_field_strength(jnp, F, omega, f, coupling)


def global_gauge_transform(A: Array, U: Array, *, algebra: LieAlgebra) -> Array:
    r"""Finite global transform ``A_mu -> U A_mu U^{-1}`` for constant ``U`` -> ``(B, d, n)``."""
    a_mat = to_matrix(A, algebra)
    u = U.astype(a_mat.dtype)
    u_dag = jnp.swapaxes(u.conj(), -1, -2)
    transformed = jnp.matmul(jnp.matmul(u, a_mat), u_dag)
    return from_matrix(transformed, algebra).astype(A.dtype)


def gauge_invariance_defect(
    F: Array,
    omega: Array,
    *,
    algebra: LieAlgebra,
    coupling: float,
    signature: tuple[int, ...],
) -> Array:
    r"""First-order change of the action density under ``delta F`` -> ``(B,)`` (~ 0)."""
    dF = gauge_variation_field_strength(F, omega, algebra=algebra, coupling=coupling)
    eta = signature_diagonal(signature, dtype=F.dtype)
    return 0.5 * jnp.einsum("m,n,Bmna,Bmna->B", eta, eta, F, dF)


__all__ = [
    "gauge_invariance_defect",
    "gauge_variation_field_strength",
    "global_gauge_transform",
    "infinitesimal_gauge_variation",
]
