# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Representation-theory tensor ops (jax): rep matrices + Casimir operators.

These materialize the pure-Python highest-weight data from
:mod:`omnibias.geometry.gauge._core.representation` as jax arrays, so an irrep's
generators plug straight into the gauge-covariant-derivative surface. The
numbers are produced by the shared numpy core, so the jax and torch twins are
bit-identical by construction.
"""

from __future__ import annotations

from typing import Any

import jax.numpy as jnp
import numpy as np
from omnibias.geometry.gauge._core import representation as _rep
from omnibias.geometry.gauge._core.lie_algebra import LieAlgebra

Array = Any


def _complex_dtype(dtype: Any) -> Any:
    return jnp.complex128 if np.dtype(dtype) == np.float64 else jnp.complex64


def _real_dtype(dtype: Any) -> Any:
    return jnp.asarray(0.0).dtype if dtype is None else dtype


def spin_matrices(two_j: int, *, dtype: Any = None) -> Array:
    r"""``su(2)`` spin-``j`` generators stacked as ``(3, d, d)`` (``two_j = 2j``).

    Component ``a`` is ``J_a`` with ``[J_a, J_b] = i \epsilon_{abc} J_c``,
    ``\sum_a J_a^2 = j(j+1) I`` and ``d = 2j + 1``.
    """
    cdt = _complex_dtype(_real_dtype(dtype))
    jx, jy, jz = _rep.su2_spin_matrices(two_j)
    return jnp.stack(
        [jnp.asarray(jx, dtype=cdt), jnp.asarray(jy, dtype=cdt), jnp.asarray(jz, dtype=cdt)]
    )


def adjoint_generators(algebra: LieAlgebra, *, dtype: Any = None) -> Array:
    r"""Adjoint generators ``(T^a_{adj})_{bc} = -i f^{abc}`` as ``(dim, dim, dim)``.

    Hermitian, satisfy ``[T^a, T^b] = i f^{abc} T^c`` and
    ``tr(T^a_{adj} T^b_{adj}) = N \delta^{ab}`` (the dual Coxeter number).
    """
    cdt = _complex_dtype(_real_dtype(dtype))
    mats = _rep.adjoint_rep_matrices(algebra.structure_constants())
    return jnp.asarray(mats, dtype=cdt)


def symmetric_power_generators(algebra: LieAlgebra, k: int, *, dtype: Any = None) -> Array:
    r"""``Sym^k`` generators of the fundamental as ``(dim, D, D)``, ``D = C(N+k-1, k)``."""
    cdt = _complex_dtype(_real_dtype(dtype))
    mats = _rep.symmetric_power_rep_matrices(algebra.generators(), k)
    return jnp.asarray(mats, dtype=cdt)


def antisymmetric_power_generators(
    algebra: LieAlgebra, k: int, *, dtype: Any = None
) -> Array:
    r"""``Lambda^k`` generators of the fundamental as ``(dim, D, D)``, ``D = C(N, k)``."""
    cdt = _complex_dtype(_real_dtype(dtype))
    mats = _rep.antisymmetric_power_rep_matrices(algebra.generators(), k)
    return jnp.asarray(mats, dtype=cdt)


def casimir_operator(matrices: Array) -> Array:
    r"""Quadratic Casimir operator ``\sum_a T^a T^a`` for stacked ``(dim, d, d)``.

    For an irrep this equals ``C_2(R) I``; used as an independent cross-check of
    :func:`casimir_eigenvalue`.
    """
    return jnp.einsum("aij,ajk->ik", matrices, matrices)


def casimir_eigenvalue(rep: _rep.Irrep, *, dtype: Any = None) -> Array:
    r"""The quadratic Casimir eigenvalue ``C_2(R)`` as a scalar array.

    Physics normalization ``tr_fund(T^a T^b) = 1/2 \delta^{ab}`` (so the ``su(N)``
    fundamental has ``C_2 = (N^2 - 1) / (2 N)``).
    """
    return jnp.asarray(float(_rep.quadratic_casimir(rep)), dtype=_real_dtype(dtype))


def dynkin_index_value(rep: _rep.Irrep, *, dtype: Any = None) -> Array:
    r"""The Dynkin index ``T(R) = C_2(R) \dim(R) / \dim(G)`` as a scalar array."""
    return jnp.asarray(float(_rep.dynkin_index(rep)), dtype=_real_dtype(dtype))


__all__ = [
    "adjoint_generators",
    "antisymmetric_power_generators",
    "casimir_eigenvalue",
    "casimir_operator",
    "dynkin_index_value",
    "spin_matrices",
    "symmetric_power_generators",
]
