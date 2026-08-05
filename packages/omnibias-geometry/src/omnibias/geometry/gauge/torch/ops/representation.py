# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Representation-theory tensor ops (torch): rep matrices + Casimir operators.

These materialize the pure-Python highest-weight data from
:mod:`omnibias.geometry.gauge._core.representation` as torch tensors, so an irrep's
generators plug straight into the gauge-covariant-derivative surface. The
numbers are produced by the shared numpy core, so the torch and jax twins are
bit-identical by construction.
"""

from __future__ import annotations

import torch
from omnibias.geometry.gauge._core import representation as _rep
from omnibias.geometry.gauge._core.lie_algebra import LieAlgebra
from torch import Tensor


def _complex_dtype(dtype: torch.dtype) -> torch.dtype:
    return torch.complex128 if dtype in (torch.float64, torch.complex128) else torch.complex64


def spin_matrices(
    two_j: int,
    *,
    dtype: torch.dtype | None = None,
    device: torch.device | None = None,
) -> Tensor:
    r"""``su(2)`` spin-``j`` generators stacked as ``(3, d, d)`` (``two_j = 2j``).

    Component ``a`` is ``J_a`` with ``[J_a, J_b] = i \epsilon_{abc} J_c``,
    ``\sum_a J_a^2 = j(j+1) I`` and ``d = 2j + 1``.
    """
    real_dt = dtype if dtype is not None else torch.get_default_dtype()
    cdt = _complex_dtype(real_dt)
    jx, jy, jz = _rep.su2_spin_matrices(two_j)
    stacked = torch.stack(
        [
            torch.as_tensor(jx, dtype=cdt, device=device),
            torch.as_tensor(jy, dtype=cdt, device=device),
            torch.as_tensor(jz, dtype=cdt, device=device),
        ]
    )
    return stacked


def adjoint_generators(
    algebra: LieAlgebra,
    *,
    dtype: torch.dtype | None = None,
    device: torch.device | None = None,
) -> Tensor:
    r"""Adjoint generators ``(T^a_{adj})_{bc} = -i f^{abc}`` as ``(dim, dim, dim)``.

    Hermitian, satisfy ``[T^a, T^b] = i f^{abc} T^c`` and
    ``tr(T^a_{adj} T^b_{adj}) = N \delta^{ab}`` (the dual Coxeter number).
    """
    real_dt = dtype if dtype is not None else torch.get_default_dtype()
    cdt = _complex_dtype(real_dt)
    mats = _rep.adjoint_rep_matrices(algebra.structure_constants())
    return torch.as_tensor(mats, dtype=cdt, device=device)


def symmetric_power_generators(
    algebra: LieAlgebra,
    k: int,
    *,
    dtype: torch.dtype | None = None,
    device: torch.device | None = None,
) -> Tensor:
    r"""``Sym^k`` generators of the fundamental as ``(dim, D, D)``, ``D = C(N+k-1, k)``."""
    real_dt = dtype if dtype is not None else torch.get_default_dtype()
    cdt = _complex_dtype(real_dt)
    mats = _rep.symmetric_power_rep_matrices(algebra.generators(), k)
    return torch.as_tensor(mats, dtype=cdt, device=device)


def antisymmetric_power_generators(
    algebra: LieAlgebra,
    k: int,
    *,
    dtype: torch.dtype | None = None,
    device: torch.device | None = None,
) -> Tensor:
    r"""``Lambda^k`` generators of the fundamental as ``(dim, D, D)``, ``D = C(N, k)``."""
    real_dt = dtype if dtype is not None else torch.get_default_dtype()
    cdt = _complex_dtype(real_dt)
    mats = _rep.antisymmetric_power_rep_matrices(algebra.generators(), k)
    return torch.as_tensor(mats, dtype=cdt, device=device)


def casimir_operator(matrices: Tensor) -> Tensor:
    r"""Quadratic Casimir operator ``\sum_a T^a T^a`` for stacked ``(dim, d, d)``.

    For an irrep this equals ``C_2(R) I``; used as an independent cross-check of
    :func:`casimir_eigenvalue`.
    """
    return torch.einsum("aij,ajk->ik", matrices, matrices)


def casimir_eigenvalue(
    rep: _rep.Irrep,
    *,
    dtype: torch.dtype | None = None,
    device: torch.device | None = None,
) -> Tensor:
    r"""The quadratic Casimir eigenvalue ``C_2(R)`` as a scalar tensor.

    Physics normalization ``tr_fund(T^a T^b) = 1/2 \delta^{ab}`` (so the ``su(N)``
    fundamental has ``C_2 = (N^2 - 1) / (2 N)``).
    """
    real_dt = dtype if dtype is not None else torch.get_default_dtype()
    return torch.as_tensor(float(_rep.quadratic_casimir(rep)), dtype=real_dt, device=device)


def dynkin_index_value(
    rep: _rep.Irrep,
    *,
    dtype: torch.dtype | None = None,
    device: torch.device | None = None,
) -> Tensor:
    r"""The Dynkin index ``T(R) = C_2(R) \dim(R) / \dim(G)`` as a scalar tensor."""
    real_dt = dtype if dtype is not None else torch.get_default_dtype()
    return torch.as_tensor(float(_rep.dynkin_index(rep)), dtype=real_dt, device=device)


__all__ = [
    "adjoint_generators",
    "antisymmetric_power_generators",
    "casimir_eigenvalue",
    "casimir_operator",
    "dynkin_index_value",
    "spin_matrices",
    "symmetric_power_generators",
]
