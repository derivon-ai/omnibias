# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Lie-algebra tensor ops (torch): structure constants, bracket, trace, matrices."""

from __future__ import annotations

import torch
from omnibias.geometry.gauge._core import kernels
from omnibias.geometry.gauge._core.lie_algebra import LieAlgebra
from torch import Tensor


def _complex_dtype(dtype: torch.dtype) -> torch.dtype:
    return torch.complex128 if dtype in (torch.float64, torch.complex128) else torch.complex64


def structure_constants(
    algebra: LieAlgebra,
    *,
    dtype: torch.dtype | None = None,
    device: torch.device | None = None,
) -> Tensor:
    """Totally antisymmetric ``f^{abc}`` as a ``(dim, dim, dim)`` real tensor."""
    dt = dtype if dtype is not None else torch.get_default_dtype()
    return torch.as_tensor(algebra.structure_constants(), dtype=dt, device=device)


def symmetric_constants(
    algebra: LieAlgebra,
    *,
    dtype: torch.dtype | None = None,
    device: torch.device | None = None,
) -> Tensor:
    """Totally symmetric ``d^{abc}`` as a ``(dim, dim, dim)`` real tensor."""
    dt = dtype if dtype is not None else torch.get_default_dtype()
    return torch.as_tensor(algebra.symmetric_constants(), dtype=dt, device=device)


def generators(
    algebra: LieAlgebra,
    *,
    dtype: torch.dtype | None = None,
    device: torch.device | None = None,
) -> Tensor:
    """The Hermitian generators ``T^a`` as a complex ``(dim, N, N)`` tensor."""
    real_dt = dtype if dtype is not None else torch.get_default_dtype()
    cdt = _complex_dtype(real_dt)
    return torch.as_tensor(algebra.generators(), dtype=cdt, device=device)


def bracket(x: Tensor, y: Tensor, algebra: LieAlgebra) -> Tensor:
    r"""Component Lie bracket ``[x, y]^c = f^{abc} x^a y^b`` for ``(B, n)`` inputs."""
    f = structure_constants(algebra, dtype=x.dtype, device=x.device)
    return kernels.lie_bracket(torch, x, y, f)


def trace_product(x: Tensor, y: Tensor) -> Tensor:
    r"""``tr(X Y) = (1/2) x^a y^a`` in the fundamental normalization -> ``(B,)``."""
    return 0.5 * torch.einsum("Ba,Ba->B", x, y)


def to_matrix(x: Tensor, algebra: LieAlgebra) -> Tensor:
    r"""Map adjoint components ``x^a`` to fundamental matrices ``x^a T^a``."""
    gen = generators(algebra, dtype=x.dtype, device=x.device)
    return kernels.to_matrix(torch, x.to(gen.dtype), gen)


def from_matrix(mat: Tensor, algebra: LieAlgebra) -> Tensor:
    r"""Project fundamental matrices back to real adjoint components ``2 Re tr(T^a X)``."""
    gen = generators(algebra, device=mat.device)
    return kernels.from_matrix(torch, mat, gen)


__all__ = [
    "bracket",
    "from_matrix",
    "generators",
    "structure_constants",
    "symmetric_constants",
    "to_matrix",
    "trace_product",
]
