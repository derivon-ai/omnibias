# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Flat, signature-aware Hodge star and field-strength dual (torch).

omnibias-geometry's :func:`hodge_star` is built from a Riemannian metric and
*ignores* the metric signature; here we provide the flat ``R^4`` Hodge star that
honours Euclidean ``(+,+,+,+)`` or Minkowski ``(-,+,+,+)`` signatures, plus the
field-strength dual ``\tilde F`` used for self-duality and the Bianchi identity.
"""

from __future__ import annotations

from typing import Any

import torch
from omnibias.geometry.gauge._core import forms, kernels
from torch import Tensor


def signature_diagonal(
    signature: tuple[int, ...],
    *,
    dtype: torch.dtype | None = None,
    device: torch.device | None = None,
) -> Tensor:
    """The flat metric diagonal ``eta`` (entries ``+/-1``) as a ``(d,)`` tensor."""
    dt = dtype if dtype is not None else torch.get_default_dtype()
    return torch.tensor(signature, dtype=dt, device=device)


def levi_civita(
    dim: int,
    *,
    dtype: torch.dtype | None = None,
    device: torch.device | None = None,
) -> Tensor:
    """The rank-``dim`` Levi-Civita symbol as a tensor."""
    dt = dtype if dtype is not None else torch.get_default_dtype()
    return torch.as_tensor(forms.levi_civita_symbol(dim), dtype=dt, device=device)


def dual_field_strength(F: Tensor, *, signature: tuple[int, ...]) -> Tensor:
    r"""Hodge dual ``\tilde F_{mu nu}^a = (1/2) eps_{mu nu rho sigma} F^{rho sigma, a}``."""
    dim = F.shape[1]
    eps = levi_civita(dim, dtype=F.dtype, device=F.device)
    eta = signature_diagonal(signature, dtype=F.dtype, device=F.device)
    return kernels.dual_field_strength(torch, F, eps, eta)


def hodge_star_flat(
    values: dict[tuple[int, ...], Any],
    degree: int,
    *,
    dim: int,
    signature: tuple[int, ...],
) -> dict[tuple[int, ...], Any]:
    r"""Flat signature-aware Hodge star of an evaluated ``k``-form (dict of tensors)."""
    return forms.hodge_star_flat(values, degree, dim, signature)


__all__ = [
    "dual_field_strength",
    "hodge_star_flat",
    "levi_civita",
    "signature_diagonal",
]
