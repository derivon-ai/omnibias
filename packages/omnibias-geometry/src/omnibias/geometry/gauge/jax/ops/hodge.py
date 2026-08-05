# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Flat, signature-aware Hodge star and field-strength dual (jax).

The jax twin of :mod:`omnibias.geometry.gauge.torch.ops.hodge`; see that module for the
mathematical detail. Provides the flat ``R^4`` Hodge star (Euclidean / Minkowski)
that omnibias-geometry's metric Hodge star cannot express.
"""

from __future__ import annotations

from typing import Any

import jax.numpy as jnp
from omnibias.geometry.gauge._core import forms, kernels

Array = Any


def signature_diagonal(signature: tuple[int, ...], *, dtype: Any = None) -> Array:
    """The flat metric diagonal ``eta`` (entries ``+/-1``) as a ``(d,)`` array."""
    return jnp.asarray(signature, dtype=dtype)


def levi_civita(dim: int, *, dtype: Any = None) -> Array:
    """The rank-``dim`` Levi-Civita symbol as an array."""
    return jnp.asarray(forms.levi_civita_symbol(dim), dtype=dtype)


def dual_field_strength(F: Array, *, signature: tuple[int, ...]) -> Array:
    r"""Hodge dual ``\tilde F_{mu nu}^a = (1/2) eps_{mu nu rho sigma} F^{rho sigma, a}``."""
    dim = F.shape[1]
    eps = levi_civita(dim, dtype=F.dtype)
    eta = signature_diagonal(signature, dtype=F.dtype)
    return kernels.dual_field_strength(jnp, F, eps, eta)


def hodge_star_flat(
    values: dict[tuple[int, ...], Any],
    degree: int,
    *,
    dim: int,
    signature: tuple[int, ...],
) -> dict[tuple[int, ...], Any]:
    r"""Flat signature-aware Hodge star of an evaluated ``k``-form (dict of arrays)."""
    return forms.hodge_star_flat(values, degree, dim, signature)


__all__ = [
    "dual_field_strength",
    "hodge_star_flat",
    "levi_civita",
    "signature_diagonal",
]
