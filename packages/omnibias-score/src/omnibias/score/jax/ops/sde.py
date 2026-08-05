# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""SDE / score operators (jax). Bit-identical twin of the torch module."""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax.numpy as jnp
from jax import Array
from omnibias.fields.jax.ops.basic import gradient, value
from omnibias.fields.jax.ops.high_order import hessian

if TYPE_CHECKING:  # pragma: no cover
    from omnibias.fields._core.state import FieldState


def _quad(a: Array, hess: Array) -> Array:
    if a.ndim == 2:
        return jnp.einsum("ij,bij->b", a, hess)
    return jnp.einsum("bij,bij->b", a, hess)


def score(state: FieldState, name: str, *, eps: float = 0.0) -> Array:
    r"""Score ``grad log p = grad p / p`` of shape ``(B, d)``."""
    p = value(state, name)
    g = gradient(state, name)
    return g / (p[..., None] + eps)


def ito_generator(
    state: FieldState, name: str, *, drift: Array, diffusion: Array,
) -> Array:
    r"""Ito generator ``L f = b . grad f + 1/2 tr(a hess f)`` of shape ``(B,)``."""
    grad = gradient(state, name)
    hess = hessian(state, name)
    return jnp.einsum("bi,bi->b", drift, grad) + 0.5 * _quad(diffusion, hess)


def fokker_planck(
    state: FieldState,
    name: str,
    *,
    drift: Array,
    diffusion: Array,
    drift_divergence: Array,
) -> Array:
    r"""Fokker-Planck adjoint ``L* p`` of shape ``(B,)`` (constant diffusion)."""
    p = value(state, name)
    gradp = gradient(state, name)
    hessp = hessian(state, name)
    transport = drift_divergence * p + jnp.einsum("bi,bi->b", drift, gradp)
    return -transport + 0.5 * _quad(diffusion, hessp)


__all__ = ["fokker_planck", "ito_generator", "score"]
