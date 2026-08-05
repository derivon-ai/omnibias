# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""JAX twin of the founding finite-difference stencil operator.

Bit-identical mirror of :mod:`omnibias.difference.torch`, built on the **shared
pure-Python** stencil numbers in :mod:`omnibias.difference._core.stencil`. Pure and
``jit`` / ``vmap`` / ``grad``-safe: ``name`` / ``order`` / ``delta`` / ``stencil``
are static, only ``z`` is traced, and no host-side coercion of a tracer occurs.

``finite_difference_tower`` is the literal multi-bias sum ``sum_k s_k sigma(z + b_k)``
(numerical); ``collapse_to_derivative`` is the closed-form ``sigma^(order)(z)`` from
the backend fastpath. Their residual ``-> 0`` as ``delta -> 0`` -- the founding
bias collapse of many biases into the derivative ``sigma^(K-1)``.
"""

from __future__ import annotations

import jax.numpy as jnp
from jax import Array
from omnibias.difference._core.stencil import Stencil, stencil_offsets, stencil_signs
from omnibias.jax.activations import get_activation


def finite_difference_tower(
    name: str, z: Array, order: int, delta: float, stencil: Stencil = "central"
) -> Array:
    """Numerical multi-bias finite-difference stencil ``sum_k s_k sigma(z + b_k)``.

    The signs and offsets come from the shared pure-Python
    :mod:`omnibias.difference._core.stencil`, so the value is bit-identical to the
    PyTorch twin up to the base activation's per-backend rounding.
    """
    spec = get_activation(name)
    signs = jnp.asarray(stencil_signs(order, delta, stencil), dtype=z.dtype)
    offsets = jnp.asarray(stencil_offsets(order, delta, stencil), dtype=z.dtype)
    values = spec.forward(z[..., None] + offsets)  # (..., K)
    return (values * signs).sum(axis=-1)


def collapse_to_derivative(name: str, z: Array, order: int) -> Array:
    """Closed-form ``sigma^(order)(z)`` via the backend fastpath (the exact limit)."""
    if order < 0:
        raise ValueError(f"order must be >= 0, got {order}")
    spec = get_activation(name)
    if spec.fastpath is None:
        raise NotImplementedError(f"activation {name!r} has no closed-form derivative kernel")
    out: Array = spec.fastpath(z, order)
    return out


def finite_difference_residual(
    name: str, z: Array, order: int, delta: float, stencil: Stencil = "central"
) -> Array:
    """``finite_difference_tower - collapse_to_derivative``; ``-> 0`` as ``delta -> 0``."""
    return finite_difference_tower(name, z, order, delta, stencil) - collapse_to_derivative(
        name, z, order
    )


__all__ = [
    "collapse_to_derivative",
    "finite_difference_residual",
    "finite_difference_tower",
]
