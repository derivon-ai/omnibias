# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""JAX twin of the Jackson q-derivative of an activation.

Bit-identical mirror of :mod:`omnibias.qcalculus.torch`, ``jit`` / ``vmap`` / ``grad``-safe
(``name`` and ``q`` are static, only ``z`` is traced). ``q_derivative`` is the batched
Jackson quotient ``(sigma(qz) - sigma(z)) / ((q-1)z)``; ``q_derivative_limit`` is the
closed-form ``sigma'(z)`` -- the exact ``q -> 1`` limit (the **distinct** q-limit, not the
``delta -> 0`` founding collapse).
"""

from __future__ import annotations

from jax import Array
from omnibias.jax.activations import get_activation


def q_derivative(name: str, z: Array, q: float) -> Array:
    r"""Batched Jackson q-derivative ``(sigma(qz) - sigma(z)) / ((q-1)z)`` (``q != 1``)."""
    if q == 1.0:
        raise ValueError("q_derivative needs q != 1 (the q -> 1 limit is q_derivative_limit)")
    spec = get_activation(name)
    return (spec.forward(q * z) - spec.forward(z)) / ((q - 1.0) * z)


def q_derivative_limit(name: str, z: Array) -> Array:
    r"""Closed-form ``sigma'(z)`` via the backend fastpath -- the exact ``q -> 1`` limit."""
    spec = get_activation(name)
    if spec.fastpath is None:
        raise NotImplementedError(f"activation {name!r} has no closed-form derivative kernel")
    out: Array = spec.fastpath(z, 1)
    return out


def q_derivative_residual(name: str, z: Array, q: float) -> Array:
    r"""``q_derivative - q_derivative_limit``; ``-> 0`` as ``q -> 1``."""
    return q_derivative(name, z, q) - q_derivative_limit(name, z)


__all__ = [
    "q_derivative",
    "q_derivative_limit",
    "q_derivative_residual",
]
