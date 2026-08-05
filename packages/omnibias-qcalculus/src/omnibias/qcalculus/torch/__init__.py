# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""PyTorch twin of the Jackson q-derivative of an activation.

``q_derivative(name, z, q)`` is the batched Jackson quotient
``(sigma(qz) - sigma(z)) / ((q-1)z)``; ``q_derivative_limit(name, z)`` is the closed-form
first derivative ``sigma'(z)`` from the backend fastpath -- the exact ``q -> 1`` limit.
Their residual ``-> 0`` as ``q -> 1`` (the **distinct** q-limit, not the ``delta -> 0``
founding collapse). Bit-identical to the JAX twin up to the base activation's libm.
"""

from __future__ import annotations

from omnibias.torch.activations.registry import get_activation
from torch import Tensor


def q_derivative(name: str, z: Tensor, q: float) -> Tensor:
    r"""Batched Jackson q-derivative ``(sigma(qz) - sigma(z)) / ((q-1)z)`` (``q != 1``).

    Evaluated away from ``z = 0``; near ``z = 0`` the quotient is ill-conditioned (the true
    value is ``sigma'(0)``, recovered by :func:`q_derivative_limit`).
    """
    if q == 1.0:
        raise ValueError("q_derivative needs q != 1 (the q -> 1 limit is q_derivative_limit)")
    spec = get_activation(name)
    return (spec.forward(q * z) - spec.forward(z)) / ((q - 1.0) * z)


def q_derivative_limit(name: str, z: Tensor) -> Tensor:
    r"""Closed-form ``sigma'(z)`` via the backend fastpath -- the exact ``q -> 1`` limit."""
    spec = get_activation(name)
    if spec.fastpath is None:
        raise NotImplementedError(f"activation {name!r} has no closed-form derivative kernel")
    out: Tensor = spec.fastpath(z, 1)
    return out


def q_derivative_residual(name: str, z: Tensor, q: float) -> Tensor:
    r"""``q_derivative - q_derivative_limit``; ``-> 0`` as ``q -> 1``."""
    return q_derivative(name, z, q) - q_derivative_limit(name, z)


__all__ = [
    "q_derivative",
    "q_derivative_limit",
    "q_derivative_residual",
]
