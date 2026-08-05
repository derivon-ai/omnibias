# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Soft logic gates (jax): product-t-norm and hard-forward threshold gates.

Bit-identical twin of :mod:`omnibias.boolean.torch.ops.gates`. The product-t-norm
gates are the multilinear extensions of the Boolean gates (exact on the cube
vertices, smooth inside); the threshold gates have a hard ``{0, 1}`` forward with
the ``sigmoid(beta z)`` surrogate backward from
:func:`omnibias.binary.jax.ops.binarize01`.
"""

from __future__ import annotations

from collections.abc import Sequence

from jax import Array
from omnibias.binary.jax.ops import binarize01

# --- product-t-norm (multilinear) gates: exact on {0,1} vertices ------------


def soft_not(a: Array) -> Array:
    """``NOT a = 1 - a``."""
    return 1.0 - a


def soft_and(a: Array, b: Array) -> Array:
    """``a AND b = a * b`` (product t-norm)."""
    return a * b


def soft_or(a: Array, b: Array) -> Array:
    """``a OR b = a + b - a*b`` (probabilistic sum)."""
    return a + b - a * b


def soft_xor(a: Array, b: Array) -> Array:
    """``a XOR b = a + b - 2*a*b``."""
    return a + b - 2.0 * a * b


def soft_nand(a: Array, b: Array) -> Array:
    """``a NAND b = 1 - a*b``."""
    return 1.0 - a * b


def soft_nor(a: Array, b: Array) -> Array:
    """``a NOR b = 1 - (a + b - a*b)``."""
    return 1.0 - (a + b - a * b)


def soft_xnor(a: Array, b: Array) -> Array:
    """``a XNOR b = 1 - (a + b - 2*a*b)``."""
    return 1.0 - (a + b - 2.0 * a * b)


def soft_implies(a: Array, b: Array) -> Array:
    """``a -> b = 1 - a + a*b``."""
    return 1.0 - a + a * b


def soft_majority3(a: Array, b: Array, c: Array) -> Array:
    """3-input majority ``ab + bc + ca - 2abc`` (exact on vertices)."""
    return a * b + b * c + c * a - 2.0 * a * b * c


# --- threshold gates: hard {0,1} forward, sigmoid surrogate backward --------


def linear_threshold(
    inputs: Sequence[Array], weights: Sequence[float], bias: float, beta: float = 10.0
) -> Array:
    """Hard linear-threshold gate ``H(sum_k w_k x_k + bias)`` with surrogate backward."""
    if len(inputs) != len(weights):
        raise ValueError("inputs and weights must have equal length")
    acc = inputs[0] * weights[0]
    for x, w in zip(inputs[1:], weights[1:], strict=False):
        acc = acc + x * w
    return binarize01(acc + bias, beta)


def threshold_and(a: Array, b: Array, beta: float = 10.0) -> Array:
    """Hard ``a AND b`` (fires iff ``a + b >= 2``); ``sigmoid(beta z)`` backward."""
    return binarize01(a + b - 1.5, beta)


def threshold_or(a: Array, b: Array, beta: float = 10.0) -> Array:
    """Hard ``a OR b`` (fires iff ``a + b >= 1``); ``sigmoid(beta z)`` backward."""
    return binarize01(a + b - 0.5, beta)


def threshold_not(a: Array, beta: float = 10.0) -> Array:
    """Hard ``NOT a`` (fires iff ``a < 0.5``); ``sigmoid(beta z)`` backward."""
    return binarize01(0.5 - a, beta)


__all__ = [
    "linear_threshold",
    "soft_and",
    "soft_implies",
    "soft_majority3",
    "soft_nand",
    "soft_nor",
    "soft_not",
    "soft_or",
    "soft_xnor",
    "soft_xor",
    "threshold_and",
    "threshold_not",
    "threshold_or",
]
