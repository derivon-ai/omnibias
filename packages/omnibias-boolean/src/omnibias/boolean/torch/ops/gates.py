# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Soft logic gates (torch): product-t-norm and hard-forward threshold gates.

Two families of differentiable gates on ``{0, 1}`` inputs:

* **product-t-norm gates** -- the multilinear extensions of the Boolean gates
  (``and = ab``, ``or = a + b - ab``, ``xor = a + b - 2ab``, ...). They are *exact*
  on the cube vertices ``{0, 1}^k`` and smooth in between, so they compose into
  differentiable circuits whose value at any Boolean input is the true gate output;
* **threshold gates** -- a hard ``{0, 1}`` forward (a linear threshold) with the
  smooth ``sigmoid(beta z)`` surrogate backward from
  :func:`omnibias.binary.torch.ops.binarize01`. Use these when you want the forward
  pass to stay exactly Boolean while still receiving a gradient.
"""

from __future__ import annotations

from collections.abc import Sequence

from omnibias.binary.torch.ops import binarize01
from torch import Tensor

# --- product-t-norm (multilinear) gates: exact on {0,1} vertices ------------


def soft_not(a: Tensor) -> Tensor:
    """``NOT a = 1 - a``."""
    return 1.0 - a


def soft_and(a: Tensor, b: Tensor) -> Tensor:
    """``a AND b = a * b`` (product t-norm)."""
    return a * b


def soft_or(a: Tensor, b: Tensor) -> Tensor:
    """``a OR b = a + b - a*b`` (probabilistic sum)."""
    return a + b - a * b


def soft_xor(a: Tensor, b: Tensor) -> Tensor:
    """``a XOR b = a + b - 2*a*b``."""
    return a + b - 2.0 * a * b


def soft_nand(a: Tensor, b: Tensor) -> Tensor:
    """``a NAND b = 1 - a*b``."""
    return 1.0 - a * b


def soft_nor(a: Tensor, b: Tensor) -> Tensor:
    """``a NOR b = 1 - (a + b - a*b)``."""
    return 1.0 - (a + b - a * b)


def soft_xnor(a: Tensor, b: Tensor) -> Tensor:
    """``a XNOR b = 1 - (a + b - 2*a*b)``."""
    return 1.0 - (a + b - 2.0 * a * b)


def soft_implies(a: Tensor, b: Tensor) -> Tensor:
    """``a -> b = 1 - a + a*b``."""
    return 1.0 - a + a * b


def soft_majority3(a: Tensor, b: Tensor, c: Tensor) -> Tensor:
    """3-input majority ``ab + bc + ca - 2abc`` (exact on vertices)."""
    return a * b + b * c + c * a - 2.0 * a * b * c


# --- threshold gates: hard {0,1} forward, sigmoid surrogate backward --------


def linear_threshold(
    inputs: Sequence[Tensor], weights: Sequence[float], bias: float, beta: float = 10.0
) -> Tensor:
    """Hard linear-threshold gate ``H(sum_k w_k x_k + bias)`` with surrogate backward."""
    if len(inputs) != len(weights):
        raise ValueError("inputs and weights must have equal length")
    acc = inputs[0] * weights[0]
    for x, w in zip(inputs[1:], weights[1:], strict=False):
        acc = acc + x * w
    return binarize01(acc + bias, beta)


def threshold_and(a: Tensor, b: Tensor, beta: float = 10.0) -> Tensor:
    """Hard ``a AND b`` (fires iff ``a + b >= 2``); ``sigmoid(beta z)`` backward."""
    return binarize01(a + b - 1.5, beta)


def threshold_or(a: Tensor, b: Tensor, beta: float = 10.0) -> Tensor:
    """Hard ``a OR b`` (fires iff ``a + b >= 1``); ``sigmoid(beta z)`` backward."""
    return binarize01(a + b - 0.5, beta)


def threshold_not(a: Tensor, beta: float = 10.0) -> Tensor:
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
