# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Bias-collapse stencils (Keras backend).

These build the init *values* (as NumPy arrays) for the K biases and
signs of a multi-bias unit. They are plain NumPy because they only run at
construction time; the live forward pass uses ``keras.ops``. Mirrors
:mod:`omnibias.torch.stencil`.
"""

from __future__ import annotations

import math
from typing import Literal

import numpy as np

import keras


def _default_dtype(dtype: str | None) -> str:
    return keras.config.floatx() if dtype is None else dtype


def identity_signs(K: int, dtype: str | None = None) -> np.ndarray:
    """Signs that satisfy Lemma 1 (identity nesting); ``sum(s) = 1``.

    For odd K the alternating pattern ``(+1, -1, +1, ..., +1)`` sums to
    one; for even K we set ``s[0] = +2`` so the sum is one while ``s[1:]``
    stays strictly alternating.

    ``dtype`` defaults to :func:`keras.config.floatx()`.
    """
    if K < 1:
        raise ValueError(f"K must be >= 1, got {K}")
    dtype = _default_dtype(dtype)
    s = np.array([(-1.0) ** k for k in range(K)], dtype=dtype)
    if K % 2 == 0:
        s[0] = 2.0
    return s


def forward_difference_signs(
    K: int, delta: float, dtype: str | None = None
) -> np.ndarray:
    """Forward-difference signs at order ``K-1`` with step ``delta``.

    ``dtype`` defaults to :func:`keras.config.floatx()`.
    """
    if K < 1:
        raise ValueError(f"K must be >= 1, got {K}")
    if delta <= 0 and K > 1:
        raise ValueError(f"delta must be > 0 for K > 1 (got delta={delta}, K={K})")
    dtype = _default_dtype(dtype)
    if K == 1:
        return np.array([1.0], dtype=dtype)
    inv_scale = 1.0 / (delta ** (K - 1))
    signs = [((-1.0) ** (K - 1 - j)) * math.comb(K - 1, j) * inv_scale for j in range(K)]
    return np.array(signs, dtype=dtype)


def central_difference_signs(
    K: int, delta: float, dtype: str | None = None
) -> np.ndarray:
    """Central-difference signs (same magnitudes as forward-difference)."""
    return forward_difference_signs(K, delta, dtype=dtype)


def forward_bias_offsets(K: int, delta: float, dtype: str | None = None) -> np.ndarray:
    """Offsets ``(0, delta, 2*delta, ..., (K-1)*delta)``.

    ``dtype`` defaults to :func:`keras.config.floatx()`.
    """
    return np.arange(K, dtype=_default_dtype(dtype)) * delta


def central_bias_offsets(K: int, delta: float, dtype: str | None = None) -> np.ndarray:
    """Symmetric offsets ``((k - (K+1)/2) * delta)_k=1..K`` (zero mean).

    ``dtype`` defaults to :func:`keras.config.floatx()`.
    """
    centre = (K + 1) / 2.0
    return (np.arange(1, K + 1, dtype=_default_dtype(dtype)) - centre) * delta


def stencil_offsets(
    K: int,
    delta: float,
    stencil: Literal["forward", "central"] = "central",
    dtype: str | None = None,
) -> np.ndarray:
    """Bias offsets for the chosen stencil."""
    if stencil == "forward":
        return forward_bias_offsets(K, delta, dtype=dtype)
    if stencil == "central":
        return central_bias_offsets(K, delta, dtype=dtype)
    raise ValueError(f"Unknown stencil {stencil!r}; expected 'forward' or 'central'.")


def stencil_signs(
    K: int,
    delta: float,
    stencil: Literal["forward", "central"] = "central",
    dtype: str | None = None,
) -> np.ndarray:
    """Signs matched to the chosen stencil (same magnitudes for both)."""
    if stencil not in ("forward", "central"):
        raise ValueError(f"Unknown stencil {stencil!r}; expected 'forward' or 'central'.")
    return forward_difference_signs(K, delta, dtype=dtype)


__all__ = [
    "central_bias_offsets",
    "central_difference_signs",
    "forward_bias_offsets",
    "forward_difference_signs",
    "identity_signs",
    "stencil_offsets",
    "stencil_signs",
]
