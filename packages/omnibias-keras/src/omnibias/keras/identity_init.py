# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Lemma 1 (identity nesting) init helpers (Keras backend).

For any base activation ``sigma`` and signs summing to one, tying all
biases gives ``sum_k s_k sigma(z + b_k) = sigma(z + b)`` bit-identically.
These helpers build the NumPy init arrays satisfying the lemma so a fresh
OMBU returns its base activation at step zero. Mirrors
:mod:`omnibias.torch.identity_init`.
"""

from __future__ import annotations

import numpy as np
from omnibias.keras.stencil import identity_signs as identity_signs

import keras

__all__ = [
    "identity_init_biases",
    "identity_init_signs",
    "identity_signs",
    "verify_identity_init",
]


def _default_dtype(dtype: str | None) -> str:
    return keras.config.floatx() if dtype is None else dtype


def identity_init_biases(
    num_channels: int,
    K: int,
    bias_value: float = 0.0,
    dtype: str | None = None,
) -> np.ndarray:
    """Tied biases for identity nesting: a ``(num_channels, K)`` array.

    ``dtype`` defaults to :func:`keras.config.floatx()`.
    """
    if num_channels < 1:
        raise ValueError(f"num_channels must be >= 1, got {num_channels}")
    if K < 1:
        raise ValueError(f"K must be >= 1, got {K}")
    return np.full((num_channels, K), bias_value, dtype=_default_dtype(dtype))


def identity_init_signs(
    num_channels: int,
    K: int,
    dtype: str | None = None,
) -> np.ndarray:
    """Signs that sum to one, broadcast across channels: ``(num_channels, K)``.

    ``dtype`` defaults to :func:`keras.config.floatx()`.
    """
    base = identity_signs(K, dtype=_default_dtype(dtype))
    return np.broadcast_to(base, (num_channels, K)).copy()


def verify_identity_init(
    biases: np.ndarray,
    signs: np.ndarray,
    atol: float = 0.0,
) -> bool:
    """True if the init satisfies Lemma 1 (tied biases, signs sum to one)."""
    biases = np.asarray(biases)
    signs = np.asarray(signs)
    if biases.shape != signs.shape:
        raise ValueError(
            f"biases and signs must have the same shape, got {biases.shape} vs {signs.shape}"
        )
    bias_constancy = float(np.abs(biases - biases[..., :1]).max())
    sign_sum_err = float(np.abs(signs.sum(axis=-1) - 1.0).max())
    return bias_constancy <= atol and sign_sum_err <= atol
