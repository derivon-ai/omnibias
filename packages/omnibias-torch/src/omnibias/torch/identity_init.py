# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Lemma 1 (identity nesting) helpers.

Lemma 1 of the multi-bias paper says: for any base activation ``sigma``
and any signs ``s_1, ..., s_K`` summing to one, tying ``b_1 = ... = b_K = b``
gives ``sum_k s_k * sigma(z + b_k) = sigma(z + b)`` exactly (bit-identical
on any IEEE-754 implementation).

These helpers build init tensors that satisfy the lemma so a fresh OMBU
returns its base activation at step zero, no matter what K is.
"""

from __future__ import annotations

from omnibias.torch.stencil import identity_signs as identity_signs

import torch
from torch import Tensor

__all__ = ["identity_init_biases", "identity_init_signs", "identity_signs", "verify_identity_init"]


def identity_init_biases(
    num_channels: int,
    K: int,
    bias_value: float = 0.0,
    dtype: torch.dtype | None = None,
) -> Tensor:
    """Tied biases for identity nesting.

    Returns a ``(num_channels, K)`` tensor with every entry equal to
    ``bias_value``. Combined with :func:`identity_init_signs`, makes the
    OMBU forward equal ``sigma(z + bias_value)`` bit-identically.

    ``dtype`` defaults to :func:`torch.get_default_dtype()` so callers
    that have set a non-default global dtype (e.g. ``float64`` for
    scientific work) get matching parameters by default.
    """
    if num_channels < 1:
        raise ValueError(f"num_channels must be >= 1, got {num_channels}")
    if K < 1:
        raise ValueError(f"K must be >= 1, got {K}")
    if dtype is None:
        dtype = torch.get_default_dtype()
    return torch.full((num_channels, K), bias_value, dtype=dtype)


def identity_init_signs(
    num_channels: int,
    K: int,
    dtype: torch.dtype | None = None,
) -> Tensor:
    """Signs that sum to one, broadcast across channels.

    ``dtype`` defaults to :func:`torch.get_default_dtype()`.
    """
    base = identity_signs(K, dtype=dtype)
    return base.unsqueeze(0).expand(num_channels, K).contiguous()


def verify_identity_init(
    biases: Tensor,
    signs: Tensor,
    atol: float = 0.0,
) -> bool:
    """Cheap sanity check: returns True if the init satisfies Lemma 1.

    Tests two conditions per channel:

    1. All biases are equal within ``atol``.
    2. The signs sum to one within ``atol``.
    """
    if biases.shape != signs.shape:
        raise ValueError(
            f"biases and signs must have the same shape, got {biases.shape} vs {signs.shape}"
        )
    bias_constancy = (biases - biases[..., :1]).abs().max().item()
    sign_sum_err = (signs.sum(dim=-1) - 1.0).abs().max().item()
    return bias_constancy <= atol and sign_sum_err <= atol
