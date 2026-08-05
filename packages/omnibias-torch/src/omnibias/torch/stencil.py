# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Bias-collapse stencils.

The K-bias multi-bias unit ``f_K(z) = sum_k s_k * sigma(z + b_k)`` becomes
the (K-1)-th derivative of ``sigma`` in a specific limit. The exact limit
depends on the choice of stencil for ``b_k`` and the matching rescaling
of the signs ``s_k``. Two stencils are supported.

**Forward-difference stencil.**

    b_k = b + (k - 1) * delta            for k = 1 .. K
    s_k = (-1)^(K-k) * binom(K-1, k-1) / delta^(K-1)

Then ``f_K(z) -> sigma^(K-1)(z + b)`` as ``delta -> 0+`` with first-order
accuracy ``O(delta)`` for sigma in C^K.

**Central-difference stencil.**

    b_k = b + (k - (K+1)/2) * delta      for k = 1 .. K
    s_k = (-1)^(K-k) * binom(K-1, k-1) / delta^(K-1)

The same scaling, but biases are centred around ``b`` instead of starting
there. Order improves to ``O(delta^2)`` for sigma in C^(K+1).

We expose plain ``torch.Tensor`` builders for the stencils so the unit
can pre-compute them once per (K, stencil) combination and only rescale
when ``delta`` is updated.

Terminology: this ``delta -> 0`` limit is *the* **founding bias collapse**
(``K`` biases collapse onto one value, yielding ``sigma^(K-1)``). Do not confuse
it with **temperature collapse**, the ``beta -> inf`` penalty in
:mod:`omnibias.convex` (a feasibility step, not a derivative). See ``docs/theory.md``.
"""

from __future__ import annotations

import math
from typing import Literal

import torch
from torch import Tensor


def identity_signs(K: int, dtype: torch.dtype | None = None) -> Tensor:
    """Signs that satisfy Lemma 1 (identity nesting).

    Returns ``s`` with ``sum(s) = 1``.

    - For odd K, the canonical alternating pattern ``(+1, -1, +1, ..., +1)``
      already sums to one.
    - For even K, alternating gives ``sum = 0``; we double the first sign
      (so ``s[0] = +2``) to bring the sum to one. The last K-1 entries
      remain strictly alternating, which keeps the unit balanced and
      matches the empirical pattern reported in the multi-bias paper's
      XOR construction (Figure 1, trained ``s ~ (+2.18, -4.42)``).

    ``dtype`` defaults to :func:`torch.get_default_dtype()` so callers
    that have set a non-default global dtype (e.g. ``float64`` for
    scientific work) get matching parameters by default.
    """
    if K < 1:
        raise ValueError(f"K must be >= 1, got {K}")
    if dtype is None:
        dtype = torch.get_default_dtype()
    s = torch.tensor([(-1.0) ** k for k in range(K)], dtype=dtype)  # +1, -1, +1, ...
    if K % 2 == 0:
        s[0] = 2.0  # makes sum = 1 while preserving alternation in s[1:]
    return s


def forward_difference_signs(
    K: int, delta: float, dtype: torch.dtype | None = None
) -> Tensor:
    """Forward-difference signs at order K-1 with step ``delta``.

    Produces ``s_k = (-1)^(K-k) * binom(K-1, k-1) / delta^(K-1)`` so that
    ``sum_k s_k * sigma(z + b + (k-1)*delta) -> sigma^(K-1)(z + b)``
    as ``delta -> 0+``.

    For ``K = 1`` returns ``[1.0]`` (the standard single-bias unit).

    ``dtype`` defaults to :func:`torch.get_default_dtype()`.
    """
    if K < 1:
        raise ValueError(f"K must be >= 1, got {K}")
    if delta <= 0 and K > 1:
        raise ValueError(f"delta must be > 0 for K > 1 (got delta={delta}, K={K})")
    if dtype is None:
        dtype = torch.get_default_dtype()
    if K == 1:
        return torch.tensor([1.0], dtype=dtype)
    inv_scale = 1.0 / (delta ** (K - 1))
    signs = [((-1.0) ** (K - 1 - j)) * math.comb(K - 1, j) * inv_scale for j in range(K)]
    return torch.tensor(signs, dtype=dtype)


def central_difference_signs(
    K: int, delta: float, dtype: torch.dtype | None = None
) -> Tensor:
    """Central-difference signs.

    Same magnitudes as :func:`forward_difference_signs`; the difference is
    in where the biases are placed (see :func:`central_bias_offsets`).
    """
    return forward_difference_signs(K, delta, dtype=dtype)


def forward_bias_offsets(K: int, delta: float, dtype: torch.dtype | None = None) -> Tensor:
    """Offsets ``(0, delta, 2*delta, ..., (K-1)*delta)``.

    ``dtype`` defaults to :func:`torch.get_default_dtype()`.
    """
    if dtype is None:
        dtype = torch.get_default_dtype()
    return torch.arange(K, dtype=dtype) * delta


def central_bias_offsets(K: int, delta: float, dtype: torch.dtype | None = None) -> Tensor:
    """Symmetric offsets ``((k - (K+1)/2) * delta)_k=1..K``.

    The mean is zero so the symmetric stencil is centred on ``b``.

    ``dtype`` defaults to :func:`torch.get_default_dtype()`.
    """
    if dtype is None:
        dtype = torch.get_default_dtype()
    centre = (K + 1) / 2.0
    return (torch.arange(1, K + 1, dtype=dtype) - centre) * delta


def stencil_offsets(
    K: int,
    delta: float,
    stencil: Literal["forward", "central"] = "central",
    dtype: torch.dtype | None = None,
) -> Tensor:
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
    dtype: torch.dtype | None = None,
) -> Tensor:
    """Signs matched to the chosen stencil (same magnitudes for both)."""
    if stencil not in ("forward", "central"):
        raise ValueError(f"Unknown stencil {stencil!r}; expected 'forward' or 'central'.")
    return forward_difference_signs(K, delta, dtype=dtype)
