# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Learnable fractional order (jax): functional twin of the torch ``LearnableOrder``.

JAX keeps parameters as plain leaves, so the reparametrisation is a pair of
functions rather than a stateful module: :func:`init_order` returns the
unconstrained ``raw`` leaf whose constrained order equals ``init``, and
:func:`constrain_order` maps ``raw`` through a sigmoid into the open band
``(lo, hi)``.  Feed ``constrain_order(raw, ...)`` into a fractional op's ``alpha``.
"""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
from jax import Array


def init_order(init: float = 0.5, *, lo: float = 0.0, hi: float = 2.0) -> Array:
    r"""The unconstrained ``raw`` leaf whose constrained order equals ``init``.

    Inverse of :func:`constrain_order`: returns ``logit((init - lo) / (hi - lo))``.
    """
    if not lo < hi:
        raise ValueError(f"require lo < hi, got lo={lo}, hi={hi}")
    if not (lo < init < hi):
        raise ValueError(f"init {init} must lie in the open interval ({lo}, {hi})")
    p = (init - lo) / (hi - lo)
    return jnp.asarray(math.log(p / (1.0 - p)))


def constrain_order(raw: Array, *, lo: float = 0.0, hi: float = 2.0) -> Array:
    r"""Map an unconstrained ``raw`` leaf into the open band ``(lo, hi)``.

    Returns ``lo + (hi - lo) * sigmoid(raw)`` -- smooth and differentiable in
    ``raw``, so the order trains end-to-end.
    """
    return lo + (hi - lo) * jax.nn.sigmoid(raw)


__all__ = ["constrain_order", "init_order"]
