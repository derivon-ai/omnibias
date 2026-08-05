# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Entropy-consistent residual loss (jax twin)."""

from __future__ import annotations

from collections.abc import Callable

import jax.numpy as jnp
from jax import Array


def entropy_consistent_residual(
    residual: Array,
    *,
    entropy_weight: Callable[[Array], Array] | None = None,
    state_for_weight: Array | None = None,
) -> Array:
    """Entropy-weighted MSE of a residual tensor.

    Parameters
    ----------
    residual
        Array of any shape.
    entropy_weight
        Optional callable ``u -> eta''(u)`` returning per-element
        non-negative weights. If ``None``, plain MSE is returned.
    state_for_weight
        State :math:`u` at which to evaluate ``eta''``. Defaults to
        ``residual``.
    """
    if entropy_weight is None:
        return jnp.mean(residual * residual)
    if state_for_weight is None:
        state_for_weight = residual
    weight = entropy_weight(state_for_weight)
    return jnp.mean(weight * residual * residual)


__all__ = ["entropy_consistent_residual"]
