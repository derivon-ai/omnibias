# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Differentiable Boolean-function design losses (jax).

Twin of :mod:`omnibias.boolean.torch.ops.design`: spectral objectives
(:func:`degree_penalty`, :func:`influence_penalty`, :func:`target_spectrum_loss`)
built on the differentiable :mod:`~omnibias.boolean.jax.ops.spectrum` engine.
"""

from __future__ import annotations

from collections.abc import Sequence

import jax.numpy as jnp
from jax import Array
from omnibias.boolean.jax.ops.spectrum import (
    influences_diff,
    mobius_coeffs,
    walsh_coeffs,
)


def degree_penalty(values: Sequence[float] | Array) -> Array:
    """``sum_S |S| * hat f(S)^2`` -- spectral energy weighted by monomial order."""
    w = walsh_coeffs(values)
    n = w.shape[0].bit_length() - 1
    orders = jnp.asarray(
        [bin(mask).count("1") for mask in range(1 << n)], dtype=w.dtype
    )
    return (orders * w**2).sum()


def influence_penalty(values: Sequence[float] | Array) -> Array:
    """Total influence ``sum_i Inf_i`` (average sensitivity), differentiable."""
    return influences_diff(values).sum()


def target_spectrum_loss(
    values: Sequence[float] | Array,
    target: Sequence[float] | Array,
    basis: str = "walsh",
) -> Array:
    """Mean-squared error between the function's spectrum and a target spectrum."""
    if basis == "walsh":
        coeffs = walsh_coeffs(values)
    elif basis == "mobius":
        coeffs = mobius_coeffs(values)
    else:
        raise ValueError(f"basis must be 'walsh' or 'mobius', got {basis!r}")
    target_a = jnp.asarray(target, dtype=coeffs.dtype)
    return ((coeffs - target_a) ** 2).mean()


__all__ = [
    "degree_penalty",
    "influence_penalty",
    "target_spectrum_loss",
]
