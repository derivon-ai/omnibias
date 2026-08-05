# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""JAX LIF / IF neuron primitives with closed-form surrogate gradients."""

from __future__ import annotations

from collections.abc import Callable

import jax
import jax.numpy as jnp
from jax import Array
from omnibias.core.spec import ActivationSpec
from omnibias.spiking._surrogates import (
    fast_sigmoid_derivative,
    gaussian_derivative,
    normalize_surrogate_kind,
)

SurrogateArg = str | ActivationSpec[Array]


def _jax_sigmoid(z: Array) -> Array:
    return jax.nn.sigmoid(z)


def _jax_exp(z: Array) -> Array:
    return jnp.exp(z)


def _dispatch_surrogate_derivative(u: Array, kind: str) -> Array:
    key = normalize_surrogate_kind(kind)
    if key == "fast_sigmoid":
        return fast_sigmoid_derivative(u, sigmoid=_jax_sigmoid)
    return gaussian_derivative(u, exp=_jax_exp)


def _resolve_surrogate_fn(
    surrogate: SurrogateArg,
) -> Callable[[Array], Array]:
    if isinstance(surrogate, str):
        key = normalize_surrogate_kind(surrogate)
        return lambda u: _dispatch_surrogate_derivative(u, key)
    if surrogate.derivative is not None:
        return surrogate.derivative
    if surrogate.fastpath is not None:
        return lambda u: surrogate.fastpath(u, 1)
    msg = f"ActivationSpec {surrogate.name!r} has no derivative or fastpath"
    raise ValueError(msg)


def surrogate_derivative(u: Array, kind: str = "fast_sigmoid") -> Array:
    """Closed-form surrogate derivative ``d sigma / du`` for kind ``kind``."""
    return _dispatch_surrogate_derivative(u, kind)


def heaviside_spike(
    v: Array,
    threshold: float = 1.0,
    *,
    surrogate: SurrogateArg = "fast_sigmoid",
    surrogate_scale: float = 4.0,
) -> Array:
    """Hard spike forward with a closed-form surrogate backward pass."""
    u = surrogate_scale * (v - threshold)
    ds_du = _resolve_surrogate_fn(surrogate)(u)
    hard = (v >= threshold).astype(v.dtype)
    return hard + ds_du * surrogate_scale * (v - jax.lax.stop_gradient(v))


def lif_step(
    v: Array,
    x: Array,
    *,
    decay: float = 0.9,
    threshold: float = 1.0,
    surrogate: SurrogateArg = "fast_sigmoid",
    surrogate_scale: float = 4.0,
) -> tuple[Array, Array]:
    """One leaky integrate-and-fire step with soft reset."""
    v_pre = decay * v + x
    s = heaviside_spike(
        v_pre,
        threshold,
        surrogate=surrogate,
        surrogate_scale=surrogate_scale,
    )
    v_out = v_pre - s * threshold
    return s, v_out


def if_step(
    v: Array,
    x: Array,
    *,
    threshold: float = 1.0,
    surrogate: SurrogateArg = "fast_sigmoid",
    surrogate_scale: float = 4.0,
) -> tuple[Array, Array]:
    """One integrate-and-fire step (``decay=1`` LIF) with soft reset."""
    return lif_step(
        v,
        x,
        decay=1.0,
        threshold=threshold,
        surrogate=surrogate,
        surrogate_scale=surrogate_scale,
    )


__all__ = [
    "heaviside_spike",
    "if_step",
    "lif_step",
    "surrogate_derivative",
]
