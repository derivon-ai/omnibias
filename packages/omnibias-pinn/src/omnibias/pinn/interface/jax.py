# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Multi-interface transmission field (JAX twin; theory 02-05)."""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence

import jax.numpy as jnp
from jax import Array
from omnibias.pinn.interface._core import Interface

_LOG2 = math.log(2.0)


def _log_cosh(z: Array) -> Array:
    az = jnp.abs(z)
    return az + jnp.log1p(jnp.exp(-2.0 * az)) - _LOG2


def _int_log_cosh(z: Array, terms: int = 24) -> Array:
    sign = jnp.sign(z)
    az = jnp.abs(z)
    acc = 0.5 * az * az - _LOG2 * az
    for k in range(1, terms + 1):
        sk = 1.0 if k % 2 == 1 else -1.0
        acc = acc + sk * (1.0 - jnp.exp(-2.0 * k * az)) / (2.0 * k * k)
    return sign * acc


def profile_array(order: int, z: Array, alpha: float) -> Array:
    az = float(alpha) * z
    n = int(order)
    if n == 0:
        return jnp.tanh(az)
    if n == 1:
        return _log_cosh(az) / float(alpha)
    if n == 2:
        return _int_log_cosh(az) / (float(alpha) ** 2)
    raise ValueError(f"unsupported profile order {order}")


def multi_interface_apply(
    x: Array,
    *,
    base: Callable[[Array], Array],
    interfaces: Sequence[Interface],
    coeffs: Array,
) -> Array:
    u = base(x)
    for g, iface in enumerate(interfaces):
        w = jnp.asarray(iface.normal, dtype=x.dtype)
        z = (x * w).sum(axis=-1) + float(iface.offset)
        n = int(iface.order) if iface.order is not None else 1
        bump = profile_array(n, z, iface.alpha)
        while bump.ndim < u.ndim:
            bump = bump[..., None]
        u = u + coeffs[g] * bump
    return u


def interface_residuals(
    interfaces: Sequence[Interface],
    coeffs: Array,
) -> dict[int, Array]:
    out: dict[int, Array] = {}
    eight = jnp.asarray(8.0, dtype=jnp.float64)
    left = jnp.tanh(-eight)
    right = jnp.tanh(eight)
    for g, iface in enumerate(interfaces):
        out[g] = coeffs[g] * (right - left) - float(iface.jump)
    return out


def hard_coeffs(interfaces: Sequence[Interface]) -> Array:
    jumps = jnp.asarray([float(i.jump) for i in interfaces], dtype=jnp.float64)
    return 0.5 * jumps


__all__ = [
    "hard_coeffs",
    "interface_residuals",
    "multi_interface_apply",
    "profile_array",
]
