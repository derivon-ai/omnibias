# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""JAX-differentiable SDF callables mirroring the numpy / torch primitives."""

from __future__ import annotations

from collections.abc import Callable

import jax.numpy as jnp
from jax import Array
from omnibias.pinn.domain._core.sdf import Box, Halfspace, Sphere

DistanceFn = Callable[[Array], Array]


def sphere_distance(center: tuple[float, ...], radius: float) -> DistanceFn:
    def _fn(coords: Array) -> Array:
        d = len(center)
        x = coords[..., :d]
        c = jnp.asarray(center, dtype=coords.dtype)
        return jnp.linalg.norm(x - c, axis=-1) - float(radius)

    return _fn


def box_distance(lo: tuple[float, ...], hi: tuple[float, ...]) -> DistanceFn:
    def _fn(coords: Array) -> Array:
        d = len(lo)
        x = coords[..., :d]
        lo_t = jnp.asarray(lo, dtype=coords.dtype)
        hi_t = jnp.asarray(hi, dtype=coords.dtype)
        center = 0.5 * (lo_t + hi_t)
        half = 0.5 * (hi_t - lo_t)
        q = jnp.abs(x - center) - half
        outside = jnp.linalg.norm(jnp.maximum(q, 0.0), axis=-1)
        inside = jnp.minimum(jnp.max(q, axis=-1), 0.0)
        return outside + inside

    return _fn


def halfspace_distance(
    normal: tuple[float, ...], point: tuple[float, ...]
) -> DistanceFn:
    def _fn(coords: Array) -> Array:
        d = len(normal)
        x = coords[..., :d]
        n = jnp.asarray(normal, dtype=coords.dtype)
        n = n / jnp.linalg.norm(n)
        p = jnp.asarray(point, dtype=coords.dtype)
        return jnp.sum((x - p) * n, axis=-1)

    return _fn


def from_primitive(sdf: Sphere | Box | Halfspace) -> DistanceFn:
    if isinstance(sdf, Sphere):
        return sphere_distance(sdf.center, sdf.radius)
    if isinstance(sdf, Box):
        return box_distance(sdf.lo, sdf.hi)
    if isinstance(sdf, Halfspace):
        return halfspace_distance(sdf.normal, sdf.point)
    raise TypeError(f"unsupported primitive type {type(sdf)!r}")


def normalize_distance(distance_fn: DistanceFn, *, eps: float = 1e-12) -> DistanceFn:
    import jax

    def _fn(coords: Array) -> Array:
        def _omega(c: Array) -> Array:
            return distance_fn(c)

        # Per-sample value and gradient via vmap of value_and_grad on a point.
        def _one(c: Array) -> Array:
            omega, g = jax.value_and_grad(lambda z: distance_fn(z[None, :])[0])(c)
            denom = jnp.sqrt(omega * omega + jnp.sum(g * g) + float(eps))
            return omega / denom

        return jax.vmap(_one)(coords)

    return _fn


__all__ = [
    "box_distance",
    "from_primitive",
    "halfspace_distance",
    "normalize_distance",
    "sphere_distance",
]
