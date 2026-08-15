# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Named linearizing transforms (JAX twin; theory 02-13)."""

from __future__ import annotations

import jax.numpy as jnp
from jax import Array
from omnibias.core.transforms_pde import cole_hopf_u, darboux_dress, miura_v


def cole_hopf_apply(x: Array, t: Array, *, nu: float = 1.0, k: float = 1.0) -> Array:
    phi = jnp.exp(k * (x + t))
    return jnp.full_like(x, cole_hopf_u(1.0, k, nu=nu)) * (phi / phi)


def cole_hopf_heat_residual(x: Array, t: Array, *, k: float = 1.0) -> Array:
    phi = jnp.exp(k * (x + t))
    return k * phi - (k * k) * phi


def miura_apply(u: Array, u_x: Array) -> Array:
    xs = jnp.asarray(u).reshape(-1).tolist()
    dxs = jnp.asarray(u_x).reshape(-1).tolist()
    vals = [miura_v(float(v), float(dv)) for v, dv in zip(xs, dxs, strict=True)]
    uu = jnp.asarray(u)
    return jnp.asarray(vals, dtype=uu.dtype).reshape(uu.shape)


def darboux_step(psi: Array, psi_x: Array, u: Array) -> Array:
    xs = jnp.asarray(psi).reshape(-1).tolist()
    dxs = jnp.asarray(psi_x).reshape(-1).tolist()
    us = jnp.asarray(u).reshape(-1).tolist()
    vals = [
        darboux_dress(float(p), float(px), float(uu))
        for p, px, uu in zip(xs, dxs, us, strict=True)
    ]
    uu = jnp.asarray(u)
    return jnp.asarray(vals, dtype=uu.dtype).reshape(uu.shape)


__all__ = ["cole_hopf_apply", "cole_hopf_heat_residual", "darboux_step", "miura_apply"]
