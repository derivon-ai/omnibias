# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""BEM-Net (JAX twin; theory 02-06)."""

from __future__ import annotations

from collections.abc import Sequence

import jax.numpy as jnp
from jax import Array
from omnibias.core.conjugate import HardyDictionary
from omnibias.pinn.bem._core import (
    KernelSpec,
    Surface,
    pde_residual_off_surface,
    single_layer,
)
from omnibias.pinn.bem._core import (
    half_plane_dtn as half_plane_dtn_core,
)


def bem_evaluate(
    x: Array,
    surface: Surface,
    density: Sequence[float],
    kernel: KernelSpec,
) -> Array:
    rows = jnp.asarray(x).reshape(-1, x.shape[-1])
    dens = tuple(float(v) for v in density)
    vals = [
        single_layer((float(pt[0]), float(pt[1])), surface, dens, kernel)
        for pt in rows.tolist()
    ]
    xx = jnp.asarray(x)
    return jnp.asarray(vals, dtype=xx.dtype).reshape(xx.shape[:-1])


def bem_pde_residual(
    x: Array,
    surface: Surface,
    density: Sequence[float],
    kernel: KernelSpec,
) -> Array:
    rows = jnp.asarray(x).reshape(-1, x.shape[-1])
    dens = tuple(float(v) for v in density)
    vals = [
        pde_residual_off_surface((float(pt[0]), float(pt[1])), surface, dens, kernel)
        for pt in rows.tolist()
    ]
    xx = jnp.asarray(x)
    return jnp.asarray(vals, dtype=xx.dtype).reshape(xx.shape[:-1])


def half_plane_dtn(dictionary: HardyDictionary, coeffs: Array, y: Array) -> Array:
    c = [float(v) for v in jnp.asarray(coeffs).reshape(-1).tolist()]
    vals = [
        half_plane_dtn_core(dictionary, c, float(yi))
        for yi in jnp.asarray(y).reshape(-1).tolist()
    ]
    yy = jnp.asarray(y)
    return jnp.asarray(vals, dtype=yy.dtype).reshape(yy.shape)


__all__ = ["bem_evaluate", "bem_pde_residual", "half_plane_dtn"]
