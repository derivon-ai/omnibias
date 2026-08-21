# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Integral-kernel DeepONet cell (jax; theory 09-14).

Volumetric apply. Not BEM-Net. founding bias collapse
(``delta -> 0``) of the window is not the default. Temperature
collapse (``beta -> inf``, feasibility) does not appear. do not
conflate the two.
"""

from __future__ import annotations

from collections.abc import Sequence

import jax.numpy as jnp
from jax import Array
from omnibias.pinn.operator._core import integral_kernel as core

DISCLAIMER = core.DISCLAIMER
IntegralKernelConfig = core.IntegralKernelConfig
honesty_payload = core.honesty_payload


def integral_cell(z: Array, b_lo: float, b_hi: float) -> Array:
    """``S(z + b_hi) - S(z + b_lo)``. Matches ``OperatorBlock(op='integral')``."""
    arr = jnp.asarray(z)
    vals = [core.integral_cell(float(v), b_lo, b_hi) for v in arr.reshape(-1).tolist()]
    return jnp.asarray(vals, dtype=arr.dtype).reshape(arr.shape)


def integral_kernel_apply(
    source: Array,
    coords: Array,
    params: Sequence[float] | None = None,
    *,
    config: IntegralKernelConfig | None = None,
    nodes: Array | None = None,
) -> Array:
    """Volumetric apply. Not BEM-Net."""
    arr = jnp.asarray(coords)
    src = [float(v) for v in jnp.asarray(source).reshape(-1).tolist()]
    xs = [float(v) for v in arr.reshape(-1).tolist()]
    ys = (
        [float(v) for v in jnp.asarray(nodes).reshape(-1).tolist()]
        if nodes is not None
        else None
    )
    out = core.integral_kernel_apply(src, xs, params, config=config, nodes=ys)
    return jnp.asarray(out, dtype=arr.dtype)


def worked_example() -> dict[str, float]:
    return core.worked_example()


__all__ = [
    "DISCLAIMER",
    "IntegralKernelConfig",
    "honesty_payload",
    "integral_cell",
    "integral_kernel_apply",
    "worked_example",
]
