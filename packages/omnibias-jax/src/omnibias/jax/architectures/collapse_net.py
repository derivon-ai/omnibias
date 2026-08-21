# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Collapse-Net (jax; theory 09-11).

Train uses a founding stencil; inference is founding bias collapse
(``delta -> 0``) to the named derivative. Temperature collapse
(``beta -> inf``, feasibility) does not appear. Do not conflate the
two.
"""

from __future__ import annotations

from omnibias.core import collapse_net as core

import jax.numpy as jnp
from jax import Array

DISCLAIMER = core.DISCLAIMER
CollapseNetConfig = core.CollapseNetConfig
honesty_payload = core.honesty_payload


def collapse_net_forward(
    x: Array,
    params: object | None = None,
    *,
    config: CollapseNetConfig | None = None,
) -> Array:
    arr = jnp.asarray(x)
    xs = [float(v) for v in arr.reshape(-1).tolist()]
    ys = [core.collapse_net_forward(v, params, config=config) for v in xs]
    return jnp.asarray(ys, dtype=arr.dtype).reshape(arr.shape)


def collapse_remainder(x: Array, *, config: CollapseNetConfig | None = None) -> Array:
    arr = jnp.asarray(x)
    xs = [float(v) for v in arr.reshape(-1).tolist()]
    ys = [core.collapse_remainder(v, config=config) for v in xs]
    return jnp.asarray(ys, dtype=arr.dtype).reshape(arr.shape)


def worked_example() -> dict[str, float]:
    return core.worked_example()


__all__ = [
    "CollapseNetConfig",
    "DISCLAIMER",
    "collapse_net_forward",
    "collapse_remainder",
    "honesty_payload",
    "worked_example",
]
