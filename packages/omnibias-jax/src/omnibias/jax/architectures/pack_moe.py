# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Pack-MoE (jax; theory 09-07).

The router is slab mass (window knob), not softmax. Expert collapse
heads may use founding bias collapse (``delta -> 0``). Temperature
collapse (``beta -> inf``, feasibility) is recorded when ``beta != 1``.
Do not conflate the two.
"""

from __future__ import annotations

from collections.abc import Sequence

from omnibias.core import pack_moe as core

import jax.numpy as jnp
from jax import Array

DISCLAIMER = core.DISCLAIMER
ExpertWindow = core.ExpertWindow
PackMoEConfig = core.PackMoEConfig
honesty_payload = core.honesty_payload


def pack_moe_forward(
    x: Array,
    experts: Sequence[float] | Array,
    windows: Sequence[ExpertWindow],
    *,
    config: PackMoEConfig | None = None,
) -> Array:
    xs = [float(v) for v in jnp.asarray(x).reshape(-1).tolist()]
    fs = [float(v) for v in jnp.asarray(experts).reshape(-1).tolist()]
    ys = [core.pack_moe_forward(xi, fs, windows, config=config) for xi in xs]
    return jnp.asarray(ys, dtype=jnp.asarray(x).dtype).reshape(jnp.asarray(x).shape)


def worked_example() -> dict[str, float]:
    return core.worked_example()


__all__ = [
    "DISCLAIMER",
    "ExpertWindow",
    "PackMoEConfig",
    "honesty_payload",
    "pack_moe_forward",
    "worked_example",
]
