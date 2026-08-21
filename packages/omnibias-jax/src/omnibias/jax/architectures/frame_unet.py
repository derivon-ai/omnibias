# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Frame-UNet (jax; theory 09-04).

Encoder collapse is founding bias collapse (``delta -> 0``). Decoder
gaps are the window knob. Temperature collapse (``beta -> inf``,
feasibility) does not appear. Do not conflate the two. Band skip is
not a collapse head. ``sigma'`` is not admissible.
"""

from __future__ import annotations

from collections.abc import Sequence

from omnibias.core import frame_unet as core

import jax.numpy as jnp
from jax import Array

DISCLAIMER = core.DISCLAIMER
FrameUNetConfig = core.FrameUNetConfig
honesty_payload = core.honesty_payload


def frame_unet_forward(
    x: Array,
    *,
    config: FrameUNetConfig | None = None,
    amps: Sequence[float] | Array | None = None,
    seed: int = 0,
) -> tuple[Array, dict[str, object]]:
    arr = jnp.asarray(x)
    xs = [float(v) for v in arr.reshape(-1).tolist()]
    coef: Sequence[float] | None
    if amps is None:
        coef = None
    else:
        coef = [float(v) for v in jnp.asarray(amps).reshape(-1).tolist()]
    ys: list[float] = []
    last: dict[str, object] = {"band": [], "collapse": [], "kinds": ("band", "collapse")}
    for xi in xs:
        y, skips = core.frame_unet_forward(xi, config=config, amps=coef, seed=seed)
        ys.append(y)
        last = skips
    return jnp.asarray(ys, dtype=arr.dtype).reshape(arr.shape), last


def worked_example() -> dict[str, float]:
    return core.worked_example()


__all__ = [
    "DISCLAIMER",
    "FrameUNetConfig",
    "frame_unet_forward",
    "honesty_payload",
    "worked_example",
]
