# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Riccati flow net (jax; theory 09-10).

Depth is Riccati time, not a DEQ and not a CNF. This forward is not
founding bias collapse (no ``delta -> 0`` pack). Temperature collapse
(``beta -> inf``, feasibility) does not appear. Do not conflate the
two.
"""

from __future__ import annotations

from omnibias.core import riccati_flow as core

import jax.numpy as jnp
from jax import Array

DISCLAIMER = core.DISCLAIMER
RiccatiFlowConfig = core.RiccatiFlowConfig
honesty_payload = core.honesty_payload


def riccati_flow(s0: Array, *, config: RiccatiFlowConfig | None = None) -> Array:
    arr = jnp.asarray(s0)
    xs = [float(v) for v in arr.reshape(-1).tolist()]
    ys = [core.riccati_flow(x, config=config) for x in xs]
    return jnp.asarray(ys, dtype=arr.dtype).reshape(arr.shape)


def worked_example() -> dict[str, float]:
    return core.worked_example()


__all__ = [
    "DISCLAIMER",
    "RiccatiFlowConfig",
    "honesty_payload",
    "riccati_flow",
    "worked_example",
]
