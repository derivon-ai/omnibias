# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""q-OMBU hybrid (jax; theory 09-15).

Named ``q -> 1`` is not founding bias collapse (``delta -> 0``) and
not temperature collapse (``beta -> inf``, feasibility). Ordinary
``sigma`` cells still use founding bias collapse. do not conflate
the two.
"""

from __future__ import annotations

from collections.abc import Sequence

import jax.numpy as jnp
from jax import Array
from omnibias.qcalculus._core import hybrid as core

DISCLAIMER = core.DISCLAIMER
QOMBUConfig = core.QOMBUConfig
honesty_payload = core.honesty_payload


def q_ombu_forward(
    x: Array,
    params: Sequence[float] | None = None,
    *,
    config: QOMBUConfig | None = None,
) -> tuple[Array, Array]:
    """Returns ``(y, limit_residual)``."""
    arr = jnp.asarray(x)
    y, residual = core.q_ombu_forward(float(arr.reshape(-1)[0]), params, config=config)
    return jnp.asarray(y, dtype=arr.dtype), jnp.asarray(residual, dtype=arr.dtype)


def worked_example() -> dict[str, float]:
    return core.worked_example()


__all__ = [
    "DISCLAIMER",
    "QOMBUConfig",
    "honesty_payload",
    "q_ombu_forward",
    "worked_example",
]
