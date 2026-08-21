# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Sharpness-augmented loss (jax; theory 09-23).

``L + mu * ritz``. Exact HVPs from founding bias collapse
(``delta -> 0``). Temperature collapse (``beta -> inf``,
feasibility) does not appear. do not conflate the two. Not 08-06.
"""

from __future__ import annotations

from collections.abc import Callable

from omnibias.core import sharp_loss as core

import jax.numpy as jnp
from jax import Array

DISCLAIMER = core.DISCLAIMER
SharpnessLossConfig = core.SharpnessLossConfig
honesty_payload = core.honesty_payload

ScalarFn = Callable[[Array], Array]


def exact_trace(loss_fn: ScalarFn, params: Array) -> float:
    """Exact ``Tr H`` from unit HVPs, not Hutchinson."""
    from omnibias.jax.optim import hvp

    p = jnp.reshape(params, (-1,))
    total = 0.0
    n = int(p.size)
    for i in range(n):
        eye = jnp.zeros_like(p).at[i].set(1.0)
        hv = jnp.reshape(hvp(loss_fn, p, eye), (-1,))
        total += float(hv[i])
    return total


def sharpness_augmented_loss(
    loss_fn: ScalarFn | None,
    theta: Array | float,
    *,
    config: SharpnessLossConfig | None = None,
) -> float:
    """``L + mu * ritz``. Artifact ``schedule_only`` is false."""
    cfg = core.DEFAULT_CONFIG if config is None else config
    if loss_fn is None:
        start = float(jnp.reshape(theta, (-1,))[0]) if isinstance(theta, Array) else float(theta)
        return core.sharpness_augmented_loss(None, start, config=cfg)
    p = theta if isinstance(theta, Array) else jnp.asarray([float(theta)])
    p = jnp.reshape(p, (-1,))
    raw = float(loss_fn(p))
    if cfg.kind == "trace":
        ritz = exact_trace(loss_fn, p)
    else:
        from omnibias.jax.optim_sharpness import sharpness_lambda_max

        ritz = sharpness_lambda_max(loss_fn, p, n_lanczos=cfg.lanczos_k)
    return raw + float(cfg.mu) * ritz


def worked_example() -> dict[str, float]:
    return core.worked_example()


__all__ = [
    "DISCLAIMER",
    "SharpnessLossConfig",
    "exact_trace",
    "honesty_payload",
    "sharpness_augmented_loss",
    "worked_example",
]
