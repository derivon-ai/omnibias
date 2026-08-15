# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Chart-coordinate bias scan (JAX twin; theory 02-08)."""

from __future__ import annotations

import jax.numpy as jnp
from jax import Array
from omnibias.core.scan import BankSpec
from omnibias.geometry._core.charts import ChartSpec
from omnibias.geometry.jax.ops.pullback import pullback_metric


def chart_scan(
    chart: ChartSpec,
    x: Array,
    direction: Array,
    offsets: BankSpec,
    *,
    metric_correction: bool = True,
) -> Array:
    d = direction / jnp.clip(jnp.linalg.norm(direction), a_min=1e-12)
    z = (x * d).sum(axis=-1)
    if metric_correction:
        g = pullback_metric(x, chart)
        gd = g @ d
        gvv = jnp.clip((gd * d).sum(axis=-1), a_min=1e-18)
        z = z * jnp.sqrt(gvv)
    off = jnp.asarray(list(offsets.offsets), dtype=jnp.asarray(x).dtype)
    return z[..., None] + off


__all__ = ["chart_scan"]
