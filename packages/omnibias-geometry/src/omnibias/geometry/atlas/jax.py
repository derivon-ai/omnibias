# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""JAX builders for a region-wise Riemannian atlas (mirrors :mod:`omnibias.geometry.atlas.torch`).

Turns an :class:`~omnibias.geometry.atlas.AtlasSpec` into a :class:`~omnibias.geometry.MetricSpec`
/ :class:`~omnibias.geometry.ManifoldSpec` whose per-point metric is the partition-of-unity
blend ``g(x) = sum_l w_l(x) G_l(x)`` (a convex combination of the region metrics, hence SPD
wherever every region metric is). Downstream jax ops are reused unchanged.

Terminology: the partition gates harden as ``beta -> inf`` -- the feasibility / temperature
sense of "collapse", distinct from the **founding bias collapse** (the multi-bias
``delta -> 0`` limit to the closed-form derivative ``sigma^(K-1)``; see ``docs/theory.md``).
"""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
from omnibias.geometry._core.manifold import ManifoldSpec, MetricSpec
from omnibias.geometry.atlas._core import AtlasSpec
from omnibias.partition._core.params import region_code_matrix


def _blended_g_point(atlas: AtlasSpec) -> object:
    params = atlas.partition
    beta = atlas.beta_value()
    W = jnp.asarray(params.W)  # (depth, d)
    t = jnp.asarray(params.t)  # (depth,)
    codes = jnp.asarray(region_code_matrix(params.depth))  # (L, depth)
    region_metrics = tuple(atlas.region_metrics)

    def g_point(x: Any) -> Any:
        z = W @ x - t  # (depth,)
        g = jax.nn.sigmoid(beta * z)  # (depth,)
        factors = codes * g[None, :] + (1.0 - codes) * (1.0 - g[None, :])  # (L, depth)
        w = jnp.prod(factors, axis=1)  # (L,) partition of unity
        out = None
        for region, gl in enumerate(region_metrics):
            term = w[region] * gl(x)
            out = term if out is None else out + term
        return out

    return g_point


def blended_metric(atlas: AtlasSpec) -> MetricSpec:
    r"""The convex-combination-of-region-metrics :class:`MetricSpec` (jax)."""
    return MetricSpec(g_point=_blended_g_point(atlas), dim=atlas.dim, name=atlas.name)


def atlas_manifold(atlas: AtlasSpec) -> ManifoldSpec:
    r"""A one-chart :class:`ManifoldSpec` carrying the blended atlas metric (jax)."""
    return ManifoldSpec(name=atlas.name, dim=atlas.dim, metric=blended_metric(atlas))


__all__ = ["atlas_manifold", "blended_metric"]
