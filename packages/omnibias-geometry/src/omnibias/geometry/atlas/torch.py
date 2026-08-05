# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Torch builders for a region-wise Riemannian atlas (a bridge on omnibias-partition).

Turns an :class:`~omnibias.geometry.atlas.AtlasSpec` into the existing
:class:`~omnibias.geometry.MetricSpec` / :class:`~omnibias.geometry.ManifoldSpec` whose
per-point metric is the partition-of-unity blend ``g(x) = sum_l w_l(x) G_l(x)`` -- a convex
combination of the region metrics, hence symmetric positive-definite wherever every region
metric is. All downstream torch ops (``christoffel`` / ``scalar_curvature`` /
``geodesic_rhs`` / ``laplace_beltrami``) then work unchanged (they take forward-mode autodiff
of this ``g_point``).

Terminology: the partition gates harden as ``beta -> inf`` -- the feasibility / temperature
sense of "collapse", distinct from the **founding bias collapse** (the multi-bias
``delta -> 0`` limit to the closed-form derivative ``sigma^(K-1)``; see ``docs/theory.md``).
"""

from __future__ import annotations

import torch
from omnibias.geometry._core.manifold import ManifoldSpec, MetricSpec
from omnibias.geometry.atlas._core import AtlasSpec
from omnibias.partition._core.params import region_code_matrix
from torch import Tensor


def _blended_g_point(atlas: AtlasSpec) -> object:
    params = atlas.partition
    beta = atlas.beta_value()
    W = torch.as_tensor(params.W, dtype=torch.float64)  # (depth, d)
    t = torch.as_tensor(params.t, dtype=torch.float64)  # (depth,)
    codes = torch.as_tensor(region_code_matrix(params.depth), dtype=torch.float64)  # (L, depth)
    region_metrics = tuple(atlas.region_metrics)

    def g_point(x: Tensor) -> Tensor:
        z = W @ x - t  # (depth,)
        g = torch.sigmoid(beta * z)  # (depth,)
        factors = codes * g.unsqueeze(0) + (1.0 - codes) * (1.0 - g.unsqueeze(0))  # (L, depth)
        w = torch.prod(factors, dim=1)  # (L,) partition of unity
        out: Tensor | None = None
        for region, gl in enumerate(region_metrics):
            term = w[region] * gl(x)
            out = term if out is None else out + term
        assert out is not None
        return out

    return g_point


def blended_metric(atlas: AtlasSpec) -> MetricSpec:
    r"""The convex-combination-of-region-metrics :class:`MetricSpec` (torch)."""
    return MetricSpec(g_point=_blended_g_point(atlas), dim=atlas.dim, name=atlas.name)


def atlas_manifold(atlas: AtlasSpec) -> ManifoldSpec:
    r"""A one-chart :class:`ManifoldSpec` carrying the blended atlas metric (torch)."""
    return ManifoldSpec(name=atlas.name, dim=atlas.dim, metric=blended_metric(atlas))


__all__ = ["atlas_manifold", "blended_metric"]
