# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Pullback metric of a learned chart (jax): ``g = J^T h J``.

Bit-identical twin of :mod:`omnibias.geometry.torch.ops.pullback`.

Given an immersion ``phi: R^d -> R^n`` (a :class:`ChartSpec`), the induced
(pullback) metric on the ``d``-dimensional domain is

.. math::

    g_{ab}(x) = \\sum_{i,j} h_{ij}(\\varphi(x))\\,
        \\frac{\\partial \\varphi^i}{\\partial x^a}\\,
        \\frac{\\partial \\varphi^j}{\\partial x^b},

with ``J = d phi / dx`` of shape ``(n, d)`` and ``h`` the ambient metric
(Euclidean identity by default, giving ``g = J^T J``). The Jacobian is taken by
forward-mode autodiff, so the metric is exact for analytic / neural charts.

:func:`metric_spec_from_chart` wraps this into a
:class:`~omnibias.geometry._core.manifold.MetricSpec`, after which the connection
and curvature operators consume it unchanged (they only read
``manifold.metric.g_point``).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
from jax import Array
from omnibias.geometry._core.manifold import MetricSpec

if TYPE_CHECKING:  # pragma: no cover
    from collections.abc import Callable

    from omnibias.geometry._core.charts import ChartSpec


def _g_point_from_chart(chart: ChartSpec) -> Callable[[Array], Array]:
    """Build the per-point pullback metric ``x -> g(x)`` of shape ``(d, d)``."""
    phi = chart.phi
    ambient = chart.ambient_metric

    def g_point(x: Array) -> Array:
        jac = jax.jacfwd(phi)(x)  # (n, d)
        if ambient is None:
            return jac.T @ jac
        h = ambient(phi(x))  # (n, n)
        return jac.T @ h @ jac

    return g_point


def metric_spec_from_chart(chart: ChartSpec) -> MetricSpec:
    """Pullback :class:`MetricSpec` induced by the immersion ``chart.phi``."""
    return MetricSpec(
        g_point=_g_point_from_chart(chart),
        dim=chart.domain_dim,
        name=f"pullback[{chart.name}]",
    )


def pullback_metric(coords: Array, chart: ChartSpec) -> Array:
    """Pullback metric ``g_ab`` of shape ``(B, d, d)`` evaluated at ``coords``."""
    return jax.vmap(_g_point_from_chart(chart))(coords)


# re-exported so callers can build immersions without importing the schema path
def euclidean_ambient_metric(dim: int) -> Callable[[Array], Array]:
    """Constant Euclidean ambient metric ``h = I_n`` (a convenience builder)."""

    def h(y: Array) -> Array:
        return jnp.eye(dim, dtype=y.dtype)

    return h


__all__ = ["euclidean_ambient_metric", "metric_spec_from_chart", "pullback_metric"]
