# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Pullback metric of a learned chart (torch): ``g = J^T h J``.

Bit-identical twin of :mod:`omnibias.geometry.jax.ops.pullback`.

Given an immersion ``phi: R^d -> R^n`` (a :class:`ChartSpec`), the induced
(pullback) metric on the ``d``-dimensional domain is

.. math::

    g_{ab}(x) = \\sum_{i,j} h_{ij}(\\varphi(x))\\,
        \\frac{\\partial \\varphi^i}{\\partial x^a}\\,
        \\frac{\\partial \\varphi^j}{\\partial x^b},

with ``J = d phi / dx`` of shape ``(n, d)`` and ``h`` the ambient metric
(Euclidean identity by default, giving ``g = J^T J``). The Jacobian comes from
``chart.jacobian`` when supplied, otherwise from forward-mode autodiff of
``phi``. Provider shape is checked; its mathematical agreement with the chart
is the caller's contract, recorded by ``chart.derivative_provenance``. Both
paths preserve live backend graphs, and further metric derivatives use autodiff.

:func:`metric_spec_from_chart` wraps this into a
:class:`~omnibias.geometry._core.manifold.MetricSpec`, after which the connection
and curvature operators consume it unchanged (they only read
``manifold.metric.g_point``).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import torch
from omnibias.geometry._core.manifold import MetricSpec
from torch import Tensor
from torch.func import jacfwd, vmap

if TYPE_CHECKING:  # pragma: no cover
    from collections.abc import Callable

    from omnibias.geometry._core.charts import ChartSpec


def _g_point_from_chart(chart: ChartSpec[Tensor]) -> Callable[[Tensor], Tensor]:
    """Build the per-point pullback metric ``x -> g(x)`` of shape ``(d, d)``."""
    phi = chart.phi
    ambient = chart.ambient_metric

    def g_point(x: Tensor) -> Tensor:
        jac = cast(Tensor, chart.jacobian(x) if chart.jacobian is not None else jacfwd(phi)(x))
        expected_shape = (chart.ambient_dim, chart.domain_dim)
        if jac.shape != expected_shape:
            raise ValueError(
                f"chart Jacobian must have shape {expected_shape}, got {tuple(jac.shape)}"
            )
        if ambient is None:
            return jac.transpose(-2, -1) @ jac
        h = cast(Tensor, ambient(phi(x)))  # (n, n)
        return jac.transpose(-2, -1) @ h @ jac

    return g_point


def metric_spec_from_chart(chart: ChartSpec[Tensor]) -> MetricSpec[Tensor]:
    """Pullback :class:`MetricSpec` induced by the immersion ``chart.phi``."""
    return MetricSpec(
        g_point=_g_point_from_chart(chart),
        dim=chart.domain_dim,
        name=f"pullback[{chart.name}]",
    )


def pullback_metric(coords: Tensor, chart: ChartSpec[Tensor]) -> Tensor:
    """Pullback metric ``g_ab`` of shape ``(B, d, d)`` evaluated at ``coords``."""
    return vmap(_g_point_from_chart(chart))(coords)


def euclidean_ambient_metric(dim: int) -> Callable[[Tensor], Tensor]:
    """Constant Euclidean ambient metric ``h = I_n`` (a convenience builder)."""

    def h(y: Tensor) -> Tensor:
        return torch.eye(dim, dtype=y.dtype, device=y.device)

    return h


__all__ = ["euclidean_ambient_metric", "metric_spec_from_chart", "pullback_metric"]
