# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Backend-agnostic chart / immersion schema (pure Python: no torch / jax).

A :class:`ChartSpec` describes an *immersion* (a "learned chart")

.. math::

    \\varphi : \\mathbb{R}^d \\to \\mathbb{R}^n, \\qquad d \\le n,

by a *per-point* callable ``phi(x)`` mapping a single domain point ``x`` of
shape ``(d,)`` to its ambient image of shape ``(n,)``. Such a map induces a
Riemannian metric on the domain -- the *pullback metric*

.. math::

    g = J^{\\top} h\\, J, \\qquad J = \\partial \\varphi / \\partial x \\in
    \\mathbb{R}^{n \\times d},

where ``h`` is the ambient metric (Euclidean identity by default). The backend
ops (:mod:`omnibias.geometry.jax.ops.pullback`,
:mod:`omnibias.geometry.torch.ops.pullback`) turn a ``ChartSpec`` into a
:class:`~omnibias.geometry._core.manifold.MetricSpec`, after which every existing
geometry operator (Christoffel, Riemann/Ricci/scalar curvature, Laplace-Beltrami,
geodesics) works on the learned manifold unchanged.

Honesty note
------------
``J`` is obtained by forward-mode autodiff of ``phi``. For an analytic (or
neural-network) chart this is the *exact* Jacobian, not a finite-difference
approximation. ``phi`` must therefore be written with backend ops so it is
``vmap``- and autodiff-compatible.

This module is pure Python (no torch / jax): it only stores callables and
metadata, exactly like :class:`omnibias.geometry._core.manifold.MetricSpec`.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Generic, TypeVar

TensorT = TypeVar("TensorT")

#: A per-point chart callable: ``x`` of shape ``(d,)`` -> image of shape ``(n,)``.
ChartFn = Callable[[Any], Any]
#: An optional ambient-metric callable: ``y`` of shape ``(n,)`` -> ``(n, n)``.
AmbientMetricFn = Callable[[Any], Any]


@dataclass(frozen=True)
class ChartSpec(Generic[TensorT]):
    """An immersion ``phi: R^d -> R^n`` whose pullback induces a metric.

    Parameters
    ----------
    phi
        Callable mapping a single domain point ``x`` of shape ``(d,)`` to its
        ambient image of shape ``(n,)``. It must be written with backend ops so
        it is ``vmap``- and autodiff-compatible.
    domain_dim
        The chart (manifold) dimension ``d``.
    ambient_dim
        The ambient dimension ``n``. Must satisfy ``n >= d`` for the pullback
        metric to be (generically) non-degenerate.
    ambient_metric
        Optional callable mapping an ambient point ``y`` of shape ``(n,)`` to the
        ambient metric matrix ``h`` of shape ``(n, n)``. ``None`` (default) means
        the Euclidean identity, i.e. ``g = J^T J``.
    name
        Human-readable label (e.g. ``"sphere_S2"``).
    """

    phi: ChartFn
    domain_dim: int
    ambient_dim: int
    ambient_metric: AmbientMetricFn | None = field(default=None)
    name: str = "chart"

    def __post_init__(self) -> None:
        if self.domain_dim < 1:
            raise ValueError(f"domain_dim must be >= 1, got {self.domain_dim}")
        if self.ambient_dim < 1:
            raise ValueError(f"ambient_dim must be >= 1, got {self.ambient_dim}")
        if self.ambient_dim < self.domain_dim:
            raise ValueError(
                f"ambient_dim {self.ambient_dim} must be >= domain_dim "
                f"{self.domain_dim} for a non-degenerate pullback metric"
            )


__all__ = ["AmbientMetricFn", "ChartFn", "ChartSpec", "TensorT"]
