# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Backend-agnostic manifold / metric schemas.

A :class:`MetricSpec` describes a Riemannian (or pseudo-Riemannian) metric on a
single coordinate chart by a *per-point* callable ``g_point(x)`` returning the
``(d, d)`` metric matrix at a single point ``x`` of shape ``(d,)``. The backend
ops batch it with ``vmap`` and obtain metric derivatives by forward-mode autodiff
of ``g_point``.

Honesty note
------------
For an *analytic* metric, autodiff of ``g_point`` yields the metric derivatives
*exactly* (to machine precision) -- this is not a finite-difference
approximation. It is, however, autodiff rather than a sigma-tower closed form;
the field-function derivatives that the geometry ops consume (``grad f``,
``hess f``) remain closed-form via the activation derivative tower. See
``GEOMETRY_DERIVATIONS.md``.

This module is pure Python (no torch / jax): it only stores the callable and
metadata, exactly like :class:`omnibias.core.spec.ActivationSpec`.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Generic, TypeVar

TensorT = TypeVar("TensorT")

#: A per-point metric callable: ``x`` of shape ``(d,)`` -> metric of shape ``(d, d)``.
MetricFn = Callable[[Any], Any]


@dataclass(frozen=True)
class MetricSpec(Generic[TensorT]):
    """A Riemannian / pseudo-Riemannian metric on one coordinate chart.

    Parameters
    ----------
    g_point
        Callable mapping a single coordinate point ``x`` of shape ``(d,)`` to
        the symmetric positive-definite (or, for ``signature`` with negative
        entries, indefinite) metric matrix ``g_ij`` of shape ``(d, d)``. It must
        be written with backend ops so it is ``vmap``- and autodiff-compatible.
    dim
        The manifold dimension ``d``.
    name
        Human-readable label.
    signature
        Optional metric signature, e.g. ``(1, 1, 1)`` (Riemannian) or
        ``(1, -1, -1, -1)`` (Lorentzian). Empty means "assume Riemannian".
    """

    g_point: MetricFn
    dim: int
    name: str = "metric"
    signature: tuple[int, ...] = field(default=())

    def __post_init__(self) -> None:
        if self.dim < 1:
            raise ValueError(f"dim must be >= 1, got {self.dim}")
        if self.signature and len(self.signature) != self.dim:
            raise ValueError(
                f"signature length {len(self.signature)} != dim {self.dim}"
            )


@dataclass(frozen=True)
class ManifoldSpec(Generic[TensorT]):
    """A manifold: a name, a dimension, and a metric on one chart.

    Parameters
    ----------
    name
        Human-readable label (e.g. ``"sphere_S2"``).
    dim
        The manifold dimension ``d`` (must match ``metric.dim``).
    metric
        The :class:`MetricSpec` for the (single) chart.

    Notes
    -----
    Multi-chart atlases (with transition maps) are intentionally deferred; a
    ``ManifoldSpec`` currently describes one chart, which is sufficient for the
    closed-form local operators (Laplace-Beltrami, curvature, geodesics).
    """

    name: str
    dim: int
    metric: MetricSpec[TensorT]

    def __post_init__(self) -> None:
        if self.dim != self.metric.dim:
            raise ValueError(
                f"ManifoldSpec.dim {self.dim} != metric.dim {self.metric.dim}"
            )


__all__ = ["ManifoldSpec", "MetricFn", "MetricSpec", "TensorT"]
