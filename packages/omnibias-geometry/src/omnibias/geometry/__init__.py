# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""omnibias-geometry: differential geometry on manifolds.

Metric tensor, Christoffel symbols, covariant derivative, the Laplace-Beltrami
operator, Riemann / Ricci / scalar curvature, geodesics, and exterior calculus
(``d``, wedge, Hodge star, codifferential), built on the ``omnibias-fields``
substrate with cross-backend (torch + jax) parity.

The pure-Python schemas (:class:`MetricSpec`, :class:`ManifoldSpec`) live in
:mod:`omnibias.geometry._core`; the backend ops live in
``omnibias.geometry.torch`` and ``omnibias.geometry.jax``.

The :mod:`omnibias.geometry.gauge` submodule (**alpha**, folded in from the former
standalone ``omnibias-gauge`` package) is the non-abelian extension: Lie algebras,
Lie-algebra-valued forms, gauge connections, the field strength ``F = dA + g[A, A]``,
the Yang-Mills operator, characteristic classes, and an SU(2) lattice Monte-Carlo.
It is imported on demand and is not part of the eager ``omnibias.geometry`` API.

.. important::

    **Bit-parity with the PyTorch twin requires 64-bit JAX** --
    ``jax.config.update("jax_enable_x64", True)`` before the first JAX array is
    created (or ``JAX_ENABLE_X64=1``). JAX otherwise truncates to ``float32``
    while PyTorch uses ``float64``, so the twins stay internally consistent but
    agree only to ``float32`` tolerance. Where a value feeds a threshold, a
    rounding step or an ``argmax``, that is enough to change the decision rather
    than just the last digits. See :mod:`omnibias.jax.precision`.
"""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError as _PkgNotFound
from importlib.metadata import version as _pkg_version

from omnibias.geometry._core import (
    AmbientMetricFn,
    ChartFn,
    ChartSpec,
    DifferentialForm,
    ManifoldSpec,
    MetricFn,
    MetricSpec,
)

try:
    __version__ = _pkg_version("omnibias-geometry")
except _PkgNotFound:  # pragma: no cover - bare source checkout
    __version__ = "0.0.0+unknown"

# Founding-idea lineage (see docs/theory.md "Two senses of collapse").
__lineage__ = "bias collapse"

__all__ = [
    "AmbientMetricFn",
    "ChartFn",
    "ChartSpec",
    "DifferentialForm",
    "ManifoldSpec",
    "MetricFn",
    "MetricSpec",
    "__lineage__",
    "__version__",
]
