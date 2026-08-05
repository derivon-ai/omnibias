# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Backend-agnostic geometry schemas (pure Python: no torch / jax)."""

from __future__ import annotations

from omnibias.geometry._core.charts import (
    AmbientMetricFn,
    ChartFn,
    ChartSpec,
)
from omnibias.geometry._core.forms import (
    DifferentialForm,
    interior_product,
    permutation_sign,
    sorted_index_sets,
    wedge,
)
from omnibias.geometry._core.integration_core import pullback_form_components
from omnibias.geometry._core.manifold import ManifoldSpec, MetricFn, MetricSpec

__all__ = [
    "AmbientMetricFn",
    "ChartFn",
    "ChartSpec",
    "DifferentialForm",
    "ManifoldSpec",
    "MetricFn",
    "MetricSpec",
    "interior_product",
    "permutation_sign",
    "pullback_form_components",
    "sorted_index_sets",
    "wedge",
]
