# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Differentiable spectral graph operators and continuous relaxations.

PyTorch/JAX kernels provide Laplacians, spectral embeddings, graph heat kernels,
Sinkhorn assignments, soft sorting, and top-k. Spectral algebra is exact for
the supplied matrix up to numerical error; soft assignment is a relaxation.
Discrete decisions and their optimality certificates belong to consumers.

Use matching float64 inputs and enable ``jax_enable_x64`` for backend parity.
Framework reductions can differ in rounding; identical bits require matching
activation values and operation order, not merely shared coefficients.
"""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError as _PkgNotFound
from importlib.metadata import version as _pkg_version

try:
    __version__ = _pkg_version("omnibias-graph")
except _PkgNotFound:  # pragma: no cover - bare source checkout
    __version__ = "0.0.0+unknown"

# Limit family exposed as package metadata.
__lineage__ = "temperature collapse"

__all__ = ["__lineage__", "__version__"]
