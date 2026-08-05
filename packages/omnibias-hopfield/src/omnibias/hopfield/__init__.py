# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""omnibias-hopfield: modern Hopfield networks and attention-as-operator.

Continuous modern Hopfield retrieval (Ramsauer et al., 2020) and scaled
dot-product attention share the same softmax kernel. This package exposes
the retrieval update, Hopfield energy, and multi-query attention with
**closed-form** log-sum-exp Jacobian and Hessian routed through the shared
:mod:`omnibias.struct` ``lse_beta`` path (temperature collapse; no autodiff
and no duplicated soft-max math).

Backend ops live under ``omnibias.hopfield.torch`` and ``omnibias.hopfield.jax``.
"""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError as _PkgNotFound
from importlib.metadata import version as _pkg_version

try:
    __version__ = _pkg_version("omnibias-hopfield")
except _PkgNotFound:  # pragma: no cover - bare source checkout
    __version__ = "0.0.0+unknown"

# Founding-idea lineage (see docs/theory.md "Two senses of collapse").
__lineage__ = "temperature collapse"

__all__ = ["__lineage__", "__version__"]
