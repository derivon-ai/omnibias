# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""omnibias-spiking: LIF / IF neuron primitives.

Spiking-neuron single-step kernels whose forward pass is a hard Heaviside
threshold and whose backward pass uses an exact closed-form surrogate
gradient — the derivative of a smooth dictionary activation from the
omnibias derivative tower (no ad-hoc surrogate code path).

Backend ops live under ``omnibias.spiking.torch`` and ``omnibias.spiking.jax``.
"""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError as _PkgNotFound
from importlib.metadata import version as _pkg_version

try:
    __version__ = _pkg_version("omnibias-spiking")
except _PkgNotFound:  # pragma: no cover - bare source checkout
    __version__ = "0.0.0+unknown"

# Founding-idea lineage (see docs/theory.md "Two senses of collapse").
__lineage__ = "bias collapse"

__all__ = ["__lineage__", "__version__"]
