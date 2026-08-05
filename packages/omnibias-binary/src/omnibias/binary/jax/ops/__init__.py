# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""JAX quantization-gradient operator surface."""

from __future__ import annotations

from omnibias.binary.jax.ops.quantize import (
    binarize,
    binarize01,
    heaviside,
    kbit_quantize,
    riccati_sigmoid_derivative,
    riccati_tanh_derivative,
    ternarize,
)
from omnibias.binary.jax.ops.surrogate import (
    binarize_curvature,
    curvature_corrected_slope,
    surrogate_jet,
    surrogate_tower,
)

__all__ = [
    "binarize",
    "binarize01",
    "binarize_curvature",
    "curvature_corrected_slope",
    "heaviside",
    "kbit_quantize",
    "riccati_sigmoid_derivative",
    "riccati_tanh_derivative",
    "surrogate_jet",
    "surrogate_tower",
    "ternarize",
]
