# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""JAX Hopfield / attention operator surface."""

from __future__ import annotations

from omnibias.hopfield.jax.ops.hopfield import (
    attention,
    hopfield_energy,
    logsumexp_hessian,
    logsumexp_jacobian,
    logsumexp_value,
    modern_hopfield_retrieve,
    softmax,
)
from omnibias.hopfield.jax.ops.jet_hopfield import JetHopfieldConfig, jet_hopfield_retrieve

__all__ = [
    "JetHopfieldConfig",
    "attention",
    "hopfield_energy",
    "jet_hopfield_retrieve",
    "logsumexp_hessian",
    "logsumexp_jacobian",
    "logsumexp_value",
    "modern_hopfield_retrieve",
    "softmax",
]
