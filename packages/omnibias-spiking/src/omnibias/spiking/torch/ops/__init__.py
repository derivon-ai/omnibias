# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Torch spiking-neuron operator surface."""

from __future__ import annotations

from omnibias.spiking.torch.ops.neuron import (
    heaviside_spike,
    if_step,
    lif_step,
    surrogate_derivative,
)

__all__ = [
    "heaviside_spike",
    "if_step",
    "lif_step",
    "surrogate_derivative",
]
