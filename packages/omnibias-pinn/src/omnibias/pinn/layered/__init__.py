# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""1-D layered transfer stacks (theory 02-11, gated).

Distinct from ``omnibias.geometry.gauge.transfer``. Continuum claim is False.
"""

from __future__ import annotations

from omnibias.core.transfer import (
    BandGapCertificate,
    Layer,
    bloch_dispersion,
    certified_band_gap,
    quarter_wave_stack,
    reflection_transmission,
    stack_matrix,
    unitarity_residual,
)

__all__ = [
    "BandGapCertificate",
    "Layer",
    "TransferStack",
    "bloch_dispersion",
    "certified_band_gap",
    "quarter_wave_stack",
    "reflection_transmission",
    "stack_matrix",
    "unitarity_residual",
]


def __getattr__(name: str) -> object:
    if name == "TransferStack":
        from omnibias.pinn.layered.torch import TransferStack

        return TransferStack
    raise AttributeError(f"module {__name__!r} has no attribute {name}")
