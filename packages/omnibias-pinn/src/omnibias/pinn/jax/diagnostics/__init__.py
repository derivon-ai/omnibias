# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Diagnostics for the jax backend (twin of the torch module)."""

from __future__ import annotations

from omnibias.pinn._core.diagnostics import (
    forecast_horizon,
    power_spectrum_per_d,
    relative_l2_per_time,
    spectral_fidelity,
)
from omnibias.pinn.jax.diagnostics.field_stability import (
    autograd_phase_check,
    derivative_stability,
)

__all__ = [
    "autograd_phase_check",
    "derivative_stability",
    "forecast_horizon",
    "power_spectrum_per_d",
    "relative_l2_per_time",
    "spectral_fidelity",
]
