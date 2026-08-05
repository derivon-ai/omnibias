# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Diagnostic scalars + soft losses for the jax backend."""

from __future__ import annotations

from omnibias.qpinn.jax.diagnostics.current import (
    continuity_residual,
    current_divergence,
    probability_current,
)
from omnibias.qpinn.jax.diagnostics.energy import (
    energy_variance,
    expectation_value,
    expected_energy,
)
from omnibias.qpinn.jax.diagnostics.norm import (
    norm_drift,
    norm_squared,
)
from omnibias.qpinn.jax.diagnostics.vortex import (
    VortexDetection,
    detect_vortices,
    detect_vortices_full,
    feynman_vortex_count,
    thomas_fermi_density_2d,
    thomas_fermi_mu_2d,
    thomas_fermi_radius_2d,
)

__all__ = [
    "VortexDetection",
    "continuity_residual",
    "current_divergence",
    "detect_vortices",
    "detect_vortices_full",
    "energy_variance",
    "expectation_value",
    "expected_energy",
    "feynman_vortex_count",
    "norm_drift",
    "norm_squared",
    "probability_current",
    "thomas_fermi_density_2d",
    "thomas_fermi_mu_2d",
    "thomas_fermi_radius_2d",
]
