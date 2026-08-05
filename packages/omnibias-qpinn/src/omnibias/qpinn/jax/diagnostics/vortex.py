# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Vortex diagnostics (jax-friendly re-exports). NumPy detector under the hood."""

from __future__ import annotations

from omnibias.qpinn._core.vortex import (
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
    "detect_vortices",
    "detect_vortices_full",
    "feynman_vortex_count",
    "thomas_fermi_density_2d",
    "thomas_fermi_mu_2d",
    "thomas_fermi_radius_2d",
]
