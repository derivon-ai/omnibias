# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Strict-conservation cage layers."""

from __future__ import annotations

from omnibias.pinn.torch.cage.conservation import (
    HardBoundaryField,
    MassFluxPotentialField,
    energy_conserving_advection,
    enstrophy_conserving_advection,
)
from omnibias.pinn.torch.cage.incompressible import (
    HelmholtzProjectionField,
    StreamfunctionField,
    VectorPotentialField,
    coulomb_gauge_loss,
    helmholtz_gauge_loss,
    is_cage_field,
)

__all__ = [
    "HardBoundaryField",
    "HelmholtzProjectionField",
    "MassFluxPotentialField",
    "StreamfunctionField",
    "VectorPotentialField",
    "coulomb_gauge_loss",
    "energy_conserving_advection",
    "enstrophy_conserving_advection",
    "helmholtz_gauge_loss",
    "is_cage_field",
]
