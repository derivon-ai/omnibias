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
from omnibias.pinn.torch.cage.fluxform import (
    FluxFormField,
    antisymmetric_pairs,
    make_flux_form_field,
)
from omnibias.pinn.torch.cage.incompressible import (
    HelmholtzProjectionField,
    StreamfunctionField,
    VectorPotentialField,
    coulomb_gauge_loss,
    helmholtz_gauge_loss,
    is_cage_field,
)
from omnibias.pinn.torch.cage.integral import (
    IntegralConservationField,
    make_integral_conservation_field,
)

__all__ = [
    "FluxFormField",
    "HardBoundaryField",
    "HelmholtzProjectionField",
    "IntegralConservationField",
    "MassFluxPotentialField",
    "StreamfunctionField",
    "VectorPotentialField",
    "antisymmetric_pairs",
    "coulomb_gauge_loss",
    "energy_conserving_advection",
    "enstrophy_conserving_advection",
    "helmholtz_gauge_loss",
    "is_cage_field",
    "make_flux_form_field",
    "make_integral_conservation_field",
]
