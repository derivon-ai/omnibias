# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Strict-conservation cage layers."""

from __future__ import annotations

from omnibias.pinn.jax.cage.conservation import (
    HardBoundaryField,
    energy_conserving_advection,
    enstrophy_conserving_advection,
    make_hard_boundary_field,
    make_mass_flux_potential_field,
)
from omnibias.pinn.jax.cage.incompressible import (
    HelmholtzProjectionField,
    StreamfunctionField,
    VectorPotentialField,
    coulomb_gauge_loss,
    helmholtz_gauge_loss,
    is_cage_field,
    make_helmholtz_projection_field,
    make_streamfunction_field,
    make_vector_potential_field,
)

__all__ = [
    "HardBoundaryField",
    "HelmholtzProjectionField",
    "StreamfunctionField",
    "VectorPotentialField",
    "coulomb_gauge_loss",
    "energy_conserving_advection",
    "enstrophy_conserving_advection",
    "helmholtz_gauge_loss",
    "is_cage_field",
    "make_hard_boundary_field",
    "make_helmholtz_projection_field",
    "make_mass_flux_potential_field",
    "make_streamfunction_field",
    "make_vector_potential_field",
]
